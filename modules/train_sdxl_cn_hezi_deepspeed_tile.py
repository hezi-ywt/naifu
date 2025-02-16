import safetensors
import torch
import os
import lightning as pl
from omegaconf import OmegaConf
from common.utils import get_class, get_latest_checkpoint, load_torch_file
from common.logging import logger
from modules.sdxl_model_cn import StableDiffusionModelCN
from modules.scheduler_utils import apply_snr_weight
from lightning.pytorch.utilities.model_summary import ModelSummary
from torch.utils.data import DataLoader
from modules.sdxl_utils import get_hidden_states_sdxl 

import random
from IndexKits.index_kits.sampler import DistributedSamplerWithStartIndex, BlockDistributedSampler
from data_loader.arrow_load_stream_for_cn_tile import TextImageArrowStream
import logging
logger = logging.getLogger("hezi")

def setup(fabric: pl.Fabric, config: OmegaConf) -> tuple:
    model = SupervisedFineTune(
        config=config, 
        device=fabric.device
        
    )
    dataset_class = get_class(config.dataset.get("name", "data.AspectRatioDataset"))

    # dataset = dataset_class(
    #     batch_size=config.trainer.batch_size,
    #     rank=fabric.global_rank,
    #     dtype=torch.float32,
    #     **config.dataset,
    # )
    # dataloader = dataset.init_dataloader()

    world_size = fabric.world_size
    dataset = TextImageArrowStream(args="args",
                                   resolution=config.trainer.resolution,
                                   random_flip=config.dataset.random_flip,
                                   log_fn=logger.info,
                                   index_file=config.dataset.index_file,
                                   multireso=config.dataset.multireso,
                                   batch_size=config.trainer.batch_size,
                                   world_size=world_size,
                                #    random_shrink_size_cond=config.trainer.batch_size.random_shrink_size_cond,
                                #    merge_src_cond=config.trainer.batch_size.merge_src_cond,
                                #    uncond_p=args.uncond_p,
                                #    text_ctx_len=args.text_len,
                                #    tokenizer=tokenizer,
                                #    uncond_p_t5=args.uncond_p_t5,
                                #    text_ctx_len_t5=args.text_len_t5,
                                #    tokenizer_t5=tokenizer_t5,
                                   )

    if config.dataset.multireso:
        sampler = BlockDistributedSampler(dataset, num_replicas=world_size, rank=fabric.global_rank, seed=config.trainer.seed,
                                          shuffle=True, drop_last=True, batch_size=config.trainer.batch_size)
    else:
        sampler = DistributedSamplerWithStartIndex(dataset, num_replicas=world_size, rank=fabric.global_rank, seed=config.trainer.seed,
                                                   shuffle=False, drop_last=True)
        
    dataloader = DataLoader(dataset, batch_size=config.trainer.batch_size, shuffle=False, sampler=sampler,
                        num_workers=config.dataset.num_workers, pin_memory=True, drop_last=True)
    
    params_to_optim = [{'params': model.model.parameters()}]

        

    optim_param = config.optimizer.params
    optimizer = get_class(config.optimizer.name)(
        params_to_optim, **optim_param
    )
    scheduler = None
    if config.get("scheduler"):
        scheduler = get_class(config.scheduler.name)(
            optimizer, **config.scheduler.params
        )
    
    if config.trainer.get("resume"):
        latest_ckpt = get_latest_checkpoint(config.trainer.checkpoint_dir)
        remainder = {}
        if latest_ckpt:
            logger.info(f"Loading weights from {latest_ckpt}")
            remainder = sd = load_torch_file(ckpt=latest_ckpt, extract=False)
            if latest_ckpt.endswith(".safetensors"):
                remainder = safetensors.safe_open(latest_ckpt, "pt").metadata()
            model.load_state_dict(sd.get("state_dict", sd))
            config.global_step = remainder.get("global_step", 0)
            config.current_epoch = remainder.get("current_epoch", 0)
        
    model.vae.to(torch.float32)
    if fabric.is_global_zero and os.name != "nt":
        print(f"\n{ModelSummary(model, max_depth=1)}\n")
        
    if hasattr(fabric.strategy, "_deepspeed_engine"):
        model, optimizer = fabric.setup(model, optimizer)
        model.get_module = lambda: model
        model._deepspeed_engine = fabric.strategy._deepspeed_engine
    elif hasattr(fabric.strategy, "_fsdp_kwargs"):
        model, optimizer = fabric.setup(model, optimizer)
        model.get_module = lambda: model
        model._fsdp_engine = fabric.strategy
    else:
        model.model, optimizer = fabric.setup(model.model, optimizer)

        
    dataloader = fabric.setup_dataloaders(dataloader)
    model._fabric_wrapped = fabric
    return model, dataset, dataloader, optimizer, scheduler

def get_sigmas(sch, timesteps, n_dim=4, dtype=torch.float32, device="cuda:0"):
    sigmas = sch.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = sch.timesteps.to(device)
    timesteps = timesteps.to(device)

    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

class SupervisedFineTune(StableDiffusionModelCN):
    def forward(self, batch):

        
        advanced = self.config.get("advanced", {})
        with torch.no_grad():
            

            self.vae.to(self.target_device)
            model_dtype = next(self.model.parameters()).dtype
            latents = self.vae.encode(batch['pixels']).latent_dist.sample().to(self.target_device).to(model_dtype)

            if torch.any(torch.isnan(latents)):
                logger.info("NaN found in latents, replacing with zeros")
            latents = latents * self.vae.config.scaling_factor
            # latents *= self.vae_scale_factor
            
            # hidden_states1, hidden_states2, pool2 = get_hidden_states_sdxl(batch["prompts"], self.text_encoders, self.tokenizers, self.target_device, self.weight_dtype)
            # hidden_states = torch.cat([hidden_states1, hidden_states2], dim=2).to(self.target_device).to(model_dtype)
            # encoder_hidden_states = [hidden_states, pool2]
            if random.random() < 0.6:
                encoder_hidden_states = self.encode_text(batch["prompts"])
            else:
                encoder_hidden_states = self.encode_text_(batch["prompts"])
            noise = torch.randn_like(latents).to(self.target_device).to(model_dtype)
            add_time_ids = torch.cat(
                [self.compute_time_ids(s, c, t) for s, c, t in zip(batch["original_size_as_tuple"], batch["crop_coords_top_left"], batch['target_size_as_tuple'])]
            ).to(self.device)

            if advanced.get("offset_noise") and random.random() < 0.5:
                offset = torch.randn(latents.shape[0], latents.shape[1], 1, 1, device=latents.device)
                noise = torch.randn_like(latents) + float(advanced.get("offset_noise_val")) * offset * random.random()

            bsz = latents.shape[0]

            # Sample a random timestep for each image
            timestep_start = advanced.get("timestep_start", 0)
            timestep_end = advanced.get("timestep_end", 1000)
            timestep_sampler_type = advanced.get("timestep_sampler_type", "uniform")


            if timestep_sampler_type == "logit_normal":  
                mu = advanced.get("timestep_sampler_mean", 0)
                sigma = advanced.get("timestep_sampler_std", 1)
                t = torch.sigmoid(mu + sigma * torch.randn(size=(bsz,), device=latents.device))
                timesteps = t * (timestep_end - timestep_start) + timestep_start  # scale to [min_timestep, max_timestep)
                timesteps = timesteps.long()
            else:
                # default impl
                timesteps = torch.randint(
                    low=timestep_start, 
                    high=timestep_end,
                    size=(bsz,),
                    dtype=torch.int64,
                    device=latents.device,
                )

                timesteps = timesteps.long()
            # timesteps = self.get_cubic_timesteps(bsz,latents,timestep_end)
            # t = torch.rand((bsz, ), device=latents.device)

            # timesteps = (1 - t**3) * timestep_end
            # print(timesteps)
            # timesteps = timesteps.long().to(latents.device)
            # print(timesteps)
                            # Cubic sampling to sample a random timestep for each image

            # timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],)).long().to(self.device)
            #(1 - (t /T)^3) * T


            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_model_input = self.noise_scheduler.add_noise(latents, noise, timesteps).to(model_dtype)

            # Predict the noise residual
            
            
        down_block_res_samples, mid_block_res_sample = self.model(
                    noisy_model_input,
                    timesteps,
                    encoder_hidden_states = encoder_hidden_states[0],
                    added_cond_kwargs={"time_ids": add_time_ids, "text_embeds": encoder_hidden_states[1]},
                    controlnet_cond=batch['conditioning_pixels'],
                    return_dict=False,
                )

        if (batch['conditioning_pixels'] == batch['pixels']).all():

            print("conditioning_pixels == pixels")
        # print(batch['conditioning_pixels'])
        # print(batch['pixels'])
        # # 保存每张图像
        # from scrips.test_util import batch_tensor_to_image
        # batch_tensor_to_image(batch['conditioning_pixels'], "output_dir1")
        # batch_tensor_to_image(batch['pixels'], "output_di2")

        
        noise_pred = self.unet(
            noisy_model_input, timesteps,  encoder_hidden_states[0], added_cond_kwargs={"time_ids": add_time_ids, "text_embeds": encoder_hidden_states[1]}, \
            down_block_additional_residuals = down_block_res_samples,
            mid_block_additional_residual = mid_block_res_sample,
            return_dict=False,
            )[0]

        # Get the target for loss depending on the prediction type
        is_v = advanced.get("v_parameterization", False)
        if is_v:
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise
        
        min_snr_gamma = advanced.get("min_snr", False)            
        if min_snr_gamma:
            # do not mean over batch dimension for snr weight or scale v-pred loss
            loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
            loss = loss.mean([1, 2, 3])

            if min_snr_gamma:
                loss = apply_snr_weight(loss, timesteps, self.noise_scheduler, advanced.min_snr_val, is_v)
                
            loss = loss.mean()  # mean over batch dimension
        else:
            loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")
        
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise FloatingPointError("Error infinite or NaN loss detected")

        return loss

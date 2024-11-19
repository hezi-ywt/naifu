from pathlib import Path
import os

from safetensors.torch import safe_open, save_file
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.utils.checkpoint
import lightning as pl


import numpy as np
from modules.scheduler_utils import apply_zero_terminal_snr, cache_snr_values
from common.utils import get_class, load_torch_file, EmptyInitWrapper, get_world_size
from common.logging import logger
import random

from lightning.pytorch.utilities import rank_zero_only
from safetensors.torch import save_file
from modules.config_sdxl_base import model_config
from diffusers.training_utils import EMAModel
from diffusers import (
    
    AutoencoderKL,
    DDPMScheduler,
    ControlNetModel,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    
)

from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection

from models.ip_adapter.ip_adapter import ImageProjModel
from models.ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from models.ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from models.ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor
# define the LightningModule

from modules.sdxl_utils import get_hidden_states_sdxl

class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, unet_added_cond_kwargs, image_embeds):
        ip_tokens = self.image_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=unet_added_cond_kwargs).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        if os.path.splitext(ckpt_path)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(ckpt_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")
    

class IPAdapter_SDXL(pl.LightningModule):
    def __init__(self, config, device) -> None:
        super().__init__()
        self.config = config
        self.target_device = device
        self.init_model()

        
    def build_models(self, init_unet=True, init_vae=True, init_conditioner=True):
        trainer_cfg = self.config.trainer
        config = self.config
        
        

        # # Load scheduler, tokenizer and models.
        # noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        # tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        # text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
        # tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
        # text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
        # vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
        # unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(trainer_cfg.get("image_encoder_path")).to(self.target_device)
        # freeze parameters of models to save more memory

        advanced = trainer_cfg.get("advanced", {})
        sd = StableDiffusionXLPipeline.from_single_file(trainer_cfg.get("sdxl_model_path"))
        self.vae = sd.vae.to(self.target_device)
        self.vae_scale_factor  = self.vae.config.scaling_factor

        unet = sd.unet.to(self.target_device)
        
        self.max_token_length = config.dataset.get("max_token_length", 77)
        self.text_encoder = sd.text_encoder.to(self.target_device)
        self.text_encoder_2 = sd.text_encoder_2.to(self.target_device)
        self.tokenizer = sd.tokenizer
        self.tokenizer_2 = sd.tokenizer_2
        #ip-adapter
        num_tokens = 4
        image_proj_model = ImageProjModel(
            cross_attention_dim=unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=num_tokens,
    )
        
        

        attn_procs = {}
        unet_sd = unet.state_dict()
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=num_tokens)
                attn_procs[name].load_state_dict(weights)
        unet.set_attn_processor(attn_procs)
        adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
        
        self.model = IPAdapter(unet, image_proj_model, adapter_modules, trainer_cfg.pretrained_ip_adapter_path)
        
        self.vae.requires_grad_(False)
        self.model.unet.requires_grad_(False)
        self.model.image_proj_model.requires_grad_(True)
        self.model.adapter_modules.requires_grad_(True)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.vae.eval()
        if config.get("train_image_encoder", False):
            self.image_encoder.requires_grad_(True)
 
        else:
            self.image_encoder.requires_grad_(False)
 
        
        
        # self.model.requires_grad_(True)

        # self.model.train()


    def init_model(self):
        advanced = self.config.get("advanced", {})
        self.build_models()
        self.noise_scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
        )

        # allow custom class
        if self.config.get("noise_scheduler"):
            scheduler_cls = get_class(self.config.noise_scheduler.name)
            self.noise_scheduler = scheduler_cls(**self.config.noise_scheduler.params)

        self.to(self.target_device)

        self.batch_size = self.config.trainer.batch_size
        self.vae_encode_bsz = self.config.get("vae_encode_batch_size", self.batch_size)
        if self.vae_encode_bsz < 0:
            self.vae_encode_bsz = self.batch_size

        if advanced.get("zero_terminal_snr", False):
            apply_zero_terminal_snr(self.noise_scheduler)

        if hasattr(self.noise_scheduler, "alphas_cumprod"):
            cache_snr_values(self.noise_scheduler, self.target_device)

    def get_module(self):
        return self.model

    @torch.no_grad()
    def get_input_ids_(self, caption, tokenizer):
        tokenizer_max_length = self.max_token_length

        input_ids = tokenizer(
            caption, padding="max_length", truncation=True, max_length=tokenizer_max_length, return_tensors="pt"
        ).input_ids
        # print(caption)

        #all_input_ids is tensor

        if tokenizer_max_length > tokenizer.model_max_length:
            input_ids = input_ids.squeeze(0)
            iids_list = []
            if tokenizer.pad_token_id == tokenizer.eos_token_id:
                # v1
                # 77以上の時は "<BOS> .... <EOS> <EOS> <EOS>" でトータル227とかになっているので、"<BOS>...<EOS>"の三連に変換する
                # 1111氏のやつは , で区切る、とかしているようだが　とりあえず単純に
                for i in range(
                    1, tokenizer_max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2
                ):  # (1, 152, 75)
                    ids_chunk = (
                        input_ids[0].unsqueeze(0),
                        input_ids[i : i + tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )
                    ids_chunk = torch.cat(ids_chunk)
                    iids_list.append(ids_chunk)
            else:
                # v2 or SDXL
                # 77以上の時は "<BOS> .... <EOS> <PAD> <PAD>..." でトータル227とかになっているので、"<BOS>...<EOS> <PAD> <PAD> ..."の三連に変換する
                for i in range(1, tokenizer_max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2):
                    ids_chunk = (
                        input_ids[0].unsqueeze(0),  # BOS
                        input_ids[i : i + tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )  # PAD or EOS
                    ids_chunk = torch.cat(ids_chunk)

                    # 末尾が <EOS> <PAD> または <PAD> <PAD> の場合は、何もしなくてよい
                    # 末尾が x <PAD/EOS> の場合は末尾を <EOS> に変える（x <EOS> なら結果的に変化なし）
                    if ids_chunk[-2] != tokenizer.eos_token_id and ids_chunk[-2] != tokenizer.pad_token_id:
                        ids_chunk[-1] = tokenizer.eos_token_id
                    # 先頭が <BOS> <PAD> ... の場合は <BOS> <EOS> <PAD> ... に変える
                    if ids_chunk[1] == tokenizer.pad_token_id:
                        ids_chunk[1] = tokenizer.eos_token_id
                    iids_list.append(ids_chunk)

            input_ids = torch.stack(iids_list)  # 3,77
        return input_ids
    @torch.no_grad()
    def get_input_ids(self, captions, tokenizer):
        input_ids_list = []
        for caption in captions:
            input_ids = self.get_input_ids_(caption, tokenizer)
            input_ids_list.append(input_ids)
        input_ids = torch.stack(input_ids_list)
        return input_ids


    @torch.no_grad()
    def pool_workaround(self,
        text_encoder, last_hidden_state: torch.Tensor, input_ids: torch.Tensor, eos_token_id: int
    ):
        r"""
        workaround for CLIP's pooling bug: it returns the hidden states for the max token id as the pooled output
        instead of the hidden states for the EOS token
        If we use Textual Inversion, we need to use the hidden states for the EOS token as the pooled output

        Original code from CLIP's pooling function:

        \# text_embeds.shape = [batch_size, sequence_length, transformer.width]
        \# take features from the eot embedding (eot_token is the highest number in each sequence)
        \# casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        ]
        """

        # input_ids: b*n,77
        # find index for EOS token

        # Following code is not working if one of the input_ids has multiple EOS tokens (very odd case)
        # eos_token_index = torch.where(input_ids == eos_token_id)[1]
        # eos_token_index = eos_token_index.to(device=last_hidden_state.device)

        # Create a mask where the EOS tokens are
        eos_token_mask = (input_ids == eos_token_id).int()

        # Use argmax to find the last index of the EOS token for each element in the batch
        eos_token_index = torch.argmax(eos_token_mask, dim=1)  # this will be 0 if there is no EOS token, it's fine
        eos_token_index = eos_token_index.to(device=last_hidden_state.device)

        # get hidden states for EOS token
        pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), eos_token_index]

        # apply projection: projection may be of different dtype than last_hidden_state
        pooled_output = text_encoder.text_projection(pooled_output.to(text_encoder.text_projection.weight.dtype))
        pooled_output = pooled_output.to(last_hidden_state.dtype)

        return pooled_output
    @torch.no_grad()
    def get_cubic_timesteps(self,bsz,latents,timestep_end):
        t = torch.rand((bsz, ), device=latents.device)

        timesteps = (1 - t**3) * timestep_end

        timesteps = timesteps.long().to(latents.device)
        return timesteps
        #Cubic sampling to sample a random timestep for each images
    
    def compute_snr(self, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        sqrt_alphas_cumprod = sqrt_alphas_cumprod[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr
    
    @torch.no_grad()
    def compute_time_ids(self, original_size, crops_coords_top_left, target_size):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        
        add_time_ids = torch.cat((original_size, crops_coords_top_left, target_size), dim=0).to(self.target_device)

        add_time_ids = add_time_ids.unsqueeze(0)
        return add_time_ids
    
    @torch.no_grad()
    def encode_text(self,prompt):

        proportion_empty_prompts = 0
        prompt_embeds_list = []

        captions = []
        for caption in prompt:
            
            if random.random() < proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) )

        # captions = prompt
        # print(captions)
        for tokenizer, text_encoder in zip(
            [self.tokenizer, self.tokenizer_2],
            [self.text_encoder, self.text_encoder_2],
            ):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return prompt_embeds, pooled_prompt_embeds
    

    
    
    @torch.no_grad()
    def encode_text_(self, prompt):
        tokenizer = self.tokenizer
        tokenizer_2 = self.tokenizer_2
        max_token_length = self.max_token_length
        text_encoder = self.text_encoder
        text_encoder_2 = self.text_encoder_2

        input_ids1 = self.get_input_ids(prompt, tokenizer).to(self.target_device)

        input_ids2 = self.get_input_ids(prompt, tokenizer_2).to(self.target_device)

        b_size = input_ids1.size()[0]
        input_ids1 = input_ids1.reshape((-1, self.tokenizer.model_max_length))  # batch_size*n, 77
        input_ids2 = input_ids2.reshape((-1, self.tokenizer_2.model_max_length))  # batch_size*n, 77
        b_size = len(prompt)

        # input_ids: b,n,77
        # pool2 = enc_out["text_embeds"]
            # text_encoder1
        enc_out = text_encoder(input_ids1, output_hidden_states=True, return_dict=True)
        hidden_states1 = enc_out["hidden_states"][11]

        # text_encoder2
        enc_out = text_encoder_2(input_ids2, output_hidden_states=True, return_dict=True)
        hidden_states2 = enc_out["hidden_states"][-2]  # penuultimate layer
        unwrapped_text_encoder2 = text_encoder_2 
        pool2 = self.pool_workaround(unwrapped_text_encoder2, enc_out["last_hidden_state"], input_ids2, tokenizer_2.eos_token_id)
        # pool2 = enc_out["hidden_states"][0]
        # b*n, 77, 768 or 1280 -> b, n*77, 768 or 1280
        n_size = 1 if max_token_length is None else max_token_length // 75
        hidden_states1 = hidden_states1.reshape((b_size, -1, hidden_states1.shape[-1]))
        hidden_states2 = hidden_states2.reshape((b_size, -1, hidden_states2.shape[-1]))
        
        # if max_token_length is not None:
        if max_token_length > tokenizer.model_max_length:
            # bs*3, 77, 768 or 1024

            states_list = [hidden_states1[:, 0].unsqueeze(1)]  # <BOS>
            for i in range(1, max_token_length, tokenizer.model_max_length):
                states_list.append(hidden_states1[:, i : i + tokenizer.model_max_length - 2])  
            states_list.append(hidden_states1[:, -1].unsqueeze(1))  # <EOS>
            hidden_states1 = torch.cat(states_list, dim=1)


            
            states_list = [hidden_states2[:, 0].unsqueeze(1)]  # <BOS>

            for i in range(1, max_token_length, tokenizer_2.model_max_length):
                chunk = hidden_states2[:, i : i + tokenizer_2.model_max_length - 2]  
           
                states_list.append(chunk)  
            states_list.append(hidden_states2[:, -1].unsqueeze(1))  

            hidden_states2 = torch.cat(states_list, dim=1)


        pool2 = pool2[::n_size]
        hidden_states = torch.cat([hidden_states1, hidden_states2], dim=2)

        

        hidden_states = hidden_states.reshape((b_size, -1, hidden_states.shape[-1]))
        return hidden_states, pool2

    def _denormlize(self, latents):
        if hasattr(self, "latents_mean"):
            # https://github.com/huggingface/diffusers/pull/7111
            latents = latents * self.latents_std / self.scale_factor + self.latents_mean
        else:
            latents = 1.0 / self.scale_factor * latents
        return latents
    
    def _normliaze(self, latents):
        if hasattr(self, "latents_mean"):
            # https://github.com/huggingface/diffusers/pull/7111
            latents = (latents - self.latents_mean) * self.scale_factor / self.latents_std
        else:
            latents = self.scale_factor * latents
        return latents

    @torch.no_grad()
    def decode_first_stage(self, z):
        
        z = self._denormlize(z)
        with torch.autocast("cuda", enabled=False):
            out = self.first_stage_model.decode(z)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        latents = []
        self.first_stage_model = self.first_stage_model.float()
        with torch.autocast("cuda", enabled=False):
            for i in range(0, x.shape[0], self.vae_encode_bsz):
                o = x[i : i + self.vae_encode_bsz]
                latents.append(self.first_stage_model.encode(o).sample())
        z = torch.cat(latents, dim=0)
        return self._normliaze(z)
    

    def generate_samples(self, logger, current_epoch, global_step):
        if hasattr(self, "_fabric_wrapped"):
            if self._fabric_wrapped.world_size > 2:
                self.generate_samples_dist(logger, current_epoch, global_step)
                return self._fabric_wrapped.barrier()
                
        return self.generate_samples_seq(logger, current_epoch, global_step)

    def generate_samples_dist(self, logger, current_epoch, global_step):
        config = self.config.sampling
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        prompts = list(config.prompts)
        images = []
        size = (config.get("height", 1024), config.get("width", 1024))
        self.model.eval()

        rank = 0
        world_size = self._fabric_wrapped.world_size
        rank = self._fabric_wrapped.global_rank

        local_prompts = prompts[rank::world_size]
        for idx, prompt in tqdm(
            enumerate(local_prompts), desc=f"Sampling (Process {rank})", total=len(local_prompts), leave=False
        ):
            image = self.sample(prompt, size=size, generator=generator)
            image[0].save(
                Path(config.save_dir)
                / f"sample_e{current_epoch}_s{global_step}_p{rank}_{idx}.png"
            )
            images.append((image[0], prompt))

        gathered_images = [None] * world_size
        dist.all_gather_object(gathered_images, images)
        
        self.model.train()
        if rank in [0, -1]:
            all_images = []
            all_prompts = []
            for entry in gathered_images:
                if isinstance(entry, list):
                    entry = entry[0]
                imgs, prompts = entry
                all_prompts.append(prompts)
                all_images.append(imgs)

            if config.use_wandb and logger and "CSVLogger" != logger.__class__.__name__:
                logger.log_image(
                    key="samples", images=all_images, caption=all_prompts, step=global_step
                )
    def on_before_backward(self, loss):
        if self.use_ema:
            if not self.ema_control.cur_decay_value:
                self.ema_control.to(self.device)
            self.ema_control.step(self.controlnet.parameters())

    # def on_train_epoch_end(self):
    #     if self.use_ema:
    #         self.ema_control.save_pretrained(self.trainer.checkpoint_callback.dirpath)
            
    @rank_zero_only
    def generate_samples_seq(self, logger, current_epoch, global_step):
        config = self.config.sampling
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        prompts = list(config.prompts)
        images = []
        size = (config.get("height", 1024), config.get("width", 1024))
        self.model.eval()

        for idx, prompt in tqdm(
            enumerate(prompts), desc="Sampling", total=len(prompts), leave=False
        ):
            image = self.sample(prompt, size=size, generator=generator)
            image[0].save(
                Path(config.save_dir)
                / f"sample_e{current_epoch}_s{global_step}_{idx}.png"
            )
            images.extend(image)

        self.model.train()
        if config.use_wandb and logger and "CSVLogger" != logger.__class__.__name__:
            logger.log_image(
                key="samples", images=images, caption=prompts, step=global_step
            )
    
    
    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")
        
    def save_checkpoint(self, model_path, metadata):
        
        
        
        weight_to_save = None
        if hasattr(self, "_fsdp_engine"):
            from lightning.fabric.strategies.fsdp import _get_full_state_dict_context
            
            weight_to_save = {}    
            world_size = self._fsdp_engine.world_size
            with _get_full_state_dict_context(self.model._forward_module, world_size=world_size):
                unet_weight = self.model._forward_module.state_dict()
                for key in unet_weight.keys():
                    weight_to_save[f"{key}"] = unet_weight[key]
                

        elif hasattr(self, "_deepspeed_engine"):
            from deepspeed import zero
            weight_to_save = {}
            with zero.GatheredParameters(self.model.parameters()):
                unet_weight = self.model.state_dict()
                for key in unet_weight.keys():
                    weight_to_save[f"{key}"] = unet_weight[key]
                

                
        else:
            weight_to_save = self.state_dict()
            
        save_weight = {}
        
        for key in weight_to_save.keys():
            if not key.startswith("unet"):
                #remove the unet prefix
                save_weight[f"{key}"] = weight_to_save[key]
            if key.startswith("adapter_modules"):
                save_weight[f"{key.replace("adapter_modules", "ip_adapter")}"] = weight_to_save[key].clone()
            if key.startswith("image_proj_model"):
                save_weight[f"{key.replace("image_proj_model", "image_proj")}"] = weight_to_save[key].clone()

        
        self._save_checkpoint(model_path, save_weight, metadata)

        if self.config.get("train_image_encoder", False):
            weight_to_save = None
            if hasattr(self, "_fsdp_engine"):
                from lightning.fabric.strategies.fsdp import _get_full_state_dict_context
                
                weight_to_save = {}    
                world_size = self._fsdp_engine.world_size
                with _get_full_state_dict_context(self.image_encoder._forward_module, world_size=world_size):
                    image_encoder_weight = self.image_encoder._forward_module.state_dict()
                    for key in image_encoder_weight.keys():
                        weight_to_save[f"{key}"] = unet_weight[key]
                    
            
            elif hasattr(self, "_deepspeed_engine"):
                from deepspeed import zero
                weight_to_save = {}
                with zero.GatheredParameters(self.image_encoder.parameters()):
                    image_encoder_weight = self.image_encoder.state_dict()
                    for key in unet_weight.keys():
                        weight_to_save[f"{key}"] = unet_weight[key]
                    
            else:
                weight_to_save = self.state_dict()
                
            save_weight = {}
            
            for key in weight_to_save.keys():
                if not key.startswith("unet"):
                    #remove the unet prefix
                    save_weight[f"{key}"] = weight_to_save[key]
                if key.startswith("adapter_modules"):
                    save_weight[f"{key.replace("adapter_modules", "ip_adapter")}"] = weight_to_save[key].clone()
                if key.startswith("image_proj_model"):
                    save_weight[f"{key.replace("image_proj_model", "image_proj")}"] = weight_to_save[key].clone()

            
            self._save_checkpoint(model_path, save_weight, metadata)
                
            


    # def save_checkpoint(self, model_path, metadata):
    #     weight_to_save = None
    #     if hasattr(self, "_fsdp_engine"):
    #         from lightning.fabric.strategies.fsdp import _get_full_state_dict_context
            
    #         weight_to_save = {}    
    #         world_size = self._fsdp_engine.world_size
    #         with _get_full_state_dict_context(self.model._forward_module, world_size=world_size):
    #             unet_weight = self.model._forward_module.state_dict()
    #             for key, value in unet_weight.items():
    #                 weight_to_save[key] = value

    #     elif hasattr(self, "_deepspeed_engine"):
    #         from deepspeed import zero
    #         with zero.GatheredParameters(self.model.parameters(), modifier_rank=None):
    #             weight_to_save = self.model.state_dict()

    #     else:
    #         weight_to_save = self.state_dict()

    #     # 使用 `safetensors` 的 `save_model` 方法
    #     from safetensors.torch import save_model
    #     if weight_to_save:
    #         save_model(weight_to_save, model_path, metadata=metadata)

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")
    
    @rank_zero_only
    def _save_checkpoint(self, model_path, state_dict, metadata):
        cfg = self.config.trainer
        # check if any keys startswith modules. if so, remove the modules. prefix
        # if any([key.startswith("module.") for key in state_dict.keys()]):
        #     state_dict = {
        #         key.replace("module.", ""): value for key, value in state_dict.items()
        #     }

        if cfg.get("save_format") == "safetensors":
            model_path += ".safetensors"
            save_file(state_dict, model_path, metadata=metadata)
        else:
            state_dict = {"state_dict": state_dict, **metadata}
            model_path += ".ckpt"
            torch.save(state_dict, model_path)
        logger.info(f"Saved model to {model_path}")

    # def save_model(ckpt_name, model, force_sync_upload=False):
    #     os.makedirs(args.output_dir, exist_ok=True)
    #     ckpt_file = os.path.join(args.output_dir, ckpt_name)

    #     accelerator.print(f"\nsaving checkpoint: {ckpt_file}")

    #     state_dict = model_util.convert_controlnet_state_dict_to_sd(model.state_dict())

    #     if save_dtype is not None:
    #         for key in list(state_dict.keys()):
    #             v = state_dict[key]
    #             v = v.detach().clone().to("cpu").to(save_dtype)
    #             state_dict[key] = v

    #     if os.path.splitext(ckpt_file)[1] == ".safetensors":
    #         from safetensors.torch import save_file

    #         save_file(state_dict, ckpt_file)
    #     else:
    #         torch.save(state_dict, ckpt_file)

    #     if args.huggingface_repo_id is not None:
    #         huggingface_util.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)

    def forward(self, batch):
        raise NotImplementedError


import safetensors
import torch
import os
import lightning as pl
import torch.nn.functional as F
from omegaconf import OmegaConf
from common.utils import get_class, get_latest_checkpoint, load_torch_file
from common.logging import logger
from lightning.pytorch.utilities.model_summary import ModelSummary
from torch.utils.data import DataLoader
from IndexKits.index_kits.sampler import DistributedSamplerWithStartIndex, BlockDistributedSampler
from data_loader.arrow2_load_stream import TextImageArrowStream

from modules.lumina2_model import Lumina2Model
from models.lumina.transport import create_transport

def setup(fabric: pl.Fabric, config: OmegaConf) -> tuple:
    model_path = config.trainer.model_path
    model = SupervisedFineTune(
        model_path=model_path, 
        config=config, 
        device=fabric.device
    )

    world_size = fabric.world_size
    dataset = TextImageArrowStream(args="args",
                                   resolution=config.trainer.resolution,
                                   random_flip=config.dataset.random_flip,
                                   log_fn=logger.info,
                                   index_file=config.dataset.index_file,
                                   multireso=config.dataset.multireso,
                                   batch_size=config.trainer.batch_size,
                                   world_size=world_size
                                   )

    if config.dataset.multireso:
        sampler = BlockDistributedSampler(dataset, num_replicas=world_size, rank=fabric.global_rank, seed=config.trainer.seed,
                                          shuffle=False, drop_last=True, batch_size=config.trainer.batch_size)
    else:
        sampler = DistributedSamplerWithStartIndex(dataset, num_replicas=world_size, rank=fabric.global_rank, seed=config.trainer.seed,
                                                   shuffle=False, drop_last=True)
        
    dataloader = DataLoader(dataset, batch_size=config.trainer.batch_size, shuffle=False, sampler=sampler,
                        num_workers=config.dataset.num_workers, pin_memory=True, drop_last=True)
    


    params_to_optim = [{"params": model.parameters()}]
    if config.advanced.get("train_text_encoder"):
        lr = config.advanced.get("text_encoder_lr", config.optimizer.params.lr)
        params_to_optim.append({"params": model.text_encoder.parameters(), "lr": lr})

    optim_param = config.optimizer.params
    optimizer = get_class(config.optimizer.name)(params_to_optim, **optim_param)
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


    if fabric.is_global_zero and os.name != "nt":
        print(f"\n{ModelSummary(model, max_depth=1)}\n")

    model.model = fabric.setup(model.model)
    optimizer = fabric.setup_optimizers(optimizer)
    
    dataloader = fabric.setup_dataloaders(dataloader)
    return model, dataset, dataloader, optimizer, scheduler


class SupervisedFineTune(Lumina2Model):
    def get_module(self):
        return self.model
    
    def forward(self, batch):
        for train_res in self.config.advanced.get("train_res", [1024]):
            trans = create_transport(
                "Linear",
                "velocity",
                None,
                None,
                None,
                snr_type=self.config.advanced.snr_type,
                do_shift=not self.config.advanced.no_shift,
                seq_len=(train_res // 16) ** 2,
            )
            images = batch["pixels"].to(self.target_device)
            prompts = batch["prompts"]
            
            with torch.no_grad():
                cap_feats, cap_mask = self.encode_prompt(prompts, self.text_encoder, self.tokenizer, 0.1)

                # 对图像进行VAE编码
                latents = self.encode_images(images)  # [B, C, H, W]
                

                # 编码文本提示
                prompt_embeds, prompt_masks = self.encode_prompt(
                    prompts, 
                    self.text_encoder,
                    self.tokenizer,
                    proportion_empty_prompts=0.1
                )

                ### muti resolution
                # 确保 latents 是 4D 张量 [B, C, H, W]
                if len(latents.shape) == 3:
                    latents = latents.unsqueeze(0)
                latents_mb_256 = self.apply_average_pool(latents, 4)  # 直接对整个批次应用下采样

            model_kwargs = dict(cap_feats=prompt_embeds, cap_mask=prompt_masks)
            loss_dict = trans.training_losses(self.model, latents, model_kwargs)
            loss_dict_256 = trans.training_losses(self.model, latents_mb_256, model_kwargs)

            loss_1024 = loss_dict["loss"].sum() / self.batch_size
            loss_256 = loss_dict_256["loss"].sum() / self.batch_size
            loss = loss_1024 + loss_256

            # 记录训练损失
            self.log("train_loss", loss, prog_bar=True)
            
            return loss

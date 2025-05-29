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
from data_loader.arrow2_load_stream_ import TextImageArrowStream
from data_loader.masked_arrow_stream import MaskedTextImageArrowStream

from modules.lumina2_model import Lumina2Model
from models.lumina.transport import create_transport, calc_masked_training_losses

import random
import math
def setup(fabric: pl.Fabric, config: OmegaConf) -> tuple:
    model_path = config.trainer.model_path
    model = SupervisedFineTune(
        model_path=model_path, 
        config=config, 
        device=fabric.device
    )

    world_size = fabric.world_size
    logger.info(f"loading dataset from {config.dataset.index_file}")
    
    # 检查是否使用带蒙版的数据集
    use_masked_dataset = config.dataset.get("use_masked_dataset", False)
    mask_field = config.dataset.get("mask_field", "mask_path")
    
    if use_masked_dataset:
        logger.info(f"使用带蒙版的数据集，蒙版字段为: {mask_field}")
        dataset = MaskedTextImageArrowStream(
                                   args="args",
                                   resolution=config.trainer.resolution,
                                   random_flip=config.dataset.random_flip,
                                   log_fn=logger.info,
                                   index_file=config.dataset.index_file,
                                   multireso=config.dataset.multireso,
                                   batch_size=config.trainer.batch_size,
                                   world_size=world_size,
                                   mask_field=mask_field
                                   )
    else:
        dataset = TextImageArrowStream(
                                   args="args",
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
                                          shuffle=True, drop_last=True, batch_size=config.trainer.batch_size)
    else:
        sampler = DistributedSamplerWithStartIndex(dataset, num_replicas=world_size, rank=fabric.global_rank, seed=config.trainer.seed,
                                                   shuffle=True, drop_last=True)
        
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

    model.model, optimizer = fabric.setup(model.model, optimizer)
    model.model.mark_forward_method("forward_with_cfg")

    if hasattr(model, "setup"):
        model.setup(fabric)
    
    dataloader = fabric.setup_dataloaders(dataloader)
    return model, dataset, dataloader, optimizer, scheduler


class SupervisedFineTune(Lumina2Model):
    def get_module(self):
        return self.model
    
    
    def get_similar_size(self, base_size):
        #获得差最小的尺寸
        base_size_list = [1024*1024, 512*512, 768*768, 1280*1280, 1536*1536]
        min_diff = float('inf')
        target_size = base_size
        for size in base_size_list:
            diff = abs(size - base_size)
            if diff < min_diff:
                min_diff = diff
                target_size = size
        return target_size
    
    def forward(self, batch):
        # base_size = batch["base_size"][0]
        images = batch["pixels"].to(self.target_device)       
        prompts = batch["prompts"]
        
        # 检查是否有蒙版图片
        masks = batch.get("masks", None)
        mask_weight = self.config.advanced.get("mask_weight", 0.8)  # 从配置中获取蒙版权重参数，默认为0.8
        
        trans = create_transport(
            "Linear",
            "velocity",
            None,
            None,
            None,
            snr_type=self.config.advanced.snr_type,
            do_shift=not self.config.advanced.no_shift,
            seq_len=(1024 // 16) ** 2,
            # seq_len=target_size//(16*16)
        )

        
        # 编码文本提示
        prompt_embeds, prompt_masks = self.encode_prompt(
            prompts, 
            self.text_encoder,
            self.tokenizer,
            proportion_empty_prompts=0.1
        )

        latents = self.encode_images(images)  # [B, C, H, W]
        
        # 如果有蒙版
        latent_masks = None
        if masks is not None:
            # [B, 1, H, W]
            masks = masks.to(self.target_device)
            
            # 下采样
            with torch.no_grad():
                masks_scaled = masks * 2.0 - 1.0
                
                # 
                # 只形状匹配，不关心内容变化，所以只平均池化来下采样
                latent_masks = F.interpolate(
                    masks, 
                    size=(latents.shape[2], latents.shape[3]), 
                    mode='bilinear', 
                    align_corners=False
                )
                
                # 确保值在 [0,1] 范围内
                latent_masks = torch.clamp(latent_masks, 0.0, 1.0)

        model_kwargs = dict(cap_feats=prompt_embeds, cap_mask=prompt_masks)
        
        loss_dict = calc_masked_training_losses(
            trans, 
            self.model, 
            latents, 
            mask=latent_masks, 
            mask_weight=mask_weight, 
            model_kwargs=model_kwargs
        )

        loss_1024 = loss_dict["loss"].sum() / self.batch_size
        loss = loss_1024
        
        # 蒙版内部和外部损失
        if latent_masks is not None and "inside_loss" in loss_dict and "outside_loss" in loss_dict:
            inside_loss = loss_dict["inside_loss"].sum() / self.batch_size
            outside_loss = loss_dict["outside_loss"].sum() / self.batch_size
            self.log("inside_loss", inside_loss, prog_bar=True)
            self.log("outside_loss", outside_loss, prog_bar=True)

        # 记录训练损失
        self.log("train_loss", loss, prog_bar=True)
        self.log("loss_1024", loss_1024, prog_bar=True)
        # self.log("loss_256", loss_256, prog_bar=True)
        
        # 添加梯度裁剪
        if hasattr(self.config.trainer, 'grad_clip') and self.config.trainer.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), 
                max_norm=self.config.trainer.grad_clip
            )
            self.log("grad_norm", grad_norm, prog_bar=True)
            
        return loss

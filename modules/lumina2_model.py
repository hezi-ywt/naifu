from pathlib import Path
import os

from safetensors.torch import safe_open, save_file
from PIL import Image
from tqdm import tqdm
import functools
from functools import partial
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.utils.checkpoint
import lightning as pl
from copy import deepcopy
import shutil
import numpy as np
from modules.scheduler_utils import apply_zero_terminal_snr, cache_snr_values
from common.utils import get_class, load_torch_file, EmptyInitWrapper, get_world_size
from common.logging import logger
import random

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from models.lumina import models
from models.lumina.transport import create_transport
from lightning.pytorch.utilities import rank_zero_only
from safetensors.torch import save_file
from modules.config_sdxl_base import model_config
from diffusers.training_utils import EMAModel
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
)

from transformers import (
    AutoTokenizer,
    AutoModel,
)



# define the LightningModule

from modules.sdxl_utils import get_hidden_states_sdxl


class Lumina2Model(pl.LightningModule):
    def __init__(self, config, device, model_path):
        super().__init__()
        self.config = config
        self.target_device = device
        self.model_path = model_path
        self.init_model()

    def init_model(self):
        self.build_models()
       

    def build_models(self):
        trainer_cfg = self.config.trainer
        config = self.config
        advanced = config.get("advanced", {})
        
        #tokenizer
        if self.config.model.get("tokenizer_path", None):
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.tokenizer_path,
                use_fast=False,
                local_files_only=True
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                subfolder="tokenizer",
                use_fast=False,
                local_files_only=True
            )
        self.tokenizer.padding_side = "right"

        #text_encoder   
        if self.config.model.get("text_encoder_path", None):
            self.text_encoder = AutoModel.from_pretrained(
                self.config.model.text_encoder_path,
                local_files_only=True,
                torch_dtype=torch.bfloat16
            ).cuda()
        else:
            self.text_encoder = AutoModel.from_pretrained(
                self.model_path,
                subfolder="text_encoder",
                local_files_only=True,
                torch_dtype=torch.bfloat16
            ).cuda()


        logger.info(f"text encoder: {type(self.text_encoder)}")
        self.cap_feat_dim = self.text_encoder.config.hidden_size
         

        # Create model:
        self.model = models.__dict__[self.config.model.model_name](
            in_channels=16,
            qk_norm=self.config.model.get("qk_norm", True),
            cap_feat_dim=self.cap_feat_dim,
        ).to(dtype=torch.float16)
        logger.info(f"DiT Parameters: {self.model.parameter_count():,}")
        self.model_patch_size = self.model.patch_size

        if self.config.trainer.get("auto_resume", False) and self.config.trainer.resume is None:
            try:
                existing_checkpoints = os.listdir(self.config.trainer.checkpoint_dir)
                if len(existing_checkpoints) > 0:
                    existing_checkpoints.sort()
                    self.config.trainer.resume = os.path.join(self.config.trainer.checkpoint_dir, existing_checkpoints[-1])
            except Exception:
                    pass
        if self.config.model.get("resume", None) is not None:
            checkpoint_path = os.path.join(
                self.config.model.resume,
                f"consolidated.00-of-01.pth",
            )
            if os.path.exists(checkpoint_path):
                logger.info(f"Resuming model weights from: {checkpoint_path}")
                self.model.load_state_dict(
                    torch.load(checkpoint_path, map_location="cpu"),
                    strict=True,
                )
            else:
                logger.warning(f"Checkpoint not found at: {checkpoint_path}")

        # Note that parameter initialization is done within the DiT constructor
        if self.config.advanced.get("use_ema", True):
            logger.info("Using EMA")
            self.model_ema = deepcopy(self.model)
        if self.config.trainer.get("resume", None) is not None:
            logger.info(f"Resuming model weights from: {self.config.model.resume}")
            self.model.load_state_dict(
                torch.load(
                    os.path.join(
                        self.config.model.resume,
                        f"consolidated.{0:02d}-of-{1:02d}.pth",
                    ),
                    map_location="cpu",
                ),
                strict=True,
            )
            logger.info(f"Resuming ema weights from: {self.config.model.resume}")
            if hasattr(self, "model_ema"):
                self.model_ema.load_state_dict(
                    torch.load(
                        os.path.join(
                            self.config.model.resume,
                            f"consolidated_ema.{0:02d}-of-{1:02d}.pth",
                        ),
                        map_location="cpu",
                    ),
                    strict=True,
                )
        elif self.config.model.get("init_from", None) is not None:
            
            logger.info(f"Initializing model weights from: {self.config.model.init_from}")
            state_dict = torch.load(
                os.path.join(
                    self.config.model.init_from,
                    f"consolidated.{0:02d}-of-{1:02d}.pth",
                ),
                map_location="cpu",
            )

            size_mismatch_keys = []
            model_state_dict = self.model.state_dict()
            for k, v in state_dict.items():
                if k in model_state_dict and model_state_dict[k].shape != v.shape:
                    size_mismatch_keys.append(k)
            for k in size_mismatch_keys:
                del state_dict[k]
            del model_state_dict

            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            missing_keys_ema, unexpected_keys_ema = self.model_ema.load_state_dict(state_dict, strict=False)
            del state_dict
            assert set(missing_keys) == set(missing_keys_ema)
            assert set(unexpected_keys) == set(unexpected_keys_ema)
            logger.info("Model initialization result:")
            logger.info(f"  Size mismatch keys: {size_mismatch_keys}")
            logger.info(f"  Missing keys: {missing_keys}")
            logger.info(f"  Unexpeected keys: {unexpected_keys}")

        # checkpointing (part1, should be called before FSDP wrapping)
        if self.config.trainer.get("checkpointing", False):
            checkpointing_list = list(self.model.get_checkpointing_wrap_module_list())
            if hasattr(self, "model_ema"):
                checkpointing_list_ema = list(self.model_ema.get_checkpointing_wrap_module_list())
        else:
            checkpointing_list = []
            checkpointing_list_ema = []

        # checkpointing (part2)
        if self.config.trainer.get("checkpointing", False):
            logger.info("apply gradient checkpointing")
            non_reentrant_wrapper = partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
            apply_activation_checkpointing(
                self.model,
                checkpoint_wrapper_fn=non_reentrant_wrapper,
                check_fn=lambda submodule: submodule in checkpointing_list,
            )
            if hasattr(self, "model_ema"):
                apply_activation_checkpointing(
                    self.model_ema,
                    checkpoint_wrapper_fn=non_reentrant_wrapper,
                    check_fn=lambda submodule: submodule in checkpointing_list_ema,
                )

        logger.info(f"model:\n{self.model}\n")
        
        if self.config.model.get("vae_path", None):
            self.vae = AutoencoderKL.from_pretrained(
                self.config.model.vae_path,
                torch_dtype=torch.bfloat16
            )
        else:
            self.vae = AutoencoderKL.from_pretrained(
                self.model_path,
                subfolder="vae",
                torch_dtype=torch.bfloat16
            )

       



        if advanced.get("latents_mean", None):
            self.latents_mean = torch.tensor(advanced.latents_mean)
            self.latents_std = torch.tensor(advanced.latents_std)
            self.latents_mean = self.latents_mean.view(1, 4, 1, 1).to(self.target_device)
            self.latents_std = self.latents_std.view(1, 4, 1, 1).to(self.target_device)
        
        self.vae.to(self.target_device)
        self.vae.requires_grad_(False)
        self.model.to(self.target_device)
        self.model.train()
        self.model.requires_grad_(True)

        # self.text_encoder.to(self.target_device)
        self.text_encoder.requires_grad_(False)
        # self.tokenizer.to(self.target_device)
        # self.tokenizer.requires_grad_(False)


    def init_model(self):
        self.build_models()
        advanced = self.config.get("advanced", {})
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
        self.vae_encode_bsz = self.config.advanced.get("vae_encode_batch_size", self.batch_size)
        if self.vae_encode_bsz < 0:
            self.vae_encode_bsz = self.batch_size

        if advanced.get("zero_terminal_snr", False):
            apply_zero_terminal_snr(self.noise_scheduler)

        if hasattr(self.noise_scheduler, "alphas_cumprod"):
            cache_snr_values(self.noise_scheduler, self.target_device)


    def apply_average_pool(self,latent, factor):
        """
        Apply average pooling to downsample the latent.

        Args:
            latent (torch.Tensor): Latent tensor with shape (1, C, H, W).
            factor (int): Downsampling factor.

        Returns:
            torch.Tensor: Downsampled latent tensor.
        """
        return F.avg_pool2d(latent, kernel_size=factor, stride=factor)
    # Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
    @torch.no_grad()
    def encode_prompt(self, prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True):
        captions = []
        for caption in prompt_batch:
            if random.random() < proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])

        with torch.no_grad():
            text_inputs = tokenizer(
                captions,
                padding=True,
                pad_to_multiple_of=8,
                max_length=256,
                truncation=True,
                return_tensors="pt",
            )

            # 将输入移动到正确的设备并设置数据类型
            text_input_ids = text_inputs.input_ids.to(self.target_device)
            prompt_masks = text_inputs.attention_mask.to(self.target_device)

            prompt_embeds = text_encoder(
                input_ids=text_input_ids,
                attention_mask=prompt_masks,
                output_hidden_states=True,
            ).hidden_states[-2]

            # 确保 prompt_embeds 的类型与 x_embedder 的 Linear 层匹配
            prompt_embeds = prompt_embeds.to(dtype=self.model.x_embedder.weight.dtype)

        return prompt_embeds, prompt_masks

    # def training_step(self, batch, batch_idx):
    #     # 获取输入数据
    #     # loss = torch.tensor(0.0, device=self.target_device)
    #     for train_res in self.config.advanced.get("train_res", [1024]):
    #         trans = create_transport(
    #             "Linear",
    #             "velocity",
    #             None,
    #             None,
    #             None,
    #             snr_type=self.config.advanced.snr_type,
    #             do_shift=not self.config.advanced.no_shift,
    #             seq_len=(train_res // 16) ** 2,
    #         )
    #         images = batch["image"].to(self.target_device)
    #         prompts = batch["prompt"]
            
    #         with torch.no_grad():
    #             cap_feats, cap_mask = self.encode_prompt(prompts, self.text_encoder, self.tokenizer, 0.1)

    #             # 对图像进行VAE编码
    #             latents = self.encode_images(images)  # [B, C, H, W]
            
    #             # 编码文本提示
    #             prompt_embeds, prompt_masks = self.encode_prompt(
    #                 prompts, 
    #                 self.text_encoder,
    #                 self.tokenizer,
    #                 proportion_empty_prompts=0.1
    #             )
    #             ### muti resolution
    #             # 确保 latents 是 4D 张量 [B, C, H, W]
    #             if len(latents.shape) == 3:
    #                 latents = latents.unsqueeze(0)
    #             latents_mb_256 = self.apply_average_pool(latents, 4)  # 直接对整个批次应用下采样

    #         model_kwargs = dict(cap_feats=prompt_embeds, cap_mask=prompt_masks)
    #         loss_dict = trans.training_losses(self.model, latents, model_kwargs)
    #         loss_dict_256 = trans.training_losses(self.model, latents_mb_256, model_kwargs)

    #         loss_1024 = loss_dict["loss"].sum() / self.batch_size
    #         loss_256 = loss_dict_256["loss"].sum() / self.batch_size
    #         loss = loss_1024 + loss_256

    #         # 记录训练损失
    #         self.log("train_loss", loss, prog_bar=True)
            
    #         return loss

    def encode_images(self, images):
        # VAE编码图像
        vae_scale = {
            "sdxl": 0.13025,
            "sd3": 1.5305,
            "ema": 0.18215,
            "mse": 0.18215,
            "cogvideox": 1.15258426,
            "flux": 0.3611,
        }["flux"]
        vae_shift = {
            "sdxl": 0.0,
            "sd3": 0.0609,
            "ema": 0.0,
            "mse": 0.0,
            "cogvideox": 0.0,
            "flux": 0.1159,
        }["flux"]
        latents = []
        
        # logger.info(f"Input images shape: {images.shape}")
        for i in range(0, images.shape[0], self.vae_encode_bsz):
            batch_latent = self.vae.encode(
                images[i:i + self.vae_encode_bsz].to(self.vae.dtype)
            ).latent_dist.mode()
            # logger.info(f"Batch latent shape after VAE: {batch_latent.shape}")
            latents.append(
                (batch_latent - vae_shift) * vae_scale
            )
        # logger.info(f"latents len: {len(latents)}")
        # 只有在真正需要连接时才使用cat
        if len(latents) == 1:
            latents = latents[0].to(dtype=self.model.x_embedder.weight.dtype)
        else:
            latents = torch.cat(latents, dim=0).to(dtype=self.model.x_embedder.weight.dtype)
        
        # logger.info(f"Final latents shape: {latents.shape}")
        return latents

    # def configure_optimizers(self):
    #     # 配置优化器
    #     optimizer = torch.optim.AdamW(
    #         self.model.parameters(),
    #         lr=self.config.optimizer.params.lr,
    #         weight_decay=self.config.optimizer.params.weight_decay,
    #         betas=(0.9, 0.999),
    #         eps=1e-8,
    #     )

    #     if self.config.trainer.get("resume", None) is not None:
    #         # 使用 Lightning 的方式获取 world_size
    #         world_size = self.trainer.world_size
    #         local_rank = self.trainer.local_rank
            
    #         opt_state_world_size = len(
    #             [x for x in os.listdir(self.config.trainer.resume) if x.startswith("optimizer.") and x.endswith(".pth")]
    #         )
    #         assert opt_state_world_size == world_size, (
    #             f"Resuming from a checkpoint with unmatched world size "
    #             f"({world_size} vs. {opt_state_world_size}) "
    #             f"is currently not supported."
    #         )
    #         logger.info(f"Resuming optimizer states from: {self.config.trainer.resume}")
    #         optimizer.load_state_dict(
    #             torch.load(
    #                 os.path.join(
    #                     self.config.trainer.resume,
    #                     f"optimizer.{local_rank:05d}-of-{world_size:05d}.pth",
    #                 ),
    #                 map_location="cpu",
    #             )
    #         )
    #         for param_group in optimizer.param_groups:
    #             param_group["lr"] = self.config.optimizer.params.lr
    #             param_group["weight_decay"] = self.config.optimizer.params.weight_decay

    #         with open(os.path.join(self.config.trainer.resume, "resume_step.txt")) as f:
    #             resume_step = int(f.read().strip())
    #     else:
    #         resume_step = 0

    #     # 配置学习率调度器
    #     if self.config.get("scheduler", None):
    #         scheduler_cls = get_class(self.config.scheduler.name)
    #         scheduler = scheduler_cls(
    #             optimizer,
    #             **self.config.scheduler.params
    #         )
            
    #         # 返回格式应该是 (list[optimizer], list[scheduler_config])
    #         return ([optimizer], [{
    #             "scheduler": scheduler,
    #             "interval": "step"
    #         }])
    #         # return [optimizer], [lr_scheduler]
    #         # return {
    #         #     "optimizer": optimizer,
    #         #     "lr_scheduler": {
    #         #         "scheduler": scheduler,
    #         #         "interval": "step"
    #         #     }
    #         # }
        
    #     # 如果没有调度器，只返回优化器
    #     return optimizer

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # 更新EMA模型
        if hasattr(self, "model_ema"):
            self.update_ema()
    

    @torch.no_grad()
    def update_ema(self, decay=0.95):
        """
        Step the EMA model towards the current model.
        """

        ema_params = OrderedDict(self.model_ema.named_parameters())
        model_params = OrderedDict(self.model.named_parameters())
        assert set(ema_params.keys()) == set(model_params.keys())

        for name, param in model_params.items():
            # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


    def save_checkpoint(self, model_path, metadata):
        weight_to_save = None
        if hasattr(self, "_fsdp_engine"):
            from lightning.fabric.strategies.fsdp import _get_full_state_dict_context
            
            weight_to_save = {}    
            world_size = self._fsdp_engine.world_size
            with _get_full_state_dict_context(self.model._forward_module, world_size=world_size):
                weight_to_save = self.model._forward_module.state_dict()
            
        elif hasattr(self, "_deepspeed_engine"):
            from deepspeed import zero
            weight_to_save = {}
            with zero.GatheredParameters(self.model.parameters()):
                weight_to_save = self.model.state_dict()
                
        else:
            weight_to_save = self.model.state_dict()
                
        self._save_checkpoint(model_path, weight_to_save, metadata)

    @rank_zero_only
    def _save_checkpoint(self, model_path, state_dict, metadata):
        cfg = self.config.trainer
        # check if any keys startswith modules. if so, remove the modules. prefix
        # if any([key.startswith("module.") for key in state_dict.keys()]):
        #     state_dict = {
        #         key.replace("module.", ""): value for key, value in state_dict.items()
        #     }

        if cfg.get("save_format") == "safetensors":

            save_file(state_dict, model_path + ".safetensors", metadata=metadata)
            if hasattr(self, "model_ema"):
                
                save_file(self.model_ema.state_dict(), model_path + "_ema.safetensors", metadata=metadata)
        elif cfg.get("save_format") == "original":
            os.makedirs(model_path, exist_ok=True)
            torch.save(state_dict, os.path.join(model_path, "consolidated.00-of-01.pth"))
            if hasattr(self, "model_ema"):
                torch.save(self.model_ema.state_dict(), os.path.join(model_path, "consolidated_ema.00-of-01.pth"))
            #copy
            arg_path = "/root/autodl-tmp/Lumina-Image-2.0/results2/NextDiT_2B_GQA_patch2_Adaln_Refiner_bs4_lr2e-4_bf16/checkpoints/0000550/model_args.pth"
            shutil.copy(arg_path, os.path.join(model_path, "model_args.pth"))
            # opt_state_fn = f"optimizer.{dist.get_rank():05d}-of-" f"{.get_world_size():05d}.pth"
            # torch.save(self.optimizer.state_dict(), os.path.join(model_path, opt_state_fn))
        else:
            state_dict = {"state_dict": state_dict, **metadata}
            model_path += ".ckpt"
            torch.save(state_dict, model_path)
        logger.info(f"Saved model to {model_path}")

    # @rank_zero_only
    # def on_save_checkpoint(self, checkpoint):
    #     # 保存模型权重
    #     save_path = os.path.join(
    #         self.config.trainer.checkpoint_dir,
    #         f"consolidated.00-of-01.pth"
    #     )
    #     torch.save(self.model.state_dict(), save_path)
        
    #     # 保存EMA模型权重
    #     if hasattr(self, "model_ema"):
    #         save_path = os.path.join(
    #             self.config.trainer.checkpoint_dir,
    #             f"consolidated_ema.00-of-01.pth" 
    #         )
    #         torch.save(self.model_ema.state_dict(), save_path)



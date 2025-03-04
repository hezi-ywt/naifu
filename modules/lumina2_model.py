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
from models.lumina.transport import create_transport, Sampler
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
from torchvision.transforms.functional import to_pil_image




class Lumina2Model(pl.LightningModule):
    def __init__(self, config, device, model_path):
        super().__init__()
        self.config = config
        self.target_device = device
        self.model_path = model_path
        self.init_model()

    def init_model(self):
        self.build_models()
        self.to(self.target_device)
        
        self.batch_size = self.config.trainer.batch_size
        self.vae_encode_bsz = self.config.advanced.get("vae_encode_batch_size", self.batch_size)
        if self.vae_encode_bsz < 0:
            self.vae_encode_bsz = self.batch_size

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

            
        if self.config.model.get("resume", None) is not None:
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
             # Note that parameter initialization is done within the DiT constructor
            if self.config.advanced.get("use_ema", True):
                logger.info("Using EMA")
                self.model_ema = deepcopy(self.model)
                
            if hasattr(self, "model_ema") and os.path.exists(os.path.join(self.config.model.resume, "consolidated_ema.00-of-01.pth")):
                logger.info(f"Resuming ema weights from: {self.config.model.resume}")
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
            self.model_ema.requires_grad_(False)  # EMA 模型不需要梯度    
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

        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

        # self.tokenizer.to(self.target_device)
        # self.tokenizer.requires_grad_(False)


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

    @torch.no_grad()
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
        
        x = [img.to(self.target_device, non_blocking=True) for img in images]
   

        for i, img in enumerate(x):
            x[i] = (self.vae.encode(img[None].bfloat16()).latent_dist.mode()[0] - vae_shift) * vae_scale
            x[i] = x[i].float()

        return x
        
        

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
            arg_path = os.path.join(self.config.trainer.model_path, "model_args.pth")
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

    @torch.inference_mode()
    def sample(
        self,
        prompt,
        negative_prompt="",
        generator=None,
        size=(1024, 1024),
        steps=25,
        guidance_scale=4.0,
        solver="euler",
        path_type="Linear",
        prediction="velocity",
        loss_weight=None,
        train_eps=None,
        sample_eps=None,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
        time_shifting_factor=1.0,
    ):
        """使用Lumina2模型生成图像样本"""
        system_prompt = "You are an assistant designed to generate anime images with the highest degree of image-text alignment based on textual prompts. <Prompt Start>  "
        system_prompt = ""
        try:
            # 切换到评估模式并标记 forward_with_cfg
            self.model.eval()
            self.vae.eval()
            
            # if hasattr(self.model, "mark_forward_method"):
            #     self.model.mark_forward_method("forward_with_cfg")
            # elif hasattr(self.model, "_forward_module") and hasattr(self.model._forward_module, "mark_forward_method"):
            #     self.model._forward_module.mark_forward_method("forward_with_cfg")
            
            # 获取模型的数据类型和设备
            dtype = self.model.x_embedder.weight.dtype
            device = self.target_device
            logger.info(f"prompt: {prompt}")
            logger.info(f"negative_prompt: {negative_prompt}")
            
            if isinstance(prompt, str):
                prompt = [system_prompt + prompt]
            else:
                prompt = [system_prompt + p for p in prompt]
            n = len(prompt)
            negative_prompt = [system_prompt +  negative_prompt] * n
            
            # 编码正面和负面提示词
            with torch.no_grad():
                cap_feats, cap_mask = self.encode_prompt(
                    prompt + negative_prompt, 
                    self.text_encoder,
                    self.tokenizer,
                    proportion_empty_prompts=0.0,
                    is_train=False
                )
                # 确保提示词嵌入的数据类型匹配
                cap_feats = cap_feats.to(device=device, dtype=dtype)
                cap_mask = cap_mask.to(device=device)
            
                # 设置latent的尺寸
                w, h = size
                latent_w, latent_h = int(w // 8), int(h // 8)
                
                # 修复 generator 设备问题
                if generator is not None:
                    if isinstance(generator, torch.Generator) and generator.device.type != "cuda":
                        device_generator = torch.Generator(device=device)
                        device_generator.manual_seed(generator.initial_seed())
                        generator = device_generator
                
                # 生成 latents 并确保数据类型匹配
                z = torch.randn([1, 16, latent_h, latent_w], generator=generator, device=device, dtype=dtype)
                z = z.repeat(n * 2, 1, 1, 1)  # 复制一份用于负面提示词
                

                
                # 设置模型参数
                model_kwargs = dict(
                    cap_feats=cap_feats,
                    cap_mask=cap_mask,
                    cfg_scale=guidance_scale,
                )
                
                # 创建采样器
                if solver == "dpm":
                    transport = create_transport(
                        "Linear",
                        "velocity",
                    )
                    sampler = Sampler(transport)
                    sample_fn = sampler.sample_dpm(
                        self.model_ema.ard_with_cfg,
                        model_kwargs=model_kwargs,
                    )
                    samples = sample_fn(
                        z, 
                        steps=steps, 
                        order=2, 
                        skip_type="time_uniform_flow", 
                        method="multistep", 
                        flow_shift=time_shifting_factor
                    )
                else:
                    transport = create_transport(
                        path_type,
                        prediction,
                        loss_weight,
                        train_eps,
                        sample_eps,
                    )
                    sampler = Sampler(transport)
                    sample_fn = sampler.sample_ode(
                        sampling_method=solver,
                        num_steps=steps,
                        atol=atol,
                        rtol=rtol,
                        reverse=reverse,
                        time_shifting_factor=time_shifting_factor
                    )
                    
                    # 确保时间步使用正确的数据类型
                    def wrapped_forward_with_cfg(*args, **kwargs):
                        # 确保所有输入张量使用正确的数据类型
                        args = tuple(a.to(dtype=dtype) if isinstance(a, torch.Tensor) else a for a in args)
                        kwargs = {k: v.to(dtype=dtype) if isinstance(v, torch.Tensor) and k != 'cap_mask' else v 
                                for k, v in kwargs.items()}
                        return self.model.forward_with_cfg(*args, **kwargs)
                    
                    samples = sample_fn(z, wrapped_forward_with_cfg, **model_kwargs)[-1]
                
                # 只保留正面提示词生成的样本
                samples = samples[:1]
                
                # VAE解码前确保数据类型匹配
                vae_dtype = self.vae.dtype  # 获取VAE的数据类型
                samples = samples.to(dtype=vae_dtype)  # 将samples转换为VAE的数据类型
                
                # VAE解码
                samples = self.vae.decode(samples / self.vae.config.scaling_factor + self.vae.config.shift_factor)[0]
                samples = (samples + 1.0) / 2.0
                samples = samples[:1]

                # 转换为PIL图像前的处理
                samples = samples.float()
                samples = torch.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=0.0)  # 处理无效值
                samples.clamp_(0.0, 1.0)  # 再次确保值在正确范围内
            
            # 转换为PIL图像
            images = []
            for sample in samples:
                # 确保图像数据是有效的
                sample = sample.cpu()
                if torch.isnan(sample).any() or torch.isinf(sample).any():
                    logger.warning("检测到无效的图像数据，将替换为有效值")
                    sample = torch.nan_to_num(sample, nan=0.0, posinf=1.0, neginf=0.0)
                    sample.clamp_(0.0, 1.0)
                
                image = to_pil_image(sample)
                images.append(image)
            

            return images

        finally:
            # 恢复训练模式
            self.model.train()
            self.vae.train()

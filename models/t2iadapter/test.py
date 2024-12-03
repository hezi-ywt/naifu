from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from diffusers.utils import load_image, make_image_grid
from controlnet_aux.processor import OpenposeDetector
import torch

# load adapter
adapter = T2IAdapter.from_pretrained(
  "/mnt/g/hf/t2i-adapter-openpose-sdxl-1.0", torch_dtype=torch.float16, varient="fp16"
).to("cuda")

# load euler_a scheduler
model_id = '/mnt/g/hf/stable-diffusion-xl-base-1.0'
euler_a = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")


from diffusers import StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_single_file("/mnt/d/sdwebui_model/Stable-diffusion/checkpoint-e2_s82000.safetensors")
pipeline.save_pretrained("/mnt/d/sdwebui_model/noob1")

pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
    "/mnt/d/sdwebui_model/noob1", adapter=adapter, scheduler=euler_a, torch_dtype=torch.float16, variant="fp16", 
).to("cuda")
pipe.enable_xformers_memory_efficient_attention()


pipe.enable_xformers_memory_efficient_attention()

open_pose = OpenposeDetector.from_pretrained("/mnt/g/hf/Annotators")

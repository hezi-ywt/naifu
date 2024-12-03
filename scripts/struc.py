import torch
from safetensors.torch import load_file

def load_model_structure(file_path):
    # 从 safetensor 文件中加载模型
    model = load_file(file_path)
    return model

def compare_model_structures(model1, model2):
    # 比较两个模型的结构
    structure1 = {k: v.shape for k, v in model1.items()}
    structure2 = {k: v.shape for k, v in model2.items()}
    
    return structure1 == structure2

# 文件路径
model1_path = '/data/controlnet-openpose-sdxl-1.0/diffusion_pytorch_model.safetensors'
model2_path = '/data/stable-diffusion-webui/extensions/sd-webui-controlnet/models/co3/checkpoint-e0_s120.safetensors'

# 加载模型
model1 = load_model_structure(model1_path)
model2 = load_model_structure(model2_path)

# 比较模型结构
are_structures_equal = compare_model_structures(model1, model2)

if are_structures_equal:
    print("两个模型的结构相同。")
else:
    print("两个模型的结构不同。")
    
    

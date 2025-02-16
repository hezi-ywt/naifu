import safetensors
from safetensors.torch import load_file
import torch

def compare_safetensors(model_path1, model_path2, tolerance=1e-5):
    """
    比较两个safetensor模型的参数是否相同
    
    Args:
        model_path1: 第一个模型的路径
        model_path2: 第二个模型的路径
        tolerance: 允许的误差范围
        
    Returns:
        bool: 如果模型参数相同返回True，否则返回False
    """
    # 加载两个模型
    model1 = load_file(model_path1)
    model2 = load_file(model_path2)
    
    # 检查键是否相同
    if model1.keys() != model2.keys():
        print("模型的参数键不匹配")
        diff_keys = set(model1.keys()).symmetric_difference(set(model2.keys()))
        print(f"不同的键: {diff_keys}")
        return False
    
    # 比较每个参数
    for key in model1.keys():
        tensor1 = model1[key]
        tensor2 = model2[key]
        
        # 检查形状是否相同
        if tensor1.shape != tensor2.shape:
            print(f"参数 {key} 的形状不同:")
            print(f"模型1形状: {tensor1.shape}")
            print(f"模型2形状: {tensor2.shape}")
            return False
        
        # 检查值是否相近
        diff = torch.max(torch.abs(tensor1 - tensor2))
        if diff > tolerance:
            print(f"参数 {key} 的值不同，最大差异: {diff}")
            return False
    
    print("两个模型的参数完全相同！")
    return True

# 使用示例
if __name__ == "__main__":
    model_path1 = "/mnt/data/neta_cn_ip/test/checkpoint-e0_s6000.safetensors"
    model_path2 = "/mnt/data/neta_cn_ip/test/checkpoint-e2_s22000.safetensors"
    compare_safetensors(model_path1, model_path2)
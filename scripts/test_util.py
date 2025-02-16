import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np


# 生成一个随机张量作为示例

def batch_tensor_to_image(tensor_images, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存每张图像
    for i in range(tensor_images.sha):
        # 选择第 i 张图像
        image_tensor = tensor_images[i]

        # 将张量范围调整到 [0, 1]
        image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())

        # 转换为 NumPy 数组并调整维度顺序
        image_array = image_tensor.permute(1, 2, 0).numpy() * 255  # 转换到 [0, 255]
        image_array = image_array.astype(np.uint8)  # 转换为 uint8

        # 使用 PIL 创建图像
        image = Image.fromarray(image_array)

        # 保存图像
        image.save(os.path.join(output_dir, f'output_image_{i}.png'))

    print(f"已保存 {tensor_images.size(0)} 张图像到 '{output_dir}' 目录")

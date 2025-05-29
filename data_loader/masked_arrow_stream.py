# 此文件扩展了arrow2_load_stream_.py里的TextImageArrowStream类，增加了对蒙版图像的支持
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from data_loader.arrow2_load_stream_ import TextImageArrowStream
from torchvision import transforms

class MaskedTextImageArrowStream(TextImageArrowStream):
    """
    扩展TextImageArrowStream类，支持加载并处理蒙版图像
    """
    
    def __init__(self, *args, mask_field="mask_path", **kwargs):
        """
        mask_field: Arrow表中存储蒙版图像路径的字段名，默认为"mask_path"，其他参数与TextImageArrowStream相同
        """
        super().__init__(*args, **kwargs)
        self.mask_field = mask_field
        self.log_fn(f"启用蒙版加载，蒙版字段为: {self.mask_field}")
        
    def load_mask(self, ind):
        """
        加载蒙版
        ind: 索引值
        Returns: 蒙版图像张量，如果没有蒙版则返回None
        """
        try:
            # 尝试从Arrow表中获取蒙版路径
            row = self.index_manager.get_row(ind)
            if self.mask_field not in row:
                return None
                
            mask_path = row[self.mask_field]
            if not mask_path:  # 空字符串或None
                return None
                
            # 加载蒙版图像
            mask_image = Image.open(mask_path).convert("L")  # 转换为灰度图
            
            # 进行与普通图像相同的裁剪和调整大小操作，确保蒙版与图像对齐
            if hasattr(self, "latest_image_transforms"):
                transforms = self.latest_image_transforms
                for transform in transforms:
                    if hasattr(transform, "__call__"):
                        mask_image = transform(mask_image)
            
            # 转换为张量并归一化到 [0,1]
            mask_tensor = transforms.ToTensor()(mask_image)
            
            # 确保蒙版为 [1, H, W] 形状
            if len(mask_tensor.shape) < 3:
                mask_tensor = mask_tensor.unsqueeze(0)
            elif mask_tensor.shape[0] > 1:
                # 如果蒙版有多个通道，取第一个通道
                mask_tensor = mask_tensor[0:1]
                
            # 二值化蒙版，大于0.5的值设为1，否则为0
            mask_tensor = (mask_tensor > 0.5).float()
            
            return mask_tensor
            
        except Exception as e:
            self.log_fn(f"加载蒙版时出错: {str(e)}")
            return None
    
    def __getitem__(self, ind):
        """
        获取数据集中的一个样本
        
        Args:
            ind: 索引值
            
        Returns:
            包含图像、提示词和可选蒙版的字典
        """
        # 获取原始数据
        result = super().__getitem__(ind)
        
        # 加载蒙版
        mask = self.load_mask(ind)
        
        # 如果有蒙版，添加到结果中
        if mask is not None:
            result["masks"] = mask
            
        return result

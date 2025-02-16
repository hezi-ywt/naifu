import pickle
import random
from pathlib import Path
import ast
import numpy as np
import re
import json
import time
from functools import partial
from PIL import Image
import random

import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset

from IndexKits.index_kits import ArrowIndexV2, MultiResolutionBucketIndexV2, MultiIndexV2
from . import eugebooru

import cv2
import numpy as np
from PIL import Image
import torch
import random

import numpy as np
import cv2

## Convert image into float32 type.
def to32F(img):
    if img.dtype == np.float32:
        return img
    return (1.0 / 255.0) * np.float32(img)

## Convert image into uint8 type.
def to8U(img):
    if img.dtype == np.uint8:
        return img
    return np.clip(np.uint8(255.0 * img), 0, 255)

## Return if the input image is gray or not.
def _isGray(I):
    return len(I.shape) == 2


## Return down sampled image.
#  @param scale (w/s, h/s) image will be created.
#  @param shape I.shape[:2]=(h, w). numpy friendly size parameter.
def _downSample(I, scale=4, shape=None):
    if shape is not None:
        h, w = shape
        return cv2.resize(I, (w, h), interpolation=cv2.INTER_NEAREST)

    h, w = I.shape[:2]
    return cv2.resize(I, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_NEAREST)


## Return up sampled image.
#  @param scale (w*s, h*s) image will be created.
#  @param shape I.shape[:2]=(h, w). numpy friendly size parameter.
def _upSample(I, scale=2, shape=None):
    if shape is not None:
        h, w = shape
        return cv2.resize(I, (w, h), interpolation=cv2.INTER_LINEAR)

    h, w = I.shape[:2]
    return cv2.resize(I, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

## Fast guide filter.
class FastGuidedFilter:
    ## Constructor.
    #  @param I Input guidance image. Color or gray.
    #  @param radius Radius of Guided Filter.
    #  @param epsilon Regularization term of Guided Filter.
    #  @param scale Down sampled scale.
    def __init__(self, I, radius=5, epsilon=0.4, scale=4):
        I_32F = to32F(I)
        self._I = I_32F
        h, w = I.shape[:2]

        I_sub = _downSample(I_32F, scale)

        self._I_sub = I_sub
        radius = int(radius / scale)

        if _isGray(I):
            self._guided_filter = GuidedFilterGray(I_sub, radius, epsilon)
        else:
            self._guided_filter = GuidedFilterColor(I_sub, radius, epsilon)

    ## Apply filter for the input image.
    #  @param p Input image for the filtering.
    def filter(self, p):
        p_32F = to32F(p)
        shape_original = p.shape[:2]

        p_sub = _downSample(p_32F, shape=self._I_sub.shape[:2])

        if _isGray(p_sub):
            return self._filterGray(p_sub, shape_original)

        cs = p.shape[2]
        q = np.array(p_32F)

        for ci in range(cs):
            q[:, :, ci] = self._filterGray(p_sub[:, :, ci], shape_original)
        return to8U(q)

    def _filterGray(self, p_sub, shape_original):
        ab_sub = self._guided_filter._computeCoefficients(p_sub)
        ab = [_upSample(abi, shape=shape_original) for abi in ab_sub]
        return self._guided_filter._computeOutput(ab, self._I)


## Guide filter.
class GuidedFilter:
    ## Constructor.
    #  @param I Input guidance image. Color or gray.
    #  @param radius Radius of Guided Filter.
    #  @param epsilon Regularization term of Guided Filter.
    def __init__(self, I, radius=5, epsilon=0.4):
        try:
            self._I = to32F(I)
            self._init_filter()
        finally:
            if hasattr(self, '_I'):
                del self._I
                
    def clean(self):
        """清理滤波器资源"""
        attributes = ['_I', '_I_mean', '_I_var', '_guided_filter']
        for attr in attributes:
            if hasattr(self, attr):
                delattr(self, attr)

    ## Apply filter for the input image.
    #  @param p Input image for the filtering.
    def filter(self, p):
        return to8U(self._guided_filter.filter(p))


## Common parts of guided filter.
#
#  This class is used by guided_filter class. GuidedFilterGray and GuidedFilterColor.
#  Based on guided_filter._computeCoefficients, guided_filter._computeOutput,
#  GuidedFilterCommon.filter computes filtered image for color and gray.
class GuidedFilterCommon:
    def __init__(self, guided_filter):
        self._guided_filter = guided_filter

    ## Apply filter for the input image.
    #  @param p Input image for the filtering.
    def filter(self, p):
        p_32F = to32F(p)
        if _isGray(p_32F):
            return self._filterGray(p_32F)

        cs = p.shape[2]
        q = np.array(p_32F)

        for ci in range(cs):
            q[:, :, ci] = self._filterGray(p_32F[:, :, ci])
        return q

    def _filterGray(self, p):
        ab = self._guided_filter._computeCoefficients(p)
        return self._guided_filter._computeOutput(ab, self._guided_filter._I)


## Guided filter for gray guidance image.
class GuidedFilterGray:
    #  @param I Input gray guidance image.
    #  @param radius Radius of Guided Filter.
    #  @param epsilon Regularization term of Guided Filter.
    def __init__(self, I, radius=5, epsilon=0.4):
        self._radius = 2 * radius + 1
        self._epsilon = epsilon
        self._I = to32F(I)
        self._initFilter()
        self._filter_common = GuidedFilterCommon(self)

    ## Apply filter for the input image.
    #  @param p Input image for the filtering.
    def filter(self, p):
        return self._filter_common.filter(p)

    def _initFilter(self):
        I = self._I
        r = self._radius
        self._I_mean = cv2.blur(I, (r, r))
        I_mean_sq = cv2.blur(I ** 2, (r, r))
        self._I_var = I_mean_sq - self._I_mean ** 2

    def _computeCoefficients(self, p):
        r = self._radius
        p_mean = cv2.blur(p, (r, r))
        p_cov = p_mean - self._I_mean * p_mean
        a = p_cov / (self._I_var + self._epsilon)
        b = p_mean - a * self._I_mean
        a_mean = cv2.blur(a, (r, r))
        b_mean = cv2.blur(b, (r, r))
        return a_mean, b_mean

    def _computeOutput(self, ab, I):
        a_mean, b_mean = ab
        return a_mean * I + b_mean


## Guided filter for color guidance image.
class GuidedFilterColor:
    #  @param I Input color guidance image.
    #  @param radius Radius of Guided Filter.
    #  @param epsilon Regularization term of Guided Filter.
    def __init__(self, I, radius=5, epsilon=0.2):
        self._radius = 2 * radius + 1
        self._epsilon = epsilon
        self._I = to32F(I)
        self._initFilter()
        self._filter_common = GuidedFilterCommon(self)

    ## Apply filter for the input image.
    #  @param p Input image for the filtering.
    def filter(self, p):
        return self._filter_common.filter(p)

    def _initFilter(self):
        I = self._I
        r = self._radius
        eps = self._epsilon

        Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]

        self._Ir_mean = cv2.blur(Ir, (r, r))
        self._Ig_mean = cv2.blur(Ig, (r, r))
        self._Ib_mean = cv2.blur(Ib, (r, r))

        Irr_var = cv2.blur(Ir ** 2, (r, r)) - self._Ir_mean ** 2 + eps
        Irg_var = cv2.blur(Ir * Ig, (r, r)) - self._Ir_mean * self._Ig_mean
        Irb_var = cv2.blur(Ir * Ib, (r, r)) - self._Ir_mean * self._Ib_mean
        Igg_var = cv2.blur(Ig * Ig, (r, r)) - self._Ig_mean * self._Ig_mean + eps
        Igb_var = cv2.blur(Ig * Ib, (r, r)) - self._Ig_mean * self._Ib_mean
        Ibb_var = cv2.blur(Ib * Ib, (r, r)) - self._Ib_mean * self._Ib_mean + eps

        Irr_inv = Igg_var * Ibb_var - Igb_var * Igb_var
        Irg_inv = Igb_var * Irb_var - Irg_var * Ibb_var
        Irb_inv = Irg_var * Igb_var - Igg_var * Irb_var
        Igg_inv = Irr_var * Ibb_var - Irb_var * Irb_var
        Igb_inv = Irb_var * Irg_var - Irr_var * Igb_var
        Ibb_inv = Irr_var * Igg_var - Irg_var * Irg_var

        I_cov = Irr_inv * Irr_var + Irg_inv * Irg_var + Irb_inv * Irb_var
        Irr_inv /= I_cov
        Irg_inv /= I_cov
        Irb_inv /= I_cov
        Igg_inv /= I_cov
        Igb_inv /= I_cov
        Ibb_inv /= I_cov

        self._Irr_inv = Irr_inv
        self._Irg_inv = Irg_inv
        self._Irb_inv = Irb_inv
        self._Igg_inv = Igg_inv
        self._Igb_inv = Igb_inv
        self._Ibb_inv = Ibb_inv

    def _computeCoefficients(self, p):
        r = self._radius
        I = self._I
        Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]

        p_mean = cv2.blur(p, (r, r))

        Ipr_mean = cv2.blur(Ir * p, (r, r))
        Ipg_mean = cv2.blur(Ig * p, (r, r))
        Ipb_mean = cv2.blur(Ib * p, (r, r))

        Ipr_cov = Ipr_mean - self._Ir_mean * p_mean
        Ipg_cov = Ipg_mean - self._Ig_mean * p_mean
        Ipb_cov = Ipb_mean - self._Ib_mean * p_mean

        ar = self._Irr_inv * Ipr_cov + self._Irg_inv * Ipg_cov + self._Irb_inv * Ipb_cov
        ag = self._Irg_inv * Ipr_cov + self._Igg_inv * Ipg_cov + self._Igb_inv * Ipb_cov
        ab = self._Irb_inv * Ipr_cov + self._Igb_inv * Ipg_cov + self._Ibb_inv * Ipb_cov
        b = p_mean - ar * self._Ir_mean - ag * self._Ig_mean - ab * self._Ib_mean

        ar_mean = cv2.blur(ar, (r, r))
        ag_mean = cv2.blur(ag, (r, r))
        ab_mean = cv2.blur(ab, (r, r))
        b_mean = cv2.blur(b, (r, r))

        return ar_mean, ag_mean, ab_mean, b_mean

    def _computeOutput(self, ab, I):
        ar_mean, ag_mean, ab_mean, b_mean = ab

        Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]

        q = (ar_mean * Ir +
             ag_mean * Ig +
             ab_mean * Ib +
             b_mean)
        return q

def pil2tensor(image: Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(t_image: torch.Tensor) -> Image:
    if isinstance(t_image, torch.Tensor):
        return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    elif isinstance(t_image, np.ndarray):
        return Image.fromarray(np.clip(255.0 * t_image.squeeze(), 0, 255).astype(np.uint8))
    elif isinstance(t_image, Image.Image):
        return t_image
    else:
        raise ValueError("Invalid input type")

def apply_gaussian_blur(image_np, ksize=5, sigmaX=1.0):
    if ksize % 2 == 0:
        ksize += 1  # ksize must be odd
    blurred_image = cv2.GaussianBlur(image_np, (ksize, ksize), sigmaX=sigmaX)
    return blurred_image
def resize_image_control(control_image, resolution):
    HH, WW, _ = control_image.shape
    crop_h = random.randint(0, HH - resolution[1])
    crop_w = random.randint(0, WW - resolution[0])
    crop_image = control_image[crop_h:crop_h+resolution[1], crop_w:crop_w+resolution[0], :]
    return crop_image, crop_w, crop_h

def apply_gaussian_blur(image_np, ksize=5, sigmaX=1.0):
    if ksize % 2 == 0:
        ksize += 1  # ksize must be odd
    blurred_image = cv2.GaussianBlur(image_np, (ksize, ksize), sigmaX=sigmaX)
    return blurred_image


def apply_guided_filter(image_np, radius, eps):
    # Convert image to float32 for the guided filter
    image_np_float = np.float32(image_np) / 255.0
    # Apply the guided filter
    filtered_image = cv2.ximgproc.guidedFilter(image_np_float, image_np_float, radius, eps)
    # Scale back to uint8
    filtered_image = np.clip(filtered_image * 255, 0, 255).astype(np.uint8)
    return filtered_image

class TTPlanet_Tile_Preprocessor_GF:
    def __init__(self, blur_strength=3.0, radius=7, eps=0.01):
        self.blur_strength = blur_strength
        self.radius = radius
        self.eps = eps

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ("FLOAT", {"default": 1.00, "min": 1.00, "max": 8.00, "step": 0.05}),
                "blur_strength": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "radius": ("INT", {"default": 7, "min": 1, "max": 20, "step": 1}),
                "eps": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 0.1, "step": 0.001}),
            },
            "optional": {}
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_output",)
    FUNCTION = 'process_image'
    CATEGORY = 'TTP_TILE'

    def process_image(self, image, scale_factor, blur_strength, radius, eps):
        ret_images = []
    
        for i in image:
            # 这里可能存在内存泄漏
            _canvas = tensor2pil(torch.unsqueeze(i, 0)).convert('RGB')
            img_np = np.array(_canvas)[:, :, ::-1]  # RGB to BGR
            
            # 建议修改为：
            try:
                _canvas = tensor2pil(torch.unsqueeze(i, 0)).convert('RGB')
                img_np = np.array(_canvas)[:, :, ::-1]
                
                # 处理图像...
                guided_filter = GuidedFilter(img_np, radius, eps)
                img_np = guided_filter.filter(img_np)
                
                # 转换回tensor
                tensor_img = pil2tensor(Image.fromarray(img_np[:, :, ::-1]))
                ret_images.append(tensor_img)
                
            finally:
                # 清理中间变量
                del _canvas
                del img_np
                if 'guided_filter' in locals():
                    del guided_filter
        
        return (torch.cat(ret_images, dim=0),)
        
class TTPlanet_Tile_Preprocessor_Simple:
    def __init__(self, blur_strength=3.0):
        self.blur_strength = blur_strength

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ("FLOAT", {"default": 2.00, "min": 1.00, "max": 8.00, "step": 0.05}),
                "blur_strength": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {}
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_output",)
    FUNCTION = 'process_image'
    CATEGORY = 'TTP_TILE'

    def process_image(self, image, scale_factor, blur_strength):
        ret_images = []
    
        for i in image:
            # Convert tensor to PIL for processing
            _canvas = tensor2pil(torch.unsqueeze(i, 0)).convert('RGB')
        
            # Convert PIL image to OpenCV format
            img_np = np.array(_canvas)[:, :, ::-1]  # RGB to BGR
        
            # Resize image first if you want blur to apply after resizing
            height, width = img_np.shape[:2]
            new_width = int(width / scale_factor)
            new_height = int(height / scale_factor)
            resized_down = cv2.resize(img_np, (new_width, new_height), interpolation=cv2.INTER_AREA)
            resized_img = cv2.resize(resized_down, (width, height), interpolation=cv2.INTER_LANCZOS4)
        
            # Apply Gaussian blur after resizing
            img_np = apply_gaussian_blur(resized_img, ksize=int(blur_strength), sigmaX=blur_strength / 2)
        
            # Convert OpenCV back to PIL and then to tensor
            _canvas = Image.fromarray(img_np[:, :, ::-1])  # BGR to RGB
            tensor_img = pil2tensor(_canvas)
            ret_images.append(tensor_img)
    
        return (torch.cat(ret_images, dim=0),)        

class TTPlanet_Tile_Preprocessor_cufoff:
    def __init__(self, blur_strength=3.0, cutoff_frequency=30, filter_strength=1.0):
        self.blur_strength = blur_strength
        self.cutoff_frequency = cutoff_frequency
        self.filter_strength = filter_strength

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ("FLOAT", {"default": 1.00, "min": 1.00, "max": 8.00, "step": 0.05}),
                "blur_strength": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "cutoff_frequency": ("INT", {"default": 100, "min": 0, "max": 256, "step": 1}),
                "filter_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            },
            "optional": {}
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_output",)
    FUNCTION = 'process_image'
    CATEGORY = 'TTP_TILE'

    
    def random_crop(self, img_np, size_factor):
        HH, WW, _ = img_np.shape
        crop_h = random.randint(0, int(HH - HH * size_factor))
        crop_w = random.randint(0, int(WW - WW * size_factor))
        crop_image = img_np[crop_h:crop_h + int(HH * size_factor), crop_w:crop_w + int(WW * size_factor), :]
        return crop_image
    
    def process_image(self, image, scale_factor, blur_strength, size_factor):
        ret_images = []
    
        for i in image:
            # Convert tensor to PIL for processing
            _canvas = tensor2pil(torch.unsqueeze(i, 0)).convert('RGB')
            img_np = np.array(_canvas)[:, :, ::-1]  # RGB to BGR

            # Apply low pass filter with new strength parameter

            

            # Resize image
            height, width = img_np.shape[:2]
            
            img_mp = self.random_crop(img_np, size_factor)
            
            new_width = int(width / scale_factor)
            new_height = int(height / scale_factor)
            resized_down = cv2.resize(img_np, (new_width, new_height), interpolation=cv2.INTER_AREA)
            resized_img = cv2.resize(resized_down, (width, height), interpolation=cv2.INTER_LANCZOS4)
            
            # Apply Gaussian blur
            img_np = apply_gaussian_blur(img_np, ksize=int(blur_strength), sigmaX=blur_strength / 2)
            
            # Convert OpenCV back to PIL and then to tensor
            pil_img = Image.fromarray(resized_img[:, :, ::-1])  # BGR to RGB
            tensor_img = pil2tensor(pil_img)
            ret_images.append(tensor_img)
        
        return (torch.cat(ret_images, dim=0),)

NODE_CLASS_MAPPINGS = {
    "TTPlanet_Tile_Preprocessor_GF": TTPlanet_Tile_Preprocessor_GF,
    "TTPlanet_Tile_Preprocessor_Simple": TTPlanet_Tile_Preprocessor_Simple,
    "TTPlanet_Tile_Preprocessor_cufoff": TTPlanet_Tile_Preprocessor_cufoff
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TTPlanet_Tile_Preprocessor_GF": "🪐TTPlanet Tile Preprocessor GF",
    "TTPlanet_Tile_Preprocessor_Simple": "🪐TTPlanet Tile Preprocessor Simple",
    "TTPlanet_Tile_Preprocessor_cufoff": "🪐TTPlanet Tile Preprocessor cufoff"
}




def coloe_shift(image:Image):
    image_np = np.array(image)
    image_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    hue_shift = random.uniform(-0.1, 0.1)  # 色相偏移
    sat_shift = random.uniform(0.5, 1.5)   # 饱和度偏移
    val_shift = random.uniform(0.5, 1.5)   # 亮度偏移
    image_hsv[:, :, 0] = (image_hsv[:, :, 0] + hue_shift * 180) % 180  # 色相范围是0-180
    image_hsv[:, :, 1] = np.clip(image_hsv[:, :, 1] * sat_shift, 0, 255)  # 饱和度范围是0-255
    image_hsv[:, :, 2] = np.clip(image_hsv[:, :, 2] * val_shift, 0, 255)  # 亮度范围是0-255
    image_rgb = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    image = Image.fromarray(image_rgb)
    return image

def apply_random_adjustments(image_np, adjust_num="random"):
    """应用随机的颜色调整组合"""
    adjustments = []
    
    # 随机选择要应用的调整数量（1-3个）
    num_adjustments = random.randint(1, 3)
    
    # 可用的调整列表
    possible_adjustments = [
        ('contrast', (0.8, 1.2)),
        ('brightness', (0.8, 1.2)),
        ('saturation', (0.7, 1.3)),
        ('hue', (-10, 10)),
        ('gamma', (0.8, 1.2))
    ]
    possible_adjustments_name = [
        'contrast',
        'brightness',
        'saturation',
        'hue',
        'gamma'
    ]
    # 随机选择调整
    if adjust_num in possible_adjustments_name:
        selected_adjustments = [possible_adjustments[possible_adjustments_name.index(adjust_num)]]
    else:
        selected_adjustments = random.sample(possible_adjustments, num_adjustments)
    
    
    
    
    image = image_np.astype(np.float32)
    
    for adjustment, (min_val, max_val) in selected_adjustments:
        if adjustment == 'contrast':
            # 对比度调整
            factor = random.uniform(min_val, max_val)
            mean = np.mean(image, axis=(0, 1), keepdims=True)
            image = (image - mean) * factor + mean
            
        elif adjustment == 'brightness':
            # 亮度调整
            factor = random.uniform(min_val, max_val)
            image = image * factor
            
        elif adjustment == 'saturation':
            # 饱和度调整
            factor = random.uniform(min_val, max_val)
            image_hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)
            image_hsv[:, :, 1] = np.clip(image_hsv[:, :, 1] * factor, 0, 255)
            image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB).astype(np.float32)
            
        elif adjustment == 'hue':
            # 色相调整
            shift = random.uniform(min_val, max_val)
            image_hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)
            image_hsv[:, :, 0] = (image_hsv[:, :, 0] + shift) % 180
            image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB).astype(np.float32)
            
        elif adjustment == 'gamma':
            # 伽马校正
            gamma = random.uniform(min_val, max_val)
            image = np.power(image / 255.0, gamma) * 255.0
            
        adjustments.append(f"{adjustment}: {factor if adjustment != 'hue' else shift:.2f}")
    
    return np.clip(image, 0, 255).astype(np.uint8), adjustments

def color_hue_variations(image_np, method="random"):
    """提供多种色相变换方法"""
    try:
        # 确保输入是正确的numpy数组格式
        if isinstance(image_np, torch.Tensor):
            if len(image_np.shape) == 4:
                image_np = image_np.squeeze(0)  # 移除batch维度
            if image_np.shape[0] == 3:
                image_np = image_np.permute(1, 2, 0)  # CHW -> HWC
            image_np = image_np.cpu().numpy()
        elif isinstance(image_np, Image.Image):
            image_np = np.array(image_np)
        
        # 确保是3通道RGB图像
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif len(image_np.shape) == 3:
            if image_np.shape[2] == 1:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            elif image_np.shape[2] == 4:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        elif len(image_np.shape) == 4:
            image_np = image_np.squeeze()  # 移除多余的维度
        
        # 确保值域在0-255之间
        if image_np.dtype == np.float32 or image_np.dtype == np.float64:
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        
        methods = ['rotate_hue', 'selective_hue', 'gradient_hue', 'complementary']
        if method in methods:
            selected_method = method
        else:
            selected_method = random.choice(methods)
            
        def rotate_hue(img, angle):
            """基础色相旋转
            angle: -180 到 180 之间的角度"""
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            hsv[:, :, 0] = (hsv[:, :, 0] + angle) % 180
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        def selective_hue(img, target_hue_range, shift):
            """选择性色相变换
            只改变特定色相范围内的颜色"""
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            hue = hsv[:, :, 0]
            mask = (hue >= target_hue_range[0]) & (hue <= target_hue_range[1])
            hsv[:, :, 0][mask] = (hsv[:, :, 0][mask] + shift) % 180
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        def gradient_hue(img):
            """渐变色相变换
            从图像左到右逐渐改变色相"""
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            height, width = img.shape[:2]
            gradient = np.linspace(0, 30, width)  # 创建水平渐变
            gradient = np.tile(gradient, (height, 1))
            hsv[:, :, 0] = (hsv[:, :, 0] + gradient) % 180
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        def complementary_shift(img):
            """互补色变换
            将部分颜色转换为其互补色"""
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            # 随机选择要转换为互补色的区域
            mask = np.random.random(img.shape[:2]) > 0.7
            hsv[:, :, 0][mask] = (hsv[:, :, 0][mask] + 90) % 180  # 互补色相差180度
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        def split_toning(img):
            """分离色调
            分别处理暗部和亮部的色相"""
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            # 基于亮度分离暗部和亮部
            shadows_mask = hsv[:, :, 2] < 128
            highlights_mask = ~shadows_mask
            
            # 为暗部和亮部分别设置不同的色相偏移
            shadow_shift = random.uniform(-30, 30)
            highlight_shift = random.uniform(-30, 30)
            
            hsv[:, :, 0][shadows_mask] = (hsv[:, :, 0][shadows_mask] + shadow_shift) % 180
            hsv[:, :, 0][highlights_mask] = (hsv[:, :, 0][highlights_mask] + highlight_shift) % 180
            
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        def color_balance(img):
            """色彩平衡
            调整RGB通道的平衡"""
            # 分别调整RGB通道的增益
            r_gain = random.uniform(0.8, 1.2)
            g_gain = random.uniform(0.8, 1.2)
            b_gain = random.uniform(0.8, 1.2)
            
            balanced = img.copy().astype(np.float32)
            balanced[:, :, 0] *= r_gain  # R通道
            balanced[:, :, 1] *= g_gain  # G通道
            balanced[:, :, 2] *= b_gain  # B通道
            
            return np.clip(balanced, 0, 255).astype(np.uint8)
        
        # 根据选择的方法应用相应的变换
        if method == 'rotate_hue':
            angle = random.uniform(-30, 30)
            return rotate_hue(image_np, angle)
        
        elif method == 'selective_hue':
            target_range = (random.randint(0, 90), random.randint(91, 180))
            shift = random.uniform(-20, 20)
            return selective_hue(image_np, target_range, shift)
        
        elif method == 'gradient_hue':
            return gradient_hue(image_np)
        
        elif method == 'complementary':
            return complementary_shift(image_np)
        
        # 添加更复杂的色相处理方法
        def advanced_hue_processing(img):
            """组合多种色相处理方法"""
            # 首先应用基础色相旋转
            result = rotate_hue(img, random.uniform(-15, 15))
            
            # 根据图像的平均亮度决定后续处理
            hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
            avg_brightness = np.mean(hsv[:, :, 2])
            
            if avg_brightness < 128:
                # 对于暗色图像，增强饱和度并应用分离色调
                result = split_toning(result)
            else:
                # 对于亮色图像，应用渐变色相
                result = gradient_hue(result)
            
            # 最后应用色彩平衡
            result = color_balance(result)
            
            return result
        
        return advanced_hue_processing(image_np)
        
    except Exception as e:
        print(f"Error in color_hue_variations: {e}")
        # 如果处理失败，返回原始图像
        if isinstance(image_np, np.ndarray):
            if len(image_np.shape) == 4:
                image_np = image_np.squeeze()
            return image_np
        return image_np

def procees_image(image):
    try:
        # 确保输入是PIL Image
        if isinstance(image, torch.Tensor):
            image = tensor2pil(image)
        
        # 色相变换
        if random.random() < 0.0001:
            try:
                # 转换为numpy数组
                img_np = np.array(image)
                # 确保是3通道RGB图像
                if len(img_np.shape) == 2:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
                elif len(img_np.shape) == 3 and img_np.shape[2] == 1:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
                elif len(img_np.shape) == 4:
                    img_np = img_np.squeeze()  # 移除多余的维度
                
                # 处理图像
                processed_image = color_hue_variations(img_np, method="random")
                if processed_image is not None:
                    image = Image.fromarray(processed_image)
            except Exception as e:
                print(f"Error in color processing: {e}")
        
        # 其他处理保持不变...
        if random.random() < 0.8:
            scale_factor = random.randint(4, 24)
            blur_strength = random.randint(3, 10)
            processor = TTPlanet_Tile_Preprocessor_Simple()
            image = processor.process_image(pil2tensor(image), scale_factor, blur_strength)[0]
        else:
            scale_factor = random.randint(8, 24)
            blur_strength = random.randint(5, 16)
            size_factor = random.uniform(0.3, 0.8)
            processor = TTPlanet_Tile_Preprocessor_cufoff()
            image = processor.process_image(pil2tensor(image), scale_factor, blur_strength, size_factor)[0]
        
        return tensor2pil(image)
        
    except Exception as e:
        print(f"Error in procees_image: {e}")
        return image  # 如果处理失败，返回原始图像
    finally:
        if 'image' in locals():
            del image

def tensor_to_numpy(tensor):
    """安全地将tensor转换为numpy数组"""
    try:
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)
        
        if tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
            
        image_np = tensor.cpu().numpy()
        
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
            
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 1:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            
        return image_np
        
    except Exception as e:
        print(f"Error in tensor_to_numpy: {e}")
        raise

class TextImageArrowStream(Dataset):
    def __init__(self,
                 args="",
                 resolution=512,
                 random_flip=None,
                 enable_CN=True,
                 log_fn=print,
                 index_file=None,
                 multireso=False,
                 batch_size=-1,
                 world_size=1,
                 random_shrink_size_cond=False,
                 merge_src_cond=False,
                 uncond_p=0.0,
                 uncond_p_t5=0.0,
                 rank=0,
                 dtype=torch.float32,
                 **kwarges
                 ):
        self.args = args
        self.resolution = resolution
        self.log_fn = lambda x: log_fn(f"    {Path(__file__).stem} | " + x)

        self.random_flip = random_flip
        # If true, the Chinese prompt from the `text_zh` column will be taken from the arrow file;
        # otherwise, the English prompt from the `text_en` column will be taken,
        # provided that `text_zh` or `text_en` exists in the arrow file.
        self.enable_CN = enable_CN
        self.index_file = index_file
        self.multireso = multireso
        self.batch_size = batch_size
        self.world_size = world_size
        self.index_manager = self.load_index()

        # clip params
        # self.uncond_p = uncond_p


        # t5 params
        # self.uncond_p_t5 = uncond_p_t5


        # size condition
        self.random_shrink_size_cond = random_shrink_size_cond
        self.merge_src_cond = merge_src_cond

        assert isinstance(resolution, int), f"resolution must be an integer, got {resolution}"
        self.flip_norm = T.Compose(
            [
                # T.RandomHorizontalFlip() if self.random_flip else T.Lambda(lambda x: x),
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ]
        )
        self.flip_norm_cn = T.Compose(
            [
                # T.RandomHorizontalFlip() if self.random_flip else T.Lambda(lambda x: x),
                T.ToTensor(),
            ]
        )
        # tag_edit
        # self.replace_to_zn = 0.2
        # self.copyright_dropout = 0.05
        # self.year_dropout = 0.005
        # self.meta_dropout = 0.005

        self.dataset_hook = {
            "tag_counter": {
                "epoch": {
                    "artist": {},
                    "character": {},
                },
            }
        }
        
        # show info
        if self.merge_src_cond:
            self.log_fn("Enable merging src condition: (oriW, oriH) --> ((WH)**0.5, (WH)**0.5)")

        self.log_fn("Enable image_meta_size condition (original_size, target_size, crop_coords)")
        self.log_fn(f"Image_transforms: {self.flip_norm}")

    def load_index(self):
        multireso = self.multireso
        index_file = self.index_file
        batch_size = self.batch_size
        world_size = self.world_size

        if multireso:
            if isinstance(index_file, (list, tuple)):
                if len(index_file) > 1:
                    raise ValueError(f"When enabling multireso, index_file should be a single file, but got {index_file}")
                index_file = index_file[0]
            index_manager = MultiResolutionBucketIndexV2(index_file, batch_size, world_size)
            self.log_fn(f"Using MultiResolutionBucketIndexV2: {len(index_manager):,}")
        else:
            if isinstance(index_file, str):
                index_file = [index_file]
            if len(index_file) == 1:
                index_manager = ArrowIndexV2(index_file[0])
                self.log_fn(f"Using ArrowIndexV2: {len(index_manager):,}")
            else:
                index_manager = MultiIndexV2(index_file)
                self.log_fn(f"Using MultiIndexV2: {len(index_manager):,}")

        return index_manager

    def shuffle(self, seed, fast=False):
        self.index_manager.shuffle(seed, fast=fast)

    def get_raw_image(self, index, image_key="image"):
        try:
            ret = self.index_manager.get_image(index, image_key)
        except Exception as e:
            self.log_fn(f'get_raw_image | Error: {e}')
            ret = Image.new("RGB", (256, 256), (255, 255, 255))
        return ret

    @staticmethod
    def random_crop_image(image, origin_size, target_size):
        aspect_ratio = float(origin_size[0]) / float(origin_size[1])
        if origin_size[0] < origin_size[1]:
            new_width = target_size[0]
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = target_size[1]
            new_width = int(new_height * aspect_ratio)

        image = image.resize((new_width, new_height), Image.LANCZOS)

        if new_width > target_size[0]:
            x_start = random.randint(0, new_width - target_size[0])
            y_start = 0
        else:
            x_start = 0
            y_start = random.randint(0, new_height - target_size[1])
        image_crop = image.crop((x_start, y_start, x_start + target_size[0], y_start + target_size[1]))
        crops_coords_top_left = (x_start, y_start)
        return image_crop, crops_coords_top_left

    def get_style(self, index):
        "Here we use a default learned embedder layer for future extension."
        style = 0
        return style

    def blurry_process(self, image):
        if random.random() < 0.85:
            image = image.filter(ImageFilter.GaussianBlur(radius=1))
        return image
    

    def tile_process(self, image):
        
        return procees_image(image)

    def get_image_with_hwxy(self, index, image_key="image"):

        original_image = self.get_raw_image(index, image_key=image_key)
        
        # 创建副本用于处理
        image_for_process = original_image.copy()
        condition_image = self.tile_process(image_for_process)
        
        # 使用原始图像获取尺寸信息
        origin_size = original_image.size
        
        if self.multireso:
            target_size = self.index_manager.get_target_size(index)
            # 创建新的副本列表进行resize和crop操作
            process_images = [original_image.copy(), condition_image]
            images, crops_coords_top_left = self.index_manager.resize_and_crop(
                process_images, 
                target_size, 
                resample=Image.LANCZOS,
                crop_type='random',
                is_list=True
            )
            
            # 对原始图像使用flip_norm
            image_tensor = self.flip_norm(images[0])
            condition_image_tensor = self.flip_norm_cn(images[1])

        if self.random_shrink_size_cond:
            origin_size = (1024 if origin_size[0] < 1024 else origin_size[0],
                           1024 if origin_size[1] < 1024 else origin_size[1])
        if self.merge_src_cond:
            val = (origin_size[0] * origin_size[1]) ** 0.5
            origin_size = (val, val)

        image_meta_size = tuple(origin_size) + tuple(target_size) + tuple(crops_coords_top_left)
        kwargs = {
            
            'origin_size': tuple(origin_size),
            'target_size': tuple(target_size),
            'crops_coords_top_left': tuple(crops_coords_top_left)

        }

        style = self.get_style(index)
        kwargs['style'] = style
            
        return image_tensor, condition_image_tensor, kwargs
        





    def get_original_text(self, ind):
        try:
            text = self.index_manager.get_attribute(ind, 'text_zh' if self.enable_CN else 'text_en')
            if text:
                text = text.replace("|||","")
                text = text.replace(", general, sensitive, questionable, explicit","")
                text = text.strip()
            else:   
                meta_info = self.index_manager.get_attribute(ind, 'meta_info')
                if meta_info:
                    if "caption_base" in meta_info:
                        text = meta_info["caption_base"].replace("|||","")
                        text = text.strip()
                    else:
                        text = ""
                else:
                    text = ""
        except Exception as e:
            
            text = ""
            
        return text

    
    def get_tags(
        self,
        ind,
        ):  
        
        try:
            meta_info = self.index_manager.get_attribute(ind, 'meta_info')

            if random.random() < 0.001:
                return ""
            if random.random() < 0.001:
                return 'Generate a random image'


            if len(meta_info) > 1:

                if random.random() < 0.85:
                    try:
                        return eugebooru.get_ata_caption(meta_info, self.dataset_hook)
                    except Exception as e:
                        print(f"Error retrieving tags: {e}, use original text")
                        return self.get_original_text(ind)
                else:
                    return self.get_original_text(ind)


            return self.get_original_text(ind)
        except Exception as e:
            return self.get_original_text(ind)
        
    def get_original_text_old(self, ind):
        text = self.index_manager.get_attribute(ind, 'text_zh' if self.enable_CN else 'text_en')
        text = str(text).strip()

        return text

    def get_text(self, ind):
        text =  self.get_original_text(ind)
        if text == '':
            text = 'Generate a random image'
        # print(f"get_text | text: {text}")
        return text

    
    def __len__(self):
        return len(self.index_manager)
    
    def __getitem__(self, ind):
        # Get text
        # text = self.get_text(ind)
        text = self.get_tags(ind)

        original_pil_image, condition_pil_image, kwargs = self.get_image_with_hwxy(ind)
        pixel = original_pil_image  
        condition = condition_pil_image
        # torch.stack(original_pil_image, dim=0).contiguous()
        target_size = kwargs["target_size"][::-1]
        origin_size = kwargs["origin_size"][::-1]
        crops_coords_top_left = kwargs["crops_coords_top_left"][::-1]
        origin_size = torch.asarray(target_size)
        target_size = torch.asarray(origin_size)
        crops_coords_top_left = torch.asarray(crops_coords_top_left)


        return {
            "prompts": text,
            "pixels": pixel,
            "conditioning_pixels": condition,
            "is_latent": False,
            "target_size_as_tuple": target_size,
            "original_size_as_tuple": origin_size,
            "crop_coords_top_left": crops_coords_top_left,
            }

    

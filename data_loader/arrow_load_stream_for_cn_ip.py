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
from PIL import ImageFilter
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

from typing import List, Tuple

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



from PIL import Image, ImageOps
from enum import Enum

class ResizeMode(Enum):
    RESIZE = 0          # 直接调整到指定大小
    CROP_AND_RESIZE = 1 # 保持比例裁剪并调整
    RESIZE_AND_FILL = 2 # 保持比例填充并调整

def resize_image(resize_mode, image, width, height):
    """
    按照指定模式调整图像大小
    
    Args:
        resize_mode: ResizeMode枚举值
        image: PIL Image对象
        width: 目标宽度
        height: 目标高度
    """
    
    if resize_mode == ResizeMode.RESIZE:
        # 模式0：直接调整到指定大小
        return image.resize((width, height), Image.LANCZOS)
    
    elif resize_mode == ResizeMode.CROP_AND_RESIZE:
        # 模式1：保持比例裁剪并调整
        ratio = width / height
        src_ratio = image.width / image.height
        
        # 计算裁剪尺寸
        src_w = width if ratio > src_ratio else image.width * height // image.height
        src_h = height if ratio <= src_ratio else image.height * width // image.width
        
        # resize并居中裁剪
        resized = image.resize((src_w, src_h), Image.LANCZOS)
        result = Image.new("RGB", (width, height))
        result.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
        
        return result
    
    else:  # ResizeMode.RESIZE_AND_FILL
        # 模式2：保持比例填充并调整
        ratio = width / height
        src_ratio = image.width / image.height
        
        # 计算resize尺寸
        src_w = width if ratio < src_ratio else image.width * height // image.height
        src_h = height if ratio >= src_ratio else image.height * width // image.width
        
        # resize
        resized = image.resize((src_w, src_h), Image.LANCZOS)
        result = Image.new("RGB", (width, height))
        
        # 居中粘贴
        result.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
        
        # 填充边缘
        if ratio < src_ratio:
            # 需要填充上下
            fill_height = height // 2 - src_h // 2
            if fill_height > 0:
                # 复制顶部和底部边缘进行填充
                top = resized.resize((width, fill_height), box=(0, 0, width, 0))
                bottom = resized.resize((width, fill_height), box=(0, resized.height, width, resized.height))
                result.paste(top, box=(0, 0))
                result.paste(bottom, box=(0, fill_height + src_h))
        else:
            # 需要填充左右
            fill_width = width // 2 - src_w // 2
            if fill_width > 0:
                # 复制左右边缘进行填充
                left = resized.resize((fill_width, height), box=(0, 0, 0, height))
                right = resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height))
                result.paste(left, box=(0, 0))
                result.paste(right, box=(fill_width + src_w, 0))
        
        return result
    
    

def load_crop_json(path):
    with open(path, 'r') as f:
        crop_json = json.load(f)
    
    return crop_json

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
        
        self.detect_json = load_crop_json("/mnt/data/jsondata/all_person_detected.json")
        self.detect_json2 = load_crop_json("/mnt/data/jsondata/detect_data_head.json")
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
        if random.random() < 0.35:
            image = image.filter(ImageFilter.GaussianBlur(radius=1))
        return image
    
    def load_crop2(self, danbooru_id):
        #"7697000": {"head_detected": [[[201, 486, 568, 875], "head", 0.4663766026496887]]}, "
        data = self.detect_json2.get(danbooru_id)
        if data:
            head_detected = data.get("head_detected")
            if head_detected:
                return head_detected
        
        return None

    def get_image_with_hwxy(self, index, image_key="image"):

        original_image = self.get_raw_image(index, image_key=image_key)
        
        # 创建副本用于处理
        image_for_process = original_image.copy()
        
        meta_info = self.index_manager.get_attribute(index, 'meta_info')
        condition_image = image_for_process
        if meta_info:
            if "danbooru_id" in meta_info:
                danbooru_id = str(meta_info["danbooru_id"]).replace("danbooru_:", "")
                # print(f"danbooru_id: {danbooru_id}")
                
                crop_datas = []
                
                crop_data1= self.detect_json.get(danbooru_id+".webp")
                if crop_data1:
                    crop_datas.extend(crop_data1)
                    
                crop_data2 = self.load_crop2(danbooru_id)
                if crop_data2:
                    crop_datas.extend(crop_data2)

                
                    
                if len(crop_datas) > 0:
                    #根据面积大小随机抽一个
                    # 计算所有边界框的面积
                    areas = []
                    for crop_data in crop_datas:
                        (xmin, ymin, xmax, ymax), _, _ = crop_data
                        area = (xmax - xmin) * (ymax - ymin)
                        areas.append(area)
                    

                    
                        # 将面积转换为概率，使用power调整权重影响
                    areas = np.array(areas)
                    areas = np.power(areas, 1)  # 添加指数调整
                    probs = areas / areas.sum()
                    
                    # 根据面积概率随机选择一个索引
                    selected_idx = np.random.choice(len(crop_datas), p=probs)
                    
                    crop_data = crop_datas[selected_idx]
                    (xmin, ymin, xmax, ymax), _, _ = crop_data
                    condition_image = image_for_process.crop((xmin, ymin, xmax, ymax)) 

                # else:
                #     print(f"get_image_with_hwxy | crop_datas is None")

        
        # 使用原始图像获取尺寸信息
        origin_size = original_image.size
        
        
        # resize and fill
        condition_image = resize_image(ResizeMode.RESIZE_AND_FILL, condition_image, origin_size[0], origin_size[1])
        if random.random() < 0.2:
            condition_image = self.blurry_process(condition_image)
        
        
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

    
    def crop_img(self, image, detection: List[Tuple[Tuple[float, float, float, float], str, float]]):

        cropped_images = []
        for _, ((xmin, ymin, xmax, ymax), _, _) in enumerate(detection):
            cropped = image.crop((xmin, ymin, xmax, ymax))
            cropped_images.append(cropped)
        return cropped_images

    
    def get_tags(
        self,
        ind,
        ):  
        
        try:
            meta_info = self.index_manager.get_attribute(ind, 'meta_info')

            if random.random() < 0.001:
                return ""
            if random.random() < 0.0001:
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

    

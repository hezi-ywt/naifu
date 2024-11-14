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
from . import eugebooru_ipa
from transformers import CLIPImageProcessor


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
                 rank=0,
                 dtype=torch.float32,
                 tokenizer=None, 
                 tokenizer_2=None, 
                 size=1024, center_crop=True, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05,
                 **kwarges,
                 ):
        
        #ipadapter
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.center_crop = center_crop
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.clip_image_processor = CLIPImageProcessor()
        
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
                T.RandomHorizontalFlip() if self.random_flip else T.Lambda(lambda x: x),
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ]
        )
        self.flip_norm_cn = T.Compose(
            [
                T.RandomHorizontalFlip() if self.random_flip else T.Lambda(lambda x: x),
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
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

    def get_image_with_hwxy(self, index, image_key="image"):

        image = self.get_raw_image(index, image_key=image_key)
        origin_size = image.size

        if self.multireso:
            target_size = self.index_manager.get_target_size(index)
            images, crops_coords_top_left = self.index_manager.resize_and_crop(
                image, target_size, resample=Image.LANCZOS, crop_type='random')
            image_tensor = self.flip_norm(images)
        else:
            raise NotImplementedError("just support multireso, Not implemented yet.")
            target_size = (self.resolution, self.resolution)
            image_crop, crops_coords_top_left = self.random_crop_image(image, origin_size, target_size)
            image_tensor = self.flip_norm(image_crop)

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

        return image_tensor, kwargs

    def get_inference_image(self, index):
                # get inference image
        prompt = self.get_tags(index)        
        condition_image = self.get_raw_image(index, image_key="image")
        clip_image = self.clip_image_processor(images=condition_image, return_tensors="pt").pixel_values
        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            prompt = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            prompt = ""
            drop_image_embed = 1
        
        return clip_image, drop_image_embed, prompt
        
    def get_tags(
        self,
        ind,
    ):  
        try:
            meta_info = self.index_manager.get_attribute(ind, 'meta_info')

            if random.random() < 0.3:
                return ""
            if random.random() < 0.001:
                return 'Generate a random image'


            if len(meta_info) > 1:


                if "danbooru_quality_tags" in meta_info:
                    if random.random() < 0.85:
                        try:
                            return eugebooru_ipa.get_ata_caption(meta_info, self.dataset_hook)
                        except Exception as e:
                            print(f"Error retrieving tags: {e}")
                            return self.index_manager.get_attribute(ind, 'tags')
                    else:
                        return self.index_manager.get_attribute(ind, 'text_zh' if self.enable_CN else 'text_en').replace("|||", "")
                    
                else:
                    return self.index_manager.get_attribute(ind, 'text_zh' if self.enable_CN else 'text_en').replace("|||", "")

            if random.random() < 0.001:
                return ""
            if random.random() < 0.001:
                return 'Generate a random image'
            return self.index_manager.get_attribute(ind, 'text_zh' if self.enable_CN else 'text_en')
        except Exception as e:
            return self.index_manager.get_attribute(ind, 'text_zh' if self.enable_CN else 'text_en').replace("|||", "")




    def get_text(self, ind):
        text =  self.get_original_text(ind)
        if text == '':
            text = 'Generate a random image'
        # print(f"get_text | text: {text}")
        return text

    def __getitem__(self, ind):
        # Get text
        # text = self.get_text(ind)


        original_pil_image, kwargs = self.get_image_with_hwxy(ind)
        pixel = original_pil_image  
        # torch.stack(original_pil_image, dim=0).contiguous()
        target_size = kwargs["target_size"][::-1]
        origin_size = kwargs["origin_size"][::-1]
        crops_coords_top_left = kwargs["crops_coords_top_left"][::-1]
        origin_size = torch.asarray(target_size)
        target_size = torch.asarray(origin_size)
        crops_coords_top_left = torch.asarray(crops_coords_top_left)
        clip_image, drop_image_embed, text = self.get_inference_image(ind)
        
        return {
            "prompts": text,
            "pixels": pixel,
            "clip_images": clip_image,
            "drop_image_embeds": drop_image_embed,
            "is_latent": False,
            "target_size_as_tuple": target_size,
            "original_size_as_tuple": origin_size,
            "crop_coords_top_left": crops_coords_top_left,
            # "extras": extras,
            }

    def __len__(self):
        return len(self.index_manager)
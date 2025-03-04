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

import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset

from IndexKits.index_kits import ArrowIndexV2, MultiResolutionBucketIndexV2, MultiIndexV2


class TextImageArrowStream(Dataset):
    def __init__(self,
                 args="",
                 resolution=1024,
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
        self.init_system_prompt()
        # clip params
        self.uncond_p = uncond_p


        # t5 params
        self.uncond_p_t5 = uncond_p_t5


        # size condition
        self.random_shrink_size_cond = random_shrink_size_cond
        self.merge_src_cond = merge_src_cond

        assert isinstance(resolution, int), f"resolution must be an integer, got {resolution}"
        self.flip_norm = T.Compose(
            [
                T.RandomHorizontalFlip() if self.random_flip else T.Lambda(lambda x: x),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                
            ]
        )
        # tag_edit
        self.replace_to_zn = 0.2
        self.copyright_dropout = 0.05
        self.year_dropout = 0.005
        self.meta_dropout = 0.005
        
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
            image, crops_coords_top_left = self.index_manager.resize_and_crop(
                image, target_size, resample=Image.LANCZOS, crop_type='random')
            image_tensor = self.flip_norm(image)
        else:
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


    def formate_tag(self, tag_list):
        tag_new = []
        tag = tag.strip()
        if len(tag) > 3:
            tag = tag.replace("_", " ")
        
        tag_new.append(tag)
        return tag_new

    def danbooru_meta_to_text(self, danbooru_meta):
        character_list = danbooru_meta.get("character",[])
        artist_list = danbooru_meta.get("artist",[])
        series_list = danbooru_meta.get("series",[])
        meta_list = danbooru_meta.get("meta",[])
        general_tag_list = danbooru_meta.get("general",[]) 
        keep_tag_list = danbooru_meta.get("keep_tags",[])
        if len(keep_tag_list) > 6:
            if random.random() < 0.3:
                general_tag_list = keep_tag_list
        rating_list = danbooru_meta.get("rating_tags",[])
        quality_list = danbooru_meta.get("quality_tags",[])
        special_tag_list = danbooru_meta.get("special_tags",[])
        all_tag_list = list(set(special_tag_list)) + list(set(character_list)) + list(set(series_list)) + list(set(artist_list)) + list(set(general_tag_list)) + list(set(meta_list)) + list(set(rating_list)) + list(set(quality_list))
        all_tag_list = self.formate_tag(all_tag_list)
        all_tag_text = ", ".join(all_tag_list)
        return all_tag_text, character_list, artist_list, series_list, rating_list, quality_list


    
    def init_system_prompt(self):
        self.system_prompt = {
            "danbooru": "You are an assistant designed to generate anime images with the highest degree of image-text alignment based on danbooru tags, the danbooru tag may include the character, the artist style, the action, etc. <Prompt Start>  ",
            "text": "You are an assistant designed to generate anime images based on textual prompts. <Prompt Start>  ",
            "caption": "You are an assistant designed to generate high-quality images with the highest degree of image-text alignment based on textual prompts. <Prompt Start> "
        }

    def build_system_prompt(self, type,artist_list,series_list,rating_list,quality_list):

        if len(artist_list) > 0:
            if "type" == "danbooru_meta":
                system_prompt = f"You are an artist named {', '.join(artist_list)}, you need to create works in your own style with the highest degree of image-text alignment based on danbooru tags, the danbooru tag may include the character, the artist style, the action, etc. <Prompt Start>  "
            elif "type" == "text":
                system_prompt = f"You are an artist named {', '.join(artist_list)}, you need to create works in your own style based on textual prompts. <Prompt Start>  "
        else:
            system_prompt = self.system_prompt[type]
            
        return system_prompt

    def get_original_text(self, ind):

        json_data = self.index_manager.get_attribute(ind, 'text_zh')
        caption_dict = {}
        tag_key_list = ['joycaption','regular_summary',"danboooru_meta","gemini_caption","tags", "tag", "caption", "doubao", "wd_tagger"]
        for tag_key in tag_key_list:
            if tag_key in json_data and json_data[tag_key] is not None:
                if len(json_data[tag_key]) > 0:

                    if tag_key == 'danboooru_meta':
                        all_tag_text, character_list, artist_list, series_list, rating_list, quality_list = self.danbooru_meta_to_text(json_data[tag_key])
                        if isinstance(all_tag_text, str):
                            caption_dict[tag_key] = [self.system_prompt["danbooru"], all_tag_text, tag_key]
                        else:
                            continue
                    elif tag_key == 'gemini_caption':
                        gemini_caption = json_data[tag_key].get("regular_summary",None)
                        if gemini_caption is not None and len(gemini_caption) > 40:
                            if isinstance(gemini_caption, str):
                                caption_dict[tag_key] = [self.system_prompt["text"] , gemini_caption, tag_key]
                            else:
                                continue
                        else:
                            continue
                    else:
                        if len(json_data[tag_key]) > 40 and isinstance(json_data[tag_key], str):
                            
                            caption_dict[tag_key] = [self.system_prompt["text"] , json_data[tag_key], tag_key]
                        else:
                            continue
                    

                        
        text = random.choice(list(caption_dict.values()))
        if random.random() < 0.5:
            try:
                caption = self.build_system_prompt(text[2],artist_list,series_list,rating_list,quality_list) + text[0] + text[1]
            except:
                caption = text[0] + text[1]
        else:
            caption = text[0] + text[1]

        return caption
    



    def get_text(self, ind):
        if random.random() < 0.001:
            text = 'Generate a random image'
        try:
            text =  self.get_original_text(ind)
        except:
            text = 'Generate a random image'
        # if random.random() < 0.1:
        #     print(f"get_text | text: {text}")
        return text

    def __getitem__(self, ind):
        # Get text
        text = self.get_text(ind)

        image_tensor, kwargs = self.get_image_with_hwxy(ind)
        pixel = image_tensor
        # torch.stack(original_pil_image, dim=0).contiguous()
        # target_size = kwargs["target_size"][::-1]
        # origin_size = kwargs["origin_size"][::-1]
        # crops_coords_top_left = kwargs["crops_coords_top_left"][::-1]
        # origin_size = torch.asarray(target_size)
        # target_size = torch.asarray(origin_size)
        # crops_coords_top_left = torch.asarray(crops_coords_top_left)


        return {
            "prompts": text,
            "pixels": pixel,
            "is_latent": False,
            # "target_size_as_tuple": target_size,
            # "original_size_as_tuple": origin_size,
            # "crop_coords_top_left": crops_coords_top_left,
            # "extras": extras,
            }

    def __len__(self):
        return len(self.index_manager)

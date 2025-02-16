import torch
import numpy as np
import hashlib
import json
import os
import random
import re
import waifuset
from waifuset import logging, tagging
from typing import Union, List, Tuple, Dict, Any, Literal
from collections import UserDict
from bisect import bisect_left

logger = logging.get_logger('eugebooru')

REFINE = False


FIXED_TAG_PREFIXES = (
    'artist:',
    'character:',
    'copyright:',
    'meta:',

    'style:',
    'year:',
    'period:',
    'quality:',
    'safety:',
    'aesthetic:',
)

AESTHETIC_TAGS = {
    'aesthetic',
    'beautiful color',
    'beautiful',
    'detailed',
}


QUALITY2NRP = {
    'amazing': 5,
    'best': 1.5,
    'high': 1.2,
    'normal': 1,
    'low': 0.8,
    'worst': 0.6,
    'horrible': 0.2,
}

QUALITY2NRP_REFINE = {
    'amazing': 5,
    'best': 1,
}

QUALITY_TO_ADD_PROB = {
    'masterpiece': 1.0,
    'amazing': 1.0,
    'best': 0.8,
    'high': 0.65,
    'great': 0.65,
    'good': 0.65,
    'normal': 0.5,
    'low': 1.0,
    'worst': 1.0,
    'horrible': 1.0,
}


QUALITY_TO_ADD_PROB_REFINE = {
    'amazing': 0.5,
    'best': 0.25,
}

DEFAULT_SCALE = {
    -1: 1.1,
    10: 1.05,
    50: 1.02,
    100: 1,
    1000: 0.999,
    2000: 0.995,
    4000: 0.99,
    6000: 0.98,
    8000: 0.97,
    10000: 0.96,
    15000: 0.95,
    20000: 0.90,
    30000: 0.85,
    40000: 0.80,
}

TAG_REWARD = {
    'amazing quality': 1.15,
    'best quality': 1.1,
    'high quality': 1.05,
    'normal quality': 1.0,
    'low quality': 0.95,
    'worst quality': 0.9,
    'horrible quality': 0.85,
    'bad anatomy': 0.95,
    'abstract': 0.98,
    'artifacts': 0.9,
    'scan artifacts': 0.9,
    'jpeg artifacts': 0.9,
}

TAGTYPE_TO_FORMAT = {
    'artist': 'by {}',
    'character': '1 {}',
    'copyright': '{} series',
    'style': '{} style',
    'quality': '{} quality',
    'safety': '{}',
    'year': 'year {}',
    'period': '{} period',
    'meta': '{}',
    'aesthetic': '{}',
}

FEATURE_TABLE = None
SEX_TABLE = None
TAG_FREQUENCY_HOOK = None  # to be initialized later


def get_img_md(img_md, dataset_hook, **kwargs) -> Dict[str, Any]:
    return img_md


def get_dataset_hook(dataset, *args, **kwargs):
    r"""
    Pre-calculate useful information of the dataset for later use.
    """
    total_tag_counter = {
        "category": {},
        "artist": {},
        "character": {},
        "style": {},
        "quality": {},
    }
    for img_key, img_md in dataset.dataset.items():
        weight = img_md.get('weight', 1)
        src_path = img_md.get('image_path') or img_md.get('cache_path')
        if not src_path:
            continue
        category = os.path.basename(os.path.dirname(src_path))
        if category not in total_tag_counter["category"]:
            total_tag_counter["category"][category] = {"count": 0, "weight": 0}
        total_tag_counter["category"][category]["count"] += 1
        total_tag_counter["category"][category]["weight"] += weight  # number of repeats

        for attr in total_tag_counter.keys():
            if attr == 'quality':
                if quality := img_md.get(attr):
                    quality = quality
                elif (aes_score := img_md.get('aesthetic_score')) is not None:
                    quality = get_quality_from_score(aes_score)
                else:
                    continue
                quality = tagging.fmt2danbooru(quality)
                if quality not in total_tag_counter[attr]:
                    total_tag_counter[attr][quality] = {"count": 0, "weight": 0}
                total_tag_counter[attr][quality]["count"] += 1
                total_tag_counter[attr][quality]["weight"] += weight

            elif tags := img_md.get(attr):
                skip = attr == 'artist' and (date := img_md.get('date')) and date < '2021-01-01'
                for tag in tags.split(', '):
                    tag = tagging.fmt2danbooru(tag)
                    if tag not in total_tag_counter[attr]:
                        total_tag_counter[attr][tag] = {"count": 0, "weight": 0}
                    if not skip:
                        total_tag_counter[attr][tag]["count"] += 1
                        total_tag_counter[attr][tag]["weight"] += weight
    total_tag_counter = {key: dict(sorted(val.items(), key=lambda x: x[1]['count'], reverse=True)) for key, val in total_tag_counter.items()}
    dataset_hook = {
        "tag_counter": {
            "total": total_tag_counter,
            "epoch": {
                "artist": {},
                "character": {},
            },
        }
    }
    return dataset_hook


def save_dataset_hook(dataset_hook, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(dataset_hook, f, indent=4)


def get_concept_multiple(img_md, dataset_hook, column, benchmark, lower_threshold=None, upper_threshold=None, max_multiple=None) -> int:
    r"""
    Get the weight of a concept based on the number of images that contain the concept.
    """
    total_tag_counter = dataset_hook["tag_counter"]["total"]
    lower_threshold = lower_threshold or 0
    upper_threshold = upper_threshold or float('inf')
    max_multiple = max_multiple or float('inf')
    values = img_md.get(column)
    concept_multiples = []
    if not values:
        return 0
    for value in values.split(', '):
        num_concept = total_tag_counter[column][tagging.fmt2danbooru(value)]['count']
        if num_concept < lower_threshold or num_concept > upper_threshold:
            continue
        concept_multiple = min(benchmark // num_concept, max_multiple)
        d = benchmark % num_concept
        p = d / num_concept
        if random.random() < p:
            concept_multiple += 1
        concept_multiples.append(concept_multiple)
    return max(concept_multiples, default=0)


def get_data_weight(img_md, dataset_hook, **kwargs) -> int:
    r"""
    Get the weight (number of repeats) of an image.
    """
    if not REFINE:
        if dataset_hook is None:
            raise ValueError("Dataset hook must be provided when not in refine mode.")

    artist_benchmark = 500
    character_benchmark = 1000

    least_data_weight = 1
    data_weight = least_data_weight

    quality = img_md.get('quality')
    score_ws3 = img_md.get('score_ws3')
    score_ws4 = img_md.get('score_ws4')
    if not quality and (score_ws3 is not None or score_ws4 is not None):
        if score_ws4 is not None:
            score_ws4 *= 10  # scale to same range as ws3
        avr_score = (score_ws3 + score_ws4) / 2 if score_ws3 is not None and score_ws4 is not None else score_ws3 or score_ws4
        quality = get_quality_from_score(avr_score)
    if quality in ('horrible',):
        return 0

    if REFINE:
        if quality not in ('best', 'amazing'):
            return 0
        data_weight *= QUALITY2NRP_REFINE.get(quality, 1)
    else:
        data_weight *= QUALITY2NRP.get(quality, 1)

    if not REFINE:
        max_concept_multiple = 30
        concept_multiple = 1
        concept_multiple = max(
            1,  # least concept multiple
            get_concept_multiple(img_md, dataset_hook=dataset_hook, column='artist', benchmark=artist_benchmark, lower_threshold=10, max_multiple=50),
            get_concept_multiple(img_md, dataset_hook=dataset_hook, column='character', benchmark=character_benchmark, lower_threshold=10, upper_threshold=500, max_multiple=50),
        )
        data_weight *= max(1, min(concept_multiple, max_concept_multiple) // 10)  # scale by 0.1

    caption = img_md.get('caption') or img_md.get('tags')
    if caption and any(tag in caption.split(', ') for tag in ('detailed', 'beautiful color', 'beautiful')):
        data_weight *= 1.2

    data_weight = max(least_data_weight, int(data_weight))
    return data_weight


def get_data_weight_for_refine(img_md, dataset_hook, **kwargs):
    quality = img_md.get('quality')
    aesthetic_score = img_md.get('aesthetic_score')

    if not quality and aesthetic_score is not None:
        quality = get_quality_from_score(aesthetic_score)

    data_weight = QUALITY2NRP_REFINE.get(quality, 0)
    return data_weight

# ============================================= get caption area =============================================


def get_caption(img_md, dataset_hook, **kwargs):
    if random.random() < 0.1:  # 10% chance to empty caption
        return ""
    captions = {}
    if (caption := img_md.get('caption')):
        captions['caption'] = caption
    if (description := img_md.get('description')):
        captions['description'] = description

    if len(captions) == 0:
        logger.warning(f"no caption found for image: {img_md['image_key']}")
        return ""

    caption_type, caption = random.choice(list(captions.items()))
    if caption_type == 'caption':
        return get_tags(img_md, dataset_hook, **kwargs)
    elif caption_type == 'description':
        return get_description(img_md, dataset_hook, **kwargs)


def get_tags(
    img_md,
    dataset_hook,
    **kwargs,
):
    r"""
    Perform a sequence of transforms on caption.
    """

    # total_tag_counter = dataset_hook["tag_counter"]["total"]
    epoch_tag_counter = dataset_hook["tag_counter"]["epoch"]

    artist_benchmark = 500
    character_benchmark = 1000
    character_feature_dropout_benchmark = 1000

    caption = img_md['caption']
    tags = caption.split(', ')

    artists = img_md.get('artist')
    characters = img_md.get('character')
    styles = img_md.get('style')
    metas = img_md.get('meta')
    safety = img_md.get('safety')
    quality = img_md.get('quality') 
    score_ws3 = img_md.get('score_ws3')
    score_ws4 = img_md.get('score_ws4')
    original_size = img_md.get('original_size')
    date = img_md.get('date')
    year = int(date.split('-')[0]) if date else img_md.get('year')
    
    # dropout artist tags with high frequency
    if artists:
        if REFINE:
            tags = remove_tags_by_type(tags, 'artist')
        else:
            artists = artists.split(", ")
            for artist in artists:
                removed = False
                atag = f"artist:{fmt2std(artist)}"
                # total_tag_weight = total_tag_counter["artist"][tagging.fmt2danbooru(artist)]['weight']
                count = epoch_tag_counter.get('artist', {}).get(artist, 1)
                if count <= 10 or (year and year <= 2020):  # Exclude too rare or outdated artists
                    try:
                        tags.remove(atag)
                        removed = True
                    except:
                        logger.warning(f"Failed to remove artist_tag {logging.yellow(atag)} because it's not in the following tags: {tags}")
                else:
                    artist_tag_dropout_prob = max(0, 1 - (artist_benchmark / count))
                    if artist_tag_dropout_prob > 0 and random.random() < artist_tag_dropout_prob:  # dropout artist tag
                        try:
                            tags.remove(atag)
                            removed = True
                        except:
                            logger.warning(f"Failed to remove artist_tag {logging.yellow(atag)} because it's not in the following tags: {tags}")
                    else:
                        try:
                            tags.remove(atag)
                            tags.insert(0, atag)
                        except:
                            logger.warning(f"Failed to remove artist_tag {logging.yellow(atag)} because it's not in the following tags: {tags}")

                if not removed:
                    if styles:
                        # Drop style tags to avoid redundancy
                        remove_tags_by_type(tags, 'style')
                    # Increment tag count
                    epoch_tag_counter.setdefault('artist', {}).setdefault(artist, 0)
                    epoch_tag_counter['artist'][artist] += 1

    if characters:
        # 50% chance to prepend a character tag
        characters = characters.split(", ")
        for character in characters:
            # Increment tag count
            epoch_tag_counter.setdefault('character', {}).setdefault(character, 0)
            epoch_tag_counter['character'][character] += 1

            ctag = f"character:{fmt2std(character)}"
            # total_tag_weight = total_tag_counter["character"][tagging.fmt2danbooru(character)]['weight']
            count = epoch_tag_counter['character'][character]

            # Will not remove any character tag
            character_tag_append_prob = max(0, 1 - (character_benchmark / count))
            if character_tag_append_prob > 0 and random.random() < character_tag_append_prob:
                try:
                    tags.remove(ctag)
                    tags.append(ctag)
                except:
                    logger.warning(f"Falied to remove character_tag {logging.green(ctag)} because it's not in the following tags: {tags}")
            else:
                try:
                    tags.remove(ctag)
                    tags.insert(0, ctag)
                except:
                    logger.warning(f"Falied to remove character_tag {logging.green(ctag)} because it's not in the following tags: {tags}")

        if len(characters) == 1:
            for character in characters:
                # Since we don't remove any character, we can remove features of all characters
                count = epoch_tag_counter['character'][character]
                dropout_prob = max(0.4, min(character_feature_dropout_benchmark / count, 1))
                tags = remove_feature_tags(tags, character, dropout_prob=dropout_prob)

    # Add quality tag
    if not quality and (score_ws3 is not None or score_ws4 is not None):
        if score_ws4 is not None:
            score_ws4 *= 10  # scale to same range as ws3
        avr_score = (score_ws3 + score_ws4) / 2 if score_ws3 is not None and score_ws4 is not None else score_ws3 or score_ws4
        quality = get_quality_from_score(avr_score)

    if REFINE:
        # In REFINE mode, only 'best' and 'amazing' quality data are used
        if random.random() < QUALITY_TO_ADD_PROB_REFINE.get(quality, 0):
            tags.append(f"{quality} quality")
    elif random.random() < QUALITY_TO_ADD_PROB.get(quality, 0):
        tags.append(f"{quality} quality")

    # Add safety tag
    if safety and random.random() < 0.5:
        tags.append(safety)

    # Add year and period tags
    if year:
        if random.random() < 0.5:
            tags.append(f"year {year}")
        if random.random() < 0.5:
            tags.append(get_period_tag_from_year(year))

    # Remove meta tags
    if metas:
        helpless_meta_tags = tagging.get_helpless_meta_tags()
        tags = [tag for tag in tags if not (tag.startswith('meta:') and random.random() < 0.5 and tagging.fmt2danbooru(tag[5:]) in helpless_meta_tags)]

    # Add resolution tags
    if original_size and random.random() < 0.5:
        width, height = original_size.split('x') if isinstance(original_size, str) else original_size
        width, height = int(width), int(height)
        reso_tag = get_reso_tag_from_size(width, height)
        if reso_tag and reso_tag not in metas:
            tags.append(f"meta:{reso_tag}")

    # Remove copyright tags
    if random.random() < 0.5:
        tags = remove_tags_by_type(tags, 'copyright')

    tags = shuffle_tags(
        tags,
        fixed_tag_dropout_prob=0,
        flex_tag_dropout_prob=0.3,
        tags_shuffle_prob=0.5,
        tags_shuffle_rate=0.5,
    )

    tags = [fmt2train(tag) for tag in tags]

    # Deduplicate tags
    # tags = deduplicate(tags)

    caption = ', '.join(tags)
    return caption


def list2str(tags):
    if isinstance(tags, list):
        return ", ".join(tags)
    return tags

def get_ata_caption(
    img_md,
    dataset_hook,
    **kwargs,
):
    if random.random() < 0.001:  # 1% 概率返回空描述
        return ""
    captions = {}
    if 'caption_base' in img_md:
        captions['tags'] = 10  # weight is 7
    if img_md.get('flo2_caption_ft_long'):
        captions['nl_1'] = 1
        
    # if img_md.get('description'):
    #     captions.append('description')

    if len(captions) == 0:
        logger.warning(f"no caption found for image: {img_md}")
        return ""

    # 按照权重/概率随机选择一个描述
    caption_type = random.choices(list(captions.keys()), weights=[val for val in captions.values()])[0]
    if caption_type == 'tags':
        caption = get_ata_tags(img_md, dataset_hook, **kwargs)
    elif caption_type == 'nl_1':
        caption = img_md['flo2_caption_ft_long']

    if random.random() < 0.0001:
        logger.debug(f"Image: {img_md.get('image_key')}, caption: {caption}")

    assert isinstance(caption, str), f"Caption must be a string, but got {type(caption)}: {caption}"

    return caption


def check2list(tags):
    if tags is None:
        print(f"tags is None: {tags}")
        return []
    
    tags = tags.split(",") if isinstance(tags, str) else tags
    for i, tag in enumerate(tags):
        tags[i] = tag.strip()
    return tags

def get_ata_tags(
    img_md,
    dataset_hook,
    **kwargs,
):
    r"""
    Perform a sequence of transforms on caption.
    """

    epoch_tag_counter = dataset_hook["tag_counter"]["epoch"]

    artist_benchmark = 500
    character_benchmark = 1000
    character_feature_dropout_benchmark = 1000

    tags = img_md.get('gen_tags')
    if tags:
        tags = check2list(tags) # 检查标签格式
        tags = tagging.alias_tags(tags)  # 更新 tag 别名
        tags = tagging.sort_tags(tags)  # 排序标签
        tags = tagging.deimplicate_tags(tags)  # 语义去重
        random.shuffle(tags)

    artists: List[str] = img_md.get('artist_tags',[])
    if artists:
        artists = check2list(artists)
    special_tags: List[str] = img_md.get('special_tags',[])
    if special_tags:
        special_tags = check2list(special_tags)
    characters: List[str] = img_md.get('character_tags',[])
    if characters:
        characters = check2list(characters)
    copyrights = img_md.get('copyright_tags',[])
    if copyrights:
        copyrights = check2list(copyrights)
    # styles = img_md.get('style_tags')
    metas: List[str] = img_md.get('meta_tags',[])
    if metas:
        metas = check2list(metas)
    safety: List[str] = img_md.get('rating_tags',[])
    if safety:
        safety = check2list(safety)[0]
    quality: List[str] = img_md.get('quality_tags',[])
    if quality:
        quality = check2list(quality)[0]
    aesthetic_score_1: float = img_md.get('aesthetic_score_1',None)
    if aesthetic_score_1 is None:
        aesthetic_score_1 = img_md.get('score_ws')
    score_ws3 = img_md.get('score_ws3')
    score_ws4 = img_md.get('score_ws4')
    original_size = img_md.get('original_size')
    date = img_md.get('date')
    year: str = img_md.get('year')
    if date:
        year = int(date.split('-')[0]) if date else None
    period: str = img_md.get('year_tags',[])
    if period:
        period = check2list(period)[0]

  
    # 前置 1girl
    if special_tags:
        added_sp = False
        if random.random() < 0.15:
            tags = [fmt2std(tag) for tag in special_tags] + tags
            added_sp = True

        # 添加角色标
        min_character_count = float('inf')  # 记录最小角色标计数，用于判断画师标是否前置
        if characters:
            for character in characters:
                # 增加计数器
                epoch_tag_counter.setdefault('character', {}).setdefault(character, 0)
                epoch_tag_counter['character'][character] += 1

                count = epoch_tag_counter['character'][character]

                # 不会删除任何角色标
                character_tag_append_prob = max(0, 1 - (character_benchmark / count))
                if character_tag_append_prob > 0 and random.random() < 0.35:  # 后置角色标
                    if random.random()<0.7:
                        tags.append(f"character:{fmt2std(character)}")
                    else:
                        tags.append(f"character:{fmt2std(character)}")
                else:
                    if random.random()<0.7:
                        tags.insert(0,f"character:{fmt2std(character)}")
                    else:
                        tags.insert(0, f"character:{fmt2std(character)}")  # 固定前置角色标
                min_character_count = min(min_character_count, count)

            if "solo" in tags:
            # 只考虑单角色情况
                for character in characters:
                    # 按概率移除所有角色的特征
                    count = epoch_tag_counter['character'][character]
                    dropout_prob = max(0.4, min(character_feature_dropout_benchmark / count, 1))
                    tags = remove_feature_tags(tags, character, dropout_prob=dropout_prob)

        # 添加系列标
        if copyrights:
            for copy in copyrights:
                if random.random() < 0.75:
                    tags.append(f"copyright:{copy}")
        # 添加画师标
        if artists:
            for artist in artists:
                has_artist = False
                count = epoch_tag_counter.get('artist', {}).get(artist, 1)
                
                if (year is not None) and (int(year) < 2020) and (random.random() < 0.3):  # 排除过于稀有或过时的艺术家
                    continue
    
                else:  # 按概率添加艺术家标签
                    artist_tag_dropout_prob = max(0, 1 - (artist_benchmark / count))
                    if artist_tag_dropout_prob > 0 and random.random() < artist_tag_dropout_prob:  # 丢弃艺术家标签
                        pass
                    else:
                        has_artist = True
                        # 如果画师比角色更冷门，则前置画师标签
                        if count <= min_character_count and random.random() <= 0.2:
                            tags.insert(0, f"artist:{fmt2std(artist)}")  # 固定前置画师标签
                        else:
                            tags.append(fmt2std(artist))

                if has_artist:
                    # 75% 概率删除风格标签
                    if random.random() < 0.65:
                        style_tags = tagging.get_style_tags()
                        tags = [tag for tag in tags if tag not in style_tags]

                    # 增加计数器
                    epoch_tag_counter.setdefault('artist', {}).setdefault(artist, 0)
                    epoch_tag_counter['artist'][artist] += 1

         # 前置 1girl
        if added_sp == False:
            if random.random() < 0.95:
                tags = [fmt2std(tag) for tag in special_tags] + tags
                added_sp = True




    # 添加 meta 标
    if metas:
        for meta in metas:
            if random.random() < 0.75:
                tags.append(f"meta:{meta}")  # 固定后置 meta 标


    # 添加分辨率标
    # if original_size and random.random() < 0.5:
    #     width, height = original_size.split('x') if isinstance(original_size, str) else original_size
    #     width, height = int(width), int(height)
    #     reso_tag = get_reso_tag_from_size(width, height)
    #     if reso_tag and reso_tag not in metas:
    #         tags.append(f"meta:{reso_tag}")

        # 添加安全标
    if safety and random.random() < 0.85:
        tags.append(safety)

    # 添加年份和时期标
    if year:
        if random.random() < 0.5:
            tags.append(f"{year}")
    if period:
        if random.random() < 0.75:
            tags.append(period)


    
    # 添加美学评分标
    if aesthetic_score_1 is not None:
        if aesthetic_score_1 >= 0.8:
            tags.append('very awa')
        elif aesthetic_score_1 <= 0.094:
            tags.append('worst aesthetic')

    # 添加质量标
    if random.random() < ATA_QUALITY_TO_ADD_PROB.get(list2str(quality), 0):
        
        tags.append(list2str(quality))


    # 标签洗牌
    tags = shuffle_tags(
        tags,
        fixed_tag_dropout_prob=0.003,
        flex_tag_dropout_prob=0.05,
        tags_shuffle_prob=0.2,
        tags_shuffle_rate=0.2,
    )

    # 精炼时提高丢弃 tags 的概率
    # tags = shuffle_tags(
    #     tags,
    #     fixed_tag_dropout_prob=0,
    #     flex_tag_dropout_prob=random_gaussian_prob(0.3, 0.25),
    #     tags_shuffle_prob=random_gaussian_prob(0.707, 0.25),  # sqrt(0.5)
    #     tags_shuffle_rate=random_gaussian_prob(0.707, 0.25),  # sqrt(0.5)
    # )

    # 格式化标签
    if random.random() <= 0.95:
        tags = [fmt2train_ata(tag) for tag in tags]

    # 去重
    tags = deduplicate(tags)

    caption = ', '.join(tags)
    return caption


TAGTYPE_TO_ATA_FORMAT = {
    'artist': '{}',
    'character': '{}',
    'copyright': '{}',
    'style': '{}',
    'quality': '{} quality',
    'safety': '{}',
    'year': 'year {}',
    'period': '{}',
    'meta': '{}',
    'aesthetic': '{}',
}

ATA_QUALITY_TO_ADD_PROB = {
    'masterpiece': 1.0,
    'amazing_quality': 1.0,
    'best_quality': 0.8,
    'high_quality': 0.65,
    'great_quality': 0.65,
    'good_quality': 0.65,
    'normal_quality': 0.5,
    'low_quality': 1.0,
    'worst_quality': 1.0,
    'horrible_quality': 1.0,
}


def fmt2train_ata(tag):
    if ':' not in tag:  # quick return
        pass
    elif tag.startswith(FIXED_TAG_PREFIXES):
        tagtype, tag = tag.split(":", 1)
        fmt = TAGTYPE_TO_ATA_FORMAT.get(tagtype)
        if fmt:
            tag = fmt.format(tag)
    return fmt2std(tag)


def get_description(img_md, dataset_hook, **kwargs):
    r"""
    Get natural language description of an image.
    """
    desc = img_md.get('description')
    is_flipped = img_md.get('is_flipped')
    if is_flipped:
        desc.replace('left', 'right').replace('right', 'left')
    return desc


def deduplicate(tags):
    r"""
    Remove duplicate tags in a tag list.
    """
    for i, tag in enumerate(tags):
        if tag in tags[:i]:
            tags[i] = None
    return [tag for tag in tags if tag is not None]


def hash_tag(tag: str, n_digits=8) -> str:
    # Deprecated
    tag = tagging.fmt2danbooru(tag)
    hash_object = hashlib.sha256(tag.encode())
    hash_hex = hash_object.hexdigest()
    unique_number = int(hash_hex, 16) % int(eval(f"1e{n_digits}"))
    return f"{unique_number:0{n_digits}d}"


def fmt2std(tag):
    if len(tag) > 3: 
        tag = tag.replace('_', ' ')
    return tag


def fmt2train(tag):
    if ':' not in tag:  # quick return
        pass
    elif tag.startswith(FIXED_TAG_PREFIXES):
        tagtype, tag = tag.split(":", 1)
        fmt = TAGTYPE_TO_FORMAT.get(tagtype)
        if fmt:
            tag = fmt.format(tag)
    return fmt2std(tag)


def get_period_tag_from_year(year):
    year = int(year)
    if year >= 2022:
        return 'newest'
    elif year >= 2019:
        return 'late'
    elif year >= 2015:
        return 'medial'
    elif year >= 2011:
        return 'early'
    else:
        return 'oldest'


def get_reso_tag_from_size(width, height):
    area = width * height
    if 589824 < area < 4194304:  # 768 * 768 < area < 2048 * 2048
        reso_tag = None
    elif area <= 147456:  # 384 * 384
        reso_tag = 'thumbnail'
    elif area <= 589824:  # 768 * 768
        reso_tag = 'lowres'
    elif area >= 16777216:  # 4096 * 4096
        reso_tag = 'absurdres'
    elif area >= 4194304:  # 2048 * 2048
        reso_tag = 'highres'
    else:
        raise ValueError(f"Invalid image size: {width}x{height}")
    return reso_tag


def get_sex_table(fp):
    global SEX_TABLE
    if SEX_TABLE is not None:
        return SEX_TABLE
    with open(fp, 'r') as f:
        SEX_TABLE = json.load(f)
    return SEX_TABLE


def remove_feature_tags(tags, character, feature_type_to_frequency_threshold=tagging.DEFAULT_FEATURE_TYPE_TO_FREQUENCY_THRESHOLD, dropout_prob=1.0):
    features = tagging.get_character_features(character, feature_type_to_frequency_threshold=feature_type_to_frequency_threshold)
    if not features:
        return tags
    tags = [tag for tag in tags if random.random() > dropout_prob or tagging.fmt2danbooru(tag) not in features]
    return tags


def remove_tags_by_type(tags: List[str], tagtype: Literal['artist', 'character', 'style', 'quality', 'safety', 'year', 'period', 'meta', 'aesthetic']) -> List[str]:
    r"""
    Remove a certain type of tags in a tag list.
    """
    tagtype = tagtype + ':'
    return [tag for tag in tags if not tag.startswith(tagtype)]


def get_quality_from_score(score):
    if score >= 8.7:
        return 'amazing'
    elif score >= 8.0:
        return 'best'
    elif score >= 7.0:
        return 'high'
    elif score >= 4.0:
        return 'normal'
    elif score >= 2.5:
        return 'low'
    elif score >= 1.3:
        return 'worst'
    else:
        return 'horrible'

# ============================================= shuffle tags area =============================================


def random_gaussian_prob(mean, std):
    return min(1, max(0, random.gauss(mean, std)))


def shuffle_tags(tags, fixed_tag_dropout_prob=0, flex_tag_dropout_prob=0, tags_shuffle_prob=0, tags_shuffle_rate=0):
    r"""
    Shuffle tags with dropout.
    @param tags: list of tags
    @param fixed_tag_dropout_prob: dropout probability of fixed tags
    @param flex_tag_dropout_prob: dropout probability of flexible tags
    @param tags_shuffle_prob: probability to shuffle tags
    @param tags_shuffle_rate: rate of tags to shuffle
    """
    # logging.debug(f"tag_count_before: {len(tags)}")
    # logging.debug(f"fixed_tag_dropout_prob: {fixed_tag_dropout_prob}")
    # logging.debug(f"flex_tag_dropout_prob: {flex_tag_dropout_prob}")
    # logging.debug(f"tags_shuffle_prob: {tags_shuffle_prob}")
    # logging.debug(f"tags_shuffle_rate: {tags_shuffle_rate}")
    if not (fixed_tag_dropout_prob or flex_tag_dropout_prob or tags_shuffle_prob or tags_shuffle_rate):
        return tags
    pattern = re.compile(r"\d(?:girl|boy|other)s?")  # extra fixed tags
    fixed_bitmap = [tag.startswith(FIXED_TAG_PREFIXES) or pattern.match(tag) for tag in tags]
    fixed_tags, flex_tags = [], []
    for i, (tag, is_fixed) in enumerate(zip(tags, fixed_bitmap)):
        if is_fixed or random.random() > tags_shuffle_rate:
            fixed_tags.append(tag)
            fixed_bitmap[i] = True
        else:
            flex_tags.append(tag)

    if tags_shuffle_prob and random.random() < tags_shuffle_prob:
        random.shuffle(flex_tags)

    # logging.debug(f"fixed_tag_count: {len(fixed_tags)}")
    # logging.debug(f"flex_tag_count: {len(flex_tags)}")

    # logging.debug(f"fixed_tags: {fixed_tags}")
    # logging.debug(f"flex_tags: {flex_tags}")

    proc_tags = []
    for is_fixed in fixed_bitmap:
        if is_fixed:
            tag = fixed_tags.pop(0)
            if random.random() > fixed_tag_dropout_prob:
                proc_tags.append(tag)
        else:
            tag = flex_tags.pop(0)
            if random.random() > flex_tag_dropout_prob:
                proc_tags.append(tag)
    tags = proc_tags
    # logging.debug(f"tag_count_after: {len(tags)}")
    return tags

# ============================================= tag-based loss weighting (from NovelAI's code) =============================================


class TagFreqScale(UserDict):
    steps: List[int]

    def __init__(
        self,
        scales: Union[List[Tuple[int, float]], Dict[int, float]] = DEFAULT_SCALE,
    ):
        if isinstance(scales, list):
            scales = dict(scales)
        super().__init__(scales)
        self.steps = sorted(self.keys())

    def __getitem__(self, key: int):
        if key not in self.data:
            key = self.steps[bisect_left(self.steps, key)]
        return self.data[key]

    def __setitem__(self, key: int, value: float):
        ret = super().__setitem__(key, value)
        self.steps = sorted(self.keys())
        return ret


class TagRewards(UserDict):
    data: Dict[str, float]

    def __init__(
        self,
        **kwargs,
    ):
        kwargs = {k: v for k, v in kwargs.items() if isinstance(v, float)}
        super().__init__(data=kwargs)

    def __getitem__(self, key: str):
        if key not in self:
            return None


# class TagCount(UserDict):
#     data: Dict[str, int]

#     def __getitem__(self, key):
#         if key not in self:
#             self[key] = 0
#         return super().__getitem__(key)

#     def increment(self, tag: str) -> int:
#         if tag in self:
#             self[tag] += 1
#         else:
#             self[tag] = 1
#         return self[tag]

#     def reset(self, tag: Optional[str] = None):
#         if tag is not None:
#             self[tag] = 0
#         else:
#             self.data: dict[str, int] = {}


class TagLoss(UserDict):
    data: Dict[str, Tuple[float, int]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TagFrequencyHook(object):
    def __init__(self):
        self.alpha = 0.2
        self.beta = 0.99
        self.strength = 1.0

        self.freq_scale: TagFreqScale = TagFreqScale()
        self.tag_rewards: TagRewards = TagRewards(**TAG_REWARD)
        self.loss_stats: TagLoss = TagLoss()

        self.epoch = 0
        self.step = 0
        self.total_loss: float = 0.0

    def get_loss_weights(self, batch, dataset_hook, loss, step, epoch):
        self.step = step
        if self.epoch != epoch:
            self.epoch = epoch
            # reset epoch tag count
            dataset_hook['tag_counter']['epoch'] = {'artist': {}, 'character': {}}

        epoch_tag_counter = dataset_hook['tag_counter']['epoch']

        batch_tags = [caption.split(', ') for caption in batch['captions']]
        batch_size = len(batch_tags)

        base_loss = [x.detach().mean().cpu().item() for x in loss]
        base_acc = sum(base_loss) / batch_size

        if self.total_loss <= 0.0:
            self.total_loss = base_acc
        else:
            # first 10 samples get more influence to warm up the stats
            batch_beta = min(self.beta, self.step / 10.0)
            self.total_loss = (self.total_loss * batch_beta) + (base_acc * (1.0 - batch_beta))

        weights = []
        for i in range(batch_size):
            img_md = batch['image_mds'][i]
            base_mult = 1

            sample_tags = [x for x in batch_tags[i]]
            sample_loss = base_loss[i]
            tag_mults = []
            base_mults = []

            adjust_tags_and_counts = []
            if artists := img_md.get('artist'):
                artists = artists.split(", ")
                adjust_tags_and_counts.extend((artist, epoch_tag_counter['artist'].get(artist, {}).get('count', 0)) for artist in artists)
            if characters := img_md.get('character'):
                characters = characters.split(", ")
                adjust_tags_and_counts.extend((character, epoch_tag_counter['character'].get(character, {}).get('count', 0)) for character in characters)

            for tag, tag_count in adjust_tags_and_counts:
                base_mults.append(self.freq_scale[tag_count])

                if tag in self.loss_stats:
                    tag_loss, tag_count = self.loss_stats[tag]
                    tag_beta = min(self.beta, tag_count / 10.0)
                    self.loss_stats[tag] = ((tag_loss * tag_beta) + (sample_loss * (1.0 - tag_beta)), tag_count + 1)
                else:
                    self.loss_stats[tag] = (sample_loss, 1.0)
                    tag_loss = sample_loss
                tag_mults.append(tag_loss)

            # apply tag rewards to loss to make images with desirable tags be learned more
            for tag in [x for x in sample_tags if x in self.tag_rewards]:
                base_mult *= self.tag_rewards[tag]

            # apply frequency adjust multiplier
            if len(base_mults) > 0:
                base_mult *= np.array(base_mults).mean()

            # grab the historical rolling average loss for these tags
            hist_loss = np.array(tag_mults).max() if len(tag_mults) > 0 else sample_loss

            # pull current batch item loss towards hist_loss with alpha strength
            target_loss = (sample_loss * (1.0 - self.alpha)) + (hist_loss * self.alpha)
            # apply rewards/punishments for frequency and good/bad tags
            target_loss *= base_mult
            # get ratio of adjusted loss to rolling average loss
            loss_weight = target_loss / sample_loss
            # adjust for global modifier strength
            loss_weight = 1.0 + self.strength * (loss_weight - 1.0)

            weights.append(torch.ones(loss.shape[1:]) * loss_weight)
            # logger.debug(f"loss_weight: {loss_weight}, adjust_tags: {adjust_tags}", write=True)
        weights = torch.stack(weights).to(loss)

        return weights


def get_tag_frequency_hook():
    global TAG_FREQUENCY_HOOK
    if TAG_FREQUENCY_HOOK is None:
        TAG_FREQUENCY_HOOK = TagFrequencyHook()
    return TAG_FREQUENCY_HOOK


def get_loss_weight(batch, loss, step, epoch):
    return get_tag_frequency_hook().get_loss_weights(batch, loss, step, epoch)


if __name__ == "__main__":
    # Example usage
    img_md = {
        'image_key': '114094326_p0',
        'caption': "1girl, 1boy, girl on top, solo focus, character:ganyu (genshin impact), meta:english commentary, blue hair, white background, simple background, from behind, hetero, sitting on lap, straddling, sex from behind, black hair, long hair, large breasts, grabbing another's breast, ass, clothed male nude female, completely nude, artist:trickortreat, cum, back, groping, pov, grabbing from behind",
        'score_ws3': 8.33844566345215,
        'score_ws4': 0.64787878,
        'artist': 'trickortreat',
        'character': "ganyu_(genshin_impact)",
        'style': 'oil_painting_(medium)',
        'quality': 'best',
        'meta': 'highres',
        'safety': 'sensitive',
    }
    dataset_hook = {
        'tag_counter': {
            'total': {
                'artist': {
                    'trickortreat': {
                        "count": 100,
                        "weight": 1000,
                    }
                },
                'character': {
                    'ganyu_(genshin_impact)': {
                        "count": 100,
                        "weight": 200,
                    }
                }
            },
            'epoch': {
                'artist': {
                    'trickortreat': 50,
                },
                'character': {
                    'ganyu_(genshin_impact)': 80,
                }
            }
        }
    }
    logger.info(get_caption(img_md, dataset_hook))
    # logger.info(logging.jsonize(dataset_hook))

    # with logger.timer('test'):
    #     for i in range(10000):
    #         get_caption(img_md, dataset_hook)
    # test: 0.72s

    # ata_md = {
    #     "special_tags": [
    #         "1girl"
    #     ],
    #     "character_tags": [
    #         "kousaka_tamaki"
    #     ],
    #     "copyright_tags": [
    #         "to_heart_(series)",
    #         "to_heart_2"
    #     ],
    #     "artist_tags": [
    #         "kyogoku_shin"
    #     ],
    #     "gen_tags": [
    #         "2000s_(style)",
    #         "orange_background",
    #         "simple_background",
    #         "striped_background",
    #         "solo",
    #         "cowboy_shot",
    #         ";p",
    #         "blush",
    #         "hands_up",
    #         "looking_at_viewer",
    #         "one_eye_closed",
    #         "smile",
    #         "standing",
    #         "tongue_out",
    #         "cat_girl",
    #         "school_uniform",
    #         "serafuku",
    #         "animal_ear_fluff",
    #         "animal_ears",
    #         "breasts",
    #         "brown_eyes",
    #         "cat_ears",
    #         "cat_tail",
    #         "large_breasts",
    #         "long_hair",
    #         "parted_bangs",
    #         "red_hair",
    #         "sidelocks",
    #         "tail",
    #         "thighs",
    #         "two_side_up",
    #         "very_long_hair",
    #         "collarbone",
    #         "kemonomimi_mode",
    #         "no_pants",
    #         "thigh_gap",
    #         "tongue",
    #         "aqua_panties",
    #         "blue_panties",
    #         "bow_panties",
    #         "panties",
    #         "pink_shirt",
    #         "shirt",
    #         "striped_panties",
    #         "thighhighs",
    #         "underwear",
    #         "white_thighhighs",
    #         "blue_bow",
    #         "blue_ribbon",
    #         "bow",
    #         "colored_stripes",
    #         "long_sleeves",
    #         "red_sailor_collar",
    #         "ribbon",
    #         "sailor_collar",
    #         "striped_clothes",
    #         "w_arms",
    #         "border",
    #         "outside_border",
    #         "white_border"
    #     ],
    #     "meta_tags": [
    #         "lowres"
    #     ],
    #     "year_tags": [
    #         "old"
    #     ],
    #     "rating_tags": [
    #         "sensitive"
    #     ],
    #     "danbooru_quality_tags": [
    #         "masterpiece"
    #     ],
    #     "res_source": 298350,
    #     "caption_base": "1girl, kousaka tamaki, to heart (series), to heart 2, kyogoku shin, |||2000s (style), orange background, simple background, striped background, solo, cowboy shot, ;p, blush, hands up, looking at viewer, one eye closed, smile, standing, tongue out, cat girl, school uniform, serafuku, animal ear fluff, animal ears, breasts, brown eyes, cat ears, cat tail, large breasts, long hair, parted bangs, red hair, sidelocks, tail, thighs, two side up, very long hair, collarbone, kemonomimi mode, no pants, thigh gap, tongue, aqua panties, blue panties, bow panties, panties, pink shirt, shirt, striped panties, thighhighs, underwear, white thighhighs, blue bow, blue ribbon, bow, colored stripes, long sleeves, red sailor collar, ribbon, sailor collar, striped clothes, w arms, border, outside border, white border, lowres, sensitive, old, masterpiece",
    #     "year": "2022",
    # }

    # dataset_hook = {
    #     'tag_counter': {
    #         'total': {
    #             'artist': {
    #                 'kyogoku_shin': {
    #                     "count": 100,
    #                     "weight": 1000,
    #                 }
    #             },
    #             'character': {
    #                 'kousaka_tamaki': {
    #                     "count": 100,
    #                     "weight": 200,
    #                 }
    #             }
    #         },
    #         'epoch': {
    #             'artist': {
    #                 'kyogoku_shin': 50,
    #             },
    #             'character': {
    #                 'kousaka_tamaki': 80,
    #             }
    #         }
    #     }
    # }

    # logger.info(get_ata_caption(ata_md, dataset_hook))

    # with logger.timer('test'):
    #     for i in logging.tqdm(range(10000)):
    #         get_ata_caption(ata_md, dataset_hook)

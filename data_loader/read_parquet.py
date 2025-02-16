import os
import pandas as pd
import time
import json

from tag_tool import year_tag, rating_tag, quality_tag, meta_tags_filter, extract_special_tags

import random
import dateutil.parser


# {'artist_tags': 'beudelb', 'caption_base': '1girl, projekt red (arknights), projekt red (light breeze) (arknights), arknights, beudelb, |||simple background, white background, solo, cropped torso, upper body, blush, hands in pockets, looking at viewer, wolf girl, animal ears, brown hair, hair between eyes, medium hair, tail, wolf ears, wolf tail, yellow eyes, official alternate costume, black one-piece swimsuit, jacket, one-piece swimsuit, open jacket, red jacket, swimsuit, zipper pull tab, open clothes, absurdres, highres, sensitive, newest, best quality', 
#  'character_tags': 'projekt_red_(arknights), projekt_red_(light_breeze)_(arknights)', 'copyright_tags': 'arknights', 'danbooru_quality_tags': 'best quality', 
#  'flo2_caption_ft_long': 'Anime girl is standing with her arms crossed. She has long brown hair with pointy ears. She is wearing a red jacket with a black shirt underneath. The shirt has a zipper on the front. The background is white.', 
#  'gen_tags': 'simple_background, white_background, solo, cropped_torso, upper_body, blush, hands_in_pockets, looking_at_viewer, wolf_girl, animal_ears, brown_hair, hair_between_eyes, medium_hair, tail, wolf_ears, wolf_tail, yellow_eyes, official_alternate_costume, black_one-piece_swimsuit, jacket, one-piece_swimsuit, open_jacket, red_jacket, swimsuit, zipper_pull_tab, open_clothes',
#  'image_height': 3792, 'image_key': 'danbooru_4668577', 'image_width': 2777, 'meta_tags': 'absurdres, highres', 'rating': 's', 'rating_tags': 'sensitive', 
#  'res_source': 10530384, 'score_as2': 0.8126193881034851, 'score_ws': 0.5031498491764068, 'special_tags': '1girl', 'year': 2021, 'year_tags': 'newest'}



def get_meta_data_by_id(df, id):
    record = df[df['id'] == id].to_dict(orient='records')
    if record:
        return record[0]
    else:
        # print(f"Record with id {id} not found.")
        return None



def format_meta_data(record):
    
    special_tags, general_tags = extract_special_tags(record['tag_string_general'].split(" "))
    date = dateutil.parser.parse(record['created_at'])
    year = date.year
    # print(record['tag_string_meta'])
    formatted_record = {
        "danbooru_id": record["id"],
        'image_key':f"danbooru_{record['id']}",
        'image_width': record['image_width'],
        'image_height': record['image_height'],
        'tag_count': record['tag_count'],
         
        "special_tags": special_tags,  
        
        'rating': record['rating'],
        'rating_tags': [rating_tag(record['rating'])] if rating_tag(record['rating']) else [],
        
        'created_at': record['created_at'],
        'year_tags': year_tag(record['created_at']),
        'year': year,
        
        "tag_string": record['tag_string'],
        
        "fav_count": record['fav_count'],
        'quality_tags': [quality_tag(record['id'], record['fav_count'], record['rating'])] if quality_tag(record['id'], record['fav_count'], record['rating']) else [],
        
        "created_at": record['created_at'],
        'tag_string_general': record['tag_string_general'],
        "gen_tags": general_tags,
        'tag_string_character': record['tag_string_character'],
        'character_tags': record['tag_string_character'].split(" "),
        'tag_string_artist': record['tag_string_artist'],
        'artist_tags': record['tag_string_artist'].split(" "),
        'tag_string_meta': record['tag_string_meta'],
        'meta_tags': meta_tags_filter(record['tag_string_meta'].split(" ")) if record['tag_string_meta']else [],
        'tag_string_copyright': record['tag_string_copyright'],
        "copyright_tags": record['tag_string_copyright'].split(" "),
    }
    
    
    
    

        
    return formatted_record

def random_list(tags):
    if isinstance(tags, str):
        tags = tags.split(",")
        tags = format_tag_list(tags)
    if len(tags) == 0:
        return []
    else:
        #reorder the tags
        tags = random.sample(tags, len(tags))
        return tags

def list2str(tags):
    if len(tags) == 0:
        return ""
    else:
        return ", ".join(tags)


def format_tag(tag_string):
    if len(tag_string) > 3:
        tag_string = tag_string.replace("_", " ")

    tag_string = tag_string.strip()
    return tag_string

def format_tag_list(tag_list):
    return [format_tag(tag) for tag in tag_list]

def add_base_caption(record):
    caption_base = random_list(record['special_tags']) + random_list(record['character_tags']) + random_list(record["copyright_tags"]) +  random_list(record['artist_tags'])  + random_list(record['gen_tags']) +  random_list(record['meta_tags']) +  random_list(record['rating_tags']) +  random_list(record['quality_tags'])
    caption_base = format_tag_list(caption_base)
    caption_base = list2str(caption_base)
    return caption_base

def add_flo2_caption(record, danbooru_flo2_caption_ft_long):
    
    caption_base = add_base_caption(record)
    if caption_base:
        record['caption_base'] = caption_base
    
    if str(record['danbooru_id']) in danbooru_flo2_caption_ft_long:
        record['flo2_caption_ft_long'] = danbooru_flo2_caption_ft_long[str(record['danbooru_id'])]

        return record
    
    else:
        
        return record


def gen_meta_by_id(df, id, danbooru_flo2_caption_ft_long):
    record = get_meta_data_by_id(df, id)
    if record:
        formatted_record = format_meta_data(record)
        return add_flo2_caption(formatted_record, danbooru_flo2_caption_ft_long)
    else:
        return None

def format2webui(tag_string):
    tags = tag_string.split(",")
    tags = [format_tag(tag) for tag in tags]
    
    for i, tag in enumerate(tags):
        tags[i] = tag.replace("(", r"\(").replace(")", r"\)")
    tags = list2str(tags)
    return tags


if __name__ == '__main__':
    start = time.time()
    # danbooru_parquets_path2 ="/mnt/data/Booru-parquets/danbooru.parquet"
    danbooru_parquets_path ="/mnt/data/danbooru_newest-all/table.parquet"
    nlp_path = "/mnt/data/Booru-parquets/danbooru_flo2_caption_ft_long.json"
    with open(nlp_path, "r") as f:
        danbooru_flo2_caption_ft_long = json.load(f)

    
    # 读取单个 Parquet 文件
    # df = pd.read_parquet(danbooru_parquets_path2)
    df = pd.read_parquet(danbooru_parquets_path)
    print(f"Time taken to read the Parquet file: {time.time() - start} seconds")
    
    for column in df.columns:
        print(f"Column: {column}")
        print(df[column].head())
        print("-" * 30)
    # 主程序逻辑
    start = time.time()
    id = 8128264
    record = get_meta_data_by_id(df, id)
    print(record)
    record = gen_meta_by_id(df, id, danbooru_flo2_caption_ft_long)
    print(record)
    print(f"Time taken: {time.time() - start} seconds")
    caption = add_base_caption(record)

    print(format2webui(caption))
    #get max id
    max_id = df['id'].max()
    
    print(f"Max id: {max_id}")


    



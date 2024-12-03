# -*- coding: utf-8 -*-
import argparse
import datetime
import gc
import os
import pandas as pd
import sys  
import pyarrow as pa
import hashlib
from PIL import Image
from tqdm import tqdm
import json

def load_meta_data(json_path):
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print(f"Metadata file not found: {json_path}")
        return {}

def parse_data(data):
    try:
        img_path = data[0]
        
        with open(img_path, "rb") as fp:
            image = fp.read()
            md5 = hashlib.md5(image).hexdigest()
        
        with Image.open(img_path) as f:
            width, height = f.size
        
        text = data[1]
        meta_info = data[2]

        return [md5, width, height, image, text, meta_info]
    
    except Exception as e:
        print(f'Error: {e}')
        return

def make_arrow_from_file_list(file_name_list, dataset_root, arrow_dir, meta_data=None, score_data=None, start_id=0, end_id=-1):
    image_ext = ['jpg', 'jpeg', 'png', 'webp', 'gif', 'tiff']
    data = []
    
    if not isinstance(dataset_root, list):
        dataset_root = [dataset_root]

    for file_name in file_name_list:
        find = False
        file_name = str(file_name)
        
        for root in dataset_root:
            if find:
                break
            for ext in image_ext:
                image_path = os.path.join(root, f"{file_name}.{ext}")
                if os.path.exists(image_path):
                    find = True
                    break
                
        if meta_data:
            meta_info = meta_data.get(file_name, None)
            if not meta_info:
                print(f"not find meta info for {file_name}")
                meta_info = {}
        else:
            meta_info = {}
        
        if score_data:
            score = score_data.get(f"danbooru_{file_name}", 0)
            meta_info["aesthetic_score_1"] = score
            
        txt_path = image_path.rsplit('.', 1)[0] + '.txt'
        if os.path.exists(txt_path):
            with open(txt_path, "r") as fp:
                text = fp.read()
        else:
            print("No corresponding text file found.")
            text = ""

        data.append([image_path, text, meta_info])

    if not os.path.exists(arrow_dir):
        os.makedirs(arrow_dir)

    if end_id < 0:
        end_id = len(data)
        print(f'start_id: {start_id}  end_id: {end_id}')

    data = data[start_id:end_id]
    num_slice = 5000
    start_sub = int(start_id / num_slice)
    sub_len = int(len(data) // num_slice)
    subs = list(range(sub_len + 1))

    for sub in tqdm(subs):
        arrow_path = os.path.join(arrow_dir, '{}.arrow'.format(str(sub + start_sub).zfill(5)))
        if os.path.exists(arrow_path):
            continue
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} start {sub + start_sub}")

        sub_data = data[sub * num_slice: (sub + 1) * num_slice]
        bs = [parse_data(item) for item in sub_data]
        bs = [b for b in bs if b]
        print(f'length of this arrow: {len(bs)}')

        columns_list = ["md5", "width", "height", "image", "text_zh", "meta_info"]
        dataframe = pd.DataFrame(bs, columns=columns_list)
        table = pa.Table.from_pandas(dataframe)

        with pa.OSFile(arrow_path, "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)

        del dataframe, table, bs
        gc.collect()

if __name__ == '__main__':
    csv_root = ["/data/sdxl/newdb", "/data/sdxl/db"]
    output_arrow_data_path = "/app/hfd/artist_arrow_dir"
    json_path = "/app/hfd/caption/danbooru_metainfos_full_20241001.json"
    score_path = "/app/hfd/caption/ws_danbooru.json"
    artist_json_path = "/app/hfd/caption/hy_artist.json"

    meta_data = load_meta_data(json_path) if json_path else None
    score_data = load_meta_data(score_path) if score_path else None
    artist_json = load_meta_data(artist_json_path) if artist_json_path else None

    for artist_name in tqdm(artist_json):
        file_list = artist_json[artist_name]["image_id"]
        os.makedirs(f"{output_arrow_data_path}/{artist_name}",exist_ok=True)        
        if "00000.arrow" in os.listdir(f"{output_arrow_data_path}/{artist_name}"):
            print(f"{output_arrow_data_path}/{artist_name} already existed, pass")
            continue
        make_arrow_from_file_list(file_list, csv_root, f"{output_arrow_data_path}/{artist_name}", meta_data, score_data)

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
import numpy as np

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
        condition_image_path = data[3]
        with open(condition_image_path, "rb") as fp:
            condition_image = fp.read()
            
        return [md5, width, height, image, condition_image, text, meta_info]
    
    except Exception as e:
        print(f'Error: {e}')
        return

def make_arrow_from_file_list(file_name_list, arrow_dir, df, meta_data=None, score_data=None, score_data2=None, nlp_caption_data=None, start_id=0, end_id=-1):
    image_ext = ['webp','jpg', 'jpeg', 'png',]
    data = []
    

    for image_path in file_name_list:
        file_name = os.path.basename(image_path).split('.')[0]

                
        if True:
            try:
                meta_info_raw = df.loc[file_name]
                meta_info = meta_info_raw.to_dict()
                for key, value in meta_info.items():
                    if isinstance(value, np.ndarray):
                        meta_info[key] = value.tolist()
            except Exception as e:
                print(f"Error: {e}")
                meta_info = {}
                            
            if not meta_info:
                print(f"not find meta info for {file_name}")
                meta_info = {}
        else:
            meta_info = {}
        
        meta_info["source_from"] = "danbooru"
        meta_info["pid"] = f"danbooru_{file_name}"
        meta_info["danbooru_pid"] = file_name

        txt_path = image_path.rsplit('.', 1)[0] + '.txt'
        if os.path.exists(txt_path):
            with open(txt_path, "r") as fp:
                text = fp.read()
        elif "caption_base" in meta_info:
            text = meta_info["caption_base"]
            
        else:
            print("No corresponding text file found")
            text = ""

        condition_image_path = image_path.rsplit('.', 1)[0] + '_pose.jpg'
        if not os.path.exists(condition_image_path):
            print(f"condition image not found: {condition_image_path}")
            continue
        data.append([image_path, text, meta_info,condition_image_path])

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

        columns_list = ["md5", "width", "height", "image", "condition_image","text_zh", "meta_info"]
        dataframe = pd.DataFrame(bs, columns=columns_list)
        table = pa.Table.from_pandas(dataframe)

        with pa.OSFile(arrow_path, "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)

        del dataframe, table, bs
        gc.collect()

def load_parquet_data(file_path, columns=None):
    try:
        rows = pd.read_parquet(file_path, columns=columns)
        for index, row in rows.iterrows():
            yield str(index), row['parsed']
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return iter([])


if __name__ == '__main__':
    
    import os
    import pandas as pd

    file_path = '/mnt/d/sdwebui_model/danbooru_metainfos_full_20241029/danbooru_metainfos_full_20241029.parquet'

    df = pd.read_parquet(file_path)

    data_path = "/mnt/g/hf/danbooru_newest-webp-4Mpixel_pose/images/0000"
    output_arrow_data_path = "/mnt/g/hf/danbooru_newest-webp-4Mpixel_pose_arrow/images_test/0_200"
    if not os.path.exists(output_arrow_data_path):
        os.makedirs(output_arrow_data_path)
    file_list = []
    for id in range(200):
        data_path = f"/mnt/g/hf/danbooru_newest-webp-4Mpixel_pose/images/{str(id).zfill(4)}"
        if not os.path.exists(data_path):
            continue

        
    
        #walk 
        for dirpath, dirnames, filenames in os.walk(data_path):
            for filename in tqdm(filenames):
                if filename.split('.')[-1].lower() in ['jpg', 'jpeg', 'png', 'webp']:
                    if filename.endswith('_pose.jpg'):
                        continue
                    file_list.append(os.path.join(dirpath, filename))
                    
                    
    print(file_list)
    print(f"Total images: {len(file_list)}")
    print(f"Start making arrow files")
    make_arrow_from_file_list(file_list, output_arrow_data_path, df)

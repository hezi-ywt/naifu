# -*- coding: utf-8 -*-
import argparse
import datetime
import gc
import os
from multiprocessing import Pool
import pandas as pd
import sys  
import pyarrow as pa
import hashlib
from PIL import Image
from tqdm import tqdm
import json
from read_parquet import gen_meta_by_id, format_meta_data
from functools import partial
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

def process_image_file(image_path, df, danbooru_flo2_caption_ft_long, score_data):
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    try:
        file_name = int(file_name)
        print(f"Find meta info for {file_name}")
        meta_info = gen_meta_by_id(df, file_name, danbooru_flo2_caption_ft_long)
    except Exception as e:
        print(f"Not find meta info for {file_name}, {e}")
        meta_info = {}

    if score_data:
        score = score_data.get(f"danbooru_{file_name}", 0)
        meta_info["aesthetic_score_1"] = score

    txt_path = image_path.rsplit('.', 1)[0] + '.txt'
    if os.path.exists(txt_path):
        with open(txt_path, "r") as fp:
            text = fp.read()
    elif meta_info and isinstance(meta_info, (list, dict)):
        text = meta_info.get("caption_base", "")
    else:
        print("No corresponding text file found.")
        return None

    return [image_path, text, meta_info]

def make_arrow_from_dir(dataset_root, arrow_dir, df, danbooru_flo2_caption_ft_long, score_data=None, start_id=0, end_id=-1):
    image_ext = ['webp', 'jpg', 'jpeg', 'png']
    data = []
    fail_num = 0

    # 收集所有图像文件路径
    for root, dirs, files in os.walk(dataset_root):
        for file in tqdm(files):
            if file.split('.')[-1].lower() in image_ext:
                image_path = os.path.join(root, file)
                data.append(image_path)

    print(f"Total image files found: {len(data)}")

    # 使用 Pool 进行并行处理
    with Pool() as pool:
        # 使用 partial 固定参数
        process_func = partial(process_image_file, df=df, danbooru_flo2_caption_ft_long=danbooru_flo2_caption_ft_long, score_data=score_data)
        results = list(tqdm(pool.imap(process_func, data), total=len(data)))

    # 过滤掉 None
    results = [result for result in results if result]
    
    print(f"fail_num: {fail_num}")
    
    if not os.path.exists(arrow_dir):
        os.makedirs(arrow_dir)

    if end_id < 0:
        end_id = len(results)
        print(f'start_id: {start_id}  end_id: {end_id}')

    results = results[start_id:end_id]
    num_slice = 5000
    start_sub = int(start_id / num_slice)
    sub_len = int(len(results) // num_slice)
    subs = list(range(sub_len + 1))
    
    with Pool() as pool:
        for sub in tqdm(subs):
            arrow_path = os.path.join(arrow_dir, '{}.arrow'.format(str(sub + start_sub).zfill(5)))
            if os.path.exists(arrow_path):
                continue
            print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} start {sub + start_sub}")

            sub_data = results[sub * num_slice: (sub + 1) * num_slice]
            bs = pool.map(parse_data, sub_data)
            bs = [b for b in bs if b]
            print(f'length of this arrow: {len(bs)}')

            columns_list = ["md5", "width", "height", "image", "text_zh", "meta_info"]
            dataframe = pd.DataFrame(bs, columns=columns_list)
            table = pa.Table.from_pandas(dataframe)

            os.makedirs(arrow_dir, exist_ok=True)
            with pa.OSFile(arrow_path, "wb") as sink:
                with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                    writer.write_table(table)

            del dataframe, table, bs
            gc.collect()

if __name__ == '__main__':
    import time
    
    csv_root = "/mnt/data/data_nieta/neta_"
    output_arrow_data_path = "/mnt/data/arrowdata/neta_"
    pool_num = 2
    
    start = time.time()
    danbooru_parquets_path = "/mnt/data/danbooru_newest-all/table.parquet"
    nlp_path = "/mnt/data/Booru-parquets/danbooru_flo2_caption_ft_long.json"
    
    with open(nlp_path, "r") as f:
        danbooru_flo2_caption_ft_long = json.load(f)

    # 读取 Parquet 文件
    df = pd.read_parquet(danbooru_parquets_path)
    print(f"Time taken to read the Parquet file: {time.time() - start} seconds")

    if os.path.isdir(csv_root):
        make_arrow_from_dir(csv_root, output_arrow_data_path, df, danbooru_flo2_caption_ft_long, score_data=None, start_id=0, end_id=-1)
    else:
        print("The input file format is not supported. Please input a CSV or JSON file.")
        sys.exit(1)
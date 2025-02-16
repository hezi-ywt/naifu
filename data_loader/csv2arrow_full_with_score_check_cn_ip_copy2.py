# -*- coding: utf-8 -*-
import argparse
import datetime
import gc
import os
from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import sys  
import pyarrow as pa
import hashlib
from PIL import Image
from tqdm import tqdm
import json
from read_parquet import gen_meta_by_id, format_meta_data
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
        
        img_path = img_path.replace(".webp", "")
        image_path = os.path.join("/mnt/data/danbooru2024-webp-4Mpixel/images_out", str(img_path)+".webp")
        if not os.path.exists(image_path):
            image_path = os.path.join("/mnt/data/danbooru_newest-webp-4Mpixel-all/images_out", str(img_path)+".webp")
        with open(image_path, "rb") as fp:
            image = fp.read()
            md5 = hashlib.md5(image).hexdigest()

        with Image.open(image_path) as f:
            width, height = f.size
        
        text = data[1]
        meta_info = data[2]

        return [md5, width, height, image, text, meta_info]
    
    except Exception as e:
        print(f'Error: {e}')
        return



def make_arrow_from_list(dataset_root, arrow_dir, df=None, df2=None, danbooru_flo2_caption_ft_long=None,score_data=None, start_id=0, end_id=-1):
    image_ext = ['webp', 'jpg', 'jpeg', 'png']
    data = []
    fail_num = 0
    
    # 使用ThreadPoolExecutor来并行处理meta信息的获取
    def process_file(file_name):
        try:
            meta_info = gen_meta_by_id(df, int(file_name), danbooru_flo2_caption_ft_long)
            if meta_info is None and df2 is not None:
                meta_info = gen_meta_by_id(df2, int(file_name), danbooru_flo2_caption_ft_long)
                if meta_info is None:
                    print(f"not find meta info for {file_name}")
                    return None

            if score_data:
                score = score_data.get(f"danbooru_{file_name}", 0)
                if score < 0.8:
                    return None
                meta_info["aesthetic_score_1"] = score

            if meta_info is not None and isinstance(meta_info, (list, dict)):
                text = meta_info.get("caption_base", "")
                if not text:
                    print(f"No corresponding text file found {file_name}.")
                    return None
                return [file_name, text, meta_info]
            return None

        except Exception as e:
            print(f"not find meta info {file_name}, {e}")
            return None

    # 使用线程池处理文件
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(
            executor.map(process_file, dataset_root),
            total=len(dataset_root),
            desc="Processing files"
        ))
    
    # 过滤掉None结果
    data = [r for r in results if r is not None]
    fail_num = len(dataset_root) - len(data)
    print(f"fail_num: {fail_num}")
    
    if not os.path.exists(arrow_dir):
        os.makedirs(arrow_dir)

    if end_id < 0:
        end_id = len(data)
        print(f'start_id:{start_id}  end_id:{end_id}')

    data = data[start_id:end_id]
    num_slice = 5000
    start_sub = int(start_id / num_slice)
    sub_len = int(len(data) // num_slice)
    subs = list(range(sub_len + 1))
    
    # 使用ThreadPool替代Pool
    with ThreadPool() as pool:
        for sub in tqdm(subs):
            arrow_path = os.path.join(arrow_dir, '{}.arrow'.format(str(sub + start_sub).zfill(5)))
            if os.path.exists(arrow_path):
                continue
            print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} start {sub + start_sub}")

            sub_data = data[sub * num_slice: (sub + 1) * num_slice]
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
    # parser = argparse.ArgumentParser(description="Convert images and metadata to Arrow format.")
    # parser.add_argument('--csv_root', type=str, required=True, help='Path to your CSV file or directory containing images.')
    # parser.add_argument('--output_arrow_data_path', type=str, required=True, help='Path for storing the created Arrow file.')
    # parser.add_argument('--pool_num', type=int, default=1, help='Number of processes for multiprocessing (default: 1).')
    # parser.add_argument('--json_path', type=str, default=None, help='Path to the JSON file containing metadata (optional).')

    # args = parser.parse_args()
    import time
    
    csv_root = "/mnt/data/data_nieta/char_old"
    output_arrow_data_path = "/mnt/data/arrowdata/person_detect"
    pool_num = 2
    score_path = None
    danbooru_parquets_path2 ="/mnt/data/Booru-parquets/danbooru.parquet"
    danbooru_parquets_path ="/mnt/data/danbooru_newest-all/table.parquet"


    detect_json_path = "/mnt/data/jsondata/all_person_detected.json"

    detect_data = load_meta_data(detect_json_path)

    detect_data_dict = {}


    


    
    
    
    start = time.time()

    nlp_path = "/mnt/data/Booru-parquets/danbooru_flo2_caption_ft_long.json"
    with open(nlp_path, "r") as f:
        danbooru_flo2_caption_ft_long = json.load(f)
    # 读取单个 Parquet 文件
    df = pd.read_parquet(danbooru_parquets_path2)
    df_add = pd.read_parquet(danbooru_parquets_path)
    
    
    print(f"Time taken to read the Parquet file: {time.time() - start} seconds")


    image_ids = list(detect_data.keys())
    for i,image_id in enumerate(image_ids):
        image_ids[i] = image_id.replace(".webp", "")
    
    print(f"len of image_ids: {len(image_ids)}")

    for i in range(0,len(image_ids),50000):
        image_ids_sub = image_ids[i:i+50000]
        make_arrow_from_list(image_ids_sub, output_arrow_data_path+f"/{i}", df, df_add,danbooru_flo2_caption_ft_long,score_data=None, start_id=0, end_id=-1)
        
    # make_arrow_from_list(image_ids, output_arrow_data_path, df, df_add,danbooru_flo2_caption_ft_long,score_data=None, start_id=0, end_id=-1)


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

def make_arrow_from_dir(dataset_root, arrow_dir, df=None, df2=None, danbooru_flo2_caption_ft_long=None,score_data=None, start_id=0, end_id=-1):
    image_ext = ['webp', 'jpg', 'jpeg', 'png']
    data = []
    fail_num = 0
    for root, dirs, files in os.walk(dataset_root):
        for file in tqdm(files):
            if file.split('.')[-1].lower() in image_ext:
                image_path = os.path.join(root, file)
                file_name = os.path.splitext(os.path.basename(image_path))[0]
                if True:
                    try:
                        if "danbooru_" in file_name:
                            file_name = file_name.split("danbooru_")[1]
                        file_name = int(file_name)
                        # print(f"find meta info for{file_name}")
                        meta_info = gen_meta_by_id(df, int(file_name),danbooru_flo2_caption_ft_long)
                        if meta_info is None and df2 is not None:
                            meta_info = gen_meta_by_id(df2, int(file_name),danbooru_flo2_caption_ft_long)
                            if meta_info is None:
                                print(f"not find meta info for{file_name}")
                                meta_info = {}
                                fail_num += 1
                                continue
                    except Exception as e:
                        print
                        print(f"not find meta info for{file_name}, {e}")
                        meta_info = {}
                        # fail_num += 1
                        # continue
                        

                        
                else:
                    meta_info = {}
                
                if score_data:

                    score = score_data.get(f"danbooru_{file_name}", 0)

                    if score < 0.8:
                        continue
                    
                    meta_info["aesthetic_score_1"] = score
                    
                txt_path = image_path.rsplit('.', 1)[0] + '.txt'
        
                if os.path.exists(txt_path):
                    with open(txt_path, "r") as fp:
                        text = fp.read()
                elif meta_info is not None and isinstance(meta_info, (list, dict)):  # 可以根据实际情况更改类型
                    # 处理 meta_info


                    if "caption_base" in meta_info:
                        text = meta_info["caption_base"]
                    else:
                        text = ""
                        print("No corresponding text file found.")
                        continue
                else:
                    print("No corresponding text file found.")
                    text = ""
                    continue

                data.append([image_path,text,meta_info])
                
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
    
    with Pool() as pool:
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
    output_arrow_data_path = "/mnt/data/arrowdata/char_old"
    pool_num = 2
    pool = Pool(pool_num)
    score_path = None
    danbooru_parquets_path2 ="/mnt/data/Booru-parquets/danbooru.parquet"
    danbooru_parquets_path ="/mnt/data/danbooru_newest-all/table.parquet"



    
    
    start = time.time()

    nlp_path = "/mnt/data/Booru-parquets/danbooru_flo2_caption_ft_long.json"
    with open(nlp_path, "r") as f:
        danbooru_flo2_caption_ft_long = json.load(f)
    # 读取单个 Parquet 文件
    df = pd.read_parquet(danbooru_parquets_path2)
    df_add = None
    
    
    print(f"Time taken to read the Parquet file: {time.time() - start} seconds")




    if os.path.isdir(csv_root):
        make_arrow_from_dir(csv_root, output_arrow_data_path, df, df_add,danbooru_flo2_caption_ft_long,score_data=None, start_id=0, end_id=-1)
    else:
        print("The input file format is not supported. Please input a CSV or JSON file.")
        sys.exit(1)

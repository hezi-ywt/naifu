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
from read_parquet import get_meta_data_by_id, format_meta_data
import time
import tarfile
import io

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
    
def read_images_from_tar(tar_path):
    # 打开 tar 文件
    with tarfile.open(tar_path, 'r') as tar:
        # 遍历 tar 文件中的所有成员
        for member in tar.getmembers():
            # 检查文件是否是图片
            if member.isfile() and member.name.lower().endswith(('webp', '.png', '.jpg', '.jpeg')):
                # 读取文件内容
                file_object = tar.extractfile(member)
                
                if file_object:  # 确保文件对象不为空
                    # 将文件内容读入内存
                    image_data = file_object.read()
                    # 使用 Pillow 打开图片
                    image = Image.open(io.BytesIO(image_data))


def make_arrow_from_tar_dir(dataset_root, arrow_dir, meta_data=None, score_data=None, start_id=0, end_id=-1):
    image_ext = ['jpg', 'jpeg', 'png', 'webp', 'gif', 'tiff']
    data = []
    
    for root, dirs, files in os.walk(dataset_root):
        for file in files:
            if file.endswith(".tar"):
                with tarfile.open(os.path.join(root, file), 'r') as tar:
                    for member in tar.getmembers():
                        if member.isfile() and member.name.lower().endswith(('webp', '.png', '.jpg', '.jpeg')):
                            file_object = tar.extractfile(member)
                            if file_object:
                                image_data = file_object.read()
                                image = Image.open(io.BytesIO(image_data))
                                md5 = hashlib.md5(image_data).hexdigest()
                                width, height = image.size
                                text = ""
                                meta_info = {}
                                data.append([md5, width, height, image_data, text, meta_info])
                image_path = os.path.join(root, file)
                file_name = os.path.splitext(os.path.basename(image_path))[0]
                if meta_data:
                    
                    meta_info = meta_data.get(file_name, None)
                    if not meta_info:
                        print(f"not find meta info for{file_name}")
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

                data.append([image_path,text,meta_info])

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




def make_arrow_from_file_list(file_name_list, dataset_root, arrow_dir, meta_data=None, score_data=None, start_id=0, end_id=-1):
    image_ext = ['jpg', 'jpeg', 'png', 'webp', 'gif', 'tiff']
    data = []
        # 确保 dataset_root 是一个列表
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
                    # print("find file")
                    find = True
                    break
                
        if meta_data:
            
            meta_info = meta_data.get(file_name, None)
            if not meta_info:
                print(f"not find meta info for{file_name}")
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

        data.append([image_path,text,meta_info])

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

    csv_root = ["/data/sdxl/newdb", "/data/sdxl/db"]
    output_arrow_data_path = "/data/artist_arrow_dir"
    pool_num = 1
    json_path = "/app/hfd/caption/danbooru_metainfos_full_20241001.json"
    score_path = "/app/hfd/caption/ws_danbooru.json"

    artisy_json_path = "/app/hfd/caption/hy_artist.json"

    pool = Pool(pool_num)

    meta_data = load_meta_data(json_path) if json_path else None
    score_data = load_meta_data(score_path) if json_path else None
    artist_json = load_meta_data(artisy_json_path) if json_path else None

    for artist_name in tqdm(artist_json):

        file_list = artist_json[artist_name]["image_id"]        
        make_arrow_from_file_list(file_list, csv_root, f"{output_arrow_data_path}/{artist_name}", meta_data, score_data)


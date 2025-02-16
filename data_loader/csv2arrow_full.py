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

def make_arrow_from_dir(dataset_root, arrow_dir, meta_data=None, start_id=0, end_id=-1):
    image_ext = ['jpg', 'jpeg', 'png', 'webp', 'gif', 'tiff']
    data = []
    
    for root, dirs, files in os.walk(dataset_root):
        for file in files:
            if file.split('.')[-1].lower() in image_ext:
                image_path = os.path.join(root, file)

                if meta_data:
                    file_name = os.path.splitext(os.path.basename(image_path))[0]
                    meta_info = meta_data.get(file_name, None)
                    if not meta_info:
                        print(f"not find meta info for{file_name}")
                        meta_info = {}
                else:
                    meta_info = {}



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
    parser = argparse.ArgumentParser(description="Convert images and metadata to Arrow format.")
    parser.add_argument('--csv_root', type=str, required=True, help='Path to your CSV file or directory containing images.')
    parser.add_argument('--output_arrow_data_path', type=str, required=True, help='Path for storing the created Arrow file.')
    parser.add_argument('--pool_num', type=int, default=1, help='Number of processes for multiprocessing (default: 1).')
    parser.add_argument('--json_path', type=str, default=None, help='Path to the JSON file containing metadata (optional).')

    args = parser.parse_args()

    csv_root = args.csv_root
    output_arrow_data_path = args.output_arrow_data_path
    pool_num = args.pool_num
    json_path = args.json_path

    meta_data = load_meta_data(json_path) if json_path else None
    pool = Pool(pool_num)
    if os.path.isdir(csv_root):
        make_arrow_from_dir(csv_root, output_arrow_data_path, meta_data)
    else:
        print("The input file format is not supported. Please input a CSV or JSON file.")
        sys.exit(1)

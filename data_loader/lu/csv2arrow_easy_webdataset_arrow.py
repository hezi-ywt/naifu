# -*- coding: utf-8 -*-
import datetime
import gc
import os
from multiprocessing import Pool
import pandas as pd
import sys  
import pyarrow as pa
from tqdm import tqdm
import hashlib
from PIL import Image

def check_json_file(json_data):
    tag_key_list = ['joycaption','regular_summary',
                        "danbooru_meta","gemini_caption",
                        "tags", "tag", "caption", 
                        "doubao", "wd_tagger", 
                        "midjourney_style_summary",
                        "structural_summary",
                        "deviantart_commission_request",
                        "creation_instructional_summary"
                        ]
    for tag_key in tag_key_list:
        if tag_key in json_data and json_data[tag_key] is not None:
            
            if tag_key == "danbooru_meta":
                # print(f"json_data[tag_key]: {json_data[tag_key]}")
                if isinstance(json_data[tag_key], dict) and len(json_data[tag_key]) > 5:
                    # print(f"find danbooru_meta")
                    return False
            if tag_key == "gemini_caption":
                if isinstance(json_data[tag_key], dict) and "regular_summary" in json_data[tag_key]:
                    if isinstance(json_data[tag_key]["regular_summary"], str) and len(json_data[tag_key]["regular_summary"]) > 20:
                        return False
            if isinstance(json_data[tag_key], str) and len(json_data[tag_key]) > 20:
                return False
    return True

def parse_data(data):
    try:
        img_path = data

        # 读取图像文件，计算 MD5 哈希值
        with open(img_path, "rb") as fp:
            image = fp.read()
            md5 = hashlib.md5(image).hexdigest()
        
        # 获取图像尺寸
        with Image.open(img_path) as f:
            width, height = f.size
        
        # 获取对应的JSON文件路径
        json_path = img_path.rsplit('.', 1)[0] + '.json'
        
        # 检查JSON文件是否存在
        if os.path.exists(json_path):
            # 读取JSON文件内容
            with open(json_path, "r", encoding='utf-8') as fp:
                import json
                json_data = json.load(fp)  # 读取整个JSON内容
                # 检查JSON数据
                if check_json_file(json_data):
                    # print(f"json_data")
                    print(f"json_data: {json_data}")
                    return 
                # 如果需要存储为字符串，可以在这里转换
                json_str = json.dumps(json_data, ensure_ascii=False)
                
        else:
            print(f"No corresponding JSON file found for {img_path}")
            return
        
        # 返回时使用json_str
        return [md5, width, height, image, json_str]
    
    except Exception as e:
        print(f'Error processing {img_path}: {e}')
        return





    
def make_arrow_from_dir(dataset_root, arrow_dir, start_id=0, end_id=-1):
    image_ext = ['jpg', 'jpeg', 'png', 'webp', 'gif', 'tiff']

    image_path_list = []
    for root, dirs, files in os.walk(dataset_root):
        for file in files:
            if file.split('.')[-1].lower() in image_ext:
                image_path = os.path.join(root, file)
                image_path_list.append(image_path)

    if not os.path.exists(arrow_dir):
        os.makedirs(arrow_dir)

    if end_id < 0:
        end_id = len(image_path_list)
    data = image_path_list
    print(f'start_id:{start_id}  end_id:{end_id}')
    data = data[start_id:end_id]

    num_slice = 5000
    start_sub = int(start_id / num_slice)
    sub_len = int(len(data) // num_slice)
    subs = list(range(sub_len + 1))
    
    with Pool() as pool:  # 使用多进程池来并行处理数据
        for sub in tqdm(subs):
            arrow_path = os.path.join(arrow_dir, '{}.arrow'.format(str(sub + start_sub).zfill(5)))
            if os.path.exists(arrow_path):
                continue
            print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} start {sub + start_sub}")

            # No `values` method because `data` is a list
            sub_data = data[sub * num_slice: (sub + 1) * num_slice]

            bs = pool.map(parse_data, sub_data)
            bs = [b for b in bs if b]
            print(f'length of this arrow:{len(bs)}')

            columns_list = ["md5", "width", "height", "image","text_zh"]
            dataframe = pd.DataFrame(bs, columns=columns_list)
            table = pa.Table.from_pandas(dataframe)

            os.makedirs(arrow_dir, exist_ok=True)  # 修正：创建 arrow 文件的目录（而不是原始数据集目录）
            with pa.OSFile(arrow_path, "wb") as sink:
                with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                    writer.write_table(table)

            del dataframe
            del table
            del bs
            gc.collect()



if __name__ == '__main__':

    # pool_num = int(sys.argv[3])
    pool = Pool(6)
    

    make_arrow_from_dir("/nieta/soso/wlop大神鬼刀_4k_filtered_webp_untar", "/nieta/data_arrow/wlop大神鬼刀_4k_filtered_webp_untar_arrow")

    # if len(sys.argv) != 4:
    #     print("Usage: python hydit/data_loader/csv2arrow.py ${csv_root} ${output_arrow_data_path} ${pool_num}")
    #     print("csv_root: The path to your created CSV file. For more details, see https://github.com/Tencent/HunyuanDiT?tab=readme-ov-file#truck-training")
    #     print("output_arrow_data_path: The path for storing the created Arrow file")
    #     print("pool_num: The number of processes, used for multiprocessing. If you encounter memory issues, you can set pool_num to 1")
    #     sys.exit(1)
    # csv_root = sys.argv[1]
    # output_arrow_data_path = sys.argv[2]

    # pool_num = int(sys.argv[3])
    # pool = Pool(pool_num)
    
    # if os.path.isdir(csv_root):
    #     make_arrow_from_dir(csv_root, output_arrow_data_path)

    
    # else:   
    #     print("The input file format is not supported. Please input a CSV or JSON file.")
    #     sys.exit(1)

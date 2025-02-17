import os
import json
import hashlib
from PIL import Image
from tqdm import tqdm

def process_image_data(json_file_path):
    # 读取源JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 使用tqdm添加进度条
    for item in tqdm(data, desc="处理图像数据"):
        try:
            image_path = item['image_path']
            
            # 获取图像文件名
            image_filename = os.path.basename(image_path)
            
            # 创建对应的JSON文件名（替换图像扩展名为.json）
            json_filename = os.path.splitext(image_filename)[0] + '.json'
            
            # 创建输出目录（与图像文件相同目录）
            output_dir = os.path.dirname(image_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # 构建输出JSON路径
            output_json_path = os.path.join(output_dir, json_filename)
            
            # 创建要保存的数据
            output_data = {
                'joycaption': item['joycaption']
            }
            
            # 保存JSON文件
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"处理文件 {image_filename} 时出错: {str(e)}")

if __name__ == "__main__":
    # 假设源JSON文件路径为 data/tpony_data.json
    json_file_path = "/data/soso/2_tpony-v7_metadata_updated.json"
    process_image_data(json_file_path)



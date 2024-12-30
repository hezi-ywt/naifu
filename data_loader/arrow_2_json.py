import json
import pyarrow as pa
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def arrow_to_json(args):
    """将单个arrow文件转换为json，不包含图像数据"""
    arrow_path, json_path = args
    
    try:
        # 读取arrow文件
        table = pa.ipc.RecordBatchFileReader(pa.memory_map(arrow_path)).read_all()
        
        # 转换每条记录
        records = []
        for i in range(len(table)):
            try:
                record = {
                    'md5': table['md5'][i].as_py(),
                    'width': table['width'][i].as_py(),
                    'height': table['height'][i].as_py(),
                    'text_zh': table['text_zh'][i].as_py(),
                    'meta_info': table['meta_info'][i].as_py()
                }
                records.append(record)
                
            except Exception as e:
                print(f"Error processing record {i} in {os.path.basename(arrow_path)}: {str(e)}")
                continue
        
        # 写入json文件
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "total": len(records),
                "records": records
            }, f, ensure_ascii=False, indent=2)
            
        return f"Successfully converted {os.path.basename(arrow_path)}"
    
    except Exception as e:
        return f"Error converting {os.path.basename(arrow_path)}: {str(e)}"

def main():
    # arrow文件目录
    arrow_dir = r"/mnt/data/arrowdata/danbooru2024-webp-4Mpixel"
    # 输出json目录
    json_dir = r"/mnt/data/jsondata/danbooru2024-webp-4Mpixel"
    
    # 创建输出目录
    os.makedirs(json_dir, exist_ok=True)
    
    # 获取所有arrow文件
    arrow_files = sorted([f for f in os.listdir(arrow_dir) if f.endswith('.arrow')])
    
    # 准备任务列表
    tasks = []
    for arrow_file in arrow_files:
        arrow_path = os.path.join(arrow_dir, arrow_file)
        json_path = os.path.join(json_dir, arrow_file.replace('.arrow', '.json'))
        tasks.append((arrow_path, json_path))
    
    # 使用进程池并行处理
    num_workers = max(1, multiprocessing.cpu_count() - 2)  # 留出2个CPU核心给系统
    print(f"Using {num_workers} workers")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(arrow_to_json, tasks),
            total=len(tasks),
            desc="Processing arrow files"
        ))
    
    # 打印结果
    for result in results:
        print(result)

if __name__ == "__main__":
    main()




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
import pymongo
import json
import asyncio
import motor.motor_asyncio
import time
from motor.motor_asyncio import AsyncIOMotorClient
# mongodb://nieta:.nietanieta.@8.153.97.53:27815/


# 创建全局连接客户端，这样就不会为每个文件创建新连接
_mongo_client = None

def get_mongo_client(mongo_uri="mongodb://nieta:.nietanieta.@8.153.97.53:27815/"):
    """获取一个全局共享的MongoDB客户端连接"""
    global _mongo_client
    if _mongo_client is None:
        # 设置连接池选项
        _mongo_client = pymongo.MongoClient(
            mongo_uri,
            maxPoolSize=10,          # 连接池大小
            socketTimeoutMS=30000,   # socket超时时间
            connectTimeoutMS=30000,  # 连接超时时间
            serverSelectionTimeoutMS=30000,  # 服务器选择超时
            waitQueueTimeoutMS=10000 # 等待队列超时
        )
    return _mongo_client

# 替换原来的异步客户端函数，使用连接池
_async_mongo_client = None

def get_async_mongo_client(mongo_uri="mongodb://nieta:.nietanieta.@8.153.97.53:27815/"):
    """获取一个全局共享的异步MongoDB客户端连接"""
    global _async_mongo_client
    if _async_mongo_client is None:
        # 设置连接池选项
        _async_mongo_client = AsyncIOMotorClient(
            mongo_uri,
            maxPoolSize=10,
            socketTimeoutMS=30000,
            connectTimeoutMS=30000,
            serverSelectionTimeoutMS=30000,
            waitQueueTimeoutMS=10000
        )
    return _async_mongo_client

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

# 创建一个同步函数包装异步查询
def get_mongo_data(img_id):
    """同步包装异步查询函数"""
    mongo_db = get_mongo_client()
    collection = mongo_db['captions']
    return collection.find_one({"_id": int(img_id)})

# 修改异步查询函数，解决"Event loop is closed"问题
def get_mongo_data_async(img_id):
    """同步包装异步MongoDB查询函数，使用新的事件循环并确保正确关闭"""
    # 每次创建新的事件循环
    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    try:
        # 运行异步查询
        result = loop.run_until_complete(find_by_id_async(img_id))
        return result
    except Exception as e:
        print(f"执行异步查询时出错: {e}")
        return None
    finally:
        # 确保事件循环被正确关闭
        try:
            # 关闭所有挂起的任务
            pending = asyncio.all_tasks(loop)
            if pending:
                for task in pending:
                    task.cancel()
                # 等待取消所有任务
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass
        
        # 关闭循环
        loop.close()

# 简化异步查询函数
async def find_by_id_async(id, max_retries=3):
    """从多个集合异步查询数据并合并结果，带有重试机制"""
    # 确保id是数字类型
    try:
        numeric_id = int(id)
    except ValueError:
        print(f"无法将ID转换为整数: {id}")
        return None
    
    # 计算gemini_captions_danbooru的集合名
    gemini_collection_name = numeric_id // 100000
    
    # 获取客户端 - 每次查询创建新客户端而不是使用全局变量
    client = AsyncIOMotorClient(
        "mongodb://nieta:.nietanieta.@8.153.97.53:27815/",
        maxPoolSize=10,
        socketTimeoutMS=30000,
        connectTimeoutMS=30000,
        serverSelectionTimeoutMS=30000,
        waitQueueTimeoutMS=10000
    )
    
    db = client["danbooru"]
    gemini_db = client["gemini_captions_danbooru"]
    
    result = None
    try:
        # 添加重试逻辑
        for attempt in range(max_retries):
            try:
                # 并行查询多个集合
                captions_future = db["captions"].find_one({"_id": numeric_id})
                pics_future = db["pics"].find_one({"_id": numeric_id})
                gemini_future = gemini_db[f"{gemini_collection_name}"].find_one({"_id": numeric_id})
                
                # 等待所有查询完成
                captions_result = await captions_future
                pics_result = await pics_future
                gemini_result = await gemini_future
                
                # 构建结果
                result = {}
                # 将captions_result的内容直接展开到结果中
                if captions_result:
                    result.update(captions_result)
                # 将pics_result作为origin_danbooru_data
                if pics_result:
                    result["origin_danbooru_data"] = pics_result
                # 将gemini_result作为gemini_caption_v2
                if gemini_result and "caption" in gemini_result:
                    result["gemini_caption_v2"] = gemini_result["caption"]
                    
                # 成功则跳出重试循环
                break
                
            except Exception as e:
                if attempt < max_retries - 1:
                    # 如果不是最后一次尝试，则等待后重试
                    wait_time = 2 ** attempt  # 指数退避
                    print(f"查询MongoDB出错，{wait_time}秒后重试 (尝试 {attempt+1}/{max_retries}): {e}")
                    await asyncio.sleep(wait_time)
                else:
                    # 最后一次尝试失败
                    print(f"查询MongoDB失败，已重试{max_retries}次: {e}")
    finally:
        # 确保关闭客户端连接
        client.close()
        
    return result

# 修改parse_data函数，添加统计信息
def parse_data(data):
    try:
        img_path = data
        # 添加状态标记
        status = {
            'query_failed': False,
            'no_data': False,
            'success': False
        }
        
        try:
            # 从文件名或路径中提取 ID
            img_id = os.path.basename(img_path).split('.')[0]
            
            try:
                # 使用新的异步查询函数获取MongoDB数据
                mongo_data = get_mongo_data_async(img_id)
                if mongo_data is None:
                    status['no_data'] = True
            except ValueError:
                print(f"无法将图像ID转换为整数: {img_id}")
                status['query_failed'] = True
                return None, status
            except Exception as e:
                print(f"查询MongoDB时出错: {e}")
                status['query_failed'] = True
                mongo_data = None
        except Exception as e:
            print(f"连接MongoDB时出错: {e}")
            status['query_failed'] = True
            mongo_data = None
        
        # 读取图像文件，计算 MD5 哈希值
        with open(img_path, "rb") as fp:
            image = fp.read()
            md5 = hashlib.md5(image).hexdigest()
        
        # 获取图像尺寸
        with Image.open(img_path) as f:
            width, height = f.size
        
        # 使用MongoDB数据
        text_zh_data = {}
        if mongo_data:
            text_zh_data.update(mongo_data)
        
        # 将合并后的数据转换为 JSON 字符串
        final_text_zh = json.dumps(text_zh_data, ensure_ascii=False)
        
        # 设置成功标记
        status['success'] = True
        # 返回时使用 final_text_zh 以及状态信息
        return [md5, width, height, image, final_text_zh], status
    
    except Exception as e:
        print(f'处理 {img_path} 时出错: {e}')
        return None, {'query_failed': True, 'no_data': False, 'success': False}

def make_arrow_from_dir(dataset_root, arrow_dir, start_id=0, end_id=-1, max_workers=1):
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

    # 统计计数器
    total_stats = {
        'total_images': len(data),
        'query_failed': 0,
        'no_data': 0,
        'success': 0,
        'processed_files': 0
    }

    num_slice = 5000
    start_sub = int(start_id / num_slice)
    sub_len = int(len(data) // num_slice)
    subs = list(range(sub_len + 1))
    
    with Pool(processes=max_workers) as pool:  # 使用多进程池来并行处理数据
        for sub in tqdm(subs):
            arrow_path = os.path.join(arrow_dir, '{}.arrow'.format(str(sub + start_sub).zfill(5)))
            if os.path.exists(arrow_path):
                continue
            print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} start {sub + start_sub}")

            # No `values` method because `data` is a list
            sub_data = data[sub * num_slice: (sub + 1) * num_slice]

            # 修改处理结果的接收方式，包含状态信息
            results = pool.map(parse_data, sub_data)
            
            # 分离数据和状态信息
            processed_results = []
            slice_stats = {
                'total': len(sub_data),
                'query_failed': 0,
                'no_data': 0,
                'success': 0
            }
            
            for result in results:
                if result is not None:
                    data_item, status = result
                    if data_item is not None:
                        processed_results.append(data_item)
                    
                    # 更新分片统计
                    if status['query_failed']:
                        slice_stats['query_failed'] += 1
                    if status['no_data']:
                        slice_stats['no_data'] += 1
                    if status['success']:
                        slice_stats['success'] += 1
            
            # 更新总统计
            total_stats['query_failed'] += slice_stats['query_failed']
            total_stats['no_data'] += slice_stats['no_data']
            total_stats['success'] += slice_stats['success']
            total_stats['processed_files'] += 1
            
            # 打印当前分片统计
            print(f"分片统计 - 总数: {slice_stats['total']}, 查询失败: {slice_stats['query_failed']}, 无数据: {slice_stats['no_data']}, 成功: {slice_stats['success']}")
            
            bs = processed_results
            print(f'length of this arrow:{len(bs)}')

            if len(bs) > 0:  # 只有在有数据时才创建Arrow文件
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
        
        # 打印总体统计
        print("\n========== 总体统计 ==========")
        print(f"总图像数: {total_stats['total_images']}")
        print(f"处理的文件数: {total_stats['processed_files']}")
        print(f"查询失败总数: {total_stats['query_failed']}")
        print(f"无数据总数: {total_stats['no_data']}")
        print(f"成功处理总数: {total_stats['success']}")
        print(f"成功率: {(total_stats['success'] / total_stats['total_images'] * 100):.2f}%")
        print("===============================")

if __name__ == '__main__':
    # 数据库连接参数
    # mongo_host = 'your_mongodb_host'
    # mongo_port = 27017
    # mongo_db_name = 'your_database_name'
    # mongo_collection = 'your_collection_name'
    
    # 并行处理数量
    pool = Pool(1)
    
    # 调用处理函数
    make_arrow_from_dir("/mnt/public/dataset/danbooru2024-webp-4Mpixel/images_untar", "/mnt/public/data_arrow/danbooru2024-webp-4Mpixel_arrow", max_workers=1)

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

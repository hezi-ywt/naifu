import pyarrow as pa
import pyarrow.ipc as ipc
import logging
import os
from tqdm import tqdm
import random
# 设置 Logger
logging.basicConfig(filename="/mnt/data/naifu/md5_process.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 读取 Arrow 文件
def read_arrow_file(file_path):
    try:
        with ipc.open_file(file_path) as file:
            table = file.read_all()
            return table
    except Exception as e:
        logger.error(f'Error reading the file: {e}')
        return None

# 获取行的元数据
def get_meta_index(table, index):
    row = {col: table[col][index].as_py() for col in table.column_names}
    return row

# 过滤不寻常的图像比例
def filter_by_image_ratio(table, md5_list):
    for row_index in range(table.num_rows):
        row = get_meta_index(table, row_index)
        md5 = row['md5']
        
        width = row['width']
        height = row['height']
        ratio = width / height
        
        if ratio > 4 or ratio < 0.25:
            logger.info(f'Row {row_index}: MD5 {md5} has an unusual aspect ratio of {ratio}')
            md5_list.append(md5)
    return md5_list

def filter_by_rating_tags(table, md5_list):
    for row_index in range(table.num_rows):
        row = get_meta_index(table, row_index)
        md5 = row['md5']
        try:
            rating = row['meta_info']['rating']
            
            if rating == 'e':
                if random.random() < 0.6:
                    md5_list.append(md5)
            if rating == 'q':
                if random.random() < 0.2:
                    md5_list.append(md5)
        except:
            continue
    return md5_list
                        
if __name__ == '__main__':
    # 主程序逻辑
    md5_list = []
    output_file_name = '/mnt/data/naifu/danbooru_unusual_aspect_ratio_md5.txt'
    processed_files = []
    # 打开输出文件以追加写入
    with open(output_file_name, 'w') as f:
        for i in tqdm(range(1, 1600), desc="Processing files", unit="file"):
            file_path = f'/mnt/data/danbooru2023-arrow/danbooru_{str(i).zfill(5)}.arrow'
            if os.path.exists(file_path):
                logger.info(f'Reading file {file_path}')
                processed_files.append(file_path)
                table = read_arrow_file(file_path)
                if table is not None:
                    md5_list = filter_by_image_ratio(table, md5_list)
                    logger.info(f'Arrow file has {len(md5_list)} images with unusual aspect ratios')

                    # 写入当前文件的 MD5 列表
                    for md5 in md5_list:
                        f.write(f'{md5}\n')
                else:
                    logger.error(f'Error reading the file {file_path}')
            else:
                logger.warning(f'File does not exist: {file_path}')

    # 最后关闭文件
    logger.info(f'Finished processing. Total unusual MD5s: {len(md5_list)}')
    
    # md5_list = []
    # output_file_name = '/mnt/data/naifu/danbooru_1000_unusual_rating_md5.txt'
    # processed_files = []
    # # 打开输出文件以追加写入
    # with open(output_file_name, 'w') as f:
    #     for i in tqdm(range(1, 1000), desc="Processing files", unit="file"):
    #         file_path = f'/mnt/data/danbooru2023-arrow/danbooru_{str(i).zfill(5)}.arrow'
    #         if os.path.exists(file_path):
    #             logger.info(f'Reading file {file_path}')
    #             processed_files.append(file_path)
    #             table = read_arrow_file(file_path)
    #             if table is not None:
    #                 md5_list = filter_by_rating_tags(table, md5_list)
    #                 logger.info(f'Arrow file has {len(md5_list)} images with unusual ratings')

    #                 # 写入当前文件的 MD5 列表
    #                 for md5 in md5_list:
    #                     f.write(f'{md5}\n')
    #             else:
    #                 logger.error(f'Error reading the file {file_path}')
    #         else:
    #             logger.warning(f'File does not exist: {file_path}')
    #     print(f'Finished processing. Total unusual MD5s: {len(md5_list)}')
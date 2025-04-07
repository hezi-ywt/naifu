import os
import tarfile
import re
import time
from pathlib import Path
from typing import Union, Optional
from tqdm import tqdm
import logging
from hfutils.operate import upload_file_to_file, upload_directory_as_archive, upload_directory_as_directory

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# HuggingFace配置
repo_id = 'heziiiii/lu2'
repo_type = "model"
hf_token = os.environ.get("HF_TOKEN")

def find_latest_epoch(base_dir):
    """查找文件夹中最新的epoch更新（-eXX中XX最大的）"""
    pattern = re.compile(r'checkpoint-e(\d+)_s\d+')
    latest_epoch = None
    latest_epoch_num = -1
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            match = pattern.match(item)
            if match:
                epoch_num = int(match.group(1))
                if epoch_num > latest_epoch_num:
                    latest_epoch_num = epoch_num
                    latest_epoch = item
    
    return latest_epoch, latest_epoch_num

def upload_checkpoint(base_dir, checkpoint_folder):
    """上传检查点文件"""
    checkpoint_path = os.path.join(base_dir, checkpoint_folder)
    
    for file in os.listdir(checkpoint_path):
        if file.endswith('lidated.00-of-01.pth'):
            local_file = os.path.join(checkpoint_path, file)
            file_in_repo = f'{os.path.basename(base_dir)}/{checkpoint_folder}/{file}'
            
            logger.info(f"上传文件: {local_file} 到 {file_in_repo}")
            
            try:
                upload_file_to_file(
                    local_file=local_file,
                    repo_id=repo_id,
                    file_in_repo=file_in_repo,
                    repo_type=repo_type,
                    hf_token=hf_token
                )
                logger.info(f"上传成功: {file_in_repo}")
            except Exception as e:
                logger.error(f"上传失败: {e}")

def find_all_checkpoints(base_dir):
    """查找文件夹中所有的epoch检查点并按epoch编号从大到小排序"""
    pattern = re.compile(r'checkpoint-e(\d+)_s\d+')
    checkpoints = []
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            match = pattern.match(item)
            if match:
                epoch_num = int(match.group(1))
                checkpoints.append((item, epoch_num))
    
    # 按epoch编号从大到小排序
    checkpoints.sort(key=lambda x: x[1], reverse=True)
    return checkpoints

def monitor_and_upload(base_dir, check_interval=300):
    """定时监控并按顺序上传检查点（先新后旧）"""
    logger.info(f"开始监控文件夹: {base_dir}")
    uploaded_epochs = set()
    
    while True:
        # 获取所有检查点并按epoch从大到小排序
        all_checkpoints = find_all_checkpoints(base_dir)
        
        # 检查是否有新的检查点需要上传
        new_checkpoints_found = False
        
        for checkpoint, epoch_num in all_checkpoints:
            if checkpoint not in uploaded_epochs:
                new_checkpoints_found = True
                logger.info(f"准备上传检查点: {checkpoint}, epoch数: {epoch_num}")
                upload_checkpoint(base_dir, checkpoint)
                uploaded_epochs.add(checkpoint)
        
        if not new_checkpoints_found:
            logger.info("没有发现新的检查点或所有检查点已上传")
        
        logger.info(f"等待{check_interval}秒后再次检查...")
        time.sleep(check_interval)

if __name__ == "__main__":
    # 设置要监控的基础目录
    base_directory = '/mnt/public/lu2/results_cosine_2e-4_bs64_infsssssss'
    
    # 开始监控并每5分钟(300秒)上传一次
    monitor_and_upload(base_directory)

# 单次上传示例代码(保留作为参考)
"""
local_file = '/mnt/public/lu2/results_cosine_2e-4_bs64_infsss/checkpoint-e12_s23973/consolidated.00-of-01.pth'
file_in_repo = 'results_cosine_2e-4_bs64_infsss/checkpoint-e12_s23973/consolidated.00-of-01.pth'
upload_file_to_file(
    local_file=local_file,
    repo_id=repo_id,
    file_in_repo=file_in_repo,
    repo_type="model",
    hf_token=os.environ.get("HF_TOKEN")
)
"""
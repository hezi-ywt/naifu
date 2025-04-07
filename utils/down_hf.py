from hfutils.operate import download_file_to_file, download_archive_as_directory, download_directory_as_directory
import time
from requests.exceptions import ConnectionError, Timeout

def download_with_retry(download_func, max_retries=5, initial_delay=1, *args, **kwargs):
    """使用重试逻辑包装下载函数"""
    retries = 0
    delay = initial_delay
    
    while retries < max_retries:
        try:
            return download_func(*args, **kwargs)
        except (ConnectionError, Timeout) as e:
            retries += 1
            if retries >= max_retries:
                print(f"下载失败，已达到最大重试次数 ({max_retries}): {e}")
                raise
            
            print(f"连接错误，{delay}秒后重试 ({retries}/{max_retries}): {e}")
            time.sleep(delay)
            # 指数退避策略
            delay *= 2

# # Download a single file from the repository
from hfutils.operate import download_file_to_file, download_archive_as_directory, download_directory_as_directory



# Download files from the repository as a directory tree
# download_directory_as_directory(
#     local_directory='/mnt/g/hf/danbooru_newest-webp-4Mpixel-all',
#     repo_id='deepghs/danbooru_newest-webp-4Mpixel-all')
import requests
import os

# 检查环境变量
print("HTTP_PROXY:", os.environ.get("HTTP_PROXY"))
print("HTTPS_PROXY:", os.environ.get("HTTPS_PROXY"))
print("SOCKS_PROXY:", os.environ.get("SOCKS_PROXY"))

# response = requests.get('https://hf-mirror.com/api/datasets/deepghs/danbooru2024-webp-4Mpixel/revision/main', verify=False)
# download_directory_as_directory(
#     local_directory='/mnt/z/hf/deepghs/danbooru2024-webp-4Mpixel/',
#     repo_id='deepghs/danbooru2024-webp-4Mpixel',
# )
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("SOCKS_PROXY", None)
# Download files from the repository as a directory tree
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
hf_token = os.environ.get("HF_TOKEN")

download_directory_as_directory(
    local_directory='/mnt/public/dataset/danbooru_newest-webp-4Mpixel-all',
    repo_id='deepghs/danbooru_newest-webp-4Mpixel-all',
    hf_token=hf_token
)

# download_with_retry(
#     download_directory_as_directory,
#     local_directory='/root/autodl-tmp/soso/danbooru_images_hy_artist_json_packed',
#     repo_id='heziiiii/soso',
#     dir_in_repo='danbooru_images_hy_artist_json_packed'
# )
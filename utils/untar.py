import tarfile
import os
from pathlib import Path
from typing import Union

def extract_tar(
    tar_path: Union[str, Path], 
    extract_path: Union[str, Path],
    remove_tar: bool = False
) -> bool:
    """
    解压tar文件到指定目录
    
    参数:
        tar_path: tar文件路径
        extract_path: 解压目标路径
        remove_tar: 解压后是否删除原tar文件
    
    返回:
        bool: 解压是否成功
    """
    try:
        # 确保输入路径是Path对象
        tar_path = Path(tar_path)
        extract_path = Path(extract_path)
        
        # 确保解压目录存在
        extract_path.mkdir(parents=True, exist_ok=True)
        
        # 打开并解压tar文件
        with tarfile.open(tar_path, mode='r:*') as tar:
            tar.extractall(path=extract_path)
        
        # 如果需要删除原tar文件
        if remove_tar and tar_path.exists():
            tar_path.unlink()
            
        return True
        
    except Exception as e:
        print(f"解压失败: {str(e)}")
        return False

def is_tar_file(file_path: Union[str, Path]) -> bool:
    """
    检查文件是否为tar格式
    
    参数:
        file_path: 要检查的文件路径
    
    返回:
        bool: 是否为tar文件
    """
    file_path = Path(file_path)
    return tarfile.is_tarfile(file_path) if file_path.exists() else False


if __name__ == "__main__":
    # extract_tar("/data/soso/ai_gen_sample.tar", "/data/soso/tpony_gen_sample")
    dir = "/mnt/public/danbooru/images"
    output_dir = "/mnt/public/danbooru/images_untar"
    for file in os.listdir(dir):
        if file.endswith(".tar"):
            print(file)
            extract_tar(os.path.join(dir, file),output_dir)

import os
import shutil
import re

def copy_images_to_directory(source_directory, target_directory):
    # 遍历源目录中的所有文件
    os.makedirs(target_directory, exist_ok=True)
    for root, dirs, files in os.walk(source_directory):
        for file in files:
            # 检查文件扩展名是否为图像格式
            if file.lower().endswith(('webp', '.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                # 构建源文件路径
                source_file = os.path.join(root, file)
                # 复制文件到源目录
                shutil.copy(source_file, target_directory)

# 使用示例
def copy_json_to_directory(source_directory, target_directory):
    os.makedirs(target_directory, exist_ok=True)
    for root, dirs, files in os.walk(source_directory):
        for file in files:
            if file.lower().endswith('.json'):
                source_file = os.path.join(root, file)
                shutil.copy(source_file, target_directory)
                
def rename_files_with_pid(source_directory, target_directory,ext=None):
    """
    从源目录复制文件到目标目录，并用文件名中的pid作为新文件名
    
    Args:
        source_directory (str): 源目录路径
        target_directory (str): 目标目录路径
    """
    # 确保目标目录存在
    os.makedirs(target_directory, exist_ok=True)
    
    # 遍历源目录中的所有文件
    for root, dirs, files in os.walk(source_directory):
        if ext is not None:
            files = [file for file in files if file.lower().endswith(ext)]
        for file in files:
            # 尝试从文件名中提取pid
            pid_match = re.search(r'\[pid=(\d+)\]', file)
            if pid_match:
                # 提取pid
                pid = pid_match.group(1)
                
                # 获取文件扩展名
                _, ext = os.path.splitext(file)
                
                # 构建新文件名
                new_filename = f"{pid}{ext}"
                
                # 构建源文件和目标文件的完整路径
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_directory, new_filename)
                
                # 复制文件并重命名
                shutil.copy(source_file, target_file)
                print(f"已复制 {file} 到 {new_filename}")

image_ext = ('.webp', '.png', '.jpg', '.jpeg', '.gif', '.bmp')


# copy_images_to_directory('/mnt/public/soso/shiertier', '/mnt/public/soso/shiertier_')
copy_json_to_directory('/mnt/public/soso/nieta/soso/shiertierjson_pid_', '/mnt/public/soso/shiertier_2')

# rename_files_with_pid('/mnt/public/soso/shiertier_', '/mnt/public/soso/shiertier_2', ext=image_ext)
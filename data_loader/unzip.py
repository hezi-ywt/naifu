import os
import tarfile

def extract_all_tars_in_folder(input_folder, output_folder):
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.tar'):
            tar_path = os.path.join(input_folder, filename)
            print(f"正在提取: {tar_path}")
            
            # 打开 tar 文件并提取
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path=output_folder)
                print(f"已提取到: {output_folder}")

# 示例用法
input_folder = '/mnt/data/danbooru2024-webp-4Mpixel/images'  # 替换为你的输入文件夹路径
output_folder = '/mnt/data/danbooru2024-webp-4Mpixel/images_out'  # 替换为输出文件夹路径
extract_all_tars_in_folder(input_folder, output_folder)


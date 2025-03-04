import os
import json
from pathlib import Path

def build_json_captions(image_dir):
    """
    遍历指定目录下的所有图像文件，读取对应的txt文件内容，
    并为每个图像创建同名的json文件保存caption信息
    
    Args:
        image_dir (str): 包含图像和txt文件的目录路径
    """
    # 支持的图像格式
    image_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    
    # 初始化计数器
    processed_count = 0
    error_count = 0
    
    print(f"开始处理目录: {image_dir}")
    
    # 获取目录下所有文件
    for file_path in Path(image_dir).rglob('*'):
        if file_path.suffix.lower() in image_extensions:
            # 获取相对路径用于显示
            rel_path = file_path.relative_to(image_dir)
            print(f"正在处理: {rel_path}")
            
            # 获取对应的txt文件路径
            txt_path = file_path.with_suffix('.txt')
            
            # 如果txt文件存在，读取内容
            if txt_path.exists():
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                    
                    # 创建json数据
                    json_data = {
                        "caption": caption
                    }
                    
                    # 保存为同名json文件
                    json_path = file_path.with_suffix('.json')
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, ensure_ascii=False, indent=2)
                        
                    processed_count += 1
                    print(f"✓ 已完成: {rel_path}")
                    
                except Exception as e:
                    error_count += 1
                    print(f"✗ 错误: {rel_path} - {str(e)}")
            else:
                error_count += 1
                print(f"✗ 未找到txt文件: {rel_path}")
    
    print(f"\n处理完成:")
    print(f"- 成功处理: {processed_count} 个文件")
    print(f"- 处理失败: {error_count} 个文件")
    return processed_count, error_count

if __name__ == "__main__":
    # 设置要处理的目录路径
    image_directory = "/nieta/soso/flux_lora"
    build_json_captions(image_directory)

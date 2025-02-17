import os
import json
import socket
import subprocess

def repath_images(data_list, base_path="/data/soso/tpony_gen_sample/"):
    """
    从指定目录遍历文件，通过 base_name 匹配并更新图片路径
    
    Args:
        data_list: 包含图片信息的字典列表
        base_path: 要遍历的基础路径
    
    Returns:
        更新后的数据列表
    """
    # 创建 base_name 到完整数据项的映射
    base_name_map = {}
    for item in data_list:
        if "base_name" in item:
            base_name_map[item["base_name"]] = item
    
    # 遍历目录下的所有文件
    for root, _, files in os.walk(base_path):
        for filename in files:
            if filename in base_name_map:
                # 找到匹配的文件，更新路径
                item = base_name_map[filename]
                if "image_path" in item:
                    # 构建新路径（使用实际找到的文件路径）
                    new_path = os.path.join(root, filename)
                    # 统一使用正斜杠
                    new_path = new_path.replace("\\", "/")
                    # 更新路径
                    item["image_path"] = new_path
            
    return data_list

def replace_path_prefix(data_list, old_prefix, new_prefix):
    """
    替换图片路径中的指定前缀
    
    Args:
        data_list: 包含图片信息的字典列表
        old_prefix: 要替换的旧路径前缀
        new_prefix: 新的路径前缀
    
    Returns:
        更新后的数据列表
    """
    # 统一路径分隔符
    old_prefix = old_prefix.replace("\\", "/")
    new_prefix = new_prefix.replace("\\", "/")
    
    for item in data_list:
        if "image_path" in item:
            old_path = item["image_path"].replace("\\", "/")
            if old_path.startswith(old_prefix):
                # 替换前缀
                new_path = old_path.replace(old_prefix, new_prefix, 1)
                item["image_path"] = new_path
    
    return data_list

def add_duso_permission(data_list):
    """
    为数据列表中的每个项目添加 duso 权限
    
    Args:
        data_list: 包含图片信息的字典列表
    
    Returns:
        更新后的数据列表
    """
    for item in data_list:
        item['permission'] = 'duso'
    return data_list

def check_port_proxy(port=None):
    """
    查看指定端口或所有端口的代理信息
    
    Args:
        port: 可选，指定要查看的端口号
    
    Returns:
        代理信息字符串
    """
    try:
        # 在Linux系统下使用 netstat 命令
        cmd = "netstat -tunlp" if os.name != 'nt' else "netstat -ano"
        result = subprocess.check_output(cmd, shell=True).decode()
        
        # 按行分割结果
        lines = result.split('\n')
        proxy_info = []
        
        for line in lines:
            if port is None or str(port) in line:
                proxy_info.append(line.strip())
        
        return "\n".join(proxy_info)
    except Exception as e:
        return f"获取端口信息失败: {str(e)}"

# 使用示例:
if __name__ == "__main__":
    base_path = "/data/soso/tpony_gen_sample"
    json_path = "/data/soso/2_tpony-v7_metadata.json"
    
    # 查看端口代理信息
    print("当前端口代理信息:")
    print(check_port_proxy())
    
    # 查看特定端口
    print("\n8080端口信息:")
    print(check_port_proxy(8080))
    
    # 示例：替换路径前缀
    old_prefix = "Z:/DATA/diffusion/artist/2_tpony-v7/2_tpony-v7"
    new_prefix = "/data/soso/tpony_gen_sample"
    
    # 从JSON文件加载数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 添加 duso 权限
    data = add_duso_permission(data)
    
    # 方法1：通过遍历目录匹配文件
    updated_data = repath_images(data, base_path)
    
    # 方法2：直接替换路径前缀
    updated_data = replace_path_prefix(data, old_prefix, new_prefix)
    
    # 保存更新后的数据
    with open(json_path.replace('.json', '_updated.json'), 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, ensure_ascii=False, indent=2)

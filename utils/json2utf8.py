#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import sys
from typing import Any, Dict, List, Union, Optional
import chardet  # 新增导入 chardet 库

def convert_json_string_to_utf8(json_str: str) -> str:
    """
    将JSON字符串转换为UTF-8编码
    
    Args:
        json_str: 输入的JSON字符串
        
    Returns:
        UTF-8编码的JSON字符串
    """
    # 解析JSON字符串为Python对象
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"无效的JSON字符串: {e}")
    
    # 重新编码为UTF-8
    return json.dumps(data, ensure_ascii=False, indent=2)

def convert_json_file_to_utf8(input_file: str, output_file: Optional[str] = None) -> None:
    """
    读取JSON文件并将其转换为UTF-8编码
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径，如果为None则覆盖原文件
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"找不到文件: {input_file}")
    
    # 如果没有指定输出文件，则使用输入文件路径
    if output_file is None:
        output_file = input_file
    
    # 读取JSON文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            json_str = f.read()
    except UnicodeDecodeError:
        # 使用 chardet 检测文件编码
        with open(input_file, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']
            
            if encoding is None or confidence < 0.7:
                # 如果检测结果不可靠，尝试其他常见编码
                encodings = ['latin-1', 'gbk', 'gb2312', 'iso-8859-1']
                for enc in encodings:
                    try:
                        json_str = raw_data.decode(enc)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError(f"无法识别文件编码: {input_file}")
            else:
                # 使用检测到的编码
                try:
                    json_str = raw_data.decode(encoding)
                except UnicodeDecodeError:
                    raise ValueError(f"使用检测到的编码 {encoding} 无法解码文件: {input_file}")
    
    # 转换为UTF-8
    utf8_json = convert_json_string_to_utf8(json_str)
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(utf8_json)

def batch_convert_directory(directory: str, pattern: str = "*.json", recursive: bool = False) -> List[str]:
    """
    批量转换目录中的所有JSON文件为UTF-8编码
    
    Args:
        directory: 目录路径
        pattern: 文件匹配模式，默认为"*.json"
        recursive: 是否递归处理子文件夹，默认为False
        
    Returns:
        成功转换的文件列表
    """
    import glob
    
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"无效的目录: {directory}")
    
    converted_files = []
    total_files = 0
    error_files = 0
    
    if recursive:
        for root, _, _ in os.walk(directory):
            file_pattern = os.path.join(root, pattern)
            files = glob.glob(file_pattern)
            total_files += len(files)
            
            for i, file_path in enumerate(files):
                try:
                    print(f"处理 [{i+1}/{len(files)}]: {file_path}")
                    convert_json_file_to_utf8(file_path)
                    converted_files.append(file_path)
                except Exception as e:
                    error_files += 1
                    print(f"转换文件 {file_path} 时出错: {e}")
    else:
        file_pattern = os.path.join(directory.rstrip('/'), pattern)
        files = glob.glob(file_pattern)
        total_files = len(files)
        
        for i, file_path in enumerate(files):
            try:
                print(f"处理 [{i+1}/{total_files}]: {file_path}")
                convert_json_file_to_utf8(file_path)
                converted_files.append(file_path)
            except Exception as e:
                error_files += 1
                print(f"转换文件 {file_path} 时出错: {e}")
    
    print(f"\n处理完成:")
    print(f"- 总文件数: {total_files}")
    print(f"- 成功处理: {len(converted_files)}")
    print(f"- 处理失败: {error_files}")
    
    return converted_files

if __name__ == "__main__":
    # 命令行接口增强
    import argparse
    
    parser = argparse.ArgumentParser(description="将JSON文件转换为UTF-8编码")
    parser.add_argument("input", help="输入文件或目录路径")
    parser.add_argument("-o", "--output", help="输出文件路径 (仅在处理单个文件时有效)")
    parser.add_argument("-r", "--recursive", action="store_true", help="递归处理子目录")
    parser.add_argument("-p", "--pattern", default="*.json", help="文件匹配模式 (默认: '*.json')")
    
    args = parser.parse_args()
    
    try:
        if os.path.isdir(args.input):
            # 目录模式 - 批量处理
            converted = batch_convert_directory(args.input, args.pattern, args.recursive)
            print(f"成功转换 {len(converted)} 个文件")
        else:
            # 单文件模式
            convert_json_file_to_utf8(args.input, args.output)
            print(f"成功将 {args.input} 转换为UTF-8编码" + 
                  (f" 并保存为 {args.output}" if args.output else ""))
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

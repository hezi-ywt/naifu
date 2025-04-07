# -*- coding: utf-8 -*-
import os
import sys
import argparse
import pyarrow as pa
import pandas as pd
from PIL import Image
import io
import json
from tqdm import tqdm
import numpy as np

def load_arrow_file(arrow_path):
    """加载Arrow文件并返回Table对象"""
    try:
        with pa.memory_map(arrow_path, 'r') as source:
            table = pa.ipc.RecordBatchFileReader(source).read_all()
        return table
    except Exception as e:
        print(f"加载Arrow文件时出错: {e}")
        return None

def display_arrow_info(table):
    """显示Arrow表的基本信息"""
    print("=== Arrow文件信息 ===")
    print(f"行数: {table.num_rows}")
    print(f"列数: {table.num_columns}")
    print("\n=== 表结构 ===")
    print(table.schema)
    
    # 显示每列的数据类型和大小
    print("\n=== 列详情 ===")
    for i, field in enumerate(table.schema):
        col = table.column(i)
        size_mb = col.nbytes / (1024 * 1024)
        print(f"{field.name}: 类型={field.type}, 大小={size_mb:.2f}MB")

def display_text_content(table, start_idx=0, end_idx=5, show_image=False, save_image=False, output_dir=None):
    """显示指定行范围的文本内容"""
    if end_idx > table.num_rows:
        end_idx = table.num_rows
        
    print(f"\n=== 显示行 {start_idx} 到 {end_idx-1} 的内容 ===")
    
    # 转换为pandas DataFrame更方便处理
    df = table.to_pandas()
    
    for idx in range(start_idx, end_idx):
        print(f"\n--- 行 {idx} ---")
        row = df.iloc[idx]
        
        # 显示基本信息
        print(f"MD5: {row['md5']}")
        print(f"尺寸: {row['width']} x {row['height']}")
        
        # 显示文本内容
        if 'text_zh' in row:
            try:
                text_data = json.loads(row['text_zh'])
                print("\n文本内容:")
                print(json.dumps(text_data, indent=2, ensure_ascii=False))
            except:
                print(f"\n文本内容: {row['text_zh']}")
        
        # 处理图像显示或保存
        if (show_image or save_image) and 'image' in row:
            try:
                img = Image.open(io.BytesIO(row['image']))
                if save_image and output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    img_path = os.path.join(output_dir, f"{row['md5']}.png")
                    img.save(img_path)
                    print(f"图像已保存到: {img_path}")
                
                if show_image:
                    img.show()
            except Exception as e:
                print(f"处理图像时出错: {e}")

def search_text(table, keyword, case_sensitive=False):
    """在文本内容中搜索关键词"""
    print(f"\n=== 搜索关键词: '{keyword}' ===")
    
    df = table.to_pandas()
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if 'text_zh' in row:
            text = row['text_zh']
            if not case_sensitive:
                text_lower = text.lower()
                keyword_lower = keyword.lower()
                if keyword_lower in text_lower:
                    results.append(idx)
            else:
                if keyword in text:
                    results.append(idx)
    
    print(f"找到 {len(results)} 条匹配结果")
    return results

def main():
    parser = argparse.ArgumentParser(description='Arrow文件内容查看工具')
    parser.add_argument('arrow_path', help='Arrow文件路径')
    parser.add_argument('--start', type=int, default=0, help='开始行索引')
    parser.add_argument('--end', type=int, default=5, help='结束行索引')
    parser.add_argument('--show-image', action='store_true', help='显示图像')
    parser.add_argument('--save-image', action='store_true', help='保存图像')
    parser.add_argument('--output-dir', help='图像保存目录')
    parser.add_argument('--search', help='搜索关键词')
    parser.add_argument('--case-sensitive', action='store_true', help='区分大小写搜索')
    
    args = parser.parse_args()
    
    table = load_arrow_file(args.arrow_path)
    if table is None:
        return
    
    display_arrow_info(table)
    
    if args.search:
        result_indices = search_text(table, args.search, args.case_sensitive)
        if result_indices and len(result_indices) > 0:
            # 只显示前10个结果
            display_indices = result_indices[:10] if len(result_indices) > 10 else result_indices
            display_text_content(table, 
                              start_idx=min(display_indices), 
                              end_idx=max(display_indices)+1, 
                              show_image=args.show_image,
                              save_image=args.save_image,
                              output_dir=args.output_dir)
    else:
        display_text_content(table, 
                          start_idx=args.start, 
                          end_idx=args.end, 
                          show_image=args.show_image,
                          save_image=args.save_image,
                          output_dir=args.output_dir)

if __name__ == '__main__':
    main() 
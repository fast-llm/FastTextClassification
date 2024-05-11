#!/usr/bin/env python
# coding=utf-8
"""
@author: AmbroseX
@contact: utils.py
@file: 
@date: 2024/04/30 17:00:29
@desc: 
"""
from datetime import timedelta
import json
import os
import time
from typing import Dict, List

from extras.loggings import get_logger

logger = get_logger(__name__)

def count_lines_in_jsonl_file(file_path: str) -> int:
    """
    计算 JSONL 文件的行数，每一行代表一个 JSON 对象。
    :param file_path: JSONL 文件的路径。
    :return: 文件中的行数。
    """
    line_count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for _ in file:
                line_count += 1
        print(f"文件中共有 {line_count} 行")
    except Exception as e:
        print(f"读取文件时出错：{file_path}，错误信息：{e}")
        raise Exception(f"读取文件时出错：{file_path}，错误信息：{e}")

    return line_count

def read_data(data_path:str,
              sep:str='\t', 
              sheet_name=None,
              label_col:str='label',
              text_col:str='review'
              )->list:
    if data_path.endswith('.jsonl'):
        res = read_jsonl(data_path)
        data = []
        for item in res:
            data.append([item[text_col], item[label_col]])
        text_index = 0
        label_index = 1
    if data_path.endswith('.json'):
        res = read_json(data_path)
        data = []
        for item in res:
            data.append([item[text_col], item[label_col]])
        text_index = 0
        label_index = 1
    elif data_path.endswith('.txt'):
        with open(data_path, 'r', encoding='utf-8') as file:
            data = [line.strip().split(sep) for line in file if line.strip()]
            text_index = 0
            label_index = 1
    elif data_path.endswith('.xlsx'):
        import pandas as pd
        data = pd.read_excel(data_path, sheet_name=sheet_name)
        data = data.values.tolist()
        headers = data[0]
        text_index = headers.index(text_col)
        label_index = headers.index(label_col)
    elif data_path.endswith('.csv'):
        import pandas as pd
        import csv
        with open(data_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            data = [row for row in reader]
            headers = data[0]
            data = data[1:]
            text_index = headers.index(text_col)
            label_index = headers.index(label_col)
    data = [{
             'text': row[text_index],
             'label': str(row[label_index])
             } for row in data if row]  # Format data
    return data

def read_txt(file_path: str) -> str:
    """
    读取txt文件内容的函数
    :param file_path: 文件路径，类型为字符串
    :return: 文件内容，类型为字符串
    """
    with open(file_path, 'r') as f:
        content = f.read()
    return content

def read_lines(file_path: str) -> list:
    """
    逐行读取txt文件内容的函数，移除空行和仅含空白字符的行
    :param file_path: 文件路径，类型为字符串
    :return: 文件的每一行（已清除空白字符）作为一个元素的列表，类型为字符串列表
    """
    with open(file_path, 'r') as f:
        # 读取所有行，并移除每行末尾的空白字符，同时过滤掉那些仅包含空白字符的行
        lines = [line.strip() for line in f if line.strip()]

    return lines


def read_xlsx(file_path:str,sheet_name:str='工作表1'):
    import pandas as pd
    import openpyxl
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df = df.replace('\xa0', ' ', regex=True)  # 这里将\xa0替换为空字符串，也可以替换为普通空格' '，根据需要选择
    # 将DataFrame的每一行作为一个列表元素放入大列表中
    rows_list = [row._asdict().values() for row in df.itertuples(index=False)]
    return rows_list

def save_xlsx(
            data:list,
            file_path:str,
            sheet_name:str='sheet1'):
    from openpyxl import Workbook
    from openpyxl.styles import Font

    wb = Workbook()
    ws = wb.active

    # 设置默认的字体样式
    font = Font(name='Calibri', size=11)
    wb.styles.fonts.append(font)
    wb.styles.named_styles['Normal'].font = font

    # 保存工作簿
    wb.save('file_path')
    pass


def read_json(file_path: str) -> dict: # type: ignore
    """
    读取JSON文件并返回字典
    :param file_path: JSON文件的路径
    :return: JSON文件内容转换后的字典
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
           data = json.load(file)
        return data
    except Exception as e:
        logger.info(f"读取JSON文件失败：{file_path}，错误信息：{e}")
        raise BaseException(f"读取JSON文件失败：{file_path}，错误信息：{e}")

def read_jsonl(file_path: str) -> List[Dict]:
    """
    读取.jsonl文件并返回包含所有JSON对象的列表。

    参数:
        file_path (str): .jsonl文件的路径。

    返回:
        List[Dict]: 包含文件中所有JSON对象的列表。
    """
    try:
        data = []  # 初始化一个空列表，用于存储解析后的JSON对象
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                json_obj = json.loads(line)  # 将每一行的字符串转换为字典
                data.append(json_obj)  # 添加到列表中
        return data
    except Exception as e:
        logger.error(f"读取JSONL文件失败：{file_path}，错误信息：{e}")
        raise BaseException(f"读取JSONL文件失败：{file_path}，错误信息：{e}")

def export_json(data: dict, file_path: str):
    """
    导出字典到指定路径的JSON文件。
    :param data: 要导出的字典数据。
    :param file_path: 目标JSON文件的路径。
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        logger.info(f"数据成功导出到JSON文件：{file_path}")
    except Exception as e:
        logger.error(f"导出JSON文件失败：{file_path}，错误信息：{e}")
        raise Exception(f"导出JSON文件失败：{file_path}，错误信息：{e}")

def export_jsonl(data: list, file_path: str):
    """
    将字典列表导出到指定路径的 JSONL 文件。
    :param data: 要导出的字典列表。
    :param file_path: 目标 JSONL 文件的路径。
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            for item in data:
                # 将每个字典转换为 JSON 字符串，并确保不转义 Unicode 字符
                json_string = json.dumps(item, ensure_ascii=False)
                # 写入 JSON 字符串到文件，并添加换行符以形成 JSONL 格式
                file.write(json_string + '\n')
        print(f"数据成功导出到 JSONL 文件：{file_path}")
    except Exception as e:
        print(f"导出 JSONL 文件失败：{file_path}，错误信息：{e}")
        raise Exception(f"导出 JSONL 文件失败：{file_path}，错误信息：{e}")

def get_file_name(file_path:str,file_type:str='.json', endwith:str ='_object')->str:
    """
    Get the file name from a file path.

    Parameters:
        file_path (str): The path to the file.

    Returns:
        str: The file name.
    """
    return os.path.basename(file_path).split(file_type)[0].replace(endwith,'')

def get_dir_of_file(file_path: str) -> str:
    """
    返回给定文件路径的所在目录。
    :param file_path: 文件的完整路径。
    :return: 包含文件的目录路径。
    """
    directory_path = os.path.dirname(file_path)
    return directory_path

def check_dir_exist(dir_path: str = "", create: bool = False) -> bool:
    """
    检查给定的路径是否存在，如果不存在，则创建对应的文件夹。
    :param path: 要检查和创建的文件夹路径。
    """
    if not os.path.exists(dir_path):
        if create:
            os.makedirs(dir_path)
            logger.info(f"创建了文件夹: {dir_path}")
            return True
        return False
    else:
        return True

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
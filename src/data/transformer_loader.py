# coding: UTF-8
import torch
import torch.nn as nn
from tqdm import tqdm
import time
from datetime import timedelta
import csv
from torch.utils.data import DataLoader, Dataset, IterableDataset
import numpy as np
from tqdm import tqdm
import json

from data.common import DatasetConfig
from data.utils_multi import get_multi_hot_label, get_multi_list_label

from extras.loggings import get_logger
from models.component.common import ModelConfig
from utils import count_lines_in_jsonl_file, read_data
logger = get_logger(__name__)


PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号, SEP并非必须


class DynamicDataset(IterableDataset):
    def __init__(self, path: str, tokenizer: "torch.nn.module", tag: str, pad_size: int, 
                 max_samples: int, num_classes: int, multi_class: bool, multi_label: bool):
        self.path = path
        self.tokenizer = tokenizer
        self.tag = tag
        self.pad_size = pad_size
        self.max_samples = max_samples
        self.num_classes = num_classes
        self.multi_class = multi_class
        self.multi_label = multi_label
        self.logger =get_logger(f"load:{tag}")
        self.lines = count_lines_in_jsonl_file(self.path)

    def __len__(self):
        return self.lines

    def __iter__(self):
        flag = 0
        line_num = 0
        with open(self.path, 'r', encoding='UTF-8') as f:
            self.logger.info(f"Loading {self.tag} dataset from \n{self.path}")
            for line in tqdm(f):
                if self.max_samples is not None and line_num >= self.max_samples:
                    break
                # Assuming each line in the file is a JSON string
                data = json.loads(line)
                content, label = data['text'], data['label']
                if flag < 2:
                    self.logger.info(f"content: {content}\nlabel: {label}")
                    flag += 1

                label = get_multi_list_label(label=label,
                                             multi_class=self.multi_class,
                                             multi_label=self.multi_label)
                label = get_multi_hot_label(doc_labels=[label],
                                            num_class=self.num_classes,
                                            dtype=torch.float)[0]

                inputs = self.tokenizer(content,
                                        truncation=True,
                                        return_tensors="pt",
                                        padding='max_length',
                                        max_length=self.pad_size)

                yield (inputs['input_ids'].squeeze(0),
                       inputs['attention_mask'].squeeze(0),
                       inputs['token_type_ids'].squeeze(0), 
                       label)
                line_num += 1

def load_dataset(path:str,
                 tokenizer:"torch.nn.module", 
                 tag:str,
                 pad_size:str,
                 max_samples:int,
                 num_classes:int,
                 multi_class:bool,
                 multi_label:bool,
                 )-> list[tuple]:
    """_summary_
    Args:
        path (str): _description_
        tokenizer (torch.nn.module): _description_
        tag (str): _description_
        SEP (str): _description_
        pad_size (str): _description_
        max_samples (int): _description_
        num_classes (int): _description_
        multi_class (bool): _description_
        multi_label (bool): _description_

    Returns:
        list[tuple]: _description_
    """
    batch_contents = []
    flag = 0
    with open(path, 'r', encoding='UTF-8', newline='') as f:
        logger.info(f"loading {tag} dataset from \n{path}")
        data = read_data(data_path=path,
                         text_col='text',
                         label_col='label'
                         )
        line_num = 0
        for line in tqdm(data):
            if max_samples is not None and line_num > max_samples:
                break
            if line:
                content, label = line['text'],line['label']
                if flag<2:
                    logger.info(f"content: {content}\nlabel: {label}")
                    flag += 1
                # 转换数据为list
                label = get_multi_list_label(label=label,
                                                multi_class=multi_class,
                                                multi_label=multi_label)
                label = get_multi_hot_label(doc_labels = [label], 
                                            num_class = num_classes, 
                                            dtype=torch.float)[0]
                
                inputs = tokenizer(content,
                                    truncation=True,  
                                    return_tensors="pt", 
                                    padding='max_length', 
                                    max_length=pad_size
                                    )
                batch_contents.append((inputs['input_ids'],
                                inputs['attention_mask'],
                                inputs['token_type_ids'], 
                                label)
                                )
                line_num+=1
    return batch_contents

def build_dataset(config: "ModelConfig") -> tuple:
    if config.streaming:
        train_dataset = DynamicDataset(
            config.train_path, tag='training',
            tokenizer=config.tokenizer,
            pad_size = config.pad_size,
            max_samples = config.max_samples,
            num_classes = config.num_classes,
            multi_class = config.multi_class,
            multi_label = config.multi_label,
        )

        val_dataset = DynamicDataset(
            config.val_path, tag = 'validation',
            tokenizer=config.tokenizer,
            pad_size = config.pad_size,
            max_samples = config.max_samples,
            num_classes = config.num_classes,
            multi_class = config.multi_class,
            multi_label = config.multi_label,
        )
        test_dataset = DynamicDataset(
            config.test_path, tag = 'test',
            tokenizer=config.tokenizer,
            pad_size = config.pad_size,
            max_samples = config.max_samples,
            num_classes = config.num_classes,
            multi_class = config.multi_class,
            multi_label = config.multi_label,
        )
    else:
        train_data = load_dataset(config.train_path, tag='training',
                            tokenizer=config.tokenizer,
                            pad_size = config.pad_size,
                            max_samples = config.max_samples,
                            num_classes = config.num_classes,
                            multi_class = config.multi_class,
                            multi_label = config.multi_label,
                            )
        val_data = load_dataset(config.val_path, tag = 'validation',
                        tokenizer=config.tokenizer,
                            pad_size = config.pad_size,
                            max_samples = config.max_samples,
                            num_classes = config.num_classes,
                            multi_class = config.multi_class,
                            multi_label = config.multi_label,
                        )
        test_data = load_dataset(config.test_path, tag = 'test',
                            tokenizer=config.tokenizer,
                            pad_size = config.pad_size,
                            max_samples = config.max_samples,
                            num_classes = config.num_classes,
                            multi_class = config.multi_class,
                            multi_label = config.multi_label,
                                )
        train_dataset = TransformerDataset(train_data)
        val_dataset = TransformerDataset(val_data)
        test_dataset =TransformerDataset(test_data)
    return train_dataset, val_dataset, test_dataset

class TransformerDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input = (self.data[idx][0],self.data[idx][1],self.data[idx][2])
        label = self.data[idx][3]
        return input, label
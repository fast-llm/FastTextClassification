# coding: UTF-8
import torch
import torch.nn as nn
from tqdm import tqdm
import time
from datetime import timedelta
import csv
from torch.utils.data import DataLoader, Dataset
from data.common import DatasetConfig
from data.utils_multi import get_multi_hot_label, get_multi_list_label

from extras.loggings import get_logger
from models.component.common import ModelConfig
logger = get_logger(__name__)


PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号, SEP并非必须


class TransformerDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input = (self.data[idx][0],self.data[idx][1],self.data[idx][2])
        label = self.data[idx][3]
        return input, label


def build_dataset(config: "ModelConfig") -> tuple:
    def load_dataset(path:str, tag:str)-> list[tuple]:
        batch_contents = []
        flag = 0
        with open(path, 'r', encoding='UTF-8', newline='') as f:
            logger.info(f"loading {tag} dataset from \n{path}")
            f = csv.reader(f, delimiter=config.SEP)
            for line in tqdm(f):
                if config.max_samples is not None and f.line_num > config.max_samples:
                    break
                if len(line) != 2:
                    logger.error(f"行格式错误: {line}")
                    continue
                content, label = line
                if flag<2:
                    logger.info(f"content: {content}\nlabel: {label}")
                    flag += 1
                # 转换数据为list
                label = get_multi_list_label(label=label,
                                             multi_class=config.multi_class,
                                             multi_label=config.multi_label)
                
                inputs = config.tokenizer(content,
                                            truncation=True,  
                                            return_tensors="pt", 
                                            padding='max_length', 
                                            max_length=config.pad_size)
                batch_contents.append((inputs['input_ids'].squeeze(0),
                                inputs['attention_mask'].squeeze(0),
                                inputs['token_type_ids'].squeeze(0),
                                label)
                                )

        return batch_contents

    train = load_dataset(config.train_path, tag='training')
    val = load_dataset(config.val_path, tag = 'validation')
    test = load_dataset(config.test_path, tag = 'test')
    return train, val, test


class DatasetIterater(object):
    def __init__(self, 
                 batches: list, 
                 batch_size: int,
                 multi_class:bool,
                 multi_label:bool, 
                 num_classes: int):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = len(batches) % self.batch_size != 0  # 检查是否有剩余数据没有填满一个batch
        self.index = 0
        self.multi_class = multi_class
        self.multi_label = multi_label
        self.num_classes = num_classes

    def _to_tensor(self, data:list)->tuple[tuple,torch.Tensor]:
        batch_input_ids = torch.LongTensor([_[0].tolist() for _ in data])
        batch_att_mask = torch.LongTensor([_[1].tolist() for _ in data])
        batch_token_type_ids = torch.LongTensor([_[2].tolist() for _ in data])
        
        labels = [_[3] for _ in data]
        y = get_multi_hot_label(labels, self.num_classes, dtype=torch.float)
        
        return (batch_input_ids, batch_att_mask, batch_token_type_ids), y


    def __next__(self):
        if self.residue and self.index == self.n_batches:
            # 处理最后一个batch，可能不满
            batches = self.batches[self.index * self.batch_size:]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            # 处理一个完整的batch
            batches = self.batches[self.index * self.batch_size:(self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config: "ModelConfig"):
    iter = DatasetIterater(dataset, 
                           config.batch_size,
                           config.multi_class,
                           config.multi_label, 
                           config.num_classes)
    return iter

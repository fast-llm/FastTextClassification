# coding: UTF-8
import torch
import torch.nn as nn
from tqdm import tqdm
import time
from datetime import timedelta
import csv
from tools.utils_multi_label import get_multi_hot_label

from extras.logging import get_logger
logger = get_logger(__name__)


PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号, SEP并非必须


def build_dataset(config,
                  n_samples=None, 
                  sep="\t", 
                  multi_label=False):
    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8', newline='') as f:
            f = csv.reader(f, delimiter=sep)
            for line in tqdm(f):
                if n_samples is not None and f.line_num > n_samples:
                    break
                if len(line) != 2:
                    logger.error(f"行格式错误: {line}")
                    continue
                content, label = line
                # if not lin:
                #     continue
                # if len(lin.split('\t')) == 2 and lin.split('\t')[1].isdigit():
                #     content, label = lin.split('\t')
                # elif len(lin.split(',')) == 2 and lin.split(',')[1].isdigit():
                #     content, label = lin.split(',')
                # else:
                #     print("line sep error, support tab or comma.")
                #     print(lin)
                #     print(lin.split('\t'))
                #     print(lin.split(','))
                #     break
                token = config.tokenizer.tokenize(content)
                token = [CLS] + token  # +cls这一步对GPT等模型来说不合适：TODO
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))  # 这里应该填充pad id而非0：TODO
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                contents.append(
                    (token_ids, int(label) if not multi_label else list(map(int, label.split(','))), seq_len, mask))
        return contents

    train = load_dataset(config.train_path, config.pad_size)
    val = load_dataset(config.val_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, val, test


class DatasetIterater(object):
    def __init__(self, 
                 batches: list, 
                 batch_size: int,
                 num_gpus: int,
                 device: str, 
                 multi_label:bool, 
                 num_classes: int):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = len(batches) % self.batch_size != 0  # 检查是否有剩余数据没有填满一个batch
        self.index = 0
        self.num_gpus = num_gpus
        self.device_ids = None
        self.multi_label = multi_label
        self.num_classes = num_classes
        self.init_device(device=device)

    def init_device(self,device:str):
        if self.num_gpus == 0:
            self.device = 'cpu'
        elif self.num_gpus >= 1:
            if ',' in device:  # 检查是否有多个GPU设备
                self.device_ids = list(map(int, device.split(',')))
                self.device = f"cuda:{self.device_ids[0]}"  # 选用第一个GPU作为主设备
            else:
                self.device = f"cuda:{device}"
    
    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        if self.multi_label:
            labels = [_[1] for _ in datas]
            y = get_multi_hot_label(labels, self.num_classes).to(self.device)
        else:
            y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)

        if self.num_gpus > 1:
            # 使用DataParallel分配到多个GPU
            x = nn.DataParallel(x, device_ids=self.device_ids)
            y = nn.DataParallel(y, device_ids=self.device_ids)
            seq_len = nn.DataParallel(seq_len, device_ids=self.device_ids)
            mask = nn.DataParallel(mask, device_ids=self.device_ids)

        return (x, seq_len, mask), y


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


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, 
                           config.batch_size,
                           config.num_gpus,
                           config.device, 
                           config.multi_label, 
                           config.num_classes)
    return iter

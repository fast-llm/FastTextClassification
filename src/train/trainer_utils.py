import os
import random
import numpy as np
from enum import Enum

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

import torch
from torch.optim import Adam, SGD, AdamW, Adagrad
import torch.nn as nn
from transformers import Trainer
from transformers.optimization import get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from transformers.utils.versions import require_version

from extras.loggings import get_logger
from hparams.data_args import DataArguments
from hparams.training_args import  TrainingArguments
from hparams.model_args import ModelArguments
from models.component.common import ModelConfig
from train.training_types import LossFnType


if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel

logger = get_logger(__name__)


# 权重初始化，默认xavier
def init_network(model, 
                 method='xavier', 
                 exclude='embedding', 
                 seed=42):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def create_config(model_args: "ModelArguments",
                 data_args: "DataArguments",
                 training_args: "TrainingArguments"
                  ) -> ModelConfig:
    model_name = model_args.model_name.lower()
    if 'ernie' in model_name :
        from models.ernie import Config
        return Config(
            model_args = model_args,
            data_args = data_args,
            training_args = training_args
        )
    elif 'bert' in model_name:
        from models.bert import Config
        return None


def create_model(config:ModelConfig) -> "PreTrainedModel":
    model_name = config.model_name.lower()
    if 'ernie' in model_name :
        from models.ernie import Model
        model = Model(
            model_path = config.model_path,
            update_all_layers = config.update_all_layers,
            multi_class = config.multi_class,
            multi_label= config.multi_label,
            num_classes=config.num_classes,
            hidden_size=config.hidden_size,
            mlp_layers_config=config.mlp_layers
        )
    else:
        raise ValueError("Model name or path is not specified.")
    return model


def create_loss_fn(loss_fn:str) -> "torch.nn.Module":
    if loss_fn.lower() == LossFnType.CROSS_ENTROPY.lower():
        loss_fn = nn.CrossEntropyLoss()
    elif loss_fn.lower() == LossFnType.BCE.lower():
        loss_fn = nn.BCELoss()
    elif loss_fn.lower() == LossFnType.BCE_WITH_LOGITS.lower():
        loss_fn = nn.BCEWithLogitsLoss()
    elif loss_fn.lower() == LossFnType.NLL.lower():
        loss_fn = nn.NLLLoss()
    elif loss_fn.lower() == LossFnType.POISSON_NLL.lower():
        loss_fn = nn.PoissonNLLLoss()
    elif loss_fn.lower() == LossFnType.KL_DIV.lower():
        loss_fn = nn.KLDivLoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn}")
    return loss_fn

def _create_optimizer(
    model: "PreTrainedModel",
    training_args: "TrainingArguments",
) -> "torch.optim.Optimizer":
    # 根据训练参数中指定的优化器类型创建优化器
    if training_args.optimizer_type.lower() == "adam":
        optimizer = Adam(model.parameters(), 
                         lr=training_args.learning_rate, 
                         weight_decay=training_args.weight_decay,
                         eps=training_args.adam_epsilon, 
                         betas=(training_args.adam_beta1, training_args.adam_beta2),
                         )
    elif training_args.optimizer_type.lower() == "adamw":
        optimizer = AdamW(model.parameters(), 
                          lr=training_args.learning_rate,
                          weight_decay=training_args.weight_decay,
                          eps=training_args.adam_epsilon, 
                          betas=(training_args.adam_beta1, training_args.adam_beta2),
                          )
    elif training_args.optimizer_type.lower() == "sgd":
        optimizer = SGD(model.parameters(), 
                        lr=training_args.learning_rate, 
                        weight_decay=training_args.weight_decay,
                        momentum=training_args.momentum
                        )
    elif training_args.optimizer_type.lower() == "adagrad":
        optimizer = Adagrad(model.parameters(), 
                            lr=training_args.learning_rate, 
                            lr_decay=training_args.lr_decay,
                            initial_accumulator_value=training_args.initial_accumulator_value,
                            eps=training_args.adam_epsilon,
                            weight_decay=training_args.weight_decay,
                            )
    else:
        raise ValueError(f"Unsupported optimizer type: {training_args.optimizer_type}")
    
    return optimizer


def create_custom_optimizer(
    model,
    training_args: "TrainingArguments",
) -> Optional["torch.optim.Optimizer"]:
    try:
        return _create_optimizer(model, training_args)
    except ValueError as e:
        logger.error(e)
        return None


def create_custom_scheduler(
    training_args: "TrainingArguments",
    num_training_steps: int,
    optimizer: Optional["torch.optim.Optimizer"] = None,
) -> None:
    if optimizer is not None:
        if training_args.lr_scheduler_type.lower() == "linear":
            return get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
        elif training_args.lr_scheduler_type.lower() == "cosine":
            return get_scheduler(
                "cosine",
                optimizer=optimizer,
                num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
        elif training_args.lr_scheduler_type.lower() == "cosine_with_restarts":
            return get_scheduler(
                "cosine_with_restarts",
                optimizer=optimizer,
                num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
        elif training_args.lr_scheduler_type.lower() == "polynomial":
            return get_scheduler(
                "polynomial",
                optimizer=optimizer,
                num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                power=training_args.power,
            )
        elif training_args.lr_scheduler_type.lower() == "constant":
            return get_scheduler(
                "constant",
                optimizer=optimizer,
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {training_args.lr_scheduler_type}")
    else:
        return None

def get_activations(inputs:torch.tensor, activate_func:str=None):
    if activate_func is None:
        return inputs
    elif activate_func == 'relu':
        inputs = torch.relu(inputs)
    elif activate_func == 'leaky_relu':
        inputs = torch.leaky_relu(inputs)
    elif activate_func == 'sigmoid':
        inputs = torch.sigmoid(inputs)
    elif activate_func == 'softmax':
        inputs = torch.softmax(inputs)
    elif activate_func == 'tanh':
        inputs = torch.tanh(inputs)
    return inputs

def calculate_num_training_steps(dataset_size: int,
                                 num_gpus:int, 
                                 per_device_train_batch_size: int, 
                                 epochs: int) -> int:
    """
    计算总的训练步骤数。

    参数:
        dataset_size (int): 数据集中的样本总数。
        batch_size (int): 每个批次的样本数。
        epochs (int): 训练的总轮数。

    返回:
        int: 总的训练步骤数。
    """
    steps_per_epoch = dataset_size // (per_device_train_batch_size*num_gpus)  # 每个epoch的步骤数
    if dataset_size % per_device_train_batch_size != 0:
        # 如果有余数，意味着最后一个批次的样本数少于batch_size，但仍需一个额外的步骤来处理它们
        steps_per_epoch += 1
    total_steps = steps_per_epoch * epochs  # 总步骤数
    return total_steps

def calculate_binary_accuracy(pred: torch.Tensor, 
                              label: torch.Tensor,
                              threshold: float = 0.5
                              ) -> tuple[torch.Tensor,int]:
    """_summary_
    Args:
        pred (torch.Tensor): _description_
        label (torch.Tensor): _description_
    Returns:
        float: _description_
    """
    label = label.to(pred.device)
    label = (label > threshold).int()
    pred = (pred > threshold).int()  # 将预测值转换为0或1
    correct = (pred == label).sum().item()  # 计算预测正确的样本数
    return correct, correct, len(label)

def calculate_multi_class_accuracy(pred: torch.Tensor, 
                                   label: torch.Tensor,
                                   threshold: float = 0.5
                                   ) -> tuple[torch.Tensor,int]:
    """_summary_
    Args:
        pred (torch.Tensor): _description_
        label (torch.Tensor): _description_
    Returns:
        float: _description_
    """
    _ , s = pred.size()
    if s>1:
        pred = pred.argmax(dim=-1)  # 获取预测结果中概率最高的类别
        label = label.to(pred.device)
        label = label.argmax(dim=-1)
        correct = (pred == label).sum().item()  # 计算预测正确的样本数
    else:
        label = label.to(pred.device)
        label = (label > threshold).int()
        pred = (pred > threshold).int()  # 将预测值转换为0或1
        correct = (pred == label).sum().item()  # 计算预测正确的样本数
    return correct, correct, len(label)

def calculate_multi_label_accuracy(pred: torch.Tensor, 
                                   label: torch.Tensor,
                                   threshold: float = 0.5
                                   ) -> tuple[torch.Tensor,int]:
    if threshold < 0 or threshold > 1:
        raise ValueError("threshold must be in 0-1")
    label = label.to(pred.device)
    label = (label > threshold).int()
    pred = (pred > threshold).int()  # 将预测值转换为0或1
    correct_all = (pred == label).all(dim=1).sum().item()  # 计算所有标签都预测正确的样本数
    correct = (pred == label).sum().item()  # 计算有标签预测正确的样本数
    total = len(label)  # 总样本数
    return correct, correct_all, total


#计算准确率
def calculate_accuracy(pred: torch.tensor,
                       label: torch.tensor,
                       threshold:float = 0.5,
                       multi_class: bool = False,
                       multi_label: bool = False):
    """_summary_

    Args:
        pred (torch.tensor): _description_ batch_size * num_class
        label (torch.tensor): _description_ batch_size * num_class
        multi_class (bool, optional): _description_. Defaults to False.
        multi_label (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if multi_class and multi_label:
        raise ValueError("multi_class and multi_label cannot be True at the same time.")
    elif multi_class:
        return calculate_multi_class_accuracy(pred, label)
    elif multi_label:
        return calculate_multi_label_accuracy(pred, label, threshold)
    else:
        return calculate_binary_accuracy(pred, label)


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现。


if __name__ == "__main__":
    x = torch.tensor([[0.9, 0.8, 0.1, 0.8], 
                      [0.1, 0.6, 0.2, 0.7]]).to('cuda:0')
    y = torch.tensor([[1, 0, 0, 0], 
                      [0, 1, 0, 0]],dtype=torch.float)
    correct, correct_all, total = calculate_accuracy(x, y,multi_class=True)
    logger.info(f"{correct}, {correct_all}, {total}")
    
    correct, correct_all, total = calculate_accuracy(x, y,
                                                     multi_class=False,
                                                     multi_label=True)
    logger.info(f"{correct}, {correct_all}, {total}")
    
    x = torch.tensor([[0.9, 0.8, 0.1], 
                      [0.1, 0.6, 0.2]]).to('cuda:0')
    y = torch.tensor([[1, 0, 0], 
                      [0, 1, 0]],dtype=torch.float)
    correct, correct_all, total = calculate_accuracy(x, y,multi_class=True)
    logger.info(f"{correct}, {correct_all}, {total}")
    
    correct, correct_all, total = calculate_accuracy(x, y,
                                                     multi_class=False,
                                                     multi_label=True)
    logger.info(f"{correct}, {correct_all}, {total}")
    
    
    x = torch.tensor([[0.9], 
                      [0.1],
                      [0.2],
                      [0.3]
                      ]).to('cuda:0')
    y = torch.tensor([[1], 
                      [0],
                      [1],
                      [1]
                      ],dtype=torch.float)
    correct, correct_all, total = calculate_accuracy(x, y, multi_class=True)
    logger.info(f"{correct}, {correct_all}, {total}")
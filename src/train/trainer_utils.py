import os
import random
import shutil
import numpy as np
from enum import Enum

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.optim import Adam, SGD, AdamW, Adagrad
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn as nn
from transformers.optimization import get_scheduler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from models.component.common import ModelConfig
from train.training_types import LossFnType
from extras.constants import SAVE_MODEL_NAME
from extras.loggings import get_logger
from hparams.data_args import DataArguments
from hparams.training_args import  TrainingArguments
from hparams.model_args import ModelArguments


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

def optimizer_AdamW_LLRD(full_model: "PreTrainedModel", 
                         mlp_lr: float = 0.001,
                         weight_decay: float = 0.01,
                         eps: float = 1e-8, 
                        betas: Tuple[float, float] = (0.9, 0.999)
                         ):
    model = full_model.bert
    mlp = full_model.fc
    
    opt_parameters = []    # To be passed to the optimizer (only parameters of the layers you want to update).
    named_parameters = list(model.named_parameters()) 

    # According to AAAMLP book by A. Thakur, we generally do not use any decay 
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    init_lr = 3.5e-5 
    head_lr = 3.6e-5
    lr = init_lr

    # === Pooler and regressor ======================================================  

    params_0 = [p for n,p in named_parameters if ("pooler" in n or "regressor" in n) 
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in named_parameters if ("pooler" in n or "regressor" in n)
                and not any(nd in n for nd in no_decay)]

    head_params = {"params": params_0, "lr": head_lr, "weight_decay": 0.0}    
    opt_parameters.append(head_params)

    head_params = {"params": params_1, "lr": head_lr, "weight_decay": 0.01}    
    opt_parameters.append(head_params)
    # cal hidden layers ==========================================================
    # 获取BertEncoder
    encoder = model.encoder
    # 获取BertEncoder的层列表
    layer_list = encoder.layer
    # 计算层数
    num_layers = len(layer_list)
    
    # === 12 Hidden layers ==========================================================

    for layer in range(num_layers-1,-1,-1):        
        params_0 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                    and any(nd in n for nd in no_decay)]
        params_1 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                    and not any(nd in n for nd in no_decay)]

        layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
        opt_parameters.append(layer_params)   

        layer_params = {"params": params_1, "lr": lr, "weight_decay": 0.01}
        opt_parameters.append(layer_params)       

        lr *= 0.9     

    # === Embeddings layer ==========================================================

    params_0 = [p for n,p in named_parameters if "embeddings" in n 
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in named_parameters if "embeddings" in n
                and not any(nd in n for nd in no_decay)]

    embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0} 
    opt_parameters.append(embed_params)

    embed_params = {"params": params_1, "lr": lr, "weight_decay": 0.01} 
    opt_parameters.append(embed_params)        
    
    # 线性层
    params_2 = [p for n,p in mlp.named_parameters() if any(nd in n for nd in no_decay)]
    mlp_params = {"params": params_2, "lr": mlp_lr, "weight_decay": weight_decay}
    opt_parameters.append(mlp_params)
    return AdamW(opt_parameters, lr=mlp_lr, eps=eps, betas=betas)

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
        return Config(
            model_args = model_args,
            data_args = data_args,
            training_args = training_args
        )
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
        return model
    elif 'bert' in model_name:
        from models.bert import Model
        model = Model(
            model_path = config.model_path,
            update_all_layers = config.update_all_layers,
            multi_class = config.multi_class,
            multi_label= config.multi_label,
            num_classes=config.num_classes,
            hidden_size=config.hidden_size,
            mlp_layers_config=config.mlp_layers
        )
        return model
    elif 'qwen' in model_name:
        from models.qwen import Model
        model = Model(
            model_path = config.model_path,
            update_all_layers = config.update_all_layers,
            multi_class = config.multi_class,
            multi_label= config.multi_label,
            num_classes=config.num_classes,
            hidden_size=config.hidden_size,
            mlp_layers_config=config.mlp_layers
        )
        return model
    raise ValueError("Model name or path is not specified.")
    


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
    elif training_args.optimizer_type.lower() == "adamw_llrd":
        optimizer = optimizer_AdamW_LLRD(model,
                                        mlp_lr=training_args.learning_rate,
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
) -> Optional["torch.optim.lr_scheduler._LRScheduler"]:
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
        elif training_args.lr_scheduler_type.lower() == "constant_with_warmup":
            return get_scheduler(
                "constant_with_warmup",
                optimizer=optimizer,
                num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
            )
        elif training_args.lr_scheduler_type.lower() == "inverse_sqrt":
            return get_scheduler(
                "inverse_sqrt",
                optimizer=optimizer,
                num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
            )
        elif training_args.lr_scheduler_type.lower() == "onecyclelr":
            return OneCycleLR(optimizer=optimizer, 
                              max_lr=25*training_args.learning_rate, 
                              epochs=training_args.epochs, 
                              steps_per_epoch=int(num_training_steps/training_args.epochs))
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
        label = label.argmax(dim=-1)
        correct = (pred == label).sum().item()  # 计算预测正确的样本数
    else:
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
    label = (label > threshold).int()
    pred = (pred > threshold).int()  # 将预测值转换为0或1
    correct_all = (pred == label).all(dim=1).sum().item()  # 计算所有标签都预测正确的样本数
    correct = (pred == label).sum().item()  # 计算有标签预测正确的样本数
    total = len(label)  # 总样本数
    return correct, correct_all, total


#计算准确率
def calculate_accuracy(pred: torch.Tensor,
                       label: torch.Tensor,
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

def prepare_text(text:str, 
                 tokenizer:"torch.nn.module", 
                 pad_size:int=768,
                 device:str=None):
    device = device if device else 'cpu'
    inputs = tokenizer(text,
                        truncation=True,  
                        return_tensors="pt", 
                        padding='max_length', 
                        max_length=pad_size
                        )
    if device:
        for key in inputs:
            inputs[key] = inputs[key].to(device)
    return inputs['input_ids'],inputs['attention_mask'],inputs['token_type_ids']


class ModelEntry:
    def __init__(self,loss:float, score:float, model_path:str):
        self.loss = loss
        self.score = score
        self.model_path = model_path
    
    def __eq__(self, other):
        return self.model_path == other.model_path

    def __hash__(self):
        return hash(self.model_path)

    def __repr__(self):
        return f"{self.model_path} (Loss: {self.loss}, Score: {self.score})"

class ModelManager:
    def __init__(self, num_best_models: int = 5):
        self.best_models = []
        self.num_best_models = num_best_models

    def update_best_models(self, new_model: ModelEntry):
        # 如果模型已存在，则先移除旧的记录
        if new_model in self.best_models:
            self.best_models.remove(new_model)

        # 添加新模型到列表
        self.best_models.append(new_model)

        # 按得分降序，损失升序排序
        self.best_models.sort(key=lambda x: (-x.score, x.loss))
        
        # 保持列表中只有最优的模型数量
        while len(self.best_models) > self.num_best_models:
            removed_model = self.best_models.pop()  # 删除性能最差的模型
            if os.path.exists(removed_model.model_path):
                shutil.rmtree(removed_model.model_path)  # 确认文件存在后再删除
                logger.info(f"Removed older model: {removed_model}")

class LogState:
    def __init__(self, epoch=None, step=None, learning_rate=None, loss=None, accuracy=None, grad_norm=None, eval_loss=None, eval_accuracy=None):
        self.log = {
            "epoch": epoch,
            "step": step,
            "learning_rate": learning_rate,
            "loss": loss,
            "accuracy": accuracy,
            "grad_norm": grad_norm,
            "eval_loss": eval_loss,
            "eval_accuracy": eval_accuracy
        }
    def to_dict(self):
        return self.log

class EarlyStopping:
    def __init__(self, patience:int=3, verbose:bool=False, delta:float=0.0):
        """_summary_
        Args:
            patience (int, optional): _description_. Defaults to 3.
            verbose (bool, optional): _description_. Defaults to False. verbose 为True，则打印详细信息
            delta (float, optional): _description_. Defaults to 0.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_loss = None
        self.early_stop = False
        self.val_acc_min = float('inf')
        self.delta = delta
        self.logger = get_logger('EarlyStopping')
        
    def __call__(self, acc:float):
        """_summary_
        初始化时，设定了self.best_score = None
        if 语句第一行判断 self.best_score 是否为初始值，如果是初始值，则将 score 赋值给 self.best_score ，然后返回True保存模型
        当目前分数比最好分数加 self.delta 小时，就认为模型没有改进，将 counter 计数器加1，当计数器值超过 patience 的时候，就令early_stop为True，让模型停止训练。
        当目前分数比最好分数加 self.delta 大时，我们认为模型有改进，将目前分数赋值给最好分数，令计数器归零。返回True
        Args:
            acc (_type_): _description_
        """
        score = acc
        if self.best_score is None:
            self.best_score = score
            # 保存模型
            return True
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.counter = 0
            # 保存模型
            return True
    



class ClassificationReport:
    def __init__(self,
                 predict_all: torch.Tensor,
                 labels_all: torch.Tensor,
                 target_names: list[str],
                 num_classes: int = 2,
                 threshold: float = 0.5,
                 digits: int = 4,
                 multi_label: bool = False):
    
        self.predict_all = predict_all
        self.labels_all = labels_all
        self.target_names = target_names
        self.num_classes = num_classes
        self.threshold = threshold
        self.digits = digits
        self.multi_label = multi_label
        
        # Ensure target names cover all classes
        self.adjusted_target_names = (self.target_names + [f'class {i}' for i in range(len(self.target_names), self.num_classes)])[:self.num_classes]
    
    def calculate_metrics(self):
        if self.multi_label:
            # 多标签处理：应用阈值
            binary_predictions = (self.predict_all > self.threshold).int()
            labels_all = (self.labels_all > self.threshold).int()
        else:
            # 二分类和多分类处理：使用 argmax
            binary_predictions = self.predict_all.argmax(dim=-1)
            labels_all = self.labels_all.argmax(dim=-1)
        
        # 计算精确度、召回率、F1分数和支持数
        precision, recall, f1, support = precision_recall_fscore_support(
            labels_all.cpu().numpy(),
            binary_predictions.cpu().numpy(),
            average=None,
            labels=list(range(self.num_classes)),
            zero_division=0
        )
        
        
        # Generate confusion matrix and calculate per-class accuracy
        cm = confusion_matrix(labels_all.cpu().numpy(), binary_predictions.cpu().numpy(), labels=np.arange(self.num_classes))
        # 防止除以零
        np.seterr(divide='ignore', invalid='ignore')
        class_accuracy = np.nan_to_num(cm.diagonal() / cm.sum(axis=1), nan=0.0)
        correct_per_class = cm.diagonal()  # Diagonal elements are the correct predictions per class
        
        total = labels_all.numel()
        correct = (binary_predictions == labels_all).sum().item()
        
        
        # 计算准确率
        accuracy = correct / total if total > 0 else 0  # 防止除以零
        return class_accuracy, correct, total, precision, recall, f1, correct_per_class, support
    
    def metrics(self):
        class_accuracy, correct, total, precision, recall, f1, correct_per_class, support  = self.calculate_metrics()
        # 构建报告字符串
        report = "          class        accuracy     precision     recall      f1-score      total     support\n\n"
        for i in range(self.num_classes):
            report += f"{self.adjusted_target_names[i]:>15}    {class_accuracy[i]:>10.{self.digits}f}    {precision[i]:>10.{self.digits}f}    {recall[i]:>10.{self.digits}f}    {f1[i]:>10.{self.digits}f}    {correct_per_class[i]:>6}    {support[i]:>6}\n"
        return report



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
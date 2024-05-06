import os
import random
import numpy as np
from enum import Enum
from transformers.utils import ExplicitEnum
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

import torch
from torch.optim import Adam, SGD, AdamW, Adagrad
import torch.nn as nn
from transformers import Trainer
from transformers.optimization import get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from transformers.utils.versions import require_version

from ..extras.logging import get_logger
from ..hparams.training_args import  TrainingArguments
from ..hparams.model_args import ModelArguments


if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel

logger = get_logger(__name__)

class ParallelMode(Enum):
    DISTRIBUTED = "distributed"
    SINGLE = "single"
    NOT_PARALLEL = "not_parallel"

class SchedulerType(ExplicitEnum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    INVERSE_SQRT = "inverse_sqrt"
    REDUCE_ON_PLATEAU = "reduce_lr_on_plateau"

class DummyOptimizer(torch.optim.Optimizer):
    r"""
    A dummy optimizer used for the GaLore algorithm.
    """

    def __init__(
        self, 
        lr: float = 1e-3, 
        optimizer_dict: Optional[Dict["torch.nn.Parameter", "torch.optim.Optimizer"]] = None
    ) -> None:
        dummy_tensor = torch.randn(1, 1)
        self.optimizer_dict = optimizer_dict
        super().__init__([dummy_tensor], {"lr": lr})

    def zero_grad(self, set_to_none: bool = True) -> None:
        pass

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        pass

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
                            initial_accumulator_value=training_args.initial_accumulator_value
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
    if optimizer is not None and isinstance(optimizer, DummyOptimizer):
        optimizer_dict = optimizer.optimizer_dict
        scheduler_dict: Dict["torch.nn.Parameter", "torch.optim.lr_scheduler.LRScheduler"] = {}

        for param in optimizer_dict.keys():
            scheduler_dict[param] = get_scheduler(
                training_args.lr_scheduler_type,
                optimizer=optimizer_dict[param],
                num_warmup_steps=training_args.get_warmup_steps(num_training_steps) * 2,
                num_training_steps=num_training_steps * 2,
            )

        def scheduler_hook(param: "torch.nn.Parameter"):
            if param.grad is not None:
                scheduler_dict[param].step()

        for param in optimizer_dict.keys():
            param.register_post_accumulate_grad_hook(scheduler_hook)

def get_activations(inputs, activate_func:str=None):
    if activate_func == 'sigmoid':
        inputs = torch.sigmoid(inputs)
    elif activate_func == 'softmax':
        inputs = torch.softmax(inputs)
    elif activate_func == 'tanh':
        inputs = torch.tanh(inputs)
    return inputs

def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现。

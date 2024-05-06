# coding: UTF-8
import os
import torch
import torch.nn as nn
from transformers import AutoModel,BertModel, BertTokenizer, AutoModelForMaskedLM, AutoTokenizer
from extras.packages import compare_versions, get_transformer_version
from models.component.modeling_ernie import ErnieModel
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from transformers import TrainerCallback
    from hparams.data_args import DataArguments
    from hparams.model_args import ModelArguments
    from hparams.training_args import TrainingArguments
from .component.common import BaseModel, MLPLayer, ModelConfig

from extras.loggings import get_logger
logger = get_logger(__name__)


class Config(ModelConfig):
    """配置参数"""

    def __init__(self, 
                 model_args: "ModelArguments",
                 data_args: "DataArguments",
                 training_args: "TrainingArguments"):
        super(Config, self).__init__(model_args, data_args, training_args)


class Model(BaseModel):
    def __init__(self,
                 model_path: str,
                 update_all_layers: bool,
                 multi_class: bool,
                 multi_label: bool,
                 num_classes: int,
                 hidden_size: int,
                 mlp_layers_config: List[MLPLayer]):
        super(Model, self).__init__(model_path,
                                    update_all_layers,
                                    multi_class,
                                    multi_label,
                                    num_classes,
                                    hidden_size,
                                    mlp_layers_config)
        
if __name__ == "__main__":
    pass
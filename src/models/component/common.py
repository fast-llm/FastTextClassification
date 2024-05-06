import os
from enum import Enum
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, List, Optional
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AutoModelForMaskedLM, AutoTokenizer

from extras.constants import get_model_info
from extras.packages import compare_versions, get_transformer_version
from extras.loggings import get_logger

if TYPE_CHECKING:
    from transformers import TrainerCallback
    from hparams.data_args import DataArguments
    from hparams.model_args import ModelArguments
    from hparams.training_args import TrainingArguments



logger = get_logger(__name__)



@dataclass
class MLPLayer:
    layer_type: str
    size: int
    activation: str
    dropout: float

class ModelConfig(object):
    """配置参数"""

    def __init__(self, 
                 model_args: "ModelArguments",
                 data_args: "DataArguments",
                 training_args: "TrainingArguments"):
        self.model_name = model_args.model_name
        self.train_path = os.path.join(data_args.dataset_dir,
                                       data_args.dataset,
                                       'data',
                                       data_args.train_file)  # 训练集
        self.val_path = os.path.join(data_args.dataset_dir,
                                       data_args.dataset,
                                       'data',
                                       data_args.val_file)  # 验证集
        self.test_path = os.path.join(data_args.dataset_dir,
                                       data_args.dataset,
                                       'data',
                                       data_args.test_file)  # 测试集
        self.class_path = os.path.join(data_args.dataset_dir,
                                       data_args.dataset,
                                       'data',
                                       data_args.class_file)    # 类别名单
        self.class_list = [x.strip() for x in open(self.class_path).readlines()]  # 类别名单
        self.vocab_path = os.path.join(data_args.dataset_dir,
                                       data_args.dataset,
                                       'data',
                                       data_args.vocab_file)
        self.save_folder = training_args.output_dir
        self.save_path = self.save_folder + self.model_name + '.ckpt'  # 模型训练结果
        self.log_path = os.path.join(self.save_folder,training_args.log_dir,self.model_name)
        self.num_gpus = training_args.num_gpus
        self.device = training_args.device # 设备

        self.require_improvement = training_args.require_improvement  # 若超过 default 1000 batch效果还没提升，则提前结束训练
        self.multi_label = False
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = training_args.epochs  # epoch数
        self.batch_size = training_args.per_device_train_batch_size  # mini-batch大小
        self.pad_size = data_args.cutoff_len  # 每句话处理成的长度(短填长切)
        self.learning_rate = training_args.learning_rate  # 学习率
        self.bert_path = model_args.model_name_or_path
        # transformers4.22开始支持ernie
        if ernie_available():
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path,
                                                           truncation=True,  
                                                           return_tensors="pt", 
                                                           padding='max_length', 
                                                           max_length=self.pad_size)  
        else:
            self.tokenizer = BertTokenizer.from_pretrained(self.bert_path,
                                                           truncation=True,  
                                                           return_tensors="pt", 
                                                           padding='max_length', 
                                                           max_length=self.pad_size)
        self.update_all_layers = training_args.update_all_layers
        logger.info(self.tokenizer)
        model_info = get_model_info(self.model_name)
        self.hidden_size = model_info.hidden_size
        self.mlp_layers = training_args.mlp_layers
        self.vocab = None
        self.n_vocab = None

def ernie_available():
    # transformers4.22开始支持ernie
    version = "4.22"
    now_version = get_transformer_version()
    return compare_versions(now_version,version)

def ernie_available():
    # transformers4.22开始支持ernie
    version = "4.22"
    now_version = get_transformer_version()
    return compare_versions(now_version,version)


def build_mlp_layers(num_classes:int,
                     hidden_size: int, 
                     mlp_layers_config:List[MLPLayer]):
    """根据配置构建多层MLP网络。
    Args:
        config: 包含模型配置的对象，例如包含隐藏层大小等信息。
        mlp_layers_config: 包含MLP层配置的列表。

    Returns:
        nn.Sequential: 构建的MLP网络。
    """
    mlp_layers = []
    input_size = hidden_size  # 输入大小为BERT模型的隐藏层大小
    for layer_config in mlp_layers_config:
        if layer_config.layer_type== "Dense":
            # 添加全连接层
            linear_layer = nn.Linear(input_size, layer_config.size)
            mlp_layers.append(linear_layer)
            input_size = layer_config.size # 更新输入大小为当前层的大小
            # 添加激活函数
            activation_fn = getattr(nn, layer_config.activation)()
            mlp_layers.append(activation_fn)
            # 添加Dropout层
            dropout_layer = nn.Dropout(p=layer_config.dropout)
            mlp_layers.append(dropout_layer)

    # 添加输出层
    output_layer = nn.Linear(layer_config.size, num_classes)
    mlp_layers.append(output_layer)

    # 组合所有层为Sequential模型
    return nn.Sequential(*mlp_layers)

if __name__ == "__main__":
    mlp_layers_config = [
        MLPLayer(
            layer_type="Dense",
            size= 512,
            activation="ReLU",
            dropout=0.2,
        ),
        MLPLayer(
            layer_type="Dense",
            size= 256,
            activation="ReLU",
            dropout=0.2,
        )
    ]

    mlp_model = build_mlp_layers(2,100,mlp_layers_config)
    print(mlp_model)
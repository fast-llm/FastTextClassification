import os
from enum import Enum
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, List, Optional
import torch.nn as nn
import torch
from transformers import AutoModel, BertTokenizer, AutoTokenizer

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
        self.log_path = training_args.log_dir
        self.num_gpus = training_args.num_gpus


        self.require_improvement = training_args.require_improvement  # 若超过 default 1000 batch效果还没提升，则提前结束训练
        
        self.max_samples = data_args.max_samples
        self.multi_class = data_args.multi_class
        self.multi_label = data_args.multi_label
        self.num_classes = training_args.num_classes  # 类别数
        
        self.num_epochs = training_args.epochs  # epoch数
        self.batch_size = training_args.per_device_train_batch_size  # mini-batch大小
        self.pad_size = data_args.cutoff_len  # 每句话处理成的长度(短填长切)
        self.learning_rate = training_args.learning_rate  # 学习率
        self.model_path = model_args.model_name_or_path
        
        self.SEP = data_args.SEP
        
        # transformers4.22开始支持ernie
        if ernie_available():
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path,
                                                           truncation=True,  
                                                           return_tensors="pt", 
                                                           padding='max_length', 
                                                           max_length=self.pad_size)  
        else:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path,
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
        
        

class BaseModel(nn.Module):
    def __init__(self, 
                 model_path: str,
                 update_all_layers: bool,
                 multi_class: bool,
                 multi_label: bool,
                 num_classes: int,
                 hidden_size: int,
                 mlp_layers_config: List[MLPLayer]):
        super(BaseModel, self).__init__()
        self.multi_class = multi_class
        self.multi_label = multi_label
        self.model = AutoModel.from_pretrained(model_path)
        if not update_all_layers:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            for param in self.model.parameters():
                param.requires_grad = True
        self.fc = build_mlp_layers(num_classes, hidden_size, mlp_layers_config)

    def forward(self, input, 
                label: Optional[torch.Tensor]= None,
                threshold: int=0.5):
        input_ids = input[0]
        attention_mask = input[1]
        token_type_ids = input[2]
        output = self.model(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)
        out = self.fc(output.pooler_output)
        
        if self.multi_class and self.multi_label:
            raise ValueError("multi_class and multi_label cannot be True at the same time.")
        elif self.multi_class:
            out = torch.softmax(out,-1)
            pred = pred.argmax(dim=-1)
            
        elif self.multi_label:
            out = torch.sigmoid(out)
            pred = (out > threshold).int()
        else:
            out = torch.sigmoid(out)
            pred = (out > threshold).int()  # 将预测值转换为0或1
        
        
        if label is None:  # 直接输出概率值作为预测结果
            return out, None
        else:   # 输出概率值和
            return out, pred


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
    
    sequence = "一直觉得金桥不错，虽然房间有些暗，但是配套设施很好，特别是有厨房着一定，对于带着孩子来往的人很方便。就是挨着火车轨道，早上晚上会有隆隆的火车声，有些吵，房间隔音一般吧。"


    model_path = '/platform_tech/xiongrongkang/models/ernie-3.0-base-zh'

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Original sequence: ",sequence)
    output = tokenizer(sequence,
                    truncation=True,  
                    return_tensors="pt", 
                    padding='max_length', 
                    max_length=768)

    model = BaseModel(
        model_path=model_path,
        update_all_layers=True,
        multi_class = True,
        multi_label = False,
        num_classes=2,
        hidden_size=768,
        mlp_layers_config=mlp_layers_config
    )
    pred = model((output['input_ids'],  output['attention_mask'], output['token_type_ids'],))
    print(pred)
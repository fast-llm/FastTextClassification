import os
import re
from pydantic import BaseModel
from typing import Optional
import torch
from transformers import AutoTokenizer
from dataclasses import dataclass, field
from extras.constants import SAVE_MODEL_NAME
from extras.misc import torch_gc
from train.trainer_utils import create_config, create_model

from hparams.data_args import DataArguments
from hparams.model_args import ModelArguments
from hparams.training_args import TrainingArguments


class TextModel(BaseModel):
    text: str

class PredictModel(BaseModel):
    text: list[str]
    pad_size: Optional[int] = 512

@dataclass
class InferenceArguments:
    model_name_or_path: str = field(
        default="ernie-3.0-base-zh",
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )
    pad_size : int = field(
        default=768,
        metadata={
            "help": "pad size"
        },
    )
    num_gpus: int = field(
        default=0,
        metadata={
            "help": "Number of GPUs to use"
        },
    )
    port: int = field(
        default=9090,
        metadata={
            "help": "Port to run the server"
        },
    )
    workers: int = field(
        default=1,
        metadata={
            "help": "Number of workers to run the server"
        },
    )
    
    def __post_init__(self):
        pass

class ModelHandler:
    def __init__(self, args:"InferenceArguments"):
        
        self.device = self.set_device(args.num_gpus)
        self.model, self.tokenizer = self.build_model(args.model_name_or_path, 
                                                      args.pad_size, 
                                                      self.device)

    def set_device(self,num_gpus:int):
        if num_gpus >= 1:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device('cpu')
        return device

    def build_model(self, model_path:str, pad_size: int,device:str):
        model_args = ModelArguments.load_from_bin(model_path)
        data_args = DataArguments.load_from_bin(model_path)
        training_args = TrainingArguments.load_from_bin(model_path)
        config = create_config(
            model_args = model_args,
            data_args = data_args,
            training_args = training_args
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                truncation=True,  
                                                return_tensors="pt", 
                                                padding='max_length', 
                                                max_length=pad_size)
        model = create_model(config)
        model.load_state_dict(torch.load(os.path.join(model_path, SAVE_MODEL_NAME),map_location=device))
        model = model.to(device)
        torch_gc()
        return model, tokenizer

    async def get_model(self):
        return self.model, self.tokenizer, self.device

# 切分句子
def cut_sent(txt):
    #先预处理去空格等
    txt = re.sub('([　 \t]+)',r" ",txt)  # blank word
    txt = txt.rstrip()       # 段尾如果有多余的\n就去掉它
    nlist = txt.split("\n") 
    nnlist = [x for x in nlist if x.strip()!='']  # 过滤掉空行
    return nnlist


def prepare_text(text:list[str]|str, tokenizer:"torch.nn.module", pad_size:int=768,device:str="cpu"):
    if isinstance(text, str):
        text = [text]
    elif not isinstance(text, list):
        raise ValueError("text must be a list or a string")
    elif not all(isinstance(t, str) for t in text):
        raise ValueError("all elements in text must be strings")
    else :
        text = text
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

def predict(model, tokenizer:"torch.nn.module",text:str, pad_size:int=512,device:str="cpu"):
    model.eval()
    with torch.no_grad():
        inputs= prepare_text(text, tokenizer, pad_size, device)

        logits, prob, pred = model(inputs,predict=True)
        prob_np = prob.cpu().numpy()  # 转换为numpy数组并确保在CPU上
        pred_np = pred.cpu().numpy()
    
    return prob_np.tolist(), pred_np.tolist()
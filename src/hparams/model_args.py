from dataclasses import asdict, dataclass, field
import os
from typing import Any, Dict, Optional
import json

import torch
from extras.constants import MODEL_ARGS_NAME
from extras.loggings import get_logger
logger = get_logger(__name__)

@dataclass
class ModelArguments:
    r"""
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune or infer.
    """
    
    model_name_or_path: str = field(
        default="ernie-3.0-base-zh",
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )
    model_name: str = field(
        default="ERNIE",
        metadata={
            "help": "The model architecture to be trained or used. "
            "The model architecture should correspond to a model class that can be loaded by `AutoModel.from_pretrained(model_name)`."
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pre-trained models downloaded from huggingface.co or modelscope.cn."},
    )
    architecture: Optional[str] = field(
        default="transformers",
        metadata={
            "help": "model architecture from DNN, transformers"
        }
    )
    embedding: Optional[str] = field(
        default="inner",
        metadata={
            "help": "embedding model from inner, outer, random"
        }
    )
    def __post_init__(self):
        self.compute_dtype = None
        self.device_map = None
        self.model_max_length = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_to_json(self, json_path: str):
        r"""Saves the content of this instance in JSON format inside `json_path`."""
        json_string = json.dumps(asdict(self), indent=4, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        r"""Creates an instance from the content of `json_path`."""
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))
    
    @classmethod
    def load_from_bin(cls, model_path: str):
        r"""Creates an instance from the content of `json_path`."""
        args_path = os.path.join(model_path,MODEL_ARGS_NAME)
        arguments = torch.load(args_path)
        arguments.model_name_or_path = model_path
        return arguments
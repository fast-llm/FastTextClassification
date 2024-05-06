import os
from collections import namedtuple, OrderedDict, defaultdict
from enum import Enum
from typing import Dict, Optional


ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOG_FILE_NAME = "trainer_log.jsonl"

V_HEAD_WEIGHTS_NAME = "value_head.bin"

V_HEAD_SAFE_WEIGHTS_NAME = "value_head.safetensors"


ModelInfo = namedtuple("ModelInfo", ["simple_name", "template", "hidden_size", "link", "download", "description"])

model_info = OrderedDict()

class DownloadSource(str, Enum):
    DEFAULT = "hf"
    MODELSCOPE = "ms"


def register_model_group(
    models: Dict[str, Dict[DownloadSource, str]],
    simple_name: str,
    hidden_size: int,
    template: str,
    link: str,
    description: str
) -> None:
    
    for full_name, path in models.items():
        info = ModelInfo(simple_name, template, hidden_size, link, path, description)
        model_info[full_name] = info


def get_model_info(name: str) -> ModelInfo:
    try:
        if name in model_info:
            return model_info[name]
        else:
            # To fix this, please use `register_model_info` to register your model
            raise Exception(f"Model {name} not found")
    except Exception as e:
        raise Exception(f"Model {name} not found") from e
    
register_model_group(
    models={
        "bert-base-uncased": {
            DownloadSource.DEFAULT: "google-bert/bert-base-uncased",
            DownloadSource.MODELSCOPE: "google-bert/bert-base-uncased",
        },
        "bert-base-cased": {
            DownloadSource.DEFAULT: "google-bert/bert-base-cased",
            DownloadSource.MODELSCOPE: "google-bert/bert-base-cased",
        },
        "bert-base-chinese": {
            DownloadSource.DEFAULT: "google-bert/bert-base-chinese",
            DownloadSource.MODELSCOPE: "google-bert/bert-base-chinese",
        },
        "bert-base-multilingual-cased": {
            DownloadSource.DEFAULT: "google-bert/bert-base-multilingual-cased",
            DownloadSource.MODELSCOPE: "google-bert/bert-base-multilingual-cased",
        }
    },
    simple_name="bert-base",
    template="bert",
    hidden_size=768,
    link='',
    description="cased是大小写敏感, uncased是大小写不敏感"
)

register_model_group(
    models={
        "bert-large-uncased": {
            DownloadSource.DEFAULT: "google-bert/bert-large-uncased",
            DownloadSource.MODELSCOPE: "google-bert/bert-large-uncased",
        },
        "bert-large-cased": {
            DownloadSource.DEFAULT: "google-bert/bert-large-cased",
            DownloadSource.MODELSCOPE: "google-bert/bert-large-cased",
        },
        "bert-large-uncased-whole-word-masking": {
            DownloadSource.DEFAULT: "google-bert/bert-large-uncased-whole-word-masking",
            DownloadSource.MODELSCOPE: "google-bert/bert-large-uncased-whole-word-masking",
        },
        "bert-large-cased-whole-word-masking": {
            DownloadSource.DEFAULT: "google-bert/bert-large-cased-whole-word-masking",
            DownloadSource.MODELSCOPE: "google-bert/bert-large-cased-whole-word-masking",
        }
    },
    simple_name="bert-large",
    template="bert",
    hidden_size=1024,
    link='',
    description="cased是大小写敏感, uncased是大小写不敏感"
)

register_model_group(
    models={
        "ernie-1.0-base-zh": {
            DownloadSource.DEFAULT: "nghuyong/ernie-1.0-base-zh",
            DownloadSource.MODELSCOPE: "nghuyong/ernie-1.0-base-zh",
        },
    },
    simple_name="ernie-1.0",
    template="bert",
    hidden_size=768,
    link='https://github.com/PaddlePaddle/ERNIE',
    description=''
)

register_model_group(
    models={
        "ernie-2.0-base-en": {
            DownloadSource.DEFAULT: "nghuyong/ernie-2.0-base-en",
            DownloadSource.MODELSCOPE: "nghuyong/ernie-2.0-base-en",
        },
        "ernie-2.0-base-zh": {
            DownloadSource.DEFAULT: "nghuyong/ernie-2.0-base-zh",
            DownloadSource.MODELSCOPE: "nghuyong/ernie-2.0-base-zh",
        },
    },
    simple_name="ernie-2.0-base",
    template="ernie",
    hidden_size=768,
    link='https://github.com/PaddlePaddle/ERNIE',
    description=''
)

register_model_group(
    models={
        "ernie-2.0-large-en": {
            DownloadSource.DEFAULT: "nghuyong/ernie-2.0-large-en",
            DownloadSource.MODELSCOPE: "nghuyong/ernie-2.0-large-en",
        },
    },
    simple_name="ernie-2.0-large",
    template="ernie",
    hidden_size=1024,
    link='https://github.com/PaddlePaddle/ERNIE',
    description=''
)
register_model_group(
    models={
        "ernie-3.0-nano-zh": {
            DownloadSource.DEFAULT: "nghuyong/ernie-3.0-nano-zh",
            DownloadSource.MODELSCOPE: "nghuyong/ernie-3.0-nano-zh",
        },
    },
    simple_name="ernie-3.0-nano",
    template="ernie",
    hidden_size=128,
    link='https://github.com/PaddlePaddle/ERNIE',
    description=''
)


register_model_group(
    models={
        "ernie-3.0-micro-zh": {
            DownloadSource.DEFAULT: "nghuyong/ernie-3.0-micro-zh",
            DownloadSource.MODELSCOPE: "nghuyong/ernie-3.0-micro-zh",
        },
    },
    simple_name="ernie-3.0-micro",
    template="ernie",
    hidden_size=256,
    link='https://github.com/PaddlePaddle/ERNIE',
    description=''
)

register_model_group(
    models={
        "ernie-3.0-mini-zh": {
            DownloadSource.DEFAULT: "nghuyong/ernie-3.0-mini-zh",
            DownloadSource.MODELSCOPE: "nghuyong/ernie-3.0-mini-zh",
        },
    },
    simple_name="ernie-3.0-mini",
    template="ernie",
    hidden_size=384,
    link='https://github.com/PaddlePaddle/ERNIE',
    description=''
)

register_model_group(
    models={
        "ernie-3.0-medium-zh": {
            DownloadSource.DEFAULT: "nghuyong/ernie-3.0-medium-zh",
            DownloadSource.MODELSCOPE: "nghuyong/ernie-3.0-medium-zh",
        },
    },
    simple_name="ernie-3.0-medium",
    template="ernie",
    hidden_size=768,
    link='https://github.com/PaddlePaddle/ERNIE',
    description=''
)

register_model_group(
    models={
        "ernie-3.0-base-zh": {
            DownloadSource.DEFAULT: "nghuyong/ernie-3.0-base-zh",
            DownloadSource.MODELSCOPE: "nghuyong/ernie-3.0-base-zh",
        },
    },
    simple_name="ernie-3.0-base",
    template="ernie",
    hidden_size=768,
    link='https://github.com/PaddlePaddle/ERNIE',
    description=''
)

register_model_group(
    models={
        "ernie-health-zh": {
            DownloadSource.DEFAULT: "nghuyong/ernie-health-zh",
            DownloadSource.MODELSCOPE: "nghuyong/ernie-health-zh",
        },
    },
    simple_name="ernie-health",
    template="ernie",
    hidden_size=768,
    link='https://github.com/PaddlePaddle/ERNIE',
    description='专注于医疗领域的任务'
)

register_model_group(
    models={
        "ernie-gram-zh": {
            DownloadSource.DEFAULT: "nghuyong/ernie-gram-zh",
            DownloadSource.MODELSCOPE: "nghuyong/ernie-gram-zh",
        }
    },
    simple_name="ernie-gram",
    template="ernie",
    hidden_size=768,
    link='https://github.com/PaddlePaddle/ERNIE',
    description='专门用于语言模型（LM）预训练的版本'
)

# Qwen系列
register_model_group(
    models={
        "Qwen1.5-0.5B-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-0.5B-Chat",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-0.5B-Chat",
        },
        "Qwen1.5-0.5B": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-0.5B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-0.5B",
        },
    },
    simple_name="Qwen1.5-0.5B",
    template="qwen1.5",
    hidden_size=1024,
    link='https://github.com/QwenLM/Qwen1.5',
    description='Qwen1.5系列'
)

register_model_group(
    models={
        "Qwen1.5-1.8B-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-1.8B-Chat",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-1.8B-Chat",
        },
        "Qwen1.5-1.8B": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-1.8B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-1.8B",
        },
    },
    simple_name="Qwen1.5-1.8B",
    template="qwen1.5",
    hidden_size=2048,
    link='https://github.com/QwenLM/Qwen1.5',
    description='Qwen1.5系列'
)

if __name__ == "__main__":
    print(ROOT_PATH)
    print(model_info["ernie-3.0-base-zh"])
    a = get_model_info("ernie-3.0-base-zh")
    print(a)
    print(a.hidden_size)
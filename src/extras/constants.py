from collections import namedtuple, OrderedDict, defaultdict
from enum import Enum
from typing import Dict, Optional


LOG_FILE_NAME = "trainer_log.jsonl"

ModelInfo = namedtuple("ModelInfo", ["simple_name", "link", "download", "description"])

model_info = OrderedDict()

class DownloadSource(str, Enum):
    DEFAULT = "hf"
    MODELSCOPE = "ms"


def register_model_group(
    models: Dict[str, Dict[DownloadSource, str]],
    simple_name: str, 
    link: str,
    description: str
) -> None:
    
    for full_name, path in models.items():
        info = ModelInfo(simple_name, link, path, description)
        model_info[full_name] = info


def get_model_info(name: str) -> ModelInfo:
    if name in model_info:
        return model_info[name]
    else:
        # To fix this, please use `register_model_info` to register your model
        return ModelInfo(
            name, "", "Register the description at fastchat/model/model_registry.py"
        )

register_model_group(
    models={
        "bert-base-uncased": {
            DownloadSource.DEFAULT: "google-bert/bert-base-uncased",
            DownloadSource.MODELSCOPE: "google-bert/bert-base-uncased",
        },
        "bert-large-uncased": {
            DownloadSource.DEFAULT: "google-bert/bert-large-uncased",
            DownloadSource.MODELSCOPE: "google-bert/bert-large-uncased",
        },
        "bert-base-cased": {
            DownloadSource.DEFAULT: "google-bert/bert-base-cased",
            DownloadSource.MODELSCOPE: "google-bert/bert-base-cased",
        },
        "bert-large-cased": {
            DownloadSource.DEFAULT: "google-bert/bert-large-cased",
            DownloadSource.MODELSCOPE: "google-bert/bert-large-cased",
        },
        "bert-base-chinese": {
            DownloadSource.DEFAULT: "google-bert/bert-base-chinese",
            DownloadSource.MODELSCOPE: "google-bert/bert-base-chinese",
        },
        "bert-base-multilingual-cased": {
            DownloadSource.DEFAULT: "google-bert/bert-base-multilingual-cased",
            DownloadSource.MODELSCOPE: "google-bert/bert-base-multilingual-cased",
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
    simple_name="bert",
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
        "ernie-2.0-large-en": {
            DownloadSource.DEFAULT: "nghuyong/ernie-2.0-large-en",
            DownloadSource.MODELSCOPE: "nghuyong/ernie-2.0-large-en",
        },
    },
    simple_name="ernie-2.0",
    link='https://github.com/PaddlePaddle/ERNIE',
    description=''
)

register_model_group(
    models={
        "ernie-3.0-base-zh": {
            DownloadSource.DEFAULT: "nghuyong/ernie-3.0-base-zh",
            DownloadSource.MODELSCOPE: "nghuyong/ernie-3.0-base-zh",
        },
        "ernie-3.0-medium-zh": {
            DownloadSource.DEFAULT: "nghuyong/ernie-3.0-medium-zh",
            DownloadSource.MODELSCOPE: "nghuyong/ernie-3.0-medium-zh",
        },
        "ernie-3.0-mini-zh": {
            DownloadSource.DEFAULT: "nghuyong/ernie-3.0-mini-zh",
            DownloadSource.MODELSCOPE: "nghuyong/ernie-3.0-mini-zh",
        },
        "ernie-3.0-micro-zh": {
            DownloadSource.DEFAULT: "nghuyong/ernie-3.0-micro-zh",
            DownloadSource.MODELSCOPE: "nghuyong/ernie-3.0-micro-zh",
        },
        "ernie-3.0-nano-zh": {
            DownloadSource.DEFAULT: "nghuyong/ernie-3.0-nano-zh",
            DownloadSource.MODELSCOPE: "nghuyong/ernie-3.0-nano-zh",
        },
        "ernie-health-zh": {
            DownloadSource.DEFAULT: "nghuyong/ernie-health-zh",
            DownloadSource.MODELSCOPE: "nghuyong/ernie-health-zh",
        },
        "ernie-gram-zh": {
            DownloadSource.DEFAULT: "nghuyong/ernie-gram-zh",
            DownloadSource.MODELSCOPE: "nghuyong/ernie-gram-zh",
        }
    },
    simple_name="ernie-3.0",
    link='https://github.com/PaddlePaddle/ERNIE',
    description=''
)

if __name__ == "__main__":
    print(model_info)
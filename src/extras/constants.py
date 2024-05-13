import os
from collections import namedtuple, OrderedDict, defaultdict
from enum import Enum
from typing import Dict, Optional


ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"

DATA_ARGS_NAME = "data_args.bin"

MODEL_ARGS_NAME = "model_args.bin"

SAVE_MODEL_NAME = 'pytorch_model.bin'

BEST_MODEL_PATH = 'best_model'

SAVE_CONFIG_NAME = 'config.json'

OPTIMIZER_NAME = "optimizer.pt"

OPTIMIZER_NAME_BIN = "optimizer.bin"

SCHEDULER_NAME = "scheduler.pt"

SCALER_NAME = "scaler.pt"

FSDP_MODEL_NAME = "pytorch_model_fsdp"

LOG_FILE_NAME = "trainer_log.jsonl"

TRAINING_PNG_NAME = "training_logs.png"

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

# google bert
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

# microsoft deberta

register_model_group(
    models={
        "deberta-v3-base": {
            DownloadSource.DEFAULT: "microsoft/deberta-v3-base",
            DownloadSource.MODELSCOPE: "microsoft/deberta-v3-base",
        }
    },
    simple_name="deberta-v3",
    template="bert",
    hidden_size=768,
    link='',
    description="""
    Deberta中，把Content和Position embedding做了解耦，Word embedding依旧由Content embedding和Position组成，而不是简单的加和，具有更强的表示能力
    """
)

## chinese-roberta系列

register_model_group(
    models={
        "chinese-roberta-wwm-ext": {
            DownloadSource.DEFAULT: "hfl/chinese-roberta-wwm-ext",
            DownloadSource.MODELSCOPE: "hfl/chinese-roberta-wwm-ext",
        }
    },
    simple_name="wwm-ext",
    template="bert",
    hidden_size=768,
    link='https://github.com/ymcui/Chinese-BERT-wwm',
    description="""
    We carried out extensive experiments on ten Chinese NLP tasks to evaluate the created Chinese pre-trained language models as well as the proposed MacBERT. 
    Experimental results show that MacBERT could achieve state-of-the-art performances on many NLP tasks, 
    and we also ablate details with several findings that may help future research. We open-source our pre-trained language models for further facilitating our research community
    """
)

register_model_group(
    models={
        "chinese-macbert-base": {
            DownloadSource.DEFAULT: "hfl/chinese-macbert-base",
            DownloadSource.MODELSCOPE: "hfl/chinese-macbert-base",
        }
    },
    simple_name="macbert-base",
    template="bert",
    hidden_size=768,
    link='https://github.com/ymcui/Chinese-BERT-wwm',
    description="""
    MacBERT is an improved BERT with novel MLM as correction pre-training task,
    which mitigates the discrepancy of pre-training and fine-tuning.
    Instead of masking with [MASK] token, which never appears in the ﬁne-tuning stage, 
    we propose to use similar words for the masking purpose.
    A similar word is obtained by using Synonyms toolkit (Wang and Hu, 2017), 
    which is based on word2vec (Mikolov et al., 2013) similarity calculations. 
    If an N-gram is selected to mask, we will ﬁnd similar words individually.
    In rare cases, when there is no similar word, we will degrade to use random word replacement.
    """
)



register_model_group(
    models={
        "chinese-roberta-wwm-ext-large": {
            DownloadSource.DEFAULT: "hfl/chinese-roberta-wwm-ext-large",
            DownloadSource.MODELSCOPE: "hfl/chinese-roberta-wwm-ext-large",
        }
    },
    simple_name="wwm-ext-large",
    template="bert",
    hidden_size=768,
    link='https://github.com/ymcui/Chinese-BERT-wwm',
    description=''
)


## albert系列

register_model_group(
    models={
        "albert-base-v1": {
            DownloadSource.DEFAULT: "google-bert/albert-base-v1",
            DownloadSource.MODELSCOPE: "google-bert/albert-base-v1",
        },
        "albert-large-v1": {
            DownloadSource.DEFAULT: "google-bert/albert-large-v1",
            DownloadSource.MODELSCOPE: "google-bert/albert-large-v1",
        },
        "albert-xlarge-v1": {
            DownloadSource.DEFAULT: "google-bert/albert-xlarge-v1",
            DownloadSource.MODELSCOPE: "google-bert/albert-xlarge-v1",
        },
        "albert-xxlarge-v1": {
            DownloadSource.DEFAULT: "google-bert/albert-xxlarge-v1",
            DownloadSource.MODELSCOPE: "google-bert/albert-xxlarge-v1",
        }
    },
    simple_name="albert-v1",
    template="albert",
    hidden_size=768,
    link='',
    description=''
)

## ernie系列
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
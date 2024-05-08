from typing import TYPE_CHECKING, Any, Dict, List, Optional
import os
import torch
from transformers import PreTrainedModel

from extras.callbacks import LogCallback
from hparams.parser import get_train_args
from train.workflow import run_train


if TYPE_CHECKING:
    from transformers import TrainerCallback

def environ_set():
    os.environ.setdefault("TOKENIZERS_PARALLELISM",'true')

def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: Optional[List["TrainerCallback"]] = None):
    model_args, data_args, training_args = get_train_args(args)
    callbacks = [LogCallback()] if callbacks is None else callbacks
    run_train(model_args, data_args, training_args, callbacks)


def main():
    environ_set()
    run_exp()

if __name__ == "__main__":
    main()
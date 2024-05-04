import logging
import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch
import transformers
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_torch_bf16_gpu_available



from hparams.data_args import DataArguments
from hparams.load_args import ModelConfig
from hparams.model_args import ModelArguments
from hparams.training_args import TrainingArguments

from extras.logging import get_logger
logger = get_logger(__name__)


_TRAIN_ARGS = [ModelArguments, DataArguments, TrainingArguments]
_TRAIN_CLS = Tuple[ModelArguments, DataArguments, TrainingArguments]



def _parse_args(parser: "HfArgumentParser", args: Optional[Dict[str, Any]] = None) -> Tuple[Any]:
    if args is not None:
        return parser.parse_dict(args)

    (*parsed_args, unknown_args) = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if unknown_args:
        logging.info(parser.format_help())
        logging.info("Got unknown args, potentially deprecated arguments: {}".format(unknown_args))
        raise ValueError("Some specified arguments are not used by the HfArgumentParser: {}".format(unknown_args))

    return (*parsed_args,)

def _set_transformers_logging(log_level: Optional[int] = logging.INFO) -> None:
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def _parse_train_args(args: Optional[Dict[str, Any]] = None) -> _TRAIN_CLS:
    parser = HfArgumentParser(_TRAIN_ARGS)
    return _parse_args(parser, args)


def get_train_args(args: Optional[Dict[str, Any]] = None) -> _TRAIN_CLS:
    logger.info(f"Parsing training args...:{args}")
    model_args, data_args, training_args = _parse_train_args(args)

    # Setup logging
    if training_args.should_log:
        _set_transformers_logging()

    if training_args.max_steps == -1 and data_args.streaming:
        raise ValueError("Please specify `max_steps` in streaming mode.")

    model_args.model_max_length = data_args.cutoff_len
    # Log on each process the small summary:
    logger.info(
        "Process rank: {}, device: {}, num_gpu: {}, distributed training: {}, compute dtype: {}".format(
            training_args.local_rank,
            training_args.device,
            training_args.num_gpus,
            training_args.parallel_mode.value == "distributed",
            str(model_args.compute_dtype),
        )
    )

    transformers.set_seed(training_args.seed)

    return model_args, data_args, training_args
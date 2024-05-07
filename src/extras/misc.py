import gc
import os
from typing import TYPE_CHECKING, Dict, Tuple
import torch
from GPUtil import showUtilization as gpu_usage
from numba import cuda
from transformers import InfNanRemoveLogitsProcessor, LogitsProcessorList, PreTrainedModel
from transformers.utils import (
    is_torch_bf16_gpu_available,
    is_torch_cuda_available,
    is_torch_mps_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)
from transformers.utils.versions import require_version

from extras.loggings import get_logger

if TYPE_CHECKING:
    from transformers import TrainerCallback
    from hparams.data_args import DataArguments
    from hparams.model_args import ModelArguments
    from hparams.training_args import TrainingArguments


_is_fp16_available = is_torch_npu_available() or is_torch_cuda_available()
try:
    _is_bf16_available = is_torch_bf16_gpu_available()
except Exception:
    _is_bf16_available = False

logger = get_logger(__name__)

class AverageMeter:
    r"""
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by 2
        if param.__class__.__name__ == "Params4bit":
            if hasattr(param, "quant_storage") and hasattr(param.quant_storage, "itemsize"):
                num_bytes = param.quant_storage.itemsize
            else:
                num_bytes = 1

            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param

def get_current_device() -> torch.device:
    r"""
    Gets the current available device.
    """
    if is_torch_xpu_available():
        device = "xpu:{}".format(os.environ.get("LOCAL_RANK", "0"))
    elif is_torch_npu_available():
        device = "npu:{}".format(os.environ.get("LOCAL_RANK", "0"))
    elif is_torch_mps_available():
        device = "mps:{}".format(os.environ.get("LOCAL_RANK", "0"))
    elif is_torch_cuda_available():
        device = "cuda:{}".format(os.environ.get("LOCAL_RANK", "0"))
    else:
        device = "cpu"

    return torch.device(device)

def get_device_count() -> int:
    r"""
    Gets the number of available GPU devices.
    """
    if not torch.cuda.is_available():
        return 0

    return torch.cuda.device_count()

def get_logits_processor() -> "LogitsProcessorList":
    r"""
    Gets logits processor that removes NaN and Inf logits.
    """
    logits_processor = LogitsProcessorList()
    logits_processor.append(InfNanRemoveLogitsProcessor())
    return logits_processor

def torch_gc() -> None:
    r"""
    Collects GPU memory.
    """
    logger.info("Collects GPU memory.")
    gpu_usage()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    logger.info("GPU Usage after emptying the cache")
    gpu_usage()

def free_gpu_cache():
    logger.info("Initial GPU Usage")
    gpu_usage()
    # torch.cuda.empty_cache()
    if torch.cuda.is_available():
        cuda.select_device(0)
        cuda.close()
        cuda.select_device(0)
    logger.info("GPU Usage after emptying the cache")
    gpu_usage()


def infer_optim_dtype(model_dtype: torch.dtype) -> torch.dtype:
    r"""
    Infers the optimal dtype according to the model_dtype and device compatibility.
    """
    if _is_bf16_available and model_dtype == torch.bfloat16:
        return torch.bfloat16
    elif _is_fp16_available:
        return torch.float16
    else:
        return torch.float32


def is_path_available(path: os.PathLike) -> bool:
    r"""
    Checks if the path is empty or not exist.
    """
    if not os.path.exists(path):
        return True
    elif os.path.isdir(path) and not os.listdir(path):
        return True
    else:
        return False



def try_download_model_from_ms(model_args: "ModelArguments") -> str:
    if not use_modelscope() or os.path.exists(model_args.model_name_or_path):
        return model_args.model_name_or_path

    try:
        from modelscope import snapshot_download

        revision = "master" if model_args.model_revision == "main" else model_args.model_revision
        return snapshot_download(model_args.model_name_or_path, revision=revision, cache_dir=model_args.cache_dir)
    except ImportError:
        raise ImportError("Please install modelscope via `pip install modelscope -U`")

def use_modelscope() -> bool:
    return bool(int(os.environ.get("USE_MODELSCOPE_HUB", "0")))


if __name__ == "__main__":
    logger.info(get_device_count())    
import json
import os
from collections import defaultdict
from typing import Any, Dict, Optional


from .constants import (
    model_info,
    DownloadSource,
)
from .misc import use_modelscope

DEFAULT_CACHE_DIR = "cache"
DEFAULT_CONFIG_DIR = "config"
DEFAULT_DATA_DIR = "data"
DEFAULT_SAVE_DIR = "saves"
USER_CONFIG = "user.config"




def get_save_dir(*args) -> os.PathLike:
    return os.path.join(DEFAULT_SAVE_DIR, *args)


def get_config_path() -> os.PathLike:
    return os.path.join(DEFAULT_CACHE_DIR, USER_CONFIG)


def get_save_path(config_path: str) -> os.PathLike:
    return os.path.join(DEFAULT_CONFIG_DIR, config_path)


def load_config() -> Dict[str, Any]:
    try:
        with open(get_config_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"lang": None, "last_model": None, "path_dict": {}, "cache_dir": None}


def save_config(lang: str, model_name: Optional[str] = None, model_path: Optional[str] = None) -> None:
    os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)
    user_config = load_config()
    user_config["lang"] = lang or user_config["lang"]
    if model_name:
        user_config["last_model"] = model_name
        user_config["path_dict"][model_name] = model_path
    with open(get_config_path(), "w", encoding="utf-8") as f:
        json.dump(user_config, f, indent=2, ensure_ascii=False)


def load_args(config_path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(get_save_path(config_path), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_args(config_path: str, config_dict: Dict[str, Any]) -> str:
    os.makedirs(DEFAULT_CONFIG_DIR, exist_ok=True)
    with open(get_save_path(config_path), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    return str(get_save_path(config_path))


def get_model_path(model_name: str) -> str:
    user_config = load_config()
    path_dict: Dict[DownloadSource, str] = model_info.get(model_name, defaultdict(str))
    model_path = user_config["path_dict"].get(model_name, None) or path_dict.get(DownloadSource.DEFAULT, None)
    if (
        use_modelscope()
        and path_dict.get(DownloadSource.MODELSCOPE)
        and model_path == path_dict.get(DownloadSource.DEFAULT)
    ):  # replace path
        model_path = path_dict.get(DownloadSource.MODELSCOPE)
    return model_path


def get_prefix(model_name: str) -> str:
    return model_name.split("-")[0]

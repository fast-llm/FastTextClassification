import importlib.metadata
import importlib.util
from distutils.version import LooseVersion

def _is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _get_package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "0.0.0"

def compare_versions(version1, version2):
    """比较两个版本号字符串的大小。

    Args:
        version1 (str): 第一个版本号字符串。
        version2 (str): 第二个版本号字符串。
    Returns:
        int: 如果version1大于version2，则返回1；如果version1小于version2，则返回-1；如果version1等于version2，则返回0。
    """
    if LooseVersion(version1) > LooseVersion(version2):
        return 1
    elif LooseVersion(version1) < LooseVersion(version2):
        return -1
    else:
        return 0

def get_transformer_version():
    return _get_package_version('transformers')

def is_fastapi_availble():
    return _is_package_available("fastapi")

def is_scikit_learn_availble():
    return _is_package_available("scikit-learn")

def is_jieba_available():
    return _is_package_available("jieba")

def is_matplotlib_available():
    return _is_package_available("matplotlib")

def is_nltk_available():
    return _is_package_available("nltk")

def is_starlette_available():
    return _is_package_available("sse_starlette")

def is_uvicorn_available():
    return _is_package_available("uvicorn")
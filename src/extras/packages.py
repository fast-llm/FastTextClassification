import importlib.metadata
import importlib.util


def _is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _get_package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "0.0.0"

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
import logging
import sys

# ANSI escape sequences for colors
class TerminalColor:
    RESET = '\033[0m'
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    DEBUG = CYAN       # 调试信息使用蓝色
    INFO = GREEN       # 信息使用绿色
    WARNING = YELLOW   # 警告使用黄色
    ERROR = RED        # 错误使用红色
    CRITICAL = MAGENTA # 严重错误使用品红
    

class ColoredFormatter(logging.Formatter):
    """自定义的Formatter，根据日志级别输出不同颜色的日志信息。"""
    def __init__(self, fmt, datefmt=None):
        super().__init__(fmt, datefmt)

    def format(self, record):
        color_map = {
            logging.DEBUG: TerminalColor.DEBUG,
            logging.INFO: TerminalColor.INFO,
            logging.WARNING: TerminalColor.WARNING,
            logging.ERROR: TerminalColor.ERROR,
            logging.CRITICAL: TerminalColor.CRITICAL
        }
        # 根据日志级别选择颜色
        record_color = color_map.get(record.levelno, TerminalColor.RESET)
        # 让基类Formatter进行实际的格式化工作
        # 让基类Formatter进行实际的格式化工作
        formatted_record = super().format(record)
        return f"{record_color}{formatted_record}{TerminalColor.RESET}"


class LoggerHandler(logging.Handler):
    r"""
    Logger handler used in Web UI.
    """

    def __init__(self):
        super().__init__()
        self.log = ""

    def reset(self):
        self.log = ""

    def emit(self, record):
        if record.name == "httpx":
            return
        log_entry = self.format(record)
        self.log += log_entry
        self.log += "\n\n"


def get_logger(name: str, logger_level=logging.INFO) -> logging.Logger:
    """获取配置了彩色输出和详细错误信息的logger。"""
    fmt = ("%(asctime)s - %(levelname)s - %(name)s - "
           "%(filename)s:%(funcName)s:%(lineno)d - %(message)s")
    datefmt = "%m/%d/%Y %H:%M:%S"
    formatter = ColoredFormatter(fmt=fmt, datefmt=datefmt)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logger_level)
    logger.addHandler(handler)
    return logger


def reset_logging() -> None:
    r"""
    Removes basic config of root logger. (unused in script)
    """
    root = logging.getLogger()
    list(map(root.removeHandler, root.handlers))
    list(map(root.removeFilter, root.filters))

if __name__ == "__main__":
    """
    在 Python 的 logging 库中，日志级别从高到低依次是：
    CRITICAL, ERROR, WARNING, INFO, DEBUG。
    默认情况下，Logger 的级别设置为 WARNING，
    这意味着只有 WARNING 和比它级别更高的日志消息（ERROR 和 CRITICAL）会被输出。
    """
    # 使用示例
    logger = get_logger('test', logging.DEBUG)
    logger.info("This is an info message.")
    logger.error("This is an error message.")
    logger.debug("This is a debug message.")
    logger.warning("This is a warning message.")
    logger.critical("This is a critical message.")
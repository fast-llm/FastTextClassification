import yaml
import json
from extras.loggings import get_logger

logger = get_logger(__name__)


class ModelConfig:
    def __init__(self, file_path:str):
        self.file_path = file_path
        self.config = None
        self._load_config()

    def _load_config(self):
        try:
            if self.file_path.endswith('yaml'):
                with open(self.file_path, 'r') as file:
                    self.config = yaml.safe_load(file)
                    if self.config is None:
                        raise ValueError("YAML file is empty or invalid.")
            elif self.file_path.endswith('json'):
                with open(self.file_path, 'r') as file:
                    self.config = json.load(file)
                    if self.config is None:
                        raise ValueError("JSON file is empty or invalid.")
            else:
                raise ValueError("Unsupported file format. Only YAML and JSON files are supported.")
        except FileNotFoundError:
            raise FileNotFoundError(f"file '{self.file_path}' not found.")

    def __str__(self):
        config_str = json.dumps(self.config, ensure_ascii=False, indent=4)
        return f"ModelConfig:\n{config_str}"
    
    def __dict__(self):
        return json.dumps(self.config, ensure_ascii=False, indent=4)
    
    def to_dict(self):
        return self.__dict__()
    
    def get_parameter(self, parameter_name):
        try:
            return self.config[parameter_name]
        except KeyError:
            raise KeyError(f"Parameter '{parameter_name}' not found in the {self.file_path} file.")


if __name__ == '__main__':
    # 示例用法
    try:
        yaml_config = ModelConfig('./examples/hparamsConfig/bert.yaml')
        logger.info(f"Parameters: {yaml_config}")
        learning_rate = yaml_config.get_parameter('training_parameters')['learning_rate']
        logger.info(f"Learning rate: {learning_rate}")
    except Exception as e:
        logger.error(f"Error:{e}")

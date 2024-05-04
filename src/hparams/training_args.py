import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

from .load_args import ModelConfig
from models.component.common import ParallelMode, MLPLayer

from extras.logging import get_logger
logger = get_logger(__name__)



@dataclass
class TrainingArguments:
    r"""
    Arguments pertaining to which techniques we are going to fine-tuning with.
    """

    train_config_path: str = field(metadata={"help": "Path to the configuration file."},)
    output_dir: str = field(metadata={"help": "output directory for the model and the training logs."})
    log_dir: str = field(default = None, metadata={"help": "output directory for the model and the training logs."})
    local_rank: str = field(default = None,metadata={"help": "local_rank for distributed training on gpus."})
    device: str = field(default = None,metadata={"help": "device for training. cpu, cuda, xpu, npu"})
    num_gpus : int = field(default = None,metadata={"help": "number of gpus to use."})

    max_steps: int = field(default=None,metadata={"help": "Maximum number of training steps."},)
    plot_loss: bool = field(default=False,metadata={"help": "Whether or not to save the training loss curves."})
    should_log: bool = field(default=True,metadata={"help": "Whether or not to log the training process."})
    
    # 训练配置
    lang: str = field(default=None, metadata={"help": "data embedding language, en, cn, multi"})
    use_word: bool = field(default=None, metadata={"help": "use word"})
    epochs: int = field(default=None,metadata={"help": "Number of training epochs."})
    require_improvement: int = field(default=None,metadata={"help": "Number of steps to wait for improvement."})
    per_device_train_batch_size: int = field(default=None,metadata={"help": "Batch size per GPU/TPU core/CPU for training."})
    per_device_eval_batch_size: int = field(default=None,metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."})
    update_all_layers: bool = field(default=None,metadata={"help": "if update all layers"})
    optimizer: str = field(default=None,metadata={"help": "Optimizer to use for training."})
    lr_scheduler_type: str = field(default=None,metadata={"help": "Type of learning rate scheduler."})
    learning_rate: float = field(default=None,metadata={"help": "The initial learning rate for Adam."})
    warmup_proportion: float = field(default=None,metadata={"help": "Proportion of training to perform linear learning rate warmup for."})
    dropout_rate: float = field(default=None,metadata={"help": "Dropout rate for the model."})
    activation: str = field(default=None,metadata={"help": "Activation function for the model."})
    seed: int = field(default=None,metadata={"help": "Random seed for initialization."})
    
    # 优化器配置
    epsilon: float = field(default=None,metadata={"help": "Epsilon for the Adam optimizer."})
    weight_decay: float = field(default=None,metadata={"help": "Weight decay to apply."})
    max_grad_norm: float = field(default=None,metadata={"help": "Max gradient norm."})
    gradient_accumulation_steps: int = field(default=None,metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."})
    
    
    # 模型结构参数
    num_classes: int = field(default=None,metadata={"help": "Number of classes in the dataset."})
    mlp_layers: List[MLPLayer] = field(default_factory=list, metadata={"help": "MLP layers configuration."})
    
    # 超参数搜索配置
    enable_search: bool = field(default=False,metadata={"help": "Whether or not to enable hyperparameter search."})
    search_strategy: str = field(default=None,metadata={"help": "Hyperparameter search strategy."})
    search_trials: int = field(default=None,metadata={"help": "Number of hyperparameter search trials."})
    search_metric: str = field(default=None,metadata={"help": "Metric to optimize during hyperparameter search."})
    gamma: float = field(default=None,metadata={"help": "Factor by which to reduce the learning rate."})
    optimizer_options: Optional[Dict[str, Any]] = field(default=None,metadata={"help": "Optimizer options."})
    learning_rate_bounds: List[float] = field(default=None,metadata={"help": "Bounds for the learning rate."})
    batch_size_options: List[int] = field(default=None,metadata={"help": "Batch size options."})
    regularization_strengths: List[float] = field(default=None,metadata={"help": "Regularization strengths."})
    dropout_rates: List[float] = field(default=None,metadata={"help": "Dropout rates."})
    activation_functions: List[str] = field(default=None,metadata={"help": "Activation functions."})
    weight_initialization: List[str] = field(default=None,metadata={"help": "Weight initialization functions."})
    
    # evaluation
    log_dir: str = field(default=None,metadata={"help": "Directory to save the logs."})
    logging_steps: int = field(default=None,metadata={"help": "Number of steps to log the training loss."})
    eval_steps: int = field(default=None,metadata={"help": "Number of steps to evaluate the model."})
    eval_metric: str = field(default=None,metadata={"help": "Evaluation metric."})
    eval_batch_size: int = field(default=None,metadata={"help": "Batch size for evaluation."})
    # save
    save_steps: int = field(default=None,metadata={"help": "Number of steps to save the model."})
    save_total_limit: int = field(default=None,metadata={"help": "Total number of checkpoints to save."})
    
    # early stopping
    early_stopping: bool = field(default=None,metadata={"help": "Whether or not to enable early stopping."})
    patience: int = field(default=None,metadata={"help": "Number of epochs to wait before early stopping."})
    delta: float = field(default=None,metadata={"help": "Minimum change in the monitored quantity to qualify as an improvement."})
    early_stop_metric: str = field(default=None,metadata={"help": "Metric to monitor for early stopping."})
    
    
    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg
        self.parallel_mode = self._get_parallel_mode()
        self._load_config()
    
    def _get_parallel_mode(self):
        if self.num_gpus > 1:
            return ParallelMode.DISTRIBUTED
        elif self.num_gpus == 1:
            return ParallelMode.SINGLE
        else:
            return ParallelMode.NOT_PARALLEL
    
    def _load_config(self):
        logger.info(f"Loading training configuration from {self.train_config_path}")
        config_data = ModelConfig(self.train_config_path)
        logger.info(f"Training configuration: {config_data}")
        
        if not self.log_dir:
            self.log_dir = config_data.get_parameter('output').get('log_dir', 'log')
        if not self.lang:
            self.lang = config_data.get_parameter("training").get('lang', 'cn')
        if not self.use_word:
            self.use_word = config_data.get_parameter("training").get('use_word', False)
        if not self.num_classes:
            self.num_classes = config_data.get_parameter("model_parameters").get('num_classes', 2)
        mlp_layers_config = config_data.get_parameter("model_parameters").get('mlp_layers', [])
        mlp_layers = []
        for layer_config in mlp_layers_config:
            layer = MLPLayer(**layer_config)
            mlp_layers.append(layer)
        self.mlp_layers = mlp_layers
        
        if not self.epochs:
            self.epochs = config_data.get_parameter("training").get('epochs', 20)
        if not self.require_improvement:
            self.require_improvement = config_data.get_parameter("training").get('require_improvement', 1000)
        if not self.per_device_train_batch_size:
            self.per_device_train_batch_size = config_data.get_parameter("training").get('per_device_train_batch_size', 4)
        if not self.per_device_eval_batch_size:
            self.per_device_eval_batch_size = config_data.get_parameter("training").get('per_device_eval_batch_size', 4)
        if not self.update_all_layers:
            self.update_all_layers = config_data.get_parameter("training").get('update_all_layers', False)
        if not self.optimizer:
            self.optimizer = config_data.get_parameter("training").get('optimizer', self.optimizer)
        if not self.lr_scheduler_type:
            self.lr_scheduler_type = config_data.get_parameter("training").get('lr_scheduler_type', self.lr_scheduler_type)
        if not self.learning_rate:
            self.learning_rate = config_data.get_parameter("training").get('learning_rate', self.learning_rate)
        if not self.warmup_proportion:
            self.warmup_proportion = config_data.get_parameter("training").get('warmup_proportion', self.warmup_proportion)
        if not self.dropout_rate:
            self.dropout_rate = config_data.get_parameter("training").get('dropout_rate', 0.1)
        if not self.activation:
            self.activation = config_data.get_parameter("training").get('activation', self.activation)
        if not self.seed:
            self.seed = config_data.get_parameter("training").get('seed', 42)
        
        if not self.epsilon:
            self.epsilon = config_data.get_parameter("optimizer_settings").get('epsilon', self.epsilon)    
        if not self.weight_decay:
            self.weight_decay = config_data.get_parameter("optimizer_settings").get('weight_decay', self.weight_decay)
        if not self.max_grad_norm:
            self.max_grad_norm = config_data.get_parameter("optimizer_settings").get('max_grad_norm', self.max_grad_norm)
        if not self.gradient_accumulation_steps:
            self.gradient_accumulation_steps = config_data.get_parameter("optimizer_settings").get('gradient_accumulation_steps', self.gradient_accumulation_steps)

        if not self.enable_search:
            self.enable_search = config_data.get_parameter("hyper_params").get('enable_search', self.enable_search)
        if not self.search_strategy:
            self.search_strategy = config_data.get_parameter("hyper_params").get('search_strategy', self.search_strategy)
        if not self.search_trials:
            self.search_trials = config_data.get_parameter("hyper_params").get('search_trials', self.search_trials)
        if not self.search_metric:
            self.search_metric = config_data.get_parameter("hyper_params").get('search_metric', self.search_metric)
        if not self.gamma:
            self.gamma = config_data.get_parameter("hyper_params").get('gamma', self.gamma)
        if not self.optimizer_options:
            self.optimizer_options = config_data.get_parameter("hyper_params").get('optimizer_options', self.optimizer_options)
        if not self.learning_rate_bounds:
            self.learning_rate_bounds = config_data.get_parameter("hyper_params").get('learning_rate_bounds', self.learning_rate_bounds)
        if not self.batch_size_options:
            self.batch_size_options = config_data.get_parameter("hyper_params").get('batch_size_options', self.batch_size_options)
        if not self.regularization_strengths:
            self.regularization_strengths = config_data.get_parameter("hyper_params").get('regularization_strengths', self.regularization_strengths)
        if not self.dropout_rates:
            self.dropout_rates = config_data.get_parameter("hyper_params").get('dropout_rates', self.dropout_rates)
        if not self.activation_functions:
            self.activation_functions = config_data.get_parameter("hyper_params").get('activation_functions', self.activation_functions)
        if not self.weight_initialization:
            self.weight_initialization = config_data.get_parameter("hyper_params").get('weight_initialization', self.weight_initialization)
        
        # early stopping
        if not self.early_stopping:
            self.early_stopping = config_data.get_parameter("early_stopping").get('early_stopping', self.early_stopping)
        if not self.patience:
            self.patience = config_data.get_parameter("early_stopping").get('patience', self.patience)
        if not self.delta:
            self.delta = config_data.get_parameter("early_stopping").get('delta', self.delta)
        if not self.early_stop_metric:
            self.early_stop_metric = config_data.get_parameter("early_stopping").get('early_stop_metric', self.early_stop_metric)

    def save_to_json(self, json_path: str):
        r"""Saves the content of this instance in JSON format inside `json_path`."""
        json_string = json.dumps(asdict(self), indent=4, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        r"""Creates an instance from the content of `json_path`."""
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))


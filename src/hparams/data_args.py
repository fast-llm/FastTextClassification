from dataclasses import dataclass, field
from typing import Literal, Optional

from extras.loggings import get_logger
from hparams.load_args import ModelConfig
from models.component.common import MLPLayer
logger = get_logger(__name__)

@dataclass
class DataArguments:
    r"""
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """
    dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The name of provided dataset(s) to use. Use commas to separate multiple datasets."},
    )
    dataset_dir: str = field(
        default="data",
        metadata={"help": "Path to the folder containing the datasets."},
    )
    data_config_path: str = field(default = None, metadata={"help": "Path to the configuration file."},)
    # 数据集参数
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of examples for each dataset."},
    )
    streaming: bool = field(
        default=None,
        metadata={"help": "Enable dataset streaming load."},
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the training dataset."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the test dataset."},
    )
    val_file: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the validation dataset."},
    )
    class_file: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the class dataset."},
    )
    vocab_file: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the vocabulary dataset."},
    )
    SEP: str = field(
        default=None,
        metadata={"help": "The separator used in the dataset."},
    )
    
    cutoff_len: int = field(
        default=None,
        metadata={"help": "The cutoff length of the model inputs after tokenization."},
    )
    do_lower_case: bool = field(
        default=None,
        metadata={"help": "Whether to lower case the input text. True for uncased models, False for cased models."},
    )
    multi_class: bool = field(
        default=None,
        metadata={"help": "Whether to use multi-class classification."},
    )
    multi_label: bool = field(
        default=None,
        metadata={"help": "Whether to use multi-label classification."},
    )
    ############
    reserved_label_len: int = field(
        default=1,
        metadata={"help": "The minimum cutoff length reserved for label after tokenization."},
    )
    train_on_prompt: bool = field(
        default=False,
        metadata={"help": "Whether to disable the mask on the prompt or not."},
    )
    
    buffer_size: int = field(
        default=16384,
        metadata={"help": "Size of the buffer to randomly sample examples from in dataset streaming."},
    )
    mix_strategy: Literal["concat", "interleave_under", "interleave_over"] = field(
        default="concat",
        metadata={"help": "Strategy to use in dataset mixing (concat/interleave) (undersampling/oversampling)."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."},
    )
    processing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the pre-processing."},
    )
    shuffle: Optional[bool] = field(
        default = None,
        metadata={"help": "shuffle data"}
    )
    drop_last: Optional[bool] = field(
        default = None,
        metadata={"help": "drop last data"}
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to ignore the tokens corresponding to padded labels in the loss computation."
        },
    )

    tokenized_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save or load the tokenized datasets."},
    )

    def __post_init__(self):
        if self.reserved_label_len >= self.cutoff_len:
            raise ValueError("`reserved_label_len` must be smaller than `cutoff_len`.")

        if self.streaming and self.max_samples is not None:
            raise ValueError("`max_samples` is incompatible with `streaming`.")

        self._load_config()
        
        if self.multi_class and self.multi_label:
            raise ValueError("Only one of `multi_class` or `multi_label` can be set to True.")

    def _load_config(self):
        logger.info(f"Loading training configuration from {self.data_config_path}")
        config_data = ModelConfig(self.data_config_path)
        
        if not self.max_samples:
            self.max_samples = config_data.get_parameter("data").get('max_samples', None)
            if self.max_samples == 'None':
                self.max_samples = None
        if not self.processing_num_workers:
            self.processing_num_workers = config_data.get_parameter("data").get("processing_num_workers", None)
        if not self.shuffle:
            self.shuffle = config_data.get_parameter("data").get("shuffle", True)
        if not self.drop_last:
            self.drop_last = config_data.get_parameter("data").get("drop_last", False)
        if not self.streaming:
            self.streaming = config_data.get_parameter("data").get('streaming', False)
        
        if not self.train_file:
            self.train_file = config_data.get_parameter("data").get('train_file', None)
        if not self.test_file:
            self.test_file = config_data.get_parameter("data").get('test_file', None)
        if not self.val_file:
            self.val_file = config_data.get_parameter("data").get('val_file', None)
        if not self.class_file:
            self.class_file = config_data.get_parameter("data").get('class_file', None)
        if not self.vocab_file:
            self.vocab_file = config_data.get_parameter("data").get('vocab_file', None)
        if not self.SEP:
            self.SEP = config_data.get_parameter("data").get('SEP', None)
        if not self.cutoff_len:
            self.cutoff_len = config_data.get_parameter("data").get('cutoff_len', None)
        self.pad_size = self.cutoff_len
        if not self.do_lower_case:
            self.do_lower_case = config_data.get_parameter("data").get('do_lower_case', None)
        if not self.multi_class:
            self.multi_class = config_data.get_parameter("data").get('multi_class', False)
        if not self.multi_label:
            self.multi_label = config_data.get_parameter("data").get('multi_label', False)
        logger.info(f"Training data configuration: {config_data}")
# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/summarization/run_summarization.py
import time
from importlib import import_module
from typing import TYPE_CHECKING, List, Optional


from data.common import DatasetConfig
from extras.constants import get_model_info
from tools.train_eval import train
from train.trainer import CustomClassifyTrainer
from .trainer_utils import create_config, create_model, setup_seed, init_network

from extras.loggings import get_logger
from utils import check_dir_exist, get_time_dif

logger = get_logger(__name__)


if TYPE_CHECKING:
    from transformers import TrainerCallback
    from hparams.data_args import DataArguments
    from hparams.model_args import ModelArguments
    from hparams.training_args import TrainingArguments

def run_train(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    setup_seed(training_args.seed)
    embedding = 'embedding_SougouNews.npz' if training_args.lang == "cn" else "glove.840B.300d.txt"
    if model_args.embedding == 'random':
        embedding = 'random'
    model_name = model_args.model_name
    
    from data.transformer_loader import build_dataset, build_iterator
    
    config = create_config(
        model_args = model_args,
        data_args = data_args,
        training_args = training_args
    )
    
    # 超参数设置
    train_data, val_data, test_data = build_dataset(config)
    start_time = time.time()
    print("Loading data...")
    # vocab, train_data, dev_data, test_data = build_dataset(config, 
    #                                                            word=training_args.use_word, 
    #                                                            multi_label=data_args.multi_label,
    #                                                            n_samples=data_args.max_samples, 
    #                                                            lang=training_args.lang,
    #                                                            sep=data_args.SEP)
    #     config.vocab = vocab
    #     config.n_vocab = len(vocab)
    check_dir_exist(training_args.output_dir, create=True)
    
    
    train_iter = build_iterator(train_data, config)
    val_iter = build_iterator(val_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    
    
    logger.info(f"Time usage: {time_dif}")

    # train
    model = create_model(config)
    if model_args.architecture == 'DNN':
        init_network(model)

    print("Training...")
    start_time = time.time()
    trainer = CustomClassifyTrainer(model,
                                    training_args=training_args,
                                    model_args=model_args,
                                    train_iter=train_iter,
                                    val_iter=val_iter,
                                    test_iter=test_iter)
    trainer.train()
    
    time_dif = get_time_dif(start_time)
    print(f"Time usage:{time_dif}")

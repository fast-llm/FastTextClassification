# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/summarization/run_summarization.py
import time
from importlib import import_module
from typing import TYPE_CHECKING, List, Optional

from transformers import DataCollatorForSeq2Seq

from extras.constants import get_model_info
from tools.train_eval import init_network, setup_seed, train

from extras.logging import get_logger
from utils import check_dir_exist
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
    
    embedding = 'embedding_SougouNews.npz' if training_args.lang == "cn" else "glove.840B.300d.txt"
    if model_args.embedding == 'random':
        embedding = 'random'
    model_name = model_args.model_name
    if model_name == 'FastText':
        from tools.utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    elif model_args.architecture == "transformers":
        from data.transformer_loader import build_dataset, build_iterator, get_time_dif
    else:
        from tools.utils import build_dataset, build_iterator, get_time_dif

    model_info = get_model_info(model_name)
    x = import_module('models.' + model_info.template)
    config = x.Config(model_args,
                        data_args,
                        training_args)
    # 超参数设置
    setup_seed(training_args.seed)
    start_time = time.time()
    print("Loading data...")
    if model_args.architecture == "transformers":
        logger.info(f"Building dataset for transformers {data_args.multi_label}")
        train_data, dev_data, test_data = build_dataset(config,
                                                        multi_label=data_args.multi_label,
                                                        n_samples=data_args.max_samples, 
                                                        sep=data_args.SEP)
    else:
        vocab, train_data, dev_data, test_data = build_dataset(config, 
                                                               word=training_args.use_word, 
                                                               multi_label=data_args.multi_label,
                                                               n_samples=data_args.max_samples, 
                                                               lang=training_args.lang,
                                                               sep=data_args.SEP)
        config.vocab = vocab
        config.n_vocab = len(vocab)
    check_dir_exist(training_args.output_dir, create=True)
    
    
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    logger.info("Time usage:", time_dif)

    # train
    model = x.Model(config)
    if model_args.architecture == 'DNN':
        init_network(model)
    print(model.parameters)
    print("Training...")
    start_time = time.time()
    train(config, model, train_iter, dev_iter, test_iter)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

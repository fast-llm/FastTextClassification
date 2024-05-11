# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/summarization/run_summarization.py
import time
from importlib import import_module
from typing import TYPE_CHECKING, List, Optional
from torch.utils.data import Dataset, DataLoader

from data.common import DatasetConfig
from data.transformer_loader import TransformerDataset
from extras.constants import get_model_info
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

    from data.transformer_loader import build_dataset
    
    config = create_config(
        model_args = model_args,
        data_args = data_args,
        training_args = training_args
    )
    
    # 超参数设置
    train_dataset, val_dataset, test_dataset = build_dataset(config)

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
    
    train_loader = DataLoader(train_dataset,
                              batch_size=config.per_device_train_batch_size,
                              num_workers=config.num_workers,
                              shuffle=config.shuffle,
                              drop_last=config.drop_last
                              )
    val_loader = DataLoader(val_dataset,
                            batch_size=config.per_device_eval_batch_size,
                            num_workers=config.num_workers,
                              shuffle=config.shuffle,
                              drop_last=config.drop_last
                            )
    test_loader = DataLoader(test_dataset,
                             batch_size=config.per_device_eval_batch_size,
                             num_workers=config.num_workers,
                              shuffle=config.shuffle,
                              drop_last=config.drop_last)
    
    time_dif = get_time_dif(start_time)
    
    
    logger.info(f"Time usage: {time_dif}")

    # train
    model = create_model(config)
    if model_args.architecture == 'DNN':
        init_network(model)

    print("Training...")
    start_time = time.time()
    trainer = CustomClassifyTrainer(model,
                                    tokenizer=config.tokenizer,
                                    data_args=data_args,
                                    training_args=training_args,
                                    model_args=model_args,
                                    train_iter=train_loader,
                                    val_iter=val_loader,
                                    test_iter=test_loader)
    trainer.train()
    
    time_dif = get_time_dif(start_time)
    print(f"Time usage:{time_dif}")

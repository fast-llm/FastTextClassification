import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import time
import numpy as np
import torch
from transformers import Trainer
from accelerate import Accelerator


from data.transformer_loader import DatasetIterater
from extras.loggings import get_logger
from hparams.data_args import DataArguments
from models.component.common import BaseModel
from utils import get_time_dif
from .trainer_utils import calculate_accuracy, calculate_num_training_steps, create_custom_optimizer, create_custom_scheduler, create_loss_fn


if TYPE_CHECKING:
    from transformers.trainer import PredictionOutput
    from ..hparams.training_args import TrainingArguments
    from ..hparams.model_args import ModelArguments

logger = get_logger(__name__)


class CustomClassifyTrainer(object):

    def __init__(self, 
                 model: BaseModel  = None,
                 model_args: "ModelArguments" = None,
                 data_args: "DataArguments"= None,
                 training_args: "TrainingArguments"= None,
                 train_iter: "DatasetIterater" = None, 
                 val_iter: "DatasetIterater" = None, 
                 test_iter: "DatasetIterater" = None):
        self.model = model
        self.training_args = training_args
        self.model_args = model_args
        self.data_args = data_args
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.test_iter = test_iter
        
        self.init_para()
        self.init_model()
        
        
    def init_para(self):
        self.epochs = self.training_args.epochs
        self.logging_step = self.training_args.logging_steps
        self.num_training_steps = calculate_num_training_steps(
            dataset_size = len(self.train_iter),
            num_gpus = self.training_args.num_gpus,
            per_device_train_batch_size=self.training_args.per_device_train_batch_size,
            epochs = self.epochs   
        )
        self.gradient_accumulation_steps = self.training_args.gradient_accumulation_steps
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps
            )
        
        self.loss_fn = create_loss_fn(self.training_args.loss_fn)
        self.multi_class = self.data_args.multi_class
        self.multi_label = self.data_args.multi_label
        self.threshold = self.training_args.threshold
        self.start_time = time.time()
    
    def init_model(self):
        import torch.nn.utils as nutils
        # 参数初始化
        
        
        # 梯度裁剪
        nutils.clip_grad_norm(self.model.parameters(), 
                              max_norm=self.training_args.max_grad_norm)

    
    def train(self):
        self.optimizer = create_custom_optimizer(self.model, self.training_args)

        self.lr_scheduler = create_custom_scheduler(self.training_args, self.num_training_steps, self.optimizer)
        
        self.model, self.optimizer, self.lr_scheduler, self.train_iter, self.val_iter,self.test_iter  = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler, self.train_iter, self.val_iter, self.test_iter)
        
        for epoch in range(self.epochs):
            self.accelerator.print(f"--------------Training Epoch {epoch}--------------")
            self.train_one_epoch(epoch)
            self.validate(epoch)

    
    def train_one_epoch(self,epoch:int):
        
        self.model.train()
        with self.accelerator.accumulate(self.model):
            avg_loss = 0.0
            total_accurate = 0.0
            num_elem = 0
            for step, (inputs, labels) in enumerate(self.train_iter):
                input_ids, attention_mask, token_type_ids = inputs
                # accelerator.print(f"input_ids device: {input_ids.device}")
                out, _ = self.model([input_ids, attention_mask, token_type_ids])
                # accelerator.print(f"out device: {out.device}")
                # accelerator.print(f"label-0 device: {labels.device}")
                labels = labels.to(out.device)
                
                # accelerator.print(f"label-1 device: {labels.device}")
                self.optimizer.zero_grad()
                
                loss = self.loss_fn(out, labels)
                gather_loss = self.accelerator.gather(loss)
                self.accelerator.backward(loss)
                
                self.optimizer.step()
                self.lr_scheduler.step()
                
            
                # accelerator.print(f"label-1 device: {labels.device}")
                out, labels = self.accelerator.gather((out, labels))
                accurate, accurate_all, total = calculate_accuracy(out, 
                                                                   labels,
                                                                   threshold=self.threshold,
                                                                   multi_class=self.multi_class,
                                                                   multi_label=self.multi_label
                                                                   )
                avg_loss += gather_loss.mean().item()
                num_elem += total
                total_accurate += accurate
                if (1+step) % self.logging_step == 0:
                    avg_loss /= self.logging_step
                    eval_metric = total_accurate / num_elem
                    self.accelerator.print(f"{accurate},{accurate_all},{total}")
                    self.accelerator.print(f"Time: {get_time_dif(self.start_time)}, Train epoch:{epoch}, step:{step}, Loss {avg_loss}, eval_metric:{100*eval_metric:.2f}%")
                    avg_loss = 0.0
                    total_accurate = 0.0
                    num_elem = 0
    
    def validate(self, epoch:int):
        self.model.eval()
        num_elem = 0
        total_accurate = 0
        avg_loss = 0
        self.accelerator.print(f"--------------Time: {get_time_dif(self.start_time)}, Validate on validation set. epoch {epoch}--------------")
        with torch.inference_mode():
            for step, (inputs, labels) in enumerate(self.val_iter):
                input_ids, attention_mask, token_type_ids = inputs
                # accelerator.print(f"input_ids device: {input_ids.device}")
                out, _ = self.model([input_ids, attention_mask, token_type_ids])
                
                labels = labels.to(out.device)
                loss = self.loss_fn(out, labels)

                out, labels,loss = self.accelerator.gather((out, labels, loss))
                accurate, accurate_all, total = calculate_accuracy(out, 
                                                                   labels,
                                                                   threshold=self.threshold,
                                                                   multi_class=self.multi_class,
                                                                   multi_label=self.multi_label
                                                                   )
                avg_loss += loss.mean().item()
                num_elem += total
                total_accurate += accurate
            eval_metric = total_accurate / num_elem
            # print
            avg_loss /= len(self.val_iter)
        self.accelerator.wait_for_everyone()
        self.accelerator.print(f"Time: {get_time_dif(self.start_time)}, epoch:{epoch}, loss:{avg_loss:.4f}, eval_metric:{100*eval_metric:.2f}%")
        self.accelerator.print(f"-----------END Evaluate on validation set. epoch {epoch}--------------")

    def test(self):
        pass
    
    def save_predictions(self) -> None:

        pass
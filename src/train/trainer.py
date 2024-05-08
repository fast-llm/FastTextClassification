import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import time
import numpy as np
import torch
from transformers import Trainer
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from extras.loggings import get_logger
from hparams.data_args import DataArguments
from models.component.common import BaseModel
from utils import get_time_dif, read_lines
from .trainer_utils import ClassificationReport, calculate_accuracy, calculate_num_training_steps, create_custom_optimizer, create_custom_scheduler, create_loss_fn


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
                 train_iter: "DataLoader" = None, 
                 val_iter: "DataLoader" = None, 
                 test_iter: "DataLoader" = None):
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
        self.num_classes = self.training_args.num_classes
        self.target_names = read_lines(os.path.join(self.data_args.dataset_dir,
                                                    self.data_args.dataset,
                                                    'data',
                                                    self.data_args.class_file))
        self.best_epoch = 0
        self.best_acc = 0
        self.start_time = time.time()
    
    def init_model(self):
        from torch.nn import utils
        # 参数初始化
        # 梯度裁剪
        utils.clip_grad_norm_(self.model.parameters(), 
                              max_norm=self.training_args.max_grad_norm)
        
        
    def train(self):
        self.optimizer = create_custom_optimizer(self.model, self.training_args)

        self.lr_scheduler = create_custom_scheduler(self.training_args, self.num_training_steps, self.optimizer)
        
        self.model, self.optimizer, self.lr_scheduler, self.train_iter, self.val_iter,self.test_iter  = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler, self.train_iter, self.val_iter, self.test_iter)
        
        for epoch in range(self.epochs):
            self.accelerator.print(f"--------------Training Epoch {epoch}--------------")
            self.train_one_epoch(epoch)
            self.accelerator.print(f"Time: {get_time_dif(self.start_time)}, Validate. epoch {epoch}")
            self.validate(epoch)
            self.accelerator.print(f"---------------------------------------------------")

        self.accelerator.print(f"Time: {get_time_dif(self.start_time)} \n"
                               "Best epoch: {self.best_epoch}, Best eval acc: {self.best_acc}")
        
        
    def train_one_epoch(self,epoch:int):
        with self.accelerator.accumulate(self.model):
            self.model.train()
            avg_loss = 0.0
            avg_accuracy = 0
            total_correct = 0
            total_num = 0
            total_pred = []
            total_labels = []
            for step, (inputs, labels) in enumerate(self.train_iter):
                input_ids, attention_mask, token_type_ids = inputs
                input_ids = input_ids.squeeze()
                attention_mask = attention_mask.squeeze()
                token_type_ids = token_type_ids.squeeze()
                
                
                self.model.zero_grad()
                # accelerator.print(f"input_ids device: {input_ids.device}")
                pred, _ = self.model([input_ids, attention_mask, token_type_ids])
                
                
                # accelerator.print(f"label-1 device: {labels.device}")
                self.optimizer.zero_grad()
                
                loss = self.loss_fn(pred, labels)
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.lr_scheduler.step()
                
                gather_loss = self.accelerator.gather(loss)
                pred, labels = self.accelerator.gather((pred, labels))
                
                
                total_pred.append(pred)
                total_labels.append(labels)
                avg_loss += gather_loss.mean().item()
                
                report = ClassificationReport(pred, 
                                            labels, 
                                            self.target_names,
                                            num_classes=2)
                class_accuracy, correct, total, total_per_class, precision, recall, f1, support = report.calculate_metrics()
                total_correct += correct
                total_num += total

                if (1+step) % self.logging_step == 0:
                    avg_loss /= self.logging_step
                    avg_accuracy = total_correct / total_num
                    self.accelerator.print(f"Time: {get_time_dif(self.start_time)}, epoch:{epoch}, step:{step}, loss {avg_loss}, avg_accuracy: {100*avg_accuracy:.2f}%")
                    avg_loss = 0.0
                    total_correct = 0
                    total_num = 0
        
        report = ClassificationReport(torch.cat(total_pred, dim=0), 
                                      torch.cat(total_labels, dim=0), 
                                      self.target_names,
                                      num_classes=self.num_classes,
                                      multi_label=self.multi_label)
        self.accelerator.print(f"----------- Training Metrics for epoch: {epoch}--------------")
        self.accelerator.print(report.metrics())
        # torch_gc()
        

        
    def validate(self, epoch:int):
        with torch.inference_mode():
            self.model.eval()
            avg_loss = 0.0
            avg_accuracy = 0
            total_correct = 0
            total_num = 0
            total_pred = []
            total_labels = []
            for step, (inputs, labels) in enumerate(self.val_iter):
                input_ids, attention_mask, token_type_ids = inputs
                input_ids = input_ids.squeeze()
                attention_mask = attention_mask.squeeze()
                token_type_ids = token_type_ids.squeeze()
                
                
                # accelerator.print(f"input_ids device: {input_ids.device}")
                pred, _ = self.model([input_ids, attention_mask, token_type_ids])
                
                loss = self.loss_fn(pred, labels)
                gather_loss = self.accelerator.gather(loss)
                pred, labels = self.accelerator.gather((pred, labels))
                
                total_pred.append(pred)
                total_labels.append(labels)
                
                avg_loss += gather_loss.mean().item()
                report = ClassificationReport(pred, 
                                            labels, 
                                            self.target_names,
                                            num_classes=2)
                class_accuracy, correct, total, total_per_class, precision, recall, f1, support = report.calculate_metrics()
                total_correct += correct
                total_num += total
        avg_loss /= step
        avg_accuracy = total_correct / total_num
        # print
        avg_loss /= len(self.val_iter)
        
        if avg_accuracy > self.best_acc:
            self.best_acc = avg_accuracy
            self.best_epoch = epoch
        
        self.accelerator.wait_for_everyone()
        report = ClassificationReport(torch.cat(total_pred, dim=0), 
                                      torch.cat(total_labels, dim=0), 
                                      target_names=self.target_names,
                                      num_classes=self.num_classes,
                                      multi_label=self.multi_label)
        self.accelerator.print(f"----------- Validation Metrics for epoch: {epoch}--------------")
        self.accelerator.print(report.metrics())
        self.accelerator.print(f"Time: {get_time_dif(self.start_time)}, epoch:{epoch}, loss: {avg_loss}, eval_metric:{100*avg_accuracy:.2f}%")

    def test(self):
        pass
    
    def save_ckp(self):
        
        pass
    
    
    def load_ckp(self):
        
        
        pass
    
    def save_predictions(self,epoch:int) -> None:
        
        pass
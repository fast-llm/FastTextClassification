from datetime import timedelta
import json
import os
import shutil
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import time
import numpy as np
import torch
from transformers import GenerationConfig, Trainer, TrainerControl, TrainerState
from transformers.tokenization_utils import PreTrainedTokenizer
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
import queue
from threading import Thread
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from extras.constants import BEST_MODEL_PATH, DATA_ARGS_NAME, LOG_FILE_NAME, MODEL_ARGS_NAME, SAVE_CONFIG_NAME, SAVE_MODEL_NAME, TRAINING_ARGS_NAME
from extras.loggings import get_logger
from extras.misc import AverageMeter
from extras.ploting import plot_data
from hparams.data_args import DataArguments
from models.component.common import BaseModel
from utils import get_time_dif, read_lines
from .trainer_utils import ClassificationReport, EarlyStopping, LogState, ModelEntry, ModelManager, calculate_accuracy, calculate_num_training_steps, create_custom_optimizer, create_custom_scheduler, create_loss_fn


if TYPE_CHECKING:
    from transformers.trainer import PredictionOutput
    from ..hparams.training_args import TrainingArguments
    from ..hparams.model_args import ModelArguments

logger = get_logger(__name__)


class CustomClassifyTrainer(object):

    def __init__(self, 
                 model: BaseModel  = None,
                 tokenizer: "PreTrainedTokenizer" = None,
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
        
        self.streaming = self.data_args.streaming
        
        self.early_stopper = EarlyStopping(patience=self.training_args.patience,
                                           verbose=self.training_args.verbose, 
                                           delta=self.training_args.delta,
                                           )
        self.tokenizer = tokenizer
        self.init_para()
        self.init_model()
        
        self.loss_meter = AverageMeter()
        self.acc_meter = AverageMeter()
        self.log_history = []
        self.elapsed_time = 0
        self.remaining_time = 0
        
        self.data_queue = queue.Queue()
        # 创建并启动绘图线程
        self.plot_thread = Thread(target=plot_data, args=(self.data_queue,))
        self.plot_thread.start()

        self.model_manager = ModelManager(self.training_args.num_best_models)
        if self.training_args.tensor_board:
            self.writer = SummaryWriter(self.training_args.tensorboard_dir)
        
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
        self.eval_steps = self.training_args.eval_steps
        self.cur_step = 0
        self.best_epoch = 0
        self.best_acc = 0
        self.start_time = time.time()
    
    def init_model(self):
        from torch.nn import utils
        # 参数初始化
        # 梯度裁剪
        if self.training_args.resume:
            if self.training_args.resume_file:
                self.load_model(self.training_args.resume_file)
            else:
                output_dir = os.path.join(self.training_args.output_dir, BEST_MODEL_PATH)
                self.load_model(output_dir)
        
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
            # If early stopping is triggered, break out of the training loop
            if self.early_stopper.early_stop:
                logger.info("Early stopping triggered.")
        self.accelerator.print(f"Time: {get_time_dif(self.start_time)} \n"
                               f"Best epoch: {self.best_epoch}, Best eval acc: {self.best_acc}")
        self.data_queue.put(None) # 发送结束信号
        self.plot_thread.join() # 等待绘图线程结束
        if self.training_args.tensor_board:
            self.writer.close()
        
    def train_one_epoch(self,epoch:int):
        with self.accelerator.accumulate(self.model):
            self.model.train()
            avg_loss = 0.0
            avg_accuracy = 0
            total_correct = 0
            total_num = 0
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()
            total_pred = []
            total_labels = []
            for step, batch in enumerate(self.train_iter):
                if self.streaming:
                    input_ids,attention_mask,token_type_ids,labels = batch
                else:
                    inputs, labels = batch
                    input_ids, attention_mask, token_type_ids = inputs
                    input_ids = input_ids.squeeze()
                    attention_mask = attention_mask.squeeze()
                    token_type_ids = token_type_ids.squeeze()
                
                
                self.model.zero_grad()
                # accelerator.print(f"input_ids device: {input_ids.device}")
                logits, _, _ = self.model([input_ids, attention_mask, token_type_ids])
                
                
                # accelerator.print(f"label-1 device: {labels.device}")
                self.optimizer.zero_grad()
                
                loss = self.loss_fn(logits, labels)
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.lr_scheduler.step()
                
                gather_loss = self.accelerator.gather(loss)
                logits, labels = self.accelerator.gather((logits, labels))
                
                
                total_pred.append(logits)
                total_labels.append(labels)
                loss_meter.update(gather_loss.mean().item())
                
                report = ClassificationReport(logits, 
                                            labels, 
                                            self.target_names,
                                            num_classes=self.num_classes)
                class_accuracy, correct, total, precision, recall, f1, correct_per_class, support = report.calculate_metrics()
                acc_meter.update(correct/total, total)
                
                # 更新tensorboard
                if self.training_args.tensor_board:
                    self.writer.add_scalar('Training/Loss', gather_loss, self.cur_step)
                    self.writer.add_scalar('Training/Accuracy', correct / total, self.cur_step)
                
                if (1+self.cur_step) % self.training_args.save_steps == 0:
                    model_path = f"checkpoint-step-{self.cur_step}"
                    output_dir = os.path.join(self.training_args.output_dir, model_path)
                    self.save_model(output_dir,avg_accuracy=acc_meter.avg,avg_loss=loss_meter.avg)
                    
                if (1+step) % self.logging_step == 0:
                    avg_loss = loss_meter.avg
                    avg_accuracy = acc_meter.avg
                    self.accelerator.print(f"epoch:{epoch+1}/{self.epochs}, step:{self.cur_step}, loss {avg_loss:.4f}, avg_accuracy: {100*avg_accuracy:.2f}%, "
                                           f"lr: {self.lr_scheduler.get_last_lr()[-1]:.4e}, "
                                           f"elapsed/remain time: {get_time_dif(self.start_time)}/{self.remaining_time}"
                                           )
                    # 主线程 保存训练日志
                    if self.accelerator.is_main_process:
                        self.log_history.append(LogState(
                            epoch=epoch,
                            step=self.cur_step,
                            learning_rate=self.lr_scheduler.get_last_lr()[0],
                            loss=avg_loss,
                            accuracy=avg_accuracy,
                            grad_norm=None,
                            eval_loss=None,
                            eval_accuracy=None,
                        ).to_dict())
                        self.update_logs()

                    loss_meter.reset()
                    acc_meter.reset()
                
                # 进行评估
                if (1+self.cur_step) % self.eval_steps == 0:
                    self.validate(epoch)
                # 迭代全局step
                self.cur_step+=1
        
        report = ClassificationReport(torch.cat(total_pred, dim=0), 
                                      torch.cat(total_labels, dim=0), 
                                      self.target_names,
                                      num_classes=self.num_classes,
                                      multi_label=self.multi_label)
        class_accuracy, correct, total, precision, recall, f1, correct_per_class, support = report.calculate_metrics()
        total_correct += correct
        total_num += total
        avg_loss = loss_meter.avg
        avg_accuracy = total_correct / total_num
        self.accelerator.print(f"----------- Training Metrics for epoch: {epoch+1}--------------")
        self.accelerator.print(report.metrics())
        self.accelerator.print(f"Time: {get_time_dif(self.start_time)}, epoch:{epoch+1}/{self.epochs}, loss: {avg_loss:.4f}, eval_metric:{100*avg_accuracy:.2f}%")
        # torch_gc()
    

    
    def update_logs(self):
        if self.accelerator.is_main_process:
            if len(self.log_history)>0:
                self.elapsed_time = get_time_dif(self.start_time)
                avg_step_time = self.elapsed_time.total_seconds() / self.cur_step if self.cur_step != 0 else 0
                estimated_total_time = avg_step_time * self.num_training_steps
                remaining_seconds = max(0, estimated_total_time - self.elapsed_time.total_seconds())
                self.remaining_time = timedelta(seconds=int(round(remaining_seconds)))
                
                logs = self.log_history[-1]
                logs['total_steps'] = self.num_training_steps
                logs["elapsed_time"] = str(self.elapsed_time)
                logs["remaining_time"] = str(self.remaining_time)
                logs['percentage'] = round(self.cur_step / self.num_training_steps * 100, 2) if self.num_training_steps != 0 else 100
                
                # 增量写入日志到文件
                log_file_path = os.path.join(self.training_args.output_dir, LOG_FILE_NAME)
                with open(log_file_path, "a") as writer:  # 使用 'a' 模式来追加日志
                    writer.write(json.dumps(logs) + "\n")
                # 将最新的日志写入到文件
                self.data_queue.put(dict(step=self.cur_step, 
                                         output_dir=self.training_args.output_dir
                                         ))
        
    def validate(self, epoch:int):
        with torch.inference_mode():
            self.model.eval()
            avg_accuracy = 0
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()
            total_pred = []
            total_labels = []
            for step,batch in enumerate(self.val_iter):
                if self.streaming:
                    input_ids,attention_mask,token_type_ids,labels = batch
                else:
                    inputs, labels = batch
                    input_ids, attention_mask, token_type_ids = inputs
                    input_ids = input_ids.squeeze()
                    attention_mask = attention_mask.squeeze()
                    token_type_ids = token_type_ids.squeeze()

                
                # accelerator.print(f"input_ids device: {input_ids.device}")
                logits, _, _ = self.model([input_ids, attention_mask, token_type_ids])
                
                loss = self.loss_fn(logits, labels)
                gather_loss = self.accelerator.gather(loss)
                logits, labels = self.accelerator.gather((logits, labels))
                
                total_pred.append(logits)
                total_labels.append(labels)
                
                loss_meter.update(gather_loss.mean().item())
                
                report = ClassificationReport(logits, 
                                            labels, 
                                            self.target_names,
                                            num_classes=self.num_classes)
                class_accuracy, correct, total, precision, recall, f1, correct_per_class, support = report.calculate_metrics()
                acc_meter.update(correct/total, total)

        avg_accuracy = acc_meter.avg
        avg_loss = loss_meter.avg
        
        if avg_accuracy > self.best_acc:
            self.best_acc = avg_accuracy
            self.best_epoch = epoch
        
        self.accelerator.wait_for_everyone()
        report = ClassificationReport(torch.cat(total_pred, dim=0), 
                                      torch.cat(total_labels, dim=0), 
                                      target_names=self.target_names,
                                      num_classes=self.num_classes,
                                      multi_label=self.multi_label)
        self.accelerator.print(f"----------- Validation Metrics for epoch: {epoch+1}--------------")
        self.accelerator.print(report.metrics())
        
        self.accelerator.print(f"Time: {get_time_dif(self.start_time)}, epoch:{epoch+1}/{self.epochs}, loss: {avg_loss:.4f}, eval_metric:{100*avg_accuracy:.2f}%")
        
        # 更新tensorboard
        if self.training_args.tensor_board:
            self.writer.add_scalar('Validation/Loss', avg_loss, self.cur_step)
            self.writer.add_scalar('Validation/Accuracy', avg_accuracy, self.cur_step)
        
        # Check if we should early stop and potentially save the model
        model_path = f"checkpoint-epoch-{epoch}"
        output_dir = os.path.join(self.training_args.output_dir, model_path)
        self.save_model(output_dir,avg_accuracy=avg_accuracy,avg_loss=avg_loss)
        if self.early_stopper(avg_accuracy):
            output_dir = os.path.join(self.training_args.output_dir, BEST_MODEL_PATH)
            self.save_model(output_dir,avg_accuracy=avg_accuracy,avg_loss=avg_loss)
        # 主线程 保存训练日志
        if self.accelerator.is_main_process:
            self.log_history.append(LogState(
                epoch=epoch,
                step=self.cur_step,
                learning_rate=self.lr_scheduler.get_last_lr()[0],
                loss=None,
                accuracy=None,
                grad_norm=None,
                eval_loss=avg_loss,
                eval_accuracy=avg_accuracy,
            ).to_dict())
            self.update_logs()

    def save_metrics(self):
            
            pass
        
    def save_state(self):
        
        pass
    
    def test(self):
        pass
        
    
    def save_model(self, output_dir: Optional[str] = None,
                   avg_accuracy: Optional[float] = None,
                    avg_loss: Optional[float] = None,
                   ):
        """
        Will save the model, so you can reload it using `from_pretrained()`.
        Will only save from the main process.
        """
        if self.accelerator.is_main_process:
            if output_dir is None:
                output_dir = self.training_args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            model_to_save = self.model.module if hasattr(self.model, "module") else self.model
            model_path = os.path.join(output_dir, SAVE_MODEL_NAME)
            config_path = os.path.join(output_dir, SAVE_CONFIG_NAME)
            tokenizer_path = output_dir
            
            # 保存模型状态字典
            torch.save(model_to_save.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")

            # 保存基础模型的配置文件
            model_to_save.bert.config.to_json_file(config_path)
            logger.info(f"Config saved to {config_path}")
            # model_to_save.config.to_json_file(config_path)
            # logger.info(f"Config saved to {config_path}")

            # 保存tokenizer
            self.tokenizer.save_pretrained(tokenizer_path)
            logger.info(f"Tokenizer saved to {tokenizer_path}")

            # 可选：保存训练参数
            training_args_path = os.path.join(output_dir, TRAINING_ARGS_NAME)
            torch.save(self.training_args, training_args_path)
            logger.info(f"Training arguments saved to {training_args_path}")
            
            # 可选：保存训练参数
            data_args_path = os.path.join(output_dir, DATA_ARGS_NAME)
            torch.save(self.data_args, data_args_path)
            logger.info(f"Data arguments saved to {data_args_path}")
            
            # 可选：保存训练参数
            model_args_path = os.path.join(output_dir, MODEL_ARGS_NAME)
            torch.save(self.model_args, model_args_path)
            logger.info(f"Model arguments saved to {model_args_path}")
            
            self.model_manager.update_best_models(ModelEntry(avg_accuracy, avg_loss, output_dir))
            

        self.accelerator.wait_for_everyone()


    
    def load_model(self, output_dir: Optional[str] = None):
        """
        加载以前保存的模型、配置和tokenizer。
        参数:
            output_dir (str): 模型和相关文件保存的目录。
        返回:
            model: 加载的模型对象。
        """
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        # 加载配置文件
        config_path = os.path.join(output_dir, SAVE_CONFIG_NAME)
        config = AutoConfig.from_json_file(config_path)

        # 加载模型状态字典
        model_path = os.path.join(output_dir, SAVE_MODEL_NAME)
        model_state_dict = torch.load(model_path, map_location=self.accelerator.device)

        # 加载tokenizer
        tokenizer_path = output_dir
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # 使用配置文件和状态字典重新创建模型
        model = AutoModel.from_config(config)
        model.load_state_dict(model_state_dict)

        return model, tokenizer
    
    def save_predictions(self,epoch:int) -> None:
        
        pass
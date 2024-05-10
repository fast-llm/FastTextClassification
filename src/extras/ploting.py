import json
import math
import os
from typing import List

from extras.constants import LOG_FILE_NAME, TRAINING_PNG_NAME
from utils import read_jsonl

from .loggings import get_logger
from .packages import is_matplotlib_available


if is_matplotlib_available():
    import matplotlib.pyplot as plt


logger = get_logger(__name__)


def smooth(scalars: List[float]) -> List[float]:
    r"""
    EMA implementation according to TensorBoard.
    """
    last = scalars[0]
    smoothed = list()
    weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5)  # a sigmoid function
    for next_val in scalars:
        smoothed_val = last * weight + (1 - weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_loss(output_dir:str,
              train_steps:list, train_losses:list, train_acc:list,
              eval_steps:list, eval_losses:list, eval_acc:list):
    # 创建图表
    fig, ax1 = plt.subplots()
   # 绘制损失曲线
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.plot(train_steps, train_losses, label='Training Loss', color='darkviolet')
    ax1.plot(eval_steps, eval_losses, label='Validation Loss', color='violet', linestyle='--')
    
    # 创建双y轴
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy')
    ax2.plot(train_steps, train_acc, label='Training Accuracy', color='deepskyblue')
    ax2.plot(eval_steps, eval_acc, label='Validation Accuracy', color='limegreen', linestyle='--')

    # 合并图例
    lns = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lns]
    ax1.legend(lns, labels, loc='best')

    # 显示图表
    plt.title('Training and Validation Metrics')
    figure_path = os.path.join(output_dir,TRAINING_PNG_NAME)
    plt.savefig(figure_path,format="png", dpi=100)
    plt.ioff()
    plt.show()
    plt.close(fig)
    logger.info(f"Figure saved at:{figure_path}")

def sort_data_by_step(steps:list, losses:list, accuracy:list):
    # 使用 zip 函数将步骤、损失和准确率打包在一起，然后进行排序
    sorted_data = sorted(zip(steps, losses, accuracy), key=lambda x: x[0])
    # 解压排序后的数据
    sorted_steps, sorted_losses, sorted_accuracy = zip(*sorted_data)
    return list(sorted_steps), list(sorted_losses), list(sorted_accuracy)

def plot_data(queue):
    try:
        data_point = queue.get()  # 从队列中获取数据
        step, output_dir = data_point.get("step"), data_point.get("output_dir")
        log_file = os.path.join(output_dir, LOG_FILE_NAME)
        if os.path.exists(log_file):
            data = read_jsonl(log_file)
            train_steps, train_losses, eval_steps, eval_losses = [], [], [], []
            train_acc, eval_acc = [], []
            for item in data:
                if item['accuracy']:
                    # 对训练数据和验证数据进行排序
                    train_steps.append(item['step'])
                    train_losses.append(item['loss'])
                    train_acc.append(item['accuracy'])
                if item['eval_loss']:
                    eval_steps.append(item['step'])
                    eval_losses.append(item['eval_loss'])
                    eval_acc.append(item['eval_accuracy'])
            if len(train_losses)>0:
                train_steps, train_losses, train_acc = sort_data_by_step(train_steps, train_losses, train_acc)
            if len(eval_losses)>0:
                eval_steps, eval_losses, eval_acc = sort_data_by_step(eval_steps, eval_losses, eval_acc)
            plot_loss(output_dir, 
                    train_steps, train_losses, train_acc, 
                    eval_steps, eval_losses, eval_acc)
    except Exception as e:
        logger.error(f"Error in plot_data: {e}")
        raise e
        
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import csv



train_file = "train.txt"    # 
val_file = "val.txt"        # 
test_file = "test.txt"      # 
SEP = "\t"

train_ratio = 0.8
val_ratio =  0.1
test_ratio = 0.1
seed = 42

label_col = 'label'
text_col = 'review'

# 读取CSV文件
data_path = "/platform_tech/xiongrongkang/Dataset/Classification/ChnSentiCorp_htl_all/input/ChnSentiCorp_htl_all.csv"
save_path = "/platform_tech/xiongrongkang/Dataset/Classification/ChnSentiCorp_htl_all/data"

df = pd.read_csv(data_path)

# 创建格式化后的列，确保所有字段都非空
df[text_col] = df[text_col].fillna('')  # 假设空文本字段用空字符串填充
df[label_col] = df[label_col].fillna('unknown')  # 假设空标签用'unknown'填充

# 假设CSV文件中的列名是 text_col 和 label_col
df['formatted'] = df[text_col] + SEP + df[label_col].astype(str)

# 切分数据集，假设训练集60%，验证集20%，测试集20%
train, temp = train_test_split(df, test_size=train_ratio, random_state=42)  # 先分出60%的训练数据
val, test = train_test_split(temp, test_size=(test_ratio)/(val_ratio+test_ratio), random_state=seed)  # 将剩下的40%数据平分为验证和测试数据

# 保存为文本文件
train['formatted'].to_csv(os.path.join(save_path, train_file), index=False, header=False, quoting=csv.QUOTE_NONE, escapechar='\\')
val['formatted'].to_csv(os.path.join(save_path, val_file), index=False, header=False, quoting=csv.QUOTE_NONE, escapechar='\\')
test['formatted'].to_csv(os.path.join(save_path, test_file), index=False, header=False, quoting=csv.QUOTE_NONE, escapechar='\\')

import os
import string
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import argparse

def main(data_path:str, save_path:str, train_file:str, val_file:str, test_file:str, 
         split_ratio:str="8 1 1", 
         num_class:int=2,
         seed:int=42, sep='\t', 
         label_col:str='label', 
         text_col:str='review'):
     train_ratio, val_ratio, test_ratio = map(int, split_ratio.split(' '))
     total_ratio = train_ratio + val_ratio + test_ratio
    
     df = pd.read_csv(data_path)

     # 创建格式化后的列，确保所有字段都非空
     df[text_col] = df[text_col].fillna('')  # 假设空文本字段用空字符串填充
     df[label_col] = df[label_col].fillna('unknown')  # 假设空标签用'unknown'填充

     # 假设CSV文件中的列名是 text_col 和 label_col
     df['formatted'] = df[text_col] + sep + df[label_col].astype(str)

     # 切分数据集，假设训练集60%，验证集20%，测试集20%
     train, temp = train_test_split(df, test_size=train_ratio/total_ratio, random_state=seed)  # 先分出60%的训练数据
     val, test = train_test_split(temp, test_size=(val_ratio)/(val_ratio+test_ratio), random_state=seed)  # 将剩下的数据分为验证和测试数据
     
     class_labels = list(string.ascii_uppercase)  # A-Z 字母列表
     data_class = pd.DataFrame({'class': range(num_class)})
     data_class['class'] = data_class['class'].apply(lambda x: class_labels[x])

     # 保存为文本文件
     train['formatted'].to_csv(os.path.join(save_path, train_file), index=False, header=False, quoting=csv.QUOTE_NONE, escapechar='\\')
     val['formatted'].to_csv(os.path.join(save_path, val_file), index=False, header=False, quoting=csv.QUOTE_NONE, escapechar='\\')
     test['formatted'].to_csv(os.path.join(save_path, test_file), index=False, header=False, quoting=csv.QUOTE_NONE, escapechar='\\')
     data_class.to_csv(os.path.join(save_path, 'class.txt'), index=False, header=False, quoting=csv.QUOTE_NONE, escapechar='\\')
     
     
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the input CSV file")
    parser.add_argument("--save_path", type=str, help="Path to save the output files")
    parser.add_argument("--num_class", type=int, default=2, help="number class of the dataset")
    parser.add_argument("--train_file", type=str,default = 'train.txt', help="Name of the training file")
    parser.add_argument("--val_file", type=str, default='val.txt', help="Name of the devidation file")
    parser.add_argument("--test_file", type=str, default='test.txt', help="Name of the testing file")
    parser.add_argument("--split_ratio", type=str, default="8 1 1", 
                        help="Ratio of training validation testing data, also '0.9 0.1 0.1' is ok")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sep", type=str, default="\t", help="Separator for the formatted data")
    parser.add_argument("--label_col", type=str, default="label", help="Name of the label column")
    parser.add_argument("--text_col", type=str, default="review", help="Name of the text column")
    args = parser.parse_args()

    main(args.data_path, args.save_path, 
         args.train_file, args.val_file, args.test_file, 
         args.split_ratio, 
         args.num_class,
         args.seed, args.sep, 
         args.label_col, 
         args.text_col)
    # usage
    # python ./src/data/convert_csv_data.py --data_path "/path/to/data.csv" --save_path "/path/to/save" --split_ratio "8 1 1" --label_col label --text_col review
import os
import string
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import argparse
from tqdm import tqdm
from utils import export_json, read_data, read_json, read_lines, export_jsonl



       
def main(data_path:str,
         sheet_name:str, 
         save_path:str, 
         train_file:str, val_file:str, test_file:str, 
         split_ratio:str="8 1 1", 
         num_class:int=2,
         seed:int=42, sep='\t', 
         label_col:str='label', 
         text_col:str='review'):
     train_ratio, val_ratio, test_ratio = map(float, split_ratio.split(' '))
     total_ratio = train_ratio + val_ratio + test_ratio
     
     data = read_data(data_path=data_path,
                    sep=sep,
                    sheet_name=sheet_name,
                    label_col=label_col,
                    text_col=text_col)
     

     train, temp = train_test_split(data, test_size=(1 - train_ratio / total_ratio), random_state=seed)
     val, test = train_test_split(temp, test_size=(test_ratio) / (val_ratio + test_ratio), random_state=seed)

     export_jsonl(train, os.path.join(save_path, train_file))
     export_jsonl(val, os.path.join(save_path, val_file))
     export_jsonl(test, os.path.join(save_path, test_file))

     # Optionally save class labels
     class_labels = [chr(65 + i) for i in range(num_class)]  # A-Z letters
     class_path = os.path.join(save_path, 'class.txt')
     with open(class_path, 'w', encoding='utf-8') as file:
          for label in class_labels:
               file.write(label + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the input CSV file or xlsx file or txt path")
    parser.add_argument("--sheet_name", type=str, default='Sheet1', help="Sheet name of the xlsx file")
    parser.add_argument("--save_path", type=str, help="Path to save the output files")
    parser.add_argument("--num_class", type=int, default=2, help="number class of the dataset")
    parser.add_argument("--train_file", type=str,default = 'train.json', help="Name of the training file")
    parser.add_argument("--val_file", type=str, default='val.json', help="Name of the validation file")
    parser.add_argument("--test_file", type=str, default='test.json', help="Name of the testing file")
    parser.add_argument("--split_ratio", type=str, default="8 1 1", 
                        help="Ratio of training validation testing data, also '0.9 0.1 0.1' is ok")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sep", type=str, default='\t', help="Separator for the formatted data")
    parser.add_argument("--label_col", type=str, default="label", help="Name of the label column")
    parser.add_argument("--text_col", type=str, default="review", help="Name of the text column")
    args = parser.parse_args()

    main(
        data_path=args.data_path, 
        sheet_name= args.sheet_name,
         save_path=args.save_path, 
         train_file=args.train_file, 
         val_file=args.val_file,
         test_file=args.test_file, 
         split_ratio=args.split_ratio, 
         num_class=args.num_class,
         seed=args.seed, sep=args.sep, 
         label_col=args.label_col, 
         text_col=args.text_col
         )
    # usage
    # python ./src/data/convert_raw_data.py --data_path "/path/to/data.csv" --save_path "/path/to/save" --split_ratio "8 1 1" --label_col label --text_col review
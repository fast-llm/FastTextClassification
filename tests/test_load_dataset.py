import os
from transformers import AutoModel, BertTokenizer, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

from data.transformer_loader import TransformerDataset, load_dataset
from data.utils_multi import get_multi_list_label
from extras.constants import ROOT_PATH

# model_path = '/platform_tech/models/ernie-3.0-base-zh/'
model_path = '/platform_tech/models/bert-base-chinese/'


train_path = os.path.join(ROOT_PATH, 'data/ChnSentiCorp_htl_all/data/train.txt' )

pad_size = 768
SEP = '\t'
max_samples = 100
num_classes = 2
multi_class = True
multi_label = False

tokenizer = AutoTokenizer.from_pretrained(model_path,
                                            truncation=True,  
                                            return_tensors="pt", 
                                            padding='max_length', 
                                            max_length=pad_size)

train_data = load_dataset(train_path, tag='training',
                        tokenizer=tokenizer,
                        SEP = SEP,
                        pad_size = pad_size,
                        max_samples = max_samples,
                        num_classes = num_classes,
                        multi_class = multi_class,
                        multi_label = multi_label,
                         )

train_dataset = TransformerDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=16, 
                          num_workers=2,
                          shuffle=True, 
                          drop_last=False)

print(train_data)


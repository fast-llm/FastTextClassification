from transformers import AutoTokenizer

class DatasetConfig:
    model_path: str
    batch_size: int
    multi_class: bool
    multi_label: bool
    num_classes: int
    learning_rate: float
    train_path:str = ''
    val_path:str = ''
    test_path:str = ''
    pad_size: int
    max_samples: int
    SEP: str
    streaming: bool
    def __init__(self,
                 max_samples=None,
                 batch_size=32,
                 multi_class=False, 
                 multi_label=False,
                 num_classes=2, 
                 learning_rate=1e-3,
                 train_path='', 
                 val_path='', 
                 test_path='',
                 per_device_train_batch_size: int = 8,
                 per_device_eval_batch_size: int = 8,
                 pad_size=128, 
                 model_path=None,
                 SEP='',
                 streaming = False
                 ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)  
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.multi_class = multi_class
        self.multi_label = multi_label
        self.num_classes = num_classes
        self.train_path = train_path
        self.learning_rate = learning_rate
        self.val_path = val_path
        self.test_path = test_path
        self.pad_size = pad_size
        self.SEP = SEP
        self.streaming = streaming
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.num_workers=1
        self.shuffle=True
        self.drop_last=False
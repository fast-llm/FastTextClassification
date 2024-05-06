
class DatasetConfig:
    batch_size: int
    num_gpus: int
    device: str
    multi_label: bool
    num_classes: int
    learning_rate: float
    train_path:str = ''
    val_path:str = ''
    test_path:str = ''
    pad_size: int
    tokenizer: int
    max_samples: int
    SEP: str
    streaming: bool
    def __init__(self,
                 max_samples=None,
                 batch_size=32, 
                 multi_label=False,
                 num_classes=2, 
                 learning_rate=1e-3,
                 train_path='', 
                 val_path='', 
                 test_path='',
                 pad_size=32, 
                 tokenizer=None,
                 SEP='',
                 streaming = False
                 ):
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.multi_label = multi_label
        self.num_classes = num_classes
        self.train_path = train_path
        self.learning_rate = learning_rate
        self.val_path = val_path
        self.test_path = test_path
        self.pad_size = pad_size
        self.tokenizer = tokenizer
        self.SEP = SEP
        self.streaming = streaming
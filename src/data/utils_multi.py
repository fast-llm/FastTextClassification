import torch

"""
数据二分类的时候有两种情况
0,1标签
类别0 一条数据表示成 [0] 或者 [1,0]

多分类的时候
0,1,2,3标签
类别2 一条数据表示成 one-hot [0,0,1,0]

多标签的时候
0,3 数据表示成 multi-hot  [1,0,0,1]

"""

def get_multi_list_label(label: str, 
                         multi_class: bool = False,
                         multi_label: bool = False) -> int | list[int]:
    """
    Convert a label string into an integer or a list of integers based on the `multi_label` flag.
    
    Args:
        label (str): The label string, which can be a single integer or multiple integers separated by commas.
        multi_label (bool, optional): Flag to determine the output format. If `True`, output will be a list of integers.
                                     If `False`, output will be a single integer. Defaults to False.

    Returns:
        int | list[int]: Depending on the `multi_label` flag, returns either a single integer or a list of integers.
    
    Raises:
        ValueError: If `multi_label` is False and the input label string contains commas, indicating multiple labels
                    which is inappropriate for a single label scenario.
                    
    Examples:
        - input: ("0", False) -> output: 0
        - input: ("0", True) -> output: [0]
        - input: ("0,3", False) -> raises ValueError("Comma found in label string but expected single label.")
        - input: ("0,3", True) -> output: [0, 3]
    """
    if multi_class and multi_label:
        raise ValueError("Only one of `multi_class` or `multi_label` can be set to True.")
    elif multi_class:
        if ',' in label:
            raise ValueError("Comma found in label string but expected single label")
        return [int(label)]
    elif multi_label:
        return list(map(int, label.split(',')))
    else:
        # 二分类的时候非0都会转换为1
        if ',' in label:
            raise ValueError("Comma found in label string but expected single label.")
        label = int(label)
        if label>1:
            return [1]
        return [int(label)]


def get_multi_hot_label(doc_labels:list[list[int]], 
                        num_class:int, 
                        dtype=torch.long)->torch.Tensor:
    """For multi-label classification
    input: 
    
    return:
    多分类的时候 0,1代表 有0,1类别，而不是第0,1类别
    多标签的时候 0代表没有,1代表有
    Generate multi-hot for input labels
    e.g. 二分类的时候
        input: [[0],[1],[0]]
        output: ""
    
    e.g.多标签 
        input: [[0,1], [2]]
        output: [[1,1,0], [0,0,1]]
    e.g.多分类
        input: [[0,1], [2]]
        output: [[1,1,0], [0,0,1]]
    """
    batch_size = len(doc_labels)
    max_label_num = max([len(x) for x in doc_labels])
    max_label = max([max(x) for x in doc_labels])
    if max_label_num>num_class:
        raise ValueError(f"label max length {max_label_num} is out of bounds for num_class {num_class}")
    if max_label >= num_class:
        raise ValueError(f"max label {max_label} is out of bounds for num_class {num_class}")
    doc_labels_extend = \
        [[doc_labels[i][0] for x in range(max_label_num)] for i in range(batch_size)]
    for i in range(0, batch_size):
        doc_labels_extend[i][0: len(doc_labels[i])] = doc_labels[i]
    y = torch.Tensor(doc_labels_extend).long()
    y_one_hot = torch.zeros(batch_size, num_class, dtype=dtype).scatter_(1, y, 1)
    return y_one_hot

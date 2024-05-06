import torch
from data.utils_multi import get_multi_hot_label, get_multi_list_label



def test_single_label():
    print("test_single_label")
    label = '0'
    label = get_multi_list_label(label=label,
                                 multi_class=False,
                                 multi_label=False)
    print(label)
    
    label = '1'
    label = get_multi_list_label(label=label,
                                 multi_class=False,
                                 multi_label=False)
    print(label)
    
    label = '2'
    label = get_multi_list_label(label=label,
                                 multi_class=False,
                                 multi_label=False)
    print(label)

def test_single_label_one_hot():
    print("test_single_label_one_hot")
    label = '0'
    label = get_multi_list_label(label=label,
                                 multi_class=False,
                                 multi_label=False)
    label = get_multi_hot_label([label],num_class=2)
    print(label)
    
    label = '1'
    label = get_multi_list_label(label=label,
                                 multi_class=False,
                                 multi_label=False)
    label = get_multi_hot_label([label],num_class=2)
    print(label)
    
    label = '2'
    label = get_multi_list_label(label=label,
                                 multi_class=False,
                                 multi_label=False)
    label = get_multi_hot_label([label],num_class=2)
    print(label)


def test_multi_class():
    print("test_multi_class")
    label = '0'
    label = get_multi_list_label(label=label,
                                 multi_class=True,
                                 multi_label=False)
    print(label)
    
    label = '1'
    label = get_multi_list_label(label=label,
                                 multi_class=True,
                                 multi_label=False)
    print(label)
    

    label = '2'
    label = get_multi_list_label(label=label,
                                 multi_class=True,
                                 multi_label=False)
    print(label)

def test_multi_class_one_hot():
    print("test_multi_class_one_hot")
    label = '0'
    label = get_multi_list_label(label=label,
                                 multi_class=True,
                                 multi_label=False)
    label = get_multi_hot_label([label],num_class=2)
    print(label)
    
    label = '1'
    label = get_multi_list_label(label=label,
                                 multi_class=True,
                                 multi_label=False)
    label = get_multi_hot_label([label],num_class=2)
    print(label)
    

    label = '2'
    label = get_multi_list_label(label=label,
                                 multi_class=True,
                                 multi_label=False)
    label = get_multi_hot_label([label],num_class=3)
    print(label)



def test_multi_label():
    print("test_multi_label")
    label = '0'
    label = get_multi_list_label(label=label,
                                 multi_class=False,
                                 multi_label=True)
    print(label)
    
    label = '1'
    label = get_multi_list_label(label=label,
                                 multi_class=False,
                                 multi_label=True)
    print(label)
    
    label = '2,3'
    label = get_multi_list_label(label=label,
                                 multi_class=False,
                                 multi_label=True)
    print(label)

def test_multi_label_one_hot():
    print("test_multi_label_one_hot")
    label = '0'
    label = get_multi_list_label(label=label,
                                 multi_class=False,
                                 multi_label=True)
    label = get_multi_hot_label([label],num_class=2)
    print(label)
    
    label = '1'
    label = get_multi_list_label(label=label,
                                 multi_class=False,
                                 multi_label=True)
    label = get_multi_hot_label([label],num_class=2)
    print(label)
    

    label = '2,3'
    label = get_multi_list_label(label=label,
                                 multi_class=False,
                                 multi_label=True)
    label = get_multi_hot_label([label],num_class=4)
    print(label)


if __name__ == "__main__":
    test_single_label()
    test_single_label_one_hot()
    test_multi_class()
    test_multi_class_one_hot()
    test_multi_label()
    test_multi_label_one_hot()
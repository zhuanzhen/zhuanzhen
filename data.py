import torch
import os
from torch.utils.data import Dataset, DataLoader
import json
from transformers import BertTokenizer
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


#  读取json文件
def read_json(input_file):
    with open(input_file, 'r', encoding='UTF-8') as f:
        reader = f.readlines()
        lines = []
        for line in reader:
            lines.append(json.loads(line.strip()))

        return lines

class BertClfDataset(Dataset):
    """
    用于BERT文本分类的数据集，该处定义其数据结构
    """

    def __init__(self, fp):
        """
        bert的输入需要以下三种数据，labels为文本类别用于训练及评测。
        :param data:
        """
        data = read_json(fp)
        self.text_a = [line['source'] for line in data]
        self.text_b = [line['target'] for line in data]
        self.labels = [int(line['labelA']) for line in data]

    def __len__(self):
        """
        数据集总大小
        :return:
        """
        return len(self.labels)

    def __getitem__(self, index):
        """
        从数据集中获取单条数据
        :param index: 索引
        :return:
        """
        return self.text_a[index], self.text_b[index], self.labels[index]


def get_loader(data, batch_size=4, shuffle=False):
    """
    定义数据加载器，可批量加载并简单处理数据
    :param data:
    :param batch_size:
    :param shuffle: 是否打乱顺序
    :return:
    """

    dataset = BertClfDataset(data)

    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_fn,num_workers=4)
    return loader


def _collate_fn(data):
    """
    在输入模型前的最后一道数据处理
    :param data:
    :return:
    """
    text_a, text_b, labels = zip(*data)
    inputs = tokenizer(text_a, text_b, padding=True, return_tensors='pt')
    labels = torch.tensor(labels, dtype=torch.long)

    return inputs, labels

import torch
import os
from torch.utils.data import Dataset, DataLoader
import json
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


# 输入的句子结构 两个句子一个标签
class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


#  读取json文件
def read_json(input_file):
    with open(input_file, 'r', encoding='UTF-8') as f:
        reader = f.readlines()
        lines = []
        for line in reader:
            lines.append(json.loads(line.strip()))
        return lines


def get_examples(lines, set_type):
    examples = []
    for (i, line) in enumerate(lines):
        guid = "%s-%s" % (set_type, i)
        text_a = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(lines[i]['sentence1']))
        text_b = None
        # 有改动
        label = (line['label']) if set_type != './data/test' else "0"
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


def get_any_examples(data_dir):
    return get_examples(read_json(data_dir), os.path.splitext(data_dir)[0])


# 获取标签
def get_labels():
    labels = []
    for i in range(2):
        # if i == 5 or i == 11:
        #     continue
        labels.append(str(i))
    return labels


class BertClfDataset(Dataset):
    """
    用于BERT文本分类的数据集，该处定义其数据结构
    """

    def __init__(self, data):
        """
        bert的输入需要以下三种数据，labels为文本类别用于训练及评测。
        :param data:
        """
        self.input_ids = data['input_ids']
        self.attention_masks = data['attention_masks']
        self.token_type_ids = data['token_type_ids']
        self.input_labels = data['input_labels']

    def __len__(self):
        """
        数据集总大小
        :return:
        """
        return len(self.input_labels)

    def __getitem__(self, index):
        """
        从数据集中获取单条数据
        :param index: 索引
        :return:
        """
        # print(self.input_ids[idex])
        return self.input_ids[index], self.attention_masks[index], self.token_type_ids[index], self.input_labels[index]


def get_loader(data, batch_size=4, shuffle=False):
    """
    定义数据加载器，可批量加载并简单处理数据
    :param data:
    :param batch_size:
    :param shuffle: 是否打乱顺序
    :return:
    """

    dataset = BertClfDataset(data)
    # print(dataset.__len__())
    # for i in range(dataset.__len__()):
    #     print(dataset.__getitem__(i))
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_fn)
    return loader


def _collate_fn(data):
    """
    在输入模型前的最后一道数据处理
    :param data:
    :return:
    """
    input_ids, attention_masks, token_type_ids, labels = zip(*data)
    input_ids = torch.tensor(input_ids).long()
    attention_masks = torch.tensor(attention_masks).long()
    token_type_ids = torch.tensor(token_type_ids).long()
    labels = torch.tensor(labels).long()
    return input_ids, attention_masks, token_type_ids, labels


def preprocess(data_dir,max_length=512):
    """
    处理数据
    :param raw_data_fn: 原始数据文件名
    :return:
    """
    examples = get_any_examples(data_dir)

    id_list = []
    attention_masks_list = []
    token_type_ids_list = []
    labels_list = []

    for example in examples:

        input_ids = example.text_a

        label_list = get_labels()
        ################# 有更改
        label_map = {label: i for i, label in enumerate(label_list)}
        label = label_map[example.label] ########有改动 将str转化为int索引

        if len(input_ids) > max_length:
            token = tokenizer.tokenize(input_ids)
            input_ids = token[:max_length - 2]

        input_len = len(input_ids)
        padding_length = max_length - input_len

        attention_masks = [1] * input_len + [0] * padding_length
        input_ids = input_ids + ([0] * padding_length)
        token_type_ids = [0] * max_length

        id_list.append(input_ids)
        attention_masks_list.append(attention_masks)
        token_type_ids_list.append(token_type_ids)
        labels_list.append(label)

    data = {'input_ids': id_list,
            'attention_masks': attention_masks_list,
            'token_type_ids': token_type_ids_list,
            'input_labels': labels_list}
    return data

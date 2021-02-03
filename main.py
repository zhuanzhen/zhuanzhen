import torch
import os
from data import get_loader, preprocess, get_any_examples
from transformers import BertForSequenceClassification, BertConfig
import numpy as np


#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# 定义配置项
cache_dir = './data'
path = './data/bert.pt'
model_name = 'bert-base-chinese'
train_dir = "./data/train.json"
test_dir = "./data/test.json"
dev_dir = "./data/dev.json"
valid_dir = "./data/dev.json"
batch_size = 8
accumulation_steps = 8
lr = 2e-3

epochs = 2  # 10
# lr = 1e-3 batchsize = 4 accuracy = 0.68999 epoch = 2
# lr = 1e-3 batchsize = 8 accuracy = 0.68999 epoch = 2
# lr = 1.5e-3 batchsize = 8 accuracy = 0.68999 epoch = 2
# lr = 2e-3 batchsize = 8 accuracy = 0.68999 epoch = 2
device = torch.device(f'cuda:{1}' if torch.cuda.is_available() else 'cpu')
print("device:", device)
def train_entry():
    # 加载数据
    train_data = preprocess(train_dir)
    #valid_data = preprocess(valid_dir)
    test_data = preprocess(valid_dir)

    train_loader = get_loader(train_data, batch_size=batch_size, shuffle=True)
    #dev_loader = get_loader(valid_data,batch_size=batch_size, shuffle=False)
    # 测试集无需打乱顺序
    test_loader = get_loader(test_data, batch_size=8, shuffle=False)

    # 加载模型及配置方法                                              15
    bert_config = BertConfig.from_pretrained(model_name, num_labels=2)  # 头条文本分类数据集为15类
    model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name, config=bert_config, cache_dir=cache_dir)
    model.to(device)
    print("model device:", model.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train(model, optimizer, train_loader)
    torch.load('/data/fangmiaoTEST/zaz/bert/bert.pt')
    test(model, test_loader)


def train(model, optimizer, data_loader):
    """
    训练，请自行实现
    :param model:
    :param optimizer:
    :param data_loader:
    :return:
    """
    min_loss = float('inf')
    for e in range(epochs):
        for step, batch in enumerate(data_loader):
            model.zero_grad()
            model.train()

            inputs = {'input_ids':batch[0],
                      'attention_mask':batch[1],
                      'token_type_ids':batch[2],
                      'labels':batch[3]
            }
            inputs['input_ids'] = inputs['input_ids'].to(device)
            inputs['labels'] = inputs['labels'].to(device)
            inputs['attention_mask'] = inputs['attention_mask'].to(device)
            inputs['token_type_ids'] = inputs['token_type_ids'].to(device)
            output = model(input_ids = inputs['input_ids'], labels = inputs['labels'])

            train_loss = output[0]
            train_loss = train_loss / accumulation_steps
            train_loss.backward()

            # 每八次更新一下网络中的参数
            if (step+1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if (step+1) % accumulation_steps == 1:
                print('Train Epoch: {} [{}/{}]  ||  train_loss: {:.6f}'.format(
                    e+1, step * batch_size, len(data_loader.dataset), train_loss.item()
                ))

        print('Train Epoch: {} || train_loss:{:.6f}.'.format(e+1, train_loss.item()))
        if train_loss < min_loss:
            min_loss = train_loss
            torch.save(model.state_dict(), '/data/fangmiaoTEST/zaz/bert/bert.pt')


def test(model, test_loader):
    """
    测试，请自行实现
    :param model:
    :param test_loader:
    :return:
    """
    accuracy = 0.0
    sums = 0.0 # 所有计算精度的总和
    nums = 0
    for step, batch in enumerate(test_loader):
        model.eval()
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]
                      }
            inputs['input_ids'] = inputs['input_ids'].to(device)
            inputs['labels'] = inputs['labels'].to(device)
            inputs['attention_mask'] = inputs['attention_mask'].to(device)
            inputs['token_type_ids'] = inputs['token_type_ids'].to(device)
            output = model(input_ids=inputs['input_ids'], labels=inputs['labels'])
            test_loss, logits = output[:2]
            preds = logits.detach().cpu().numpy()
            predict_label = np.argmax(preds, axis=1)
            predict_label = torch.tensor(predict_label)
            predict_label = predict_label.to(device)
            #计算精度
            sum = 0.0  # 每一次的和
            nums = nums + predict_label.numel()  # 计算预测的次数
            sum = predict_label.eq(inputs['labels']).float().sum()
            sum = sum.item()
            sums = sums + sum
            accuracy = sums/nums
            # print("the number of true result of this epoch is(sum) :", sum)
            # print("all the true result is(sums) :", sums)
            # print("inputs['labels']:", inputs['labels'].shape)
            # print("number of inputs is:", nums)
            # print("test_loss:", test_loss)
            print("accuracy of test is-->sum/nums :", accuracy)

    # for *x, y in test_loader:
    #     outputs = model(*x, labels=y)
    #     print(outputs[0])



if __name__ == '__main__':
    train_entry()
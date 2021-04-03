import torch
import os
import numpy as np
from data import get_loader
from transformers import BertForSequenceClassification, BertConfig
from sklearn import metrics

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# 定义配置项
cache_dir = './data'
ckpt_path = './data/ssbertA.pt'
model_name = 'bert-base-chinese'
train_dir = "./sohu2021_open_data_clean/短短匹配A类/train.txt"
test_dir = "./sohu2021_open_data_clean/短短匹配A类/test_with_id.txt.txt"
dev_dir = "./sohu2021_open_data_clean/短短匹配A类/valid.txt"
valid_dir = "./sohu2021_open_data_clean/短短匹配A类/valid.txt"
batch_size = 16
log_steps = 50
lr = 1e-5

epochs = 3  # 10

device = torch.device(f'cuda:{2}' if torch.cuda.is_available() else 'cpu')
print("device:", device)


def train_entry():
    # 加载数据
    train_loader = get_loader(train_dir, batch_size=batch_size, shuffle=True)
    # dev_loader = get_loader(valid_data,batch_size=batch_size, shuffle=False)
    # 测试集无需打乱顺序
    test_loader = get_loader(dev_dir, batch_size=8, shuffle=False)

    # 加载模型及配置方法                                              15
    bert_config = BertConfig.from_pretrained(model_name, num_labels=2)
    model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name,
                                                          config=bert_config)
    model.to(device)
    print("model device:", model.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train(model, optimizer, train_loader, test_loader)
    model.load_state_dict(torch.load(ckpt_path))
    test(model, test_loader)


def train(model, optimizer, data_loader, test_loader):
    """
    训练，请自行实现
    :param model:
    :param optimizer:
    :param data_loader:
    :return:
    """
    best_acc = 0.0
    for e in range(epochs):
        train_acc = [0.0]
        train_loss = [0.0]
        model.train()
        for step, batch in enumerate(data_loader):
            optimizer.zero_grad()
            x, y = batch
            x.to(device)
            y = y.to(device)
            outputs = model(**x, labels=y, return_dict=True)
            outputs.loss.backward()
            optimizer.step()

            train_acc.append(torch.mean((y == torch.argmax(outputs.logits, dim=-1)).float()).cpu().item())
            train_loss.append(outputs.loss.cpu().item())
            # recall
            recall = metrics.recall_score(y.cpu().detach().numpy() , torch.argmax(outputs.logits, dim=-1).cpu().detach().numpy() , average='macro')

            if (step + 1) % log_steps == 0:
                print('Train Epoch: {} [{}/{}]  ||  train_loss: {:.6f} || train_acc: {:.6f} || recall: {:.6f}'.format(
                    e + 1, step * batch_size, len(data_loader.dataset), np.mean(train_loss[-log_steps:]),
                    np.mean(train_acc[-log_steps:]), recall
                ))

        print('Train Epoch: {} || train_loss: {:.6f} || train_acc: {:.6f} '.format(e + 1, np.mean(train_loss),
                                                                                  np.mean(train_acc)))

        test_acc = test(model, test_loader)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), ckpt_path)
            print('model saved.')


def test(model, test_loader):
    """
    测试，请自行实现
    :param model:
    :param test_loader:
    :return:
    """
    accuracy = 0.0
    loss = 0.0
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            x, y = batch
            x.to(device)
            y = y.to(device)
            outputs = model(**x, labels=y, return_dict=True)
            accuracy += torch.sum(y == torch.argmax(outputs.logits, dim=-1)).cpu().item()
            loss += outputs.loss.cpu().item()
        print(f'valid loss: {loss / len(test_loader)}')
        print(f'valid acc: {accuracy / test_loader.dataset.__len__()}')
    return accuracy


if __name__ == '__main__':
    train_entry()

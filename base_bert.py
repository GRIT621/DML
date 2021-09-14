from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, train_test_split
from torch import nn, optim
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn import metrics
import random
import os

def setup_seed(seed):
   torch.manual_seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.enabled = True
setup_seed(666)


device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

from sklearn.preprocessing import OneHotEncoder
# train_data= AG_NEWS(root="/home/lsj0920/mpl-pytorch-main/dataset",split=("train","test"))



max_len = 64

batch_size = 128
learning_rate = 2e-6
weight_decay = 1e-5
epochs = 70
EARLY_STOP = True
EARLY_STOPPING_STEPS = 10
import pandas as pd
from torch.utils.data import TensorDataset
import torch
from torch.utils.data import DataLoader
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
data = pd.read_csv("./dataset/AG_NEWS/train.csv", names = ['label','title','text'])
devdata = pd.read_csv("./dataset/AG_NEWS/test.csv", names = ['label','title','text'])
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertTokenizer, BertModel, BertConfig, BertForSequenceClassification

#
#
# def weights_init(m):
#     classname = m.__class__.__name__
#
#     if classname.find('Conv1d') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#         if m.bias is not None:
#             m.bias.data.fill_(0)
#     elif classname.find('Conv2d') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#         if m.bias is not None:
#             m.bias.data.fill_(0)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)
#     elif classname.find('Linear') != -1:
#         m.weight.data.normal_(0, 0.02)
#         m.bias.data.fill_(0)
#

encoder = BertModel.from_pretrained(
    'bert-base-cased', num_labels=4).to(device)
# model.apply(weights_init)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased',do_lower_case=False)

def tokenizen(sent):
    lg = 32
    # sent = tokenizer.encode(sent)
    # return  sent[:lg - 2]  + [0] * (lg - len(sent[:lg - 2]) - 2)
    encoded_dict = tokenizer(sent, max_length=max_len, padding='max_length', truncation=True, )
    return encoded_dict['input_ids'],  encoded_dict['attention_mask']
def modifylabel(row):
    return row - 1


data = data.sample(frac = 0.01,random_state = 666)

data['train_data'] = data['text'].apply(tokenizen)

devdata['dev_data'] = devdata['text'].apply(tokenizen)

data["label"] = data["label"].apply(modifylabel)
devdata["label"] = devdata["label"].apply(modifylabel)
train_data, test_data, train_label, test_label = train_test_split(data['train_data'].values.tolist(),
                                                                    data['label'].values.tolist(), test_size=80,train_size=100,
                                                                    random_state=666)


train_data = torch.tensor(train_data, dtype=torch.long)
train_label = torch.tensor(train_label, dtype=torch.long)
test_data = torch.tensor(test_data, dtype=torch.long)
test_label = torch.tensor(test_label, dtype=torch.long)

x = devdata['dev_data']
y = devdata['label']

dev_data = torch.tensor(x, dtype=torch.long)
dev_label = torch.tensor(y, dtype=torch.long)

train_dataset = TensorDataset(train_data, train_label)
train_loader = DataLoader(dataset=train_dataset, batch_size=32)




test_dataset = TensorDataset(train_data, train_label)
test_loader = DataLoader(dataset=test_dataset, batch_size=32)
dev_dataset = TensorDataset(dev_data, dev_label)
dev_loader = DataLoader(dataset=dev_dataset, batch_size=32)

class Model(nn.Module):
    def __init__(self,num_classes = 4,max_len = 32, drop=0.5):
        super().__init__()
        self.bn = nn.BatchNorm1d(64)
        self.encoder = encoder

        self.embedding_size = encoder.pooler.dense.out_features #bert-base
#         self.embedding_size = encoder.pooler.out_features #albert
        self.max_len = max_len
        self.fc = nn.Linear(self.embedding_size,num_classes)
        self.dropout = nn.Dropout(drop)
#         self.sig = nn.Sigmoid()
    def forward(self, x):


        x = encoder(input_ids=x[:, 0, :], attention_mask=x[:,1, :])[0]

        # x = self.bn(x)
        x = x.mean(1)
        #print(x.size())
#         x = self.gru(x)

        x = self.dropout(x)
        #x = attention_module(x)
        x = self.fc(x)
        return x
model = Model().to(device)
def train_fn(model, optimizer, criterion, dataloader):
    model.train()  # 启用batchnormalization和dropout
    final_loss = 0

    for data in dataloader:
        optimizer.zero_grad()  # 梯度参数初始化为0
        inputs, targets = data[0].to(device), data[1].to(device)

        outputs = model(inputs)

        # print("$$$$",outputs.size(),targets.size())
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        final_loss += loss.item()

    final_loss /= len(dataloader)

    return final_loss


def valid_fn(model, criterion, dataloader):
    model.eval()  # 不启用batchnormalization和dropout
    final_loss = 0
    valid_preds = []
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    for data in dataloader:
        inputs, targets = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        labels = targets.data.cpu().numpy()
        predic = torch.max(outputs.data, 1)[1].cpu().numpy()
        loss = criterion(outputs, targets)
        labels_all = np.append(labels_all, labels)
        predict_all = np.append(predict_all, predic)

        final_loss += loss.item()
        valid_preds.append(outputs.detach().cpu().numpy())  # detach()阻断反向传播、cpu()将数据移植cpu上、numpy()将tensor转换为numpy
    mif1 = f1_score(labels_all, predict_all, average='micro')  # 调用并输出计算的值
    maf1 = f1_score(labels_all, predict_all, average='macro')
    acc = accuracy_score(labels_all, predict_all)
    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)  # concatenate()进行拼接操作

    return final_loss, valid_preds, acc,mif1, maf1


def inference_fn(model, dataloader):
    model.eval()
    preds = []
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    for data in dataloader:
        inputs, targets = data[0].to(device), data[1].to(device)
        with torch.no_grad():
            outputs = model(inputs)
            labels = targets.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()


            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
        # preds.append(outputs.sigmoid().detach().cpu().numpy())
    mif1 = f1_score(labels_all, predict_all, average='micro')  # 调用并输出计算的值
    maf1 = f1_score(labels_all, predict_all, average='macro')
    report = metrics.classification_report(labels_all, predict_all, digits=4)
    confusion = metrics.confusion_matrix(labels_all, predict_all)


    return report, confusion,mif1,maf1
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()
early_stopping_steps = EARLY_STOPPING_STEPS
early_step = 0
best_loss = np.inf

for epoch in range(epochs):
    train_loss = train_fn(model, optimizer, criterion, train_loader)
    print(f"EPOCH: {epoch}, train_loss: {train_loss}")
    valid_loss, valid_preds, acc, mif1, maf1 = valid_fn(model, criterion, test_loader)
    print(f"EPOCH: {epoch}, valid_loss: {valid_loss}, acc:{acc}, mif1{mif1}, mafi{maf1}")


    if valid_loss < best_loss:
        best_loss = valid_loss
#         torch.save(model.state_dict(), f"res-model/albert_epoch_{epoch}.pth") #state_dict存放模型权重及参数
        early_step = 0

    elif EARLY_STOP:
        early_step += 1
        if early_step >= early_stopping_steps:
            break

report, confusion, mif1, maf1 = inference_fn(model, dev_loader)
print(report,confusion,mif1,maf1)
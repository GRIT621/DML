import numpy as np
import pandas as pd
import torch
import random
from torch import nn, optim
import torch.nn.functional as F
import math
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertTokenizer, BertModel, BertConfig
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from transformers import AlbertModel, AlbertTokenizer,AlbertForSequenceClassification


max_len = 32
seed = 666
batch_size = 16
learning_rate = 2e-6
weight_decay = 1e-5
epochs = 50
EARLY_STOP = True
EARLY_STOPPING_STEPS = 5



class Model(nn.Module):
    def __init__(self,args,drop=0.7):
        super().__init__()
        self.encoder = BertModel.from_pretrained('bert-base-cased')

        self.embedding_size = self.encoder.pooler.dense.out_features #bert-base
#         self.embedding_size = encoder.pooler.out_features #albert
        self.fc = nn.Linear(self.embedding_size, args.num_classes)
        self.dropout = nn.Dropout(drop)
#         self.sig = nn.Sigmoid()
    def forward(self, x):
        x = self.encoder(input_ids=x[:, 0, :], attention_mask=x[:,1, :])[0]
        #print(x.size())
#         x = self.gru(x)

        x = x.mean(1)
        x = self.dropout(x)
        #x = attention_module(x)
        x = self.fc(x)
        return x
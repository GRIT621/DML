import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
import json
from torch.utils.data import Dataset, DataLoader
import torch
import random
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertTokenizer, BertModel, BertConfig, BertForSequenceClassification
from sklearn.model_selection import KFold, train_test_split
import os
import csv
# from torchtext.data.utils import get_tokenizer


logger = logging.getLogger(__name__)



logger = logging.getLogger(__name__)

pretrained_weights = 'bert-base-cased'

model_class = BertForSequenceClassification
tokenizer_class = BertTokenizer

tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=False)
print(tokenizer)
def loadModelTokenizer(num_labels, language = 'english'):
    model_class = BertForSequenceClassification
    tokenizer_class = BertTokenizer
    if language == 'english':
        pretrained_weights = 'bert-base-uncased'
    if language == 'german':
        pretrained_weights = 'bert-base-german-cased'

    tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=False)
    model = model_class.from_pretrained(pretrained_weights)
    return model, tokenizer


class Base_dataset(Dataset):

    def __init__(self,args,model,mode,train_data= None,train_label = None,test_data = None,test_label= None):
        self.dataset = args.dataset

        self.model = model
        self.mode = mode
        self.max_len = args.max_len
        if self.dataset == 'base':
            args.num_classes = 2
        elif self.dataset == 'tianchi':
            args.num_classes = 14
        elif self.dataset == 'AGNews':
            args.num_classes = 4
        if self.mode == "dev":
            self.test_data = test_data
            self.test_label = test_label
            print("test data %d" ,len(test_data))

        # num_samples = 140000
        # class_num = np.zeros(args.num_classes, dtype=np.int)
        # index = list(range(140000))
        # random.shuffle(index)
        # labeled_num = int(1 * 140000)
        # labeled_idx = []
        # unlabeled_idx = []
        #
        # for i in index:
        #     label = train_label[i]
        #     if class_num[label] < int(num_samples / num_classes) and len(labeled_idx) < labeled_num:
        #         labeled_idx.append(i)
        #         class_num[label] = class_num[label]+1
        #     else:
        #         unlabeled_idx.append(i)

            # if self.mode == "labeled":
            # 	pred_idx = labeled_idx
            #
            # elif self.mode == "unlabeled":
            # 	pred_idx = unlabeled_idx

        else:
            n = len(train_label)
            if self.model == "bert":
                self.train_data_l = train_data[:args.num_labeled]
                self.train_data_u = train_data[args.num_labeled:]
                self.train_label_l = train_label[:args.num_labeled]
                self.train_label_u = train_label[args.num_labeled:]
                self.test_data = test_data
                self.test_label = test_label
            elif self.model == "TextRCNN":
                self.train_data_l = train_data[:n//2][0]
                self.train_data_u = train_data[n//2:][0]
                self.train_label_l = train_label[:n//2]
                self.train_label_u = train_label[n//2:]
                self.test_data = test_data[0]
                self.test_label = test_label


            print("labeled data %d, unlabeled data %d" ,args.num_labeled, n -args.num_labeled)

    def __getitem__(self, index):

        if self.mode == 'labeled':
            text, target = self.train_data_l[index], self.train_label_l[index]

            text1 = torch.tensor(text)
            return text1, target
        elif self.mode == 'unlabeled':
            text, target = self.train_data_u[index], self.train_label_u[index]
            text1 = torch.tensor(text)
            text2 = torch.tensor(text)

            return text1, text2, target
        elif self.mode == 'test' or self.mode == "dev":
            text, target = self.test_data[index], self.test_label[index]
            text1 = torch.tensor(text)
            return text1, target

    def __len__(self):
        if self.mode == 'labeled':
            return len(self.train_data_l)
        elif self.mode == 'unlabeled':
            return len(self.train_data_u)
        else:
            return len(self.test_data)


def get_base(args):

    def tokenizen(row):
        tokenizer = BertTokenizer.from_pretrained("/home/lsj0920/mpl-pytorch-main/model/bert_base/bert-base-chinese-vocab.txt")
        encoded_dict = tokenizer(row, max_length=64, padding='max_length', truncation=True, )
        # input_ids token_type_ids attention_mask
        return encoded_dict['input_ids'], encoded_dict[
            'attention_mask']  # 使用分词器进行编码，‘input_ids’为单词在词典中的编码，‘attention_mask’指定对哪些词进行self attention
    data = pd.read_csv('/home/lsj0920/mpl-pytorch-main/dataset/base/task1_train.csv')
    data = data.set_index(keys='id').reset_index(drop=True)  # inputs_ids
    data = data.sample(frac = 0.01)
    data['train_id'] = data['joke'].apply(tokenizen)


    train_data, test_data, train_label, test_label = train_test_split(data['train_id'].values,
                                                                              data['label'].values, test_size=0.1,
                                                                              random_state=2333)
    #     dict = {"train_data": train_data.tolist(),
    #             "test_data":test_data.tolist(),
    #             "train_label":train_label.tolist(),
    #             "test_label":test_label.tolist()}
    #     with open("data_dict.json", "w") as f:
    #         json_dict = json.dumps(dict)
    #         f.write(json_dict)
    #
    # elif args.test == 'test':
    #     with open("data_dict.json", 'r+') as f:
    #         dict = json.load(f)
    #         train_data = np.array(dict["train_data"])
    #         test_data = np.array(dict["test_data"])
    #         train_label = np.array(dict["train_label"])
    #         test_label = np.array(dict["test_label"])


    train_labeled_dataset = Base_dataset(dataset=args.dataset, max_len=args.max_len,model =args.model, mode="labeled",
                                                  train_data=train_data,test_data=test_data, train_label=train_label, test_label=test_label)
    train_unlabeled_dataset =Base_dataset(dataset=args.dataset, max_len=args.max_len,model =args.model, mode="unlabeled",
                                                   train_data=train_data,test_data=test_data, train_label=train_label, test_label=test_label)
    test_dataset =Base_dataset(dataset=args.dataset, max_len=args.max_len,model = args.model,mode="test",
                                        train_data=train_data,test_data=test_data, train_label=train_label, test_label=test_label)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all training data
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


def x_u_split_test(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = []
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        np.random.shuffle(idx)
        labeled_idx.extend(idx[:label_per_class])
        unlabeled_idx.extend(idx[label_per_class:])
    labeled_idx = np.array(labeled_idx)
    unlabeled_idx = np.array(unlabeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])

    np.random.shuffle(labeled_idx)
    np.random.shuffle(unlabeled_idx)
    return labeled_idx, unlabeled_idx


def get_tianchi(args):
    def tokenizen(row):
        max_length = 200
        row = list(map(int, row.split()))
        n = len(row)
        if n < max_length-2:
            row = [101]+ row + [102] + [0]* (max_length-n-2)
            mask = [1]*(n+2) +[0]*(max_length-n-2)
        else:
            row = [101]+ row[:max_length-2] + [102]
            mask = [1]*(max_length)
        return row,mask
    data = pd.read_csv('/home/lsj0920/mpl-pytorch-main/dataset/tianchi/train.csv')


    data['train_id'] = data['text'].apply(tokenizen)
    train_data, test_data, train_label, test_label = train_test_split(data['train_id'].values,
                                                                               data['label'].values, test_size=0.1,
                                                                               random_state=2333)

    # (self, dataset, args, model, mode, train_data, train_label, test_data, test_label):

    train_labeled_dataset = Base_dataset(args,model =args.model, mode="labeled",
                                                  train_data=train_data,test_data=test_data, train_label=train_label, test_label=test_label)
    train_unlabeled_dataset =Base_dataset(args,model =args.model, mode="unlabeled",
                                                   train_data=train_data,test_data=test_data, train_label=train_label, test_label=test_label)
    test_dataset =Base_dataset(args,model = args.model,mode="test",
                                        train_data=train_data,test_data=test_data, train_label=train_label, test_label=test_label)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def get_tianchi_dev(args):
    def tokenizen(row):
        max_length = 200
        row = list(map(int, row.split()))
        n = len(row)
        if n < max_length-2:
            row = [101]+ row + [102] + [0]* (max_length-n-2)
            mask = [1]*(n+2) +[0]*(max_length-n-2)
        else:
            row = [101]+ row[:max_length-2] + [102]
            mask = [1]*(max_length)
        return row,mask

    data = pd.read_csv('/home/lsj0920/mpl-pytorch-main/dataset/tianchi/test_a.csv')
    data['train_id'] = data['text'].apply(tokenizen)
    data['label'] = [0] * 50000




    # (self, dataset, args, model, mode, train_data, train_label, test_data, test_label):

    # train_labeled_dataset = Base_dataset(args,model =args.model, mode="labeled",
    #                                               train_data=train_data,test_data=test_data, train_label=train_label, test_label=test_label)
    # train_unlabeled_dataset =Base_dataset(args,model =args.model, mode="unlabeled",
    #                                                train_data=train_data,test_data=test_data, train_label=train_label, test_label=test_label)
    dev_dataset = Base_dataset(args,model = args.model, mode="dev", test_data= data['train_id'],test_label=data['label'])

    return dev_dataset

def get_AGNews(args):
    pretrained_weights = 'bert-base-cased'

    model_class = BertForSequenceClassification
    tokenizer_class = BertTokenizer

    tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=False)

    def tokenizen(sent):

        # sent = tokenizer.encode(sent)
        #
        # return  sent[:lg - 2]  + [0] * (lg - len(sent[:lg - 2]) - 2)
        encoded_dict = tokenizer(sent, max_length=args.max_len, padding='max_length', truncation=True, )
        return encoded_dict['input_ids'],  encoded_dict['attention_mask']



    data = pd.read_csv('/home/lsj0920/MPL/dataset/AG_NEWS/train.csv',names = ["label","title","text"])
    test_data = pd.read_csv('/home/lsj0920/MPL/dataset/AG_NEWS/test.csv',names = ["label","title","text"])
    test_data['train_id'] = test_data['title'] + " " + test_data["text"]
    test_data['train_id'] = test_data['train_id'].apply(tokenizen)
    def modifylabel(row):
        return int(row) - 1
    test_data["label"] = test_data["label"].apply(modifylabel)
    testdata, testlabel = torch.tensor(test_data['train_id'] , dtype=torch.long), torch.tensor(test_data['label'] , dtype=torch.long)
    test_dataset = Base_dataset(args, model=args.model, mode="test",
                                train_data=testdata, test_data=testdata, train_label=testlabel,
                                test_label=testlabel)
    if args.evaluate == True:
        return test_dataset

    if args.num_labeled == 100:
        data2 = data.sample(frac= 0.3, random_state=2333)
    if args.num_labeled == 1000:
        data2 = data.sample(frac= 0.3, random_state=2333)
    if args.num_labeled == 10000:
        data2 = data.sample(frac= 0.4, random_state=2333)
    data2['train_id'] = data2['title'] + " " + data2["text"]
    data2['train_id'] = data2['train_id'].apply(tokenizen)
    data2["label"] = data2["label"].apply(modifylabel)

    train_data, dev_data, train_label, dev_label = train_test_split(data2['train_id'].values,
                                                                              data2['label'].values, test_size=int(0.8 * args.num_labeled),train_size=args.num_labeled+args.num_unlabeled,
                                                                              random_state=2333)


    train_labeled_dataset = Base_dataset(args,model =args.model, mode="labeled",
                                                  train_data=train_data,test_data=dev_data, train_label=train_label, test_label=dev_label)
    train_unlabeled_dataset =Base_dataset(args,model =args.model, mode="unlabeled",
                                                   train_data=train_data,test_data=dev_data, train_label=train_label, test_label=dev_label)
    dev_dataset =Base_dataset(args,model = args.model,mode="test",
                                        train_data=train_data,test_data=dev_data, train_label=train_label, test_label=dev_label)


    return train_labeled_dataset, train_unlabeled_dataset, dev_dataset,test_dataset


def get_Yelp(args):
    pretrained_weights = 'bert-base-cased'

    model_class = BertForSequenceClassification
    tokenizer_class = BertTokenizer

    tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=False)

    def tokenizen(sent):

        # sent = tokenizer.encode(sent)
        #
        # return  sent[:lg - 2]  + [0] * (lg - len(sent[:lg - 2]) - 2)
        encoded_dict = tokenizer(sent, max_length=args.max_len, padding='max_length', truncation=True, )
        return encoded_dict['input_ids'],  encoded_dict['attention_mask']



    data = pd.read_csv('/home/lsj0920/MPL/dataset/Yelp/yelp_review_full_csv/train.csv',names = ["label","text"])
    test_data = pd.read_csv('/home/lsj0920/MPL/dataset/Yelp/yelp_review_full_csv/test.csv',names = ["label","text"])
    #136参数来源
    if args.num_labeled == 100:
        data2 = data.sample(frac= 0.04, random_state=args.seed)
        logger.info(f"frac:0.04")
    if args.num_labeled == 1000:
        data2 = data.sample(frac= 0.04, random_state=args.seed)
        logger.info(f"frac:0.04")
    if args.num_labeled == 10000:
        data2 = data.sample(frac= 0.5, random_state=args.seed)
        logger.info(f"frac:0.5")
    # 256参数来源
    # if args.num_labeled == 100:
    #     data2 = data.sample(frac= 0.04, random_state=args.seed)
    #     logger.info(f"frac:0.04")
    # if args.num_labeled == 1000:
    #     data2 = data.sample(frac= 0.08, random_state=args.seed)
    #     logger.info(f"frac:0.04")
    # if args.num_labeled == 10000:
    #     data2 = data.sample(frac= 0.5, random_state=args.seed)
    #     logger.info(f"frac:0.2")
    def modifylabel(row):
        return row - 1

    test_data['train_id'] = test_data['text'].apply(tokenizen)
    test_data["label"] = test_data["label"].apply(modifylabel)
    devdata, devlabel = torch.tensor(test_data['train_id'] , dtype=torch.long), torch.tensor(dev_data['label'] , dtype=torch.long)
    test_dataset = Base_dataset(args, model=args.model, mode="test",
                                train_data=devdata, test_data=devdata, train_label=devlabel,
                                test_label=devlabel)
    if args.evaluate == True:
        return test_dataset

    data2['train_id'] = data2['text'].apply(tokenizen)
    data2["label"] = data2["label"].apply(modifylabel)



    train_data, test_data, train_label, test_label = train_test_split(data2['train_id'].values,
                                                                              data2['label'].values, test_size=int(0.8 * args.num_labeled),train_size=args.num_labeled+args.num_unlabeled,
                                                                              random_state=args.seed)

    #
    # # (self, dataset, args, model, mode, train_data, train_label, test_data, test_label):

    train_labeled_dataset = Base_dataset(args,model =args.model, mode="labeled",
                                                  train_data=train_data,test_data=test_data, train_label=train_label, test_label=test_label)
    train_unlabeled_dataset =Base_dataset(args,model =args.model, mode="unlabeled",
                                                   train_data=train_data,test_data=test_data, train_label=train_label, test_label=test_label)
    # dev_dataset =Base_dataset(args,model = args.model,mode="test",
    #                                     train_data=train_data,test_data=test_data, train_label=train_label, test_label=test_label)


    return train_labeled_dataset, train_unlabeled_dataset, test_dataset,test_dataset
def get_Yahoo(args):
    pretrained_weights = 'bert-base-cased'

    model_class = BertForSequenceClassification
    tokenizer_class = BertTokenizer

    tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=False)

    def tokenizen(sent):

        # sent = tokenizer.encode(sent)
        #
        # return  sent[:lg - 2]  + [0] * (lg - len(sent[:lg - 2]) - 2)
        encoded_dict = tokenizer(sent, max_length=args.max_len, padding='max_length', truncation=True, )
        return encoded_dict['input_ids'],  encoded_dict['attention_mask']



    data = pd.read_csv('/home/lsj0920/mpl-pytorch-main-yelp/dataset/Yahoo/train.csv',names = ["no","label","text"])
    dev_data = pd.read_csv('/home/lsj0920/mpl-pytorch-main-yelp/dataset/Yahoo/test.csv',names = ["no","label","text"])
    #136参数来源
    if args.num_labeled == 100:
        data2 = data.sample(frac= 0.04, random_state=args.seed)
        logger.info(f"frac:0.04")
    if args.num_labeled == 1000:
        data2 = data.sample(frac= 0.04, random_state=args.seed)
        logger.info(f"frac:0.04")
    if args.num_labeled == 10000:
        data2 = data.sample(frac= 0.05, random_state=args.seed)
        logger.info(f"frac:0.5")
    # 256参数来源
    # if args.num_labeled == 100:
    #     data2 = data.sample(frac= 0.04, random_state=args.seed)
    #     logger.info(f"frac:0.04")
    # if args.num_labeled == 1000:
    #     data2 = data.sample(frac= 0.08, random_state=args.seed)
    #     logger.info(f"frac:0.04")
    # if args.num_labeled == 10000:
    #     data2 = data.sample(frac= 0.5, random_state=args.seed)
    #     logger.info(f"frac:0.2")
    def modifylabel(row):
        return row - 1

    dev_data['train_id'] = dev_data['text'].apply(tokenizen)
    dev_data["label"] = dev_data["label"].apply(modifylabel)
    devdata, devlabel = torch.tensor(dev_data['train_id'] , dtype=torch.long), torch.tensor(dev_data['label'] , dtype=torch.long)
    dev_dataset = Base_dataset(args, model=args.model, mode="test",
                                train_data=devdata, test_data=devdata, train_label=devlabel,
                                test_label=devlabel)
    if args.evaluate == True:
        return dev_dataset

    data2['train_id'] = data2['text'].apply(tokenizen)
    data2["label"] = data2["label"].apply(modifylabel)



    train_data, test_data, train_label, test_label = train_test_split(data2['train_id'].values,
                                                                              data2['label'].values, test_size=int(0.8 * args.num_labeled),train_size=args.num_labeled+args.num_unlabeled,
                                                                              random_state=args.seed)

    #
    # # (self, dataset, args, model, mode, train_data, train_label, test_data, test_label):

    train_labeled_dataset = Base_dataset(args,model =args.model, mode="labeled",
                                                  train_data=train_data,test_data=test_data, train_label=train_label, test_label=test_label)
    train_unlabeled_dataset =Base_dataset(args,model =args.model, mode="unlabeled",
                                                   train_data=train_data,test_data=test_data, train_label=train_label, test_label=test_label)
    test_dataset =Base_dataset(args,model = args.model,mode="test",
                                        train_data=train_data,test_data=test_data, train_label=train_label, test_label=test_label)


    return train_labeled_dataset, train_unlabeled_dataset, test_dataset,dev_dataset

DATASET_GETTERS = {'base':get_base,
                   'tianchi':get_tianchi,
                   'tianchi_dev':get_tianchi_dev,
                   'AGNews':get_AGNews,
                   'Yelp':get_Yelp,
                   'Yahoo':get_Yahoo

                   }

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
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertTokenizer, BertConfig, BertForSequenceClassification
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import os
# from torchtext.data.utils import get_tokenizer


logger = logging.getLogger(__name__)



logger = logging.getLogger(__name__)

pretrained_weights = 'bert-base-cased'

model_class = BertForSequenceClassification
tokenizer_class = BertTokenizer

tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=False)

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
                # self.train_label_l2=pd.Series(self.train_label_l)
                # print(self.train_label_l2.value_counts())
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

            # return text1, text2, target
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%#
            return text1,text2, target
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%#
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

def get_kg(args):
    class InputFeatures(object):
        """A single set of features of data."""

        def __init__(self, input_ids, input_mask, segment_ids, label_id=None):
            self.input_ids = input_ids
            self.input_mask = input_mask
            self.segment_ids = segment_ids
            self.label_id = label_id

    class InputExample(object):
        """A single training/test example for simple sequence classification."""

        def __init__(self, guid, text_a, text_b=None, text_c=None, label="0"):
            """Constructs a InputExample.
            Args:
                guid: Unique id for the example.
                text_a: string. The untokenized text of the first sequence. For single
                sequence tasks, only this sequence must be specified.
                text_b: (Optional) string. The untokenized text of the second sequence.
                Only must be specified for sequence pair tasks.
                text_c: (Optional) string. The untokenized text of the third sequence.
                Only must be specified for sequence triple tasks.
                label: (Optional) string. The label of the example. This should be
                specified for train and dev examples, but not for test examples.
            """
            self.guid = guid
            self.text_a = text_a
            self.text_b = text_b
            self.text_c = text_c
            self.label = label

    class DataProcessor(object):
        """Base class for data converters for sequence classification data sets."""

        def get_train_examples(self, data_dir):
            """Gets a collection of `InputExample`s for the train set."""
            raise NotImplementedError()

        def get_dev_examples(self, data_dir):
            """Gets a collection of `InputExample`s for the dev set."""
            raise NotImplementedError()

        def get_labels(self, data_dir):
            """Gets the list of labels for this data set."""
            raise NotImplementedError()

        @classmethod
        def _read_jsonl(cls, input_file):
            """Reads a tab separated value file."""
            triple_data = []
            with open(input_file, "r", encoding="utf-8") as f:
                json_data = f.readlines()
            if "dev" in input_file:
                for data in json_data:
                    data_dict = json.loads(data)
                    triple_data.append(
                        [data_dict['triple_id'], data_dict['subject'], data_dict['predicate'], data_dict['object']])
            else:
                for data in json_data:
                    data_dict = json.loads(data)
                    triple_data.append(
                        [data_dict['subject'], data_dict['predicate'], data_dict['object'], data_dict['salience']])
            return triple_data

    class KGProcessor(DataProcessor):
        """Processor for knowledge graph data set."""

        def __init__(self):
            self.labels = set()

        def get_train_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_jsonl(os.path.join(data_dir)), "train", data_dir)

        def get_dev_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_jsonl(os.path.join(data_dir, "dev_triple.jsonl")), "dev", data_dir)

        def get_labels(self, data_dir):
            """Gets all labels (0, 1) for triples in the knowledge graph."""
            return ["0", "1"]

        def get_train_triples(self, data_dir):
            """Gets training triples."""
            triple_data = self._read_jsonl(os.path.join(data_dir, "train_triple.jsonl"))
            return triple_data[:-1]

        def get_dev_triples(self, data_dir):
            """Gets validation triples."""
            return self._read_jsonl(os.path.join(data_dir, "dev_triple.jsonl"))

        def _create_examples(self, lines, set_type, data_dir):
            """Creates examples for the training and dev sets."""
            examples = []

            for (i, line) in enumerate(lines):
                if set_type == "dev" or set_type == "test":

                    triple_id = line[0]
                    text_a = line[1]
                    text_b = line[2]
                    text_c = line[3]

                    examples.append(InputExample(guid=triple_id, text_a=text_a, text_b=text_b, text_c=text_c))

                elif set_type == "train":
                    guid = "%s-%s" % (set_type, i)
                    text_a = line[0]
                    text_b = line[1]
                    text_c = line[2]
                    label = line[3]

                    examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, label=label))
            return examples

    def _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_length):
        """Truncates a sequence triple in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b) and len(tokens_a) > len(tokens_c):
                tokens_a.pop()
            elif len(tokens_b) > len(tokens_a) and len(tokens_b) > len(tokens_c):
                tokens_b.pop()
            elif len(tokens_c) > len(tokens_a) and len(tokens_c) > len(tokens_b):
                tokens_c.pop()
            else:
                tokens_c.pop()

    def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, print_info=True,
                                     set_type="train"):
        """Loads a data file into a list of `InputBatch`s."""
        label_map = {label: i for i, label in enumerate(label_list)}
        triple_ids = []
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0 and print_info:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))

            tokens_a = tokenizer.tokenize(example.text_a)

            tokens_b = None
            tokens_c = None

            if set_type == "dev":
                triple_ids.append(example.guid)

            if example.text_b and example.text_c:
                tokens_b = tokenizer.tokenize(example.text_b)
                tokens_c = tokenizer.tokenize(example.text_c)
                # Modifies `tokens_a`, `tokens_b` and `tokens_c`in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP], [SEP] with "- 4"
                # _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
                _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0   0   0   0  0     0 0
            # (c) for sequence triples:
            #  tokens: [CLS] Steve Jobs [SEP] founded [SEP] Apple Inc .[SEP]
            #  type_ids: 0 0 0 0 1 1 0 0 0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence or the third sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)
            if tokens_c:
                tokens += tokens_c + ["[SEP]"]
                segment_ids += [0] * (len(tokens_c) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            if type(example.label) == str:


                label_id = label_map[example.label]
            else:
                label_id = example.label

            if ex_index < 5 and print_info:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("label: %s (id = %d)" % (example.label, label_id))

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
        if set_type == "dev":
            return features, triple_ids
        else:
            return features

    processor = KGProcessor()
    label_list = processor.get_labels("/home/lsj0920/kg/data/train_triple.jsonl")
    train_examples = processor.get_train_examples("/home/lsj0920/kg/data/train_triple.jsonl")

    label_list_uda = processor.get_labels("/home/lsj0920/kg/data/new_train_triple.jsonl")
    train_examples_uda = processor.get_train_examples("/home/lsj0920/kg/data/new_train_triple.jsonl")

    test_label_list = processor.get_labels("/home/lsj0920/kg/data/test_triple.jsonl")
    test_examples = processor.get_train_examples("/home/lsj0920/kg/data/test_triple.jsonl")

    train_features = convert_examples_to_features(
            train_examples, label_list, args.max_len, tokenizer)

    train_features_uda = convert_examples_to_features(
            train_examples_uda, label_list_uda, args.max_len, tokenizer)

    test_features = convert_examples_to_features(
            test_examples, test_label_list, args.max_len, tokenizer)


    all_input_ids = torch.tensor([(f.input_ids,f.input_mask) for f in train_features], dtype=torch.long)
    # all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    # all_input_data = ((all_input_ids,all_input_mask))
    # all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

    uda_input_ids = torch.tensor([(f.input_ids,f.input_mask) for f in train_features_uda], dtype=torch.long)
    uda_label_ids = torch.tensor([f.label_id for f in train_features_uda], dtype=torch.long)

    test_data = torch.tensor([(f.input_ids,f.input_mask) for f in test_features], dtype=torch.long)
    test_label = torch.tensor([f.label_id for f in test_features], dtype=torch.long)

    train_data = all_input_ids
    train_label = all_label_ids
    # train_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    train_data, test_data, train_label, test_label = train_test_split(all_input_ids,
                                                                               all_label_ids, test_size=0.1,
                                                                               random_state=2333)

    train_data = torch.cat((train_data,uda_input_ids),0)
    train_label= torch.cat((train_label,uda_label_ids),0)

    # (self, dataset, args, model, mode, train_data, train_label, test_data, test_label):

    train_labeled_dataset = Base_dataset(args,model =args.model, mode="labeled",
                                                  train_data=train_data,test_data=test_data, train_label=train_label, test_label=test_label)
    train_unlabeled_dataset =Base_dataset(args,model =args.model, mode="unlabeled",
                                                   train_data=train_data,test_data=test_data, train_label=train_label, test_label=test_label)
    test_dataset =Base_dataset(args,model = args.model,mode="test",
                                        train_data=train_data,test_data=test_data, train_label=train_label, test_label=test_label)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset,test_dataset

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



    data = pd.read_csv('./dataset/AG_NEWS/train.csv',names = ["label","title","text"])
    test_data = pd.read_csv('./dataset/AG_NEWS/test.csv',names = ["label","title","text"])
    test_data['train_id'] = test_data['title'] + " " + test_data["text"]
    test_data['train_id'] = test_data['train_id'].apply(tokenizen)
    def modifylabel(row):
        return row - 1
    test_data["label"] = test_data["label"].apply(modifylabel)
    testdata, testlabel = torch.tensor(test_data['train_id'] , dtype=torch.long), torch.tensor(test_data['label'] , dtype=torch.long)
    test_dataset = Base_dataset(args, model=args.model, mode="test",
                                train_data=testdata, test_data=testdata, train_label=testlabel,
                                test_label=testlabel)
    if args.evaluate == True:
        return test_dataset

    if args.num_labeled == 100:
        data2 = data.sample(frac= 0.4, random_state=2333)
        #0。3   not 0.2
    if args.num_labeled == 1000:
        # data2 = data.sample(frac= 0.3, random_state=2333)
        data2 = data.sample(frac= 0.4, random_state=2333)
    if args.num_labeled == 10000:
        #0.4
        data2 = data.sample(frac= 0.6, random_state=2333)
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



    data = pd.read_csv('./dataset/Yelp/train.csv',names = ["label","text"])
    dev_data = pd.read_csv('./dataset/Yelp/test.csv',names = ["label","text"])
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


def get_Yahoo(args):
    pretrained_weights = 'bert-base-uncased'

    model_class = BertForSequenceClassification
    tokenizer_class = BertTokenizer

    tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=False)

    def tokenizen(sent):

        # sent = tokenizer.encode(sent)
        #
        # return  sent[:lg - 2]  + [0] * (lg - len(sent[:lg - 2]) - 2)
        try:

            encoded_dict = tokenizer(sent, max_length=args.max_len, padding='max_length', truncation=True, )
            return encoded_dict['input_ids'],  encoded_dict['attention_mask']
        except:
            return None,None



    data = pd.read_csv('./dataset/Yahoo/train.csv',names = ["no","label","text"])
    dev_data = pd.read_csv('./dataset/Yahoo/test.csv',names = ["no","label","text"])
    #136参数来源
    if args.num_labeled == 100:
        data2 = data.sample(frac= 0.08, random_state=args.seed)
        logger.info(f"frac:0.08")
    if args.num_labeled == 1000:
        data2 = data.sample(frac= 0.08, random_state=args.seed)
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
                   'Yahoo':get_Yahoo,
                   'kg':get_kg,

                   }

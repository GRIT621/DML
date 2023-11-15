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
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from .meta_base import MetaModule,MetaLinear
from .MetaBert import MetaBertModel
import warnings

class Model(nn.Module):
    def __init__(self,args):
        super().__init__()

        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.embedding_size = self.encoder.pooler.dense.out_features #bert-base
        self.fc = nn.Linear(self.embedding_size, args.num_classes)
        self.dropout = nn.Dropout(args.drop)

#         self.sig = nn.Sigmoid()
    def forward(self, x, NTM_required=False):

        x = self.encoder(input_ids=x[:, 0, :], attention_mask=x[:,1, :])[0]
        x_feature = x.mean(1)
        x_feature = self.dropout(x_feature)
        x_class = self.fc(x_feature)

        return x_class,x_feature

class MetaModel(MetaModule):
    def __init__(self,args):
        super().__init__()

        self.encoder = MetaBertModel.from_pretrained('bert-base-uncased')
        self.embedding_size = self.encoder.pooler.dense.weight.shape[0] #bert-base
        self.fc = MetaLinear(self.embedding_size, args.num_classes)
        self.dropout = nn.Dropout(args.drop)

    def forward(self, x, NTM_required=False):

        x = self.encoder(input_ids=x[:, 0, :], attention_mask=x[:,1, :])[0]
        x_feature = x.mean(1)
        x_feature = self.dropout(x_feature)
        x_class = self.fc(x_feature)

        return x_class,x_feature

    def meta_zero_grad(self, set_to_none: bool = False) -> None:
        r"""Sets gradients of all model parameters to zero. See similar function
        under :class:`torch.optim.Optimizer` for more context.

        Arguments:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                See :meth:`torch.optim.Optimizer.zero_grad` for details.
        """
        if getattr(self, '_is_replica', False):
            warnings.warn(
                "Calling .zero_grad() from a module created with nn.DataParallel() has no effect. "
                "The parameters are copied (in a differentiable manner) from the original module. "
                "This means they are not leaf nodes in autograd and so don't accumulate gradients. "
                "If you need gradients in your forward method, consider using autograd.grad instead.")

        for p in self.buffers():
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    else:
                        p.grad.requires_grad_(False)
                    p.grad.zero_()


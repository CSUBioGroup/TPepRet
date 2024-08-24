#coding=utf-8
import warnings
warnings.filterwarnings("ignore")

import os
import random
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from sklearn.model_selection import StratifiedKFold

import sys
from load_data import *
from data_encode import *

USE_CUDA = torch.cuda.is_available()

# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
random_seed = 0
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if USE_CUDA:
    torch.cuda.manual_seed(0)


LOG_FILE = './log/log_file.log'

threshold = 0.5
NUM_WORKERS = 4
NUM_EPOCHS = 100
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
EMBED_SIZE = 6
K_Fold = 5

import retnet
layers = 24
hidden_dim = 256
ffn_size = 512
heads = 8


class Network_conn(nn.Module):
    def __init__(self):
        super(Network_conn, self).__init__()
        self.gru5 = nn.GRU(28, 128,
                           batch_first=True,
                           bidirectional=True
                           )
        self.gru6 = nn.GRU(28, 128,
                           batch_first=True,
                           bidirectional=True
                           )
        self.retnet = retnet.RetNet(layers, hidden_dim, ffn_size, heads, double_v_dim=True)
        self.full_conn = nn.Sequential(
            nn.Linear(34 * hidden_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    # def forward(self, peps1, cdr1, peps2, cdr2, peps3, cdr3):
    def forward(self, peps3, cdr3):
        peps3 = self.gru5(peps3)[0]  # x1:(batch , 14,  hidden_dim)
        cdr3 = self.gru6(cdr3)[0]  # x1:(batch , 20,  hidden_dim)
        x = torch.cat((peps3, cdr3), 1)
        x = self.retnet(x)
        x = self.full_conn(torch.flatten(x, start_dim=1))
        return x

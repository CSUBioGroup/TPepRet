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

model_save_dir_name = os.path.basename(sys.argv[0])[0:-3]
make_dirs_for_models(model_save_dir_name, random_seed)
MODEL_SAVE_PATH = os.path.dirname(os.path.abspath(__file__)) + '/../models/' + model_save_dir_name + '/random_seed_' + str(random_seed) + '/'
make_dirs_for_models_pretrained(model_save_dir_name, random_seed)
MODEL_SAVE_PATH_PRETRAINED = os.path.dirname(os.path.abspath(__file__)) + '/../pretrained_models/' + model_save_dir_name + '/random_seed_' + str(random_seed) + '/'

train_data_file_path = os.path.dirname(os.path.abspath(__file__)) + '/../data/train.txt'
independent_data_file_path = os.path.dirname(os.path.abspath(__file__)) + '/../data/seen.txt'

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

    def forward(self, peps3, cdr3):
        peps3 = self.gru5(peps3)[0]  # x1:(batch , 14,  hidden_dim)
        cdr3 = self.gru6(cdr3)[0]  # x1:(batch , 20,  hidden_dim)
        x = torch.cat((peps3, cdr3), 1)
        # print('x', x.shape)
        x = self.retnet(x)
        x = self.full_conn(torch.flatten(x, start_dim=1))
        return x



if "__main__" == __name__:
    print('1')
    cdr3_seq_list, pep_seq_list, label_list = get_data_from_file(train_data_file_path)
    independent_cdr3_seq_list, independent_pep_seq_list, independent_label_list = get_data_from_file(independent_data_file_path)
    print('2')
    AUC_total = []
    recall_total = []
    precision_total = []
    fold = 1
    kf = StratifiedKFold(n_splits=K_Fold, shuffle=True, random_state=random_seed)  # not KFold
    for train_index, test_index in kf.split(pep_seq_list, cdr3_seq_list, label_list):

        train_dataloader, test_dataloader = getDataLoader_distribute(train_index, test_index, pep_seq_list, cdr3_seq_list, label_list, BATCH_SIZE, NUM_WORKERS)

        independent_dataset = MyDataSet_distribute(independent_pep_seq_list, independent_cdr3_seq_list, independent_label_list)
        independent_dataloader = tud.DataLoader(independent_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        model = Network_conn()
        print("CPU")
        if USE_CUDA:
            print('using cuda')
            model = model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        # loss_func = nn.CrossEntropyLoss()
        loss_func = nn.BCELoss()
        if USE_CUDA:
            loss_func = loss_func.cuda()

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=4)

        best_model_name = train_distribute_and_save_pretrained_models(train_dataloader, test_dataloader, model, loss_func, optimizer, scheduler, NUM_EPOCHS, USE_CUDA, fold, MODEL_SAVE_PATH, MODEL_SAVE_PATH_PRETRAINED, threshold)

        model_test = Network_conn()
        if USE_CUDA:
            print('using cuda')
            model_test = model_test.cuda()

        AUC, recall, precision = test_EL_distribute(model_test, independent_dataloader, fold, best_model_name, USE_CUDA, threshold)

        # for classification
        AUC_total.append(AUC)
        recall_total.append(recall)
        precision_total.append(precision)
        fold += 1

    AUC_average = np.mean(AUC_total)
    recall_average = np.mean(recall_total)
    precision_average = np.mean(precision_total)

    print("AUC_average:{:.3f}\trecall_average:{:.3f}\tprecision_average:{:.3f}\n".format
          (AUC_average, recall_average, precision_average))
    print("#################################")
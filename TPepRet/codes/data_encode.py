import numpy as np
import torch
import torch.utils.data as tud
from utils import *
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score


# 不在20个碱基内的用X表示
aa = {"C": 0, "S": 1, "T": 2, "P": 3, "A": 4, "G": 5, "N": 6, "D": 7, "E": 8, "Q": 9, "H": 10, "R": 11, "K": 12,
      "M": 13, "I": 14, "L": 15, "V": 16, "F": 17, "Y": 18, "W": 19}

aa_blosum50={"A":0,"R":1,"N":2,"D":3,"C":4,"Q":5,"E":6,"G":7,"H":8,"I":9,"L":10,"K":11,"M":12,"F":13,"P":14,"S":15,"T":16,"W":17,"Y":18,"V":19}

AAfea_phy_dict = get_AAfea_phy()
blosum50_matrix = blosum50()
blosum62_matrix = blosum62()

embedding_dict = get_embedding()

def encode_seq(seq, ENCODING_TYPE):
    encoded_seq = []
    if ENCODING_TYPE == 'AAfea_phy_BLOSUM62':
        for residue in seq:
            encoded_residue_tmp2 = []
            encoded_residue_tmp = []
            if residue not in AAfea_phy_dict.keys():
                for i in range(28):
                    encoded_residue_tmp2.append(0)
            else:
                encoded_residue_tmp2 = AAfea_phy_dict[residue]
            if residue not in aa.keys():
                for i in range(20):
                    encoded_residue_tmp.append(0)
            else:
                residue_idx = aa[residue]
                encoded_residue_tmp = blosum62_matrix[residue_idx]
            encoded_residue = encoded_residue_tmp2 + encoded_residue_tmp
            encoded_seq.append(encoded_residue)
    elif ENCODING_TYPE == 'AAfea_phy':
        for residue in seq:
            if residue not in AAfea_phy_dict.keys():
                encoded_residue = []
                for i in range(28):
                    encoded_residue.append(0)
            else:
                encoded_residue = AAfea_phy_dict[residue]
            encoded_seq.append(encoded_residue)

    elif ENCODING_TYPE == 'encoded':
        encoded_seq = seq

    elif ENCODING_TYPE == 'num':
        for residue in seq:
            if residue in aa.keys():
                encoded_seq.append(aa[residue])
            else:
                encoded_seq.append(20)
    elif ENCODING_TYPE == 'one-hot':
        for residue in seq:
            encoded_residue = []
            if residue not in aa.keys():
                for i in range(20):
                    encoded_residue.append(0)
            else:
                residue_idx = aa[residue]
                for i in range(20):
                    if i == residue_idx:
                        encoded_residue.append(1)
                    else:
                        encoded_residue.append(0)
            encoded_seq.append(encoded_residue)
    elif ENCODING_TYPE == 'BLOSUM50':
        for residue in seq:
            if residue not in aa_blosum50.keys():
                encoded_residue = []
                for i in range(20):
                    encoded_residue.append(0)
            else:
                residue_idx = aa_blosum50[residue]
                encoded_residue = blosum50_matrix[residue_idx]
            encoded_seq.append(encoded_residue)
    elif ENCODING_TYPE == 'BLOSUM62':
        for residue in seq:
            if residue not in aa.keys():
                encoded_residue = []
                for i in range(20):
                    encoded_residue.append(0)
            else:
                residue_idx = aa[residue]
                encoded_residue = blosum62_matrix[residue_idx]
            encoded_seq.append(encoded_residue)
    elif ENCODING_TYPE == 'embedding':
        for residue in seq:
            if residue not in aa.keys():
                encoded_residue = embedding_dict['X']
            else:
                encoded_residue = embedding_dict[residue]
            encoded_seq.append(encoded_residue) # [len(residue), 6]
    else:
        print("wrong ENCODING_TYPE!")
    return encoded_seq


def encode_seq_list(seq_list, ENCODING_TYPE):
    encoded_seq_list = []
    for seq in seq_list:
        encoded_seq_list.append(encode_seq(seq, ENCODING_TYPE))
    return encoded_seq_list


class MyDataSet_distribute(tud.Dataset):
    def __init__(self, train_peps, train_cdr3, train_labels):
        super(MyDataSet_distribute, self).__init__()

        ENCODING_TYPE_PEP3 = 'AAfea_phy'
        ENCODING_TYPE_ALLELE3 = 'AAfea_phy'

        encoded_train_peps3 = encode_seq_list(train_peps, ENCODING_TYPE_PEP3)
        encoded_train_cdr33 = encode_seq_list(train_cdr3, ENCODING_TYPE_ALLELE3)
        self.encoded_peps3 = torch.Tensor(encoded_train_peps3).float()
        self.encoded_cdr33 = torch.Tensor(encoded_train_cdr33).float()

        self.labels = torch.Tensor(train_labels).reshape(-1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # return self.encoded_peps1[index], self.encoded_cdr31[index], self.encoded_peps2[index], self.encoded_cdr32[index], self.encoded_peps3[index], self.encoded_cdr33[index], self.labels[index]
        return self.encoded_peps3[index], self.encoded_cdr33[index], self.labels[index]


def test_independent_only_return_list_triple(model_test, independent_dataloader, fold, best_model_name, USE_CUDA, threshold):
    if USE_CUDA:
        model_test.load_state_dict(torch.load(best_model_name))
    else:
        model_test.load_state_dict(torch.load(best_model_name,map_location='cpu'))
    real_labels = []
    pred_prob = []
    predict_labels = []
    predic_keys = []

    # for i, (X1, X2, X3, X4, X5, X6, test_labels) in enumerate(independent_dataloader):
    for i, (X1, X2, test_labels) in enumerate(independent_dataloader):
        if USE_CUDA:
            X1 = X1.cuda()
            X2 = X2.cuda()

        predic_key_tmp = torch.cat((X1,X2), 1).tolist()
        predic_key = []
        for idx in range(len(predic_key_tmp)):
            pred_prob_list = predic_key_tmp[idx]
            key_list = ''
            for item in pred_prob_list:
                key_list += str(item)
            predic_key.append(key_list)

        model_test.eval()
        # output = model_test(X1, X2, X3, X4, X5, X6)
        output = model_test(X1, X2)

        predic_keys += predic_key # 顺序

        pred_prob += output.tolist()

        real_labels += test_labels

    return predic_keys,  pred_prob, real_labels

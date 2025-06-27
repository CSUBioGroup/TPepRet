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


def getDataLoader_distribute(train_index, test_index, pep_seq_list, cdr3_seq_list, label_list, BATCH_SIZE, NUM_WORKERS):

    train_peps = np.array(pep_seq_list)[train_index].tolist()
    train_cdr3 = np.array(cdr3_seq_list)[train_index].tolist()
    train_labels = np.array(label_list)[train_index].tolist()

    train_dataset = MyDataSet_distribute(train_peps, train_cdr3, train_labels)
    train_dataloader = tud.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    test_peps = np.array(pep_seq_list)[test_index].tolist()
    test_cdr3 = np.array(cdr3_seq_list)[test_index].tolist()
    test_labels = np.array(label_list)[test_index].tolist()

    test_dataset = MyDataSet_distribute(test_peps, test_cdr3, test_labels)
    test_dataloader = tud.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    return train_dataloader, test_dataloader


def train_distribute_and_save_pretrained_models(train_dataloader, test_dataloader, model, loss_func, optimizer, scheduler, NUM_EPOCHS, USE_CUDA, fold, model_save_path, MODEL_SAVE_PATH_PRETRAINED, threshold):
    best = 0
    model_name = ''
    for epoch in range(NUM_EPOCHS):
        batch_idx = 0
        for i, (X5, X6, train_labels) in enumerate(train_dataloader):
            batch_idx = i
            if USE_CUDA:
                X5 = X5.cuda()
                X6 = X6.cuda()
                train_labels = train_labels.cuda()

            optimizer.zero_grad()
            model.train()
            output = model(X5, X6)

            loss = loss_func(output, train_labels)

            loss.backward()
            optimizer.step()

        print("learning rate is ", optimizer.param_groups[0]["lr"])
        AUC, recall, precision, test_loss = validate_distribute(test_dataloader, model, loss_func, epoch, USE_CUDA, threshold)
        scheduler.step(test_loss)
        if AUC > best:
            best = AUC
            model_name = "validate_param_fold" + str(fold) + "epoch" + str(epoch) + "_batch" + str(batch_idx) + '.pkl'

        if epoch == 20:
            torch.save(model.state_dict(), MODEL_SAVE_PATH_PRETRAINED + '20_epoch/' + model_name)
        if epoch == 50:
            torch.save(model.state_dict(), MODEL_SAVE_PATH_PRETRAINED + '50_epoch/' + model_name)
        if epoch == 100:
            torch.save(model.state_dict(), MODEL_SAVE_PATH_PRETRAINED + '100_epoch/' + model_name)
        if epoch == 150:
            torch.save(model.state_dict(), MODEL_SAVE_PATH_PRETRAINED + '150_epoch/' + model_name)

    torch.save(model.state_dict(), model_save_path +  model_name)

    print("最佳模型：", model_name)
    return model_save_path + model_name

def validate_distribute(test_dataloader, model, loss_func, epoch, USE_CUDA, threshold):
    test_loss = 0
    predict_labels = []
    real_labels = []
    pred_pros = []
    for i, (X5, X6, test_labels) in enumerate(test_dataloader):
        if USE_CUDA:
            X5 = X5.cuda()
            X6 = X6.cuda()
            test_labels = test_labels.cuda()

        model.eval()
        output = model(X5, X6)
        output_list = output.cpu().detach().numpy().tolist()
        output_class = []
        for item in output_list:
            if float(item[0]) > threshold:
                output_class.append([1])
            else:
                output_class.append([0])

        loss = loss_func(output, test_labels)

        test_loss += float(loss)

        real_labels += test_labels.cpu().tolist()
        pred_pros += output.cpu().tolist()
        predict_labels += output_class

    AUC = roc_auc_score(real_labels, pred_pros)
    recall = metrics.recall_score(real_labels, predict_labels, average='binary')
    precision = metrics.precision_score(real_labels, predict_labels, average='binary')
    print('验证集第{}轮_AUC:{:.3f} \t recall:{:.3f}\t'
          'precision:{:.3f}\ttest_loss:{:.3f}\n'.format(epoch, AUC, recall, precision, test_loss))

    return AUC, recall, precision, test_loss

# 独立测试集
def test_EL_distribute(model_test, independent_dataloader, fold, best_model_name, USE_CUDA, threshold):
    model_test.load_state_dict(torch.load(best_model_name))
    real_labels = []
    pred_pros = []
    predict_labels = []

    for i, (X5, X6, test_labels) in enumerate(independent_dataloader):
        if USE_CUDA:
            X5 = X5.cuda()
            X6 = X6.cuda()
            test_labels = test_labels.cuda()

        model_test.eval()
        output = model_test(X5, X6)
        output_list = output.cpu().detach().numpy().tolist()
        output_class = []
        for item in output_list:
            if float(item[0]) > threshold:
                output_class.append([1])
            else:
                output_class.append([0])

        real_labels += test_labels.cpu().tolist()
        pred_pros += output.cpu().tolist()
        predict_labels += output_class


    AUC = roc_auc_score(real_labels, pred_pros)
    recall = metrics.recall_score(real_labels, predict_labels, average='binary')
    precision = metrics.precision_score(real_labels, predict_labels, average='binary')
    print('第{}折_AUC:{:.3f} \trecall:{:.3f}\t'
          'precision:{:.3f}\n'.format(fold, AUC, recall, precision))


    print('分类报告:\n', metrics.classification_report(real_labels, predict_labels))
    return AUC,  recall, precision
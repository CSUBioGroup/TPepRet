import re
import os


def get_data_from_file_origin(data_file_path):
    cdr3_seq_list_tmp, pep_seq_list_tmp = [], []
    label_list = []

    in_f = open(data_file_path, 'r')
    for line in in_f:
        cols = re.split('[\t\n]', line)
        cdr3_seq_list_tmp.append(cols[0])
        pep_seq_list_tmp.append(cols[1])
        label_list.append(int(cols[2]))
    in_f.close()

    return cdr3_seq_list_tmp, pep_seq_list_tmp, label_list

def get_data_from_file(data_file_path):
    cdr3_seq_list, pep_seq_list = [], []
    cdr3_seq_list_tmp, pep_seq_list_tmp = [], []
    label_list = []

    in_f = open(data_file_path, 'r')
    for line in in_f:
        cols = re.split('[\t\n]', line)
        cdr3_seq_list_tmp.append(cols[0])
        pep_seq_list_tmp.append(cols[1])
        label_list.append(int(cols[2]))
    in_f.close()

    for item in cdr3_seq_list_tmp:
        if len(item) >= 20:
            pseq_cdr3_seq = item[0:20]
        else:
            pseq_cdr3_seq = item + 'X' * (20 - len(item))
        cdr3_seq_list.append(pseq_cdr3_seq)

    for item in pep_seq_list_tmp:
        peplen = len(item)
        if peplen < 14:
            insert_idx = int((peplen + 1) / 2)
            pseq_pep_seq = item[0:insert_idx] + 'X' * (14 - peplen) + item[insert_idx:]
        else:
            pseq_pep_seq = item[0:7] + item[-7:]
        pep_seq_list.append(pseq_pep_seq)
    return cdr3_seq_list, pep_seq_list, label_list

# configue
from TPepRet import *


USE_CUDA = torch.cuda.is_available()
random_seed = 0
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if USE_CUDA:
    torch.cuda.manual_seed(0)

def independent_test_distribute(independent_data_file_path, model_save_dir_name, out_file_path):
    MODEL_SAVE_PATH = '../models/' + model_save_dir_name + '/'

    independent_cdr3_seq_list_origin, independent_pep_seq_list_origin, label_list_origin = \
        get_data_from_file_origin(independent_data_file_path)

    independent_cdr3_seq_list, independent_pep_seq_list, label_list = \
        get_data_from_file(independent_data_file_path)

    files = os.listdir(MODEL_SAVE_PATH)
    latest_files = []
    for i in range(5):
        fold_num = i + 1
        matched_file = []
        matched_file_tmp = []
        for file in files:
            matched_file_tmp.append(re.findall('validate_param_fold' + str(fold_num) + r'epoch.*', file))


        for item in matched_file_tmp:
            if item == []:
                continue
            else:
                matched_file.append(item[0])
        if len(matched_file) >= 2:
            mtime = os.path.getmtime(MODEL_SAVE_PATH + matched_file[0])
            latest_file_idx = 0
            for idx in range(1, len(matched_file)):
                if os.path.getmtime(MODEL_SAVE_PATH + matched_file[idx]) > mtime:
                    mtime = os.path.getmtime(MODEL_SAVE_PATH + matched_file[idx])
                    latest_file_idx = idx
            latest_files.append(MODEL_SAVE_PATH + matched_file[latest_file_idx])
        elif len(matched_file) == 1:
            latest_files.append(MODEL_SAVE_PATH + matched_file[0])
        else:
            continue

    print('latest_files', latest_files)
    list_probs_folds = []

    for item in latest_files:
        independent_dataset = MyDataSet_distribute(independent_pep_seq_list, independent_cdr3_seq_list, label_list)
        independent_dataloader = tud.DataLoader(independent_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                                num_workers=0)

        model_test = Network_conn()
        if USE_CUDA:
            print('using cuda')
            model_test = model_test.cuda()

        item_name = re.split('[/]', item)[-1]
        fold = int(item_name[len('validate_param_fold')])
        predic_keys, pred_prob, real_labels_tensor = test_independent_only_return_list_triple(model_test, independent_dataloader, fold,
                                                                          item, USE_CUDA, threshold)

        list_pred_prob_fold = list(map(list, zip(*pred_prob)))[0]

        list_probs_folds.append(list_pred_prob_fold)

    list_probs_folds_trans = list(map(list, zip(*list_probs_folds)))
    print(np.array(list_probs_folds_trans).shape)

    list_probs_folds_means = []
    for item in list_probs_folds_trans:
        list_probs_folds_means.append(np.mean(item))

    of = open(out_file_path, 'w')
    for i, prob in enumerate(list_probs_folds_means):
        of.write(independent_cdr3_seq_list_origin[i] + '\t' + independent_pep_seq_list_origin[i] + '\t' + str(prob) + '\t' + str(int(real_labels_tensor[i].tolist()[0])) + '\n')
    of.close()

    return independent_pep_seq_list_origin, list_probs_folds_means, real_labels_tensor



if __name__ == "__main__":

    # configue
    independent_data_file_path = '../for_prediction/'
    test_files = os.listdir(independent_data_file_path)
    # print(test_files)
    for test_file in test_files:
        if os.path.isdir(independent_data_file_path + test_file):
            continue
        if test_file == 'desktop.ini':
            continue
        model_save_dir_name = 'TPepRet'  # select models
        out_file_path = '../for_prediction/outputs/' + test_file
        pep_list, preds, labels_tensor = independent_test_distribute(independent_data_file_path + test_file, model_save_dir_name,
                                                                     out_file_path)




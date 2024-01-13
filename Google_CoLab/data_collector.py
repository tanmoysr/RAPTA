import json
import pickle
import numpy as np
import torch
import math
import time
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import re
import configure as config

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
else:
    device = torch.device("cpu")
# device = torch.device("cpu")

def cell_type_converter(test_str):
    temp = re.compile("([a-zA-Z]+)([0-9]+)")
    try:
        res = temp.match(test_str).groups()
        gate_type = res[0]
        if res[0] == 'MUX':
            number_of_inputs = int(res[1][0])
        else:
            number_of_inputs = sum(int(digit) for digit in res[1])
    except:
        gate_type = test_str
        number_of_inputs = 0

    return gate_type, number_of_inputs

def createFromProcessed(X, X_sublabel, Y, saving_Path_Train, saving_Path_Test):
    input_dim = config.original_input_dim  # input dimension 45/44

    X_arr = np.array(X)
    X_l = list(X_arr[:, 0])
    X_d = list(X_arr[:, 1])
    X_c = list(X_arr[:, 2])

    # Padding
    seq_lens_l = [len(_) for _ in X_l]  # 16~64, length 2598
    max_len_l = max([len(_) for _ in X_l])  # 24 as we are considering only gate
    # seq_batch_l = torch.tensor([_ + [[0] * input_dim] * (max_len_l - len(_)) for _ in X_l]) # [2598, 64, 24] with zero padding
    seq_batch_l = []
    for _ in X_l:
        pad_list = [[0] * input_dim] * (max_len_l - len(_))
        seq_batch_l.append(pad_list + _)

    seq_lens_d = [len(_) for _ in X_d]  # 16~64, length 2598
    max_len_d = max([len(_) for _ in X_d])  # 24 as we are considering only gate
    # seq_batch_d = torch.tensor([_[0] + [[[0] * input_dim] * (max_len_d - len(_)) for _ in X_d] + _[1:]]) # [2598, 64, 24] with zero padding
    seq_batch_d = []
    for _ in X_d:
        pad_list = [[0] * input_dim] * (max_len_d - len(_))
        seq_batch_d.append(_[0:1] + pad_list + _[1:])

    seq_lens_c = [len(_) for _ in X_c]  # 16~64, length 2598
    max_len_c = max([len(_) for _ in X_c])  # 24 as we are considering only gate
    # seq_batch_c = torch.tensor([_ + [[0] * input_dim] * (max_len_c - len(_)) for _ in X_c]) # [2598, 64, 24] with zero padding
    seq_batch_c = []
    for _ in X_c:
        pad_list = [[0] * input_dim] * (max_len_c - len(_))
        seq_batch_c.append(pad_list + _)

    X_l_arr = np.array(seq_batch_l)
    X_d_arr = np.array(seq_batch_d)
    X_c_arr = np.array(seq_batch_c)
    enc = preprocessing.OneHotEncoder()
    feature_array = np.array(config.gate_type_list).reshape(-1, 1)
    enc.fit(feature_array)
    encoded_gate_type_l_flat = enc.transform(X_l_arr[:, :, 1].reshape(-1, 1)).toarray()
    encoded_gate_type_d_flat = enc.transform(X_d_arr[:, :, 1].reshape(-1, 1)).toarray()
    encoded_gate_type_c_flat = enc.transform(X_c_arr[:, :, 1].reshape(-1, 1)).toarray()
    '''Deleting features
    *****************old****************
    1 = cell_max_rise_delay->invalid from 8/23/20
    2 = cell_min_rise_delay->invalid from 8/23/20
    5 = cell_min_fall_delay->invalid from 8/23/20
    6 = cell_max_fall_delay->invalid from 8/23/20
    14 = cell_voltage->10 from 8/23/20
    16 = setup->12 from 8/23/20
    *************new*********************
    12 = cell_voltage
    14 = setup
    25 = MRDL
    27 = M9
    28 = M8
    '''
    X_l_del = np.delete(X_l_arr, [1, 12, 25, 27, 28], axis=2)
    X_d_del = np.delete(X_d_arr, [1, 12, 25, 27, 28], axis=2)
    X_c_del = np.delete(X_c_arr, [1, 12, 25, 27, 28], axis=2)

    encoded_gate_type_l = encoded_gate_type_l_flat.reshape(X_l_del.shape[0], X_l_del.shape[1],
                                                           encoded_gate_type_l_flat.shape[1])
    encoded_gate_type_d = encoded_gate_type_d_flat.reshape(X_d_del.shape[0], X_d_del.shape[1],
                                                           encoded_gate_type_d_flat.shape[1])
    encoded_gate_type_c = encoded_gate_type_c_flat.reshape(X_c_del.shape[0], X_c_del.shape[1],
                                                           encoded_gate_type_c_flat.shape[1])

    X_l_final = list(np.concatenate((encoded_gate_type_l, X_l_del), axis=2).astype(np.float))
    X_d_final = list(np.concatenate((encoded_gate_type_d, X_d_del), axis=2).astype(np.float))
    X_c_final = list(np.concatenate((encoded_gate_type_c, X_c_del), axis=2).astype(np.float))

    # X_l_del = list(seq_batch_l)  # [1, 2, 5, 6, 14, 16]
    # X_d_del = list(seq_batch_d)
    # X_c_del = list(seq_batch_c)

    '''Creating separate features
    14 = cell_voltage->12
    16 = setup->14
    '''
    X_cellV = np.vstack(np.array(seq_batch_d)[:, 0, 12])  # Cell voltage
    X_setup = np.vstack(np.array(seq_batch_d)[:, -1, 14])  # setup
    X_VS = list(np.hstack((X_cellV, X_setup)))

    X_new = []
    for i in range(len(X_d_final)):
        X_new_path = []
        X_new_path.append(X_l_final[i])
        X_new_path.append(X_d_final[i])
        X_new_path.append(X_c_final[i])
        # X_new_path.append(list(seq_batch_l)[i])
        # X_new_path.append(list(seq_batch_d)[i])
        # X_new_path.append(list(seq_batch_c)[i])
        X_new_path.append(X_VS[i])
        X_new_path.append(X_sublabel[i])
        X_new_path.append(Y[i])

        X_new.append(X_new_path)
    X_new = np.array(X_new) # l, d, c, VS, sublabel, label
    # np.random.shuffle(X_new) # shuffle the set
    X_train, X_test = train_test_split(X_new, test_size=0.2, random_state=42, shuffle=True) # Split the set
    pickle.dump(X_train, open(saving_Path_Train, 'wb'))  # saving as pk file
    pickle.dump(X_test, open(saving_Path_Test, 'wb'))  # saving as pk file

def file_processing(old_file, new_file):
    tmp = open(old_file, 'r')
    lines = tmp.read().split(',\n]')
    tmp.close()
    for i_ in range(len(lines)):

        lines[i_] = lines[i_].replace("'",'"')
        lines[i_] = lines[i_].replace("nan",'"NaN"')
        lines[i_] = lines[i_].replace("}],\\n\]", "}]\\n\]")
        # lines[i_] = lines[i_].replace("}]", "}],") # added for fifth trial data
    result = '\n]'.join(lines)
    out_file = open(new_file, "w")
    out_file.write(result)
    out_file.close()

def read_data(X_path, Y_path):
    X = pickle.load(open(X_path, 'rb'))
    Y = pickle.load(open(Y_path, 'rb'))
    return X, Y

def preprocess_data(input_path, processed_saved_path, fp):
    X = []
    X_subLabels = []
    Y = []
    # Slk =[] # slack_pt, added for fifth trial data
    with open(input_path) as json_file:
        data = json.load(json_file)
        for _ in data[1:]:

            whole_seq = []
            l_seq = []
            c_seq = []
            d_seq = []

            path_sublabel = []
            path_sublabel.append(float(_[1][-1]['sub_label_LP']))  # l
            path_sublabel.append(float(_[3][-1]['sub_label_DP']))  # d
            path_sublabel.append(float(_[2][-1]['sub_label_CP']))  # c

            pair_L = np.array_split(np.array(_[1][:-1]), (math.ceil(len(_[1][:-1]) / 2)))
            pair_D = np.array_split(np.array(_[3][:-1]), (math.ceil(len(_[3][:-1]) / 2)))
            pair_C = np.array_split(np.array(_[2][:-1]), (math.ceil(len(_[2][:-1]) / 2)))

            for __ in pair_L:  # seq
                node = []
                for ___ in __:
                    for k, v in ___.items():
                        #         print(v)
                        if k == 'cell_type':
                            gate_type, number_of_inputs = cell_type_converter(v)
                            node.append(gate_type)
                            node.append(number_of_inputs)
                        elif v in ['true', 'false']:
                            node.append(float(bool(v)))
                        elif v == 'pin':
                            node.append(0.0)
                        elif v == 'NaN' or v == 'nan':
                            node.append(0.0)
                            continue
                        elif v == 'INFINITY':
                            continue
                        elif v == 'rise':
                            node.append(1.0)
                            node.append(0.0)  # setup
                            continue
                        elif v == 'fall':
                            node.append(-1.0)
                            node.append(0.0)  # setup
                            continue
                        try:
                            node.append(float(v))
                        #             print(k)
                        except:
                            #                 print(k)
                            #             print(v)
                            continue
                if len(node) < config.total_features: # gate 16 + net 28 = 44
                    node = node + [0.] * (config.total_features - len(node))
                    # node = node[0:15] + [0.] * 2 + node[15:] + [0.] * (43 - len(node)) # rise_fall, setup
                l_seq.append(node)
            for __ in pair_C:  # seq
                node = []
                for ___ in __:
                    for k, v in ___.items():
                        #         print(v)
                        if k == 'cell_type':
                            gate_type, number_of_inputs = cell_type_converter(v)
                            node.append(gate_type)
                            node.append(number_of_inputs)
                        elif v in ['true', 'false']:
                            node.append(float(bool(v)))
                        elif v == 'pin':
                            node.append(0.0)
                        elif v == 'NaN' or v == 'nan':
                            node.append(0.0)
                            continue
                        elif v == 'INFINITY':
                            continue
                        elif v == 'rise':
                            node.append(1.0)
                            node.append(0.0) # setup
                            continue
                        elif v == 'fall':
                            node.append(-1.0)
                            node.append(0.0)  # setup
                            continue
                        try:
                            node.append(float(v))
                        #             print(k)
                        except:
                            #                 print(k)
                            #             print(v)
                            continue
                if len(node) < config.total_features: #45
                    node = node + [0.] * (config.total_features - len(node))
                    # node = node[0:15] + [0.]*2 + node[15:] + [0.] * (43 - len(node))
                c_seq.append(node)
            for __ in pair_D:  # seq
                node = []
                for ___ in __:
                    for k, v in ___.items():
                        #         print(v)
                        if k == 'cell_type':
                            gate_type, number_of_inputs = cell_type_converter(v)
                            node.append(gate_type)
                            node.append(number_of_inputs)
                        elif v in ['true', 'false']:
                            node.append(float(bool(v)))
                        elif v == 'pin':
                            node.append(0.0)
                        elif v == 'NaN' or v == 'nan':
                            node.append(0.0)
                            continue
                        elif v == 'INFINITY':
                            continue
                        elif v == 'rise':
                            node.append(1.0)
                            node.append(0.0) # setup
                            continue
                        elif v == 'fall':
                            node.append(-1.0)
                            node.append(0.0)  # setup
                            continue
                        elif k == 'setup': # last gate of data path has this feature
                            # path_sublabel.append(float(v))
                            node.append(0.0)  # rise_fall
                            node.append(float(v))  # setup
                        try:
                            node.append(float(v))
                        #             print(k)
                        except:
                            #                 print(k)
                            #             print(v)
                            continue
                if len(node) < config.total_features: #45
                    node = node + [0.] * (config.total_features - len(node)) #44
                d_seq.append(node)
            whole_seq.append(l_seq)  # only gate are appending
            whole_seq.append(d_seq)  # only gate are appending
            whole_seq.append(c_seq)  # only gate are appending
            X.append(whole_seq)
            X_subLabels.append(path_sublabel) # l, d, c, setup
            Y.append(float(_[0]['label']))
            # print((float(_[0]['label'])))
            # Slk.append(float(_[0]['pt_slack']))

    pickle.dump(X, open(processed_saved_path +fp+'_X_gate.pk', 'wb'))
    pickle.dump(X_subLabels, open(processed_saved_path + fp + '_X_sublabels.pk', 'wb'))
    pickle.dump(Y, open(processed_saved_path +fp+'_Y_gate.pk', 'wb'))
    pickle.dump(X, open(config.X_data_path + fp + '_X_gate.pk', 'wb'))
    pickle.dump(X_subLabels, open(config.X_data_path + fp + '_X_sublabels.pk', 'wb'))
    pickle.dump(Y, open(config.X_data_path + fp + '_Y_gate.pk', 'wb'))
    # pickle.dump(Slk, open(processed_saved_path + fp + '_Slk_gate.pk', 'wb'))

def tensor_maker(X_train, std_scaler=None, sc_y=None):
    print('Running tensormaker')
    # data processing
    # data_collector.preprocess_data(config.processed_file, config.processed_saved_path)

    # reading processed data
    # X, Y = data_collector.read_data(config.X_data_path, config.Y_data_path)

    # data splitting to l, d, c
    X_arr = X_train
    X_l = list(X_arr[:, 0])
    X_d = list(X_arr[:, 1])
    X_c = list(X_arr[:, 2])
    X_VS = np.stack(X_train[:, 3], axis=0)
    X_sublabel = list(X_arr[:, 4])
    Y = list(X_arr[:, 5])

    # converting to array
    X_la = np.array(X_l).reshape((np.array(X_l).shape[0], -1))
    X_da = np.array(X_d).reshape((np.array(X_d).shape[0], -1))
    X_ca = np.array(X_c).reshape((np.array(X_c).shape[0], -1))
    X_VSa = np.array(X_VS)
    X_sublabela = np.array(X_sublabel).reshape((np.array(X_sublabel).shape[0], -1))
    Ya = np.array(Y).reshape((np.array(Y).shape[0], -1))

    # seq_batch_l_normalized = X_la.reshape(X_la.shape[0],-1,config.input_dim)
    # seq_batch_d_normalized = X_da.reshape(X_da.shape[0],-1,config.input_dim)
    # seq_batch_c_normalized = X_ca.reshape(X_ca.shape[0],-1,config.input_dim)
    # VS_normalized = X_VSa.astype(np.float)
    # # print(type(X_VSa))
    # Y_normalized = Ya
    # X_sub_normalized = X_sublabela
    # sc_y = None
    # std_scaler = None

    # merging
    args = (X_la, X_da, X_ca, X_VSa)
    X_merge = np.concatenate(args, axis=1)
    # Standardization
    # std_scaler = StandardScaler()
    # # std_scaler =MinMaxScaler()
    # std_X_merge = std_scaler.fit_transform(X_merge)
    if std_scaler==None:
        std_scaler = StandardScaler()
        # std_scaler =MinMaxScaler()
        std_X_merge = std_scaler.fit_transform(X_merge)
        print('Train Input Data Fit Transformed')
    else:
        std_X_merge = std_scaler.transform(X_merge)
        print ('Test Input Data Transformed')

    # Splitting
    seq_batch_l_normalized = std_X_merge[:, range(0, X_la.shape[1])].reshape(X_merge.shape[0],-1,config.input_dim)
    seq_batch_d_normalized = std_X_merge[:, range(X_la.shape[1], X_la.shape[1] + X_da.shape[1])].reshape(X_merge.shape[0],-1,config.input_dim)
    seq_batch_c_normalized = std_X_merge[:,
                             range(X_la.shape[1] + X_da.shape[1], X_la.shape[1] + X_da.shape[1] + X_ca.shape[1])].reshape(X_merge.shape[0],-1,config.input_dim)
    VS_normalized = std_X_merge[:, range(X_la.shape[1] + X_da.shape[1] + X_ca.shape[1],
                                         X_la.shape[1] + X_da.shape[1] + X_ca.shape[1] + X_VSa.shape[1])]
    # X_sub_normalized = std_X_merge[:, range(X_la.shape[1] + X_da.shape[1] + X_ca.shape[1] + X_VSa.shape[1],
    #                                         X_la.shape[1] + X_da.shape[1] + X_ca.shape[1] + X_VSa.shape[1] +
    #                                         X_sublabela.shape[1])].reshape(X_merge.shape[0],-1)
    # Y_normalized = std_X_merge[:,
    #                range(X_la.shape[1] + X_da.shape[1] + X_ca.shape[1] + X_VSa.shape[1] + X_sublabela.shape[1],
    #                      std_X_merge.shape[1])].flatten()

    # Y_normalized = Ya.flatten()
    # sc_y = StandardScaler()  # Need to use later for inverse normalization
    # # sc_y = MinMaxScaler()
    # Y_normalized = sc_y.fit_transform(Ya.reshape(-1, 1)).flatten()
    if sc_y == None:
        sc_y = StandardScaler()  # Need to use later for inverse normalization
        # sc_y = MinMaxScaler()
        Y_normalized = sc_y.fit_transform(Ya.reshape(-1, 1)).flatten()  # Standardization
        print('Train Target Data Fit Transformed')
    else:
        Y_normalized = sc_y.transform(Ya.reshape(-1, 1)).flatten()  # Standardization
        print('Test Target Data Transformed')

    X_sub_normalized = X_sublabela
    for i in range(X_sub_normalized.shape[1]):
        X_sub_normalized[:, i] = sc_y.transform(X_sub_normalized[:, i].reshape(-1, 1)).flatten()
    # X_sub = torch.FloatTensor(X_sub_normalized).to(device)  # CPU Tensor, LDCS


    # tensor making
    seq_tensor_l = torch.FloatTensor(seq_batch_l_normalized).to(device)
    seq_tensor_d = torch.FloatTensor(seq_batch_d_normalized).to(device)
    seq_tensor_c = torch.FloatTensor(seq_batch_c_normalized).to(device)
    seq_tensor_vs = torch.FloatTensor(VS_normalized).to(device)
    target = torch.FloatTensor(Y_normalized).to(device)
    X_sub = torch.FloatTensor(X_sub_normalized).to(device)  # CPU Tensor, LDCS

    seq_lens_d = [len(_) for _ in X_d]
    seq_lens_d = torch.LongTensor(seq_lens_d)

    return seq_tensor_l, seq_tensor_d, seq_tensor_c, seq_tensor_vs, seq_lens_d, X_sub, target, std_scaler, sc_y

if __name__ == "__main__":
    '''At first run this script for running_option == 2, then we can run either for option 6 or 7'''
    if config.running_option == 1: # single file
        fp=config.fp[0:-1]
        raw_input_file = config.processed_saved_path + fp + '/jason.json'
        processed_saved_path = config.processed_saved_path + fp + '/'
        processed_file = config.processed_saved_path + fp + '/processed.json'
        print(' starting file processing {}'.format(fp))
        file_processing_tic = time.perf_counter()
        file_processing(raw_input_file, processed_file)
        file_processing_toc = time.perf_counter()
        print('file_processed {}'.format(fp))
        preprocess_data(processed_file, processed_saved_path, fp)
        preprocess_data_toc = time.perf_counter()
        print('preprocess_data {}, file processing time {}, preprocessing_data {}'.format(fp, (
                    file_processing_toc - file_processing_tic), (preprocess_data_toc - file_processing_tic)))
        X = pickle.load(open(config.exp_data_path + fp + '_X_gate.pk', 'rb'))
        X_sublabel = pickle.load(open(config.exp_data_path + fp + '_X_sublabels.pk', 'rb'))
        Y = pickle.load(open(config.exp_data_path + fp + '_Y_gate.pk', 'rb'))
        saving_Path_Train = config.exp_data_path + fp + '_train.pk'
        saving_Path_Test = config.exp_data_path + fp + '_test.pk'
        print(' starting file processing {}'.format(fp))
        createFromProcessed(X, X_sublabel, Y, saving_Path_Train, saving_Path_Test)
        print('preprocess_data {}'.format(fp))
    elif config.running_option == 2: # multiple files
        # for multiple file
        for fp in config.filepaths:
            raw_input_file = config.processed_saved_path+fp+'/jason.json'
            processed_saved_path = config.processed_saved_path+fp+'/'
            processed_file = config.processed_saved_path+ fp + '/processed.json'
            print(' starting file processing {}'.format(fp))
            file_processing_tic=time.perf_counter()
            file_processing(raw_input_file, processed_file)
            file_processing_toc = time.perf_counter()
            print('file_processed {}'.format(fp))
            preprocess_data(processed_file, processed_saved_path, fp)
            preprocess_data_toc = time.perf_counter()
            print('preprocess_data {}, file processing time {}, preprocessing_data {}'.format(fp, (file_processing_toc-file_processing_tic), (preprocess_data_toc-file_processing_tic)))

    elif config.running_option == 6:
        for fp in config.filepaths:
            X = pickle.load(open(config.exp_data_path+fp+'_X_gate.pk', 'rb'))
            X_sublabel = pickle.load(open(config.exp_data_path+fp+'_X_sublabels.pk', 'rb'))
            Y = pickle.load(open(config.exp_data_path+fp+'_Y_gate.pk', 'rb'))
            saving_Path_Train=config.exp_data_path+fp+'_train.pk'
            saving_Path_Test = config.exp_data_path + fp + '_test.pk'
            print(' starting file processing {}'.format(fp))
            createFromProcessed(X, X_sublabel, Y, saving_Path_Train, saving_Path_Test)
            print('preprocess_data {}'.format(fp))

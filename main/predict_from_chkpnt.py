import pickle
import time
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import glob
import configure as config
import data_collector
import model
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
    print("Number of GPU {}".format(torch.cuda.device_count()))
    print("GPU type {}".format(torch.cuda.get_device_name(0)))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')
    # print(torch.backends.cudnn.version())
else:
    device = torch.device("cpu")
    print("Running on the CPU")
# device = torch.device("cpu")

def predict_from_chkpnt(chkpoint_filename, std_scalar_filename, std_y_filename, exp_name, training_time_filename, shuffle_cond=False):

    checkpoint = torch.load(chkpoint_filename)
    std_scaler_train =  pickle.load(open(std_scalar_filename, 'rb'))
    sc_y_train =  pickle.load(open(std_y_filename, 'rb'))
    training_time = pickle.load(open(training_time_filename, 'rb'))

    rapta_predict = model.RAPTA(config.input_dim, config.hidden_dim, config.layer_dim, config.output_dim)
    rapta_predict.load_state_dict(checkpoint)
    rapta_predict.to(device)
    rapta_predict.eval()  # to set dropout and batch
    print('Model loaded successfully')
    error = nn.MSELoss()  # mean Square root error; ((input-target)**2).mean()

    v_list = ['0.78', '0.79', '0.81', '0.83', '0.85', '0.87', '0.89', '0.91',
              '0.93', '0.95', '0.97', '0.99', '1.01', '1.03', '1.05']

    if len(chkpoint_filename.split('_'))<9 and chkpoint_filename.split('_')[4] in v_list:
        filelist_X_test = [config.X_data_path + 'v_' + chkpoint_filename.split('_')[4] + '_all.pk']
    else:
        filelist_X_test = sorted(glob.glob(config.X_test_path_multi))

    for i in range(len(filelist_X_test)):
        # reading processed data
        X_test_single = pickle.load(open(filelist_X_test[i], 'rb'))
        if len(filelist_X_test)==1:
            fp = 'v_' + chkpoint_filename.split('_')[4] + '_'
        else:
            fp = exp_name+'_'+filelist_X_test[i].split('\\')[-1][0:6] + '_'
        seq_tensor_l_test, seq_tensor_d_test, seq_tensor_c_test, seq_tensor_vs_test, seq_lens_d_test, X_sub_test, target_test, std_scaler_test, sc_y_test = data_collector.tensor_maker(X_test_single, std_scaler_train, sc_y_train)

        sc_y = sc_y_test

        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_sub_test, seq_tensor_l_test, seq_tensor_d_test, seq_tensor_c_test,
                                           seq_tensor_vs_test,
                                           seq_lens_d_test, target_test),
            batch_size=len(target_test), shuffle=shuffle_cond, drop_last=True)

        # running model
        print('Starting files set {}'.format(fp))
        print('X file {}'.format(filelist_X_test[i].split('\\')[-1]))

        # Calculate Accuracy
        correct = 0
        total = 0
        for all_sub, images_l, images_d, images_c, images_vs, _, labels in test_loader:
            _ = _.to(device)
            all_sub = all_sub.to(device)
            images_l = images_l.to(device)
            images_d = images_d.to(device)
            images_c = images_c.to(device)
            images_vs = images_vs.to(device)
            labels = labels.to(device)

            all_sub = Variable(all_sub)
            images_l = Variable(images_l)
            images_d = Variable(images_d)
            images_c = Variable(images_c)
            images_vs = Variable(images_vs)
            # Forward propagation
            testing_time_tic = time.perf_counter()
            outputs_all = rapta_predict(images_l, images_d, images_c, images_vs)
            testing_time_toc = time.perf_counter()
            testing_time = testing_time_toc - testing_time_tic
            outputs = torch.flatten(outputs_all[0])
            outputs_l = torch.flatten(outputs_all[1])
            outputs_d = torch.flatten(outputs_all[2])
            outputs_c = torch.flatten(outputs_all[3])

            outputs_original_scale = sc_y.inverse_transform(outputs.cpu().detach().numpy().reshape(-1, 1)).flatten()
            outputs_l_original_scale = sc_y.inverse_transform(outputs_l.cpu().detach().numpy().reshape(-1, 1)).flatten()
            outputs_d_original_scale = sc_y.inverse_transform(outputs_d.cpu().detach().numpy().reshape(-1, 1)).flatten()
            outputs_c_original_scale = sc_y.inverse_transform(outputs_c.cpu().detach().numpy().reshape(-1, 1)).flatten()
            if shuffle_cond==True:
                labels_original_scale = sc_y.inverse_transform(labels.cpu().detach().numpy().reshape(-1, 1)).flatten()
            else:
                labels_original_scale = np.array(list(X_test_single[:, 5])).reshape((np.array(list(X_test_single[:, 5])).shape[0], -1)).flatten()
            pickle.dump(outputs_original_scale, open(config.model_chkpnt_path + fp + 'output.pk', 'wb'))
            pickle.dump(outputs_l_original_scale, open(config.model_chkpnt_path + fp + 'output_l.pk', 'wb'))
            pickle.dump(outputs_d_original_scale, open(config.model_chkpnt_path + fp + 'output_d.pk', 'wb'))
            pickle.dump(outputs_c_original_scale, open(config.model_chkpnt_path + fp + 'output_c.pk', 'wb'))
            pickle.dump(labels_original_scale, open(config.model_chkpnt_path + fp + 'labels.pk', 'wb'))
            pickle.dump(training_time, open(config.model_chkpnt_path + fp + 'training_time2.pk', 'wb'))
            pickle.dump(testing_time, open(config.model_chkpnt_path + fp + 'testing_time.pk', 'wb'))

            print(' Model saved')
            correct += error(outputs, labels)
            total += 1
        test_loss = correct/total
        print('Model from {}, Test data {}, Loss: {} '.format(exp_name, fp, test_loss))

def predict_from_chkpnt_cust():
    combined_setup_list = ['all_datasets',
                           'v_0.78_v_1.05',
                           'v_0.78_v_0.91_v_1.05',
                           'v_0.78_v_0.83_v_0.91_v_0.97_v_1.05',
                           'v_0.78_v_0.83_v_0.87_v_0.91_v_0.93_v_0.97_v_1.01_v_1.05']
    filepaths_train = [['v_0.78', 'v_0.79', 'v_0.81', 'v_0.83', 'v_0.85', 'v_0.87', 'v_0.89', 'v_0.91', 'v_0.93', 'v_0.95', 'v_0.97', 'v_0.99', 'v_1.01', 'v_1.03', 'v_1.05'],
                       ['v_0.78', 'v_1.05'],
                       ['v_0.78', 'v_0.91', 'v_1.05'],
                       ['v_0.78', 'v_0.83', 'v_0.91', 'v_0.97', 'v_1.05'],
                       ['v_0.78', 'v_0.83', 'v_0.87', 'v_0.91', 'v_0.93', 'v_0.97', 'v_1.01', 'v_1.05']]
    exp_name_list = ['expAll', 'exp2set', 'exp3set', 'exp5set', 'exp8set']

    # return [torch.flatten(outputs).tolist(), labels]
    for i in range(len(filepaths_train)):
        voltage_lists = filepaths_train[i]
        X_test = []
        for j in range(len(voltage_lists)):
            X_test_single = pickle.load(open(config.exp_data_path + voltage_lists[j] + '_test.pk', 'rb'))
            X_test += list(X_test_single)
        print('Lenght of test set {}'.format(len(X_test)))
        chkpoint_filename = config.model_chkpnt_path + combined_setup_list[i] + '_model_min_loss.ckpt'
        std_scalar_filename = config.model_chkpnt_path + combined_setup_list[i] + '_std_scaler_train.pk'
        std_y_filename = config.model_chkpnt_path + combined_setup_list[i] + '_sc_y_train.pk'
        training_time_filename = config.model_chkpnt_path + combined_setup_list[i] + '_training_time.pk'
        exp_name = exp_name_list[i]
        checkpoint = torch.load(chkpoint_filename)
        std_scaler_train =  pickle.load(open(std_scalar_filename, 'rb'))
        sc_y_train =  pickle.load(open(std_y_filename, 'rb'))
        training_time = pickle.load(open(training_time_filename, 'rb'))

        rapta_predict = model.RAPTA(config.input_dim, config.hidden_dim, config.layer_dim, config.output_dim)
        rapta_predict.load_state_dict(checkpoint)
        rapta_predict.to(device)
        rapta_predict.eval()  # to set dropout and batch
        print('Model loaded successfully')
        error = nn.MSELoss()  # mean Square root error; ((input-target)**2).mean()

        fp = exp_name_list[i]+'_'+combined_setup_list[i]+'_'
        seq_tensor_l_test, seq_tensor_d_test, seq_tensor_c_test, seq_tensor_vs_test, seq_lens_d_test, X_sub_test, target_test, std_scaler_test, sc_y_test = data_collector.tensor_maker(np.array(X_test), std_scaler_train, sc_y_train)

        sc_y = sc_y_test

        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_sub_test, seq_tensor_l_test, seq_tensor_d_test, seq_tensor_c_test,
                                           seq_tensor_vs_test,
                                           seq_lens_d_test, target_test),
            batch_size=config.batch_size, shuffle=True, drop_last=True)

        del X_sub_test, seq_tensor_l_test, seq_tensor_d_test, seq_tensor_c_test, seq_tensor_vs_test, seq_lens_d_test, target_test
        torch.cuda.empty_cache()

        # Calculate Accuracy
        correct = 0
        total = 0
        # Iterate through test dataset
        outputs_original_scale = []
        outputs_l_original_scale = []
        outputs_d_original_scale = []
        outputs_c_original_scale = []
        labels_original_scale = []
        testing_time = 0
        for all_sub, images_l, images_d, images_c, images_vs, _, labels in test_loader:
            _ = _.to(device)
            all_sub = all_sub.to(device)
            images_l = images_l.to(device)
            images_d = images_d.to(device)
            images_c = images_c.to(device)
            images_vs = images_vs.to(device)
            labels = labels.to(device)

            all_sub = Variable(all_sub)
            images_l = Variable(images_l)
            images_d = Variable(images_d)
            images_c = Variable(images_c)
            images_vs = Variable(images_vs)
            # Forward propagation
            testing_time_tic = time.perf_counter()
            outputs_all = rapta_predict(images_l, images_d, images_c, images_vs)
            testing_time_toc = time.perf_counter()
            testing_time += testing_time_toc - testing_time_tic
            outputs = torch.flatten(outputs_all[0])
            outputs_l = torch.flatten(outputs_all[1])
            outputs_d = torch.flatten(outputs_all[2])
            outputs_c = torch.flatten(outputs_all[3])

            outputs_original_scale += sc_y.inverse_transform(outputs.cpu().detach().numpy().reshape(-1, 1)).flatten().tolist()
            outputs_l_original_scale += sc_y.inverse_transform(outputs_l.cpu().detach().numpy().reshape(-1, 1)).flatten().tolist()
            outputs_d_original_scale += sc_y.inverse_transform(outputs_d.cpu().detach().numpy().reshape(-1, 1)).flatten().tolist()
            outputs_c_original_scale += sc_y.inverse_transform(outputs_c.cpu().detach().numpy().reshape(-1, 1)).flatten().tolist()
            labels_original_scale += sc_y.inverse_transform(labels.cpu().detach().numpy().reshape(-1, 1)).flatten().tolist()
            print(' Model saved')
            # Total number of labels
            total += labels.size(0)
            correct += error(outputs, labels)
            del _, all_sub, images_l, images_d, images_c, images_vs
            torch.cuda.empty_cache()
        pickle.dump(outputs_original_scale, open(config.model_chkpnt_path + fp + 'output.pk', 'wb'))
        pickle.dump(outputs_l_original_scale, open(config.model_chkpnt_path + fp + 'output_l.pk', 'wb'))
        pickle.dump(outputs_d_original_scale, open(config.model_chkpnt_path + fp + 'output_d.pk', 'wb'))
        pickle.dump(outputs_c_original_scale, open(config.model_chkpnt_path + fp + 'output_c.pk', 'wb'))
        pickle.dump(labels_original_scale, open(config.model_chkpnt_path + fp + 'labels.pk', 'wb'))
        pickle.dump(training_time, open(config.model_chkpnt_path + fp + 'training_time2.pk', 'wb'))
        pickle.dump(testing_time, open(config.model_chkpnt_path + fp + 'testing_time.pk', 'wb'))
        test_loss = correct/total
        print('Model from {}, Test data {}, Loss: {} '.format(exp_name, fp, test_loss))

def plotting_result(otpt, labels):
    predict_plt, = plt.plot(otpt, color='b', label='prediction')
    original_plt, = plt.plot(labels, color='g', label='labels')
    plt.legend([predict_plt, original_plt], ['prediction', 'labels'])
    plt.xlabel('path')
    plt.ylabel('time')
    plt.show()

if __name__ == "__main__":
    if config.running_option == 1:
        combined_setup_list = [config.fp]

        exp_name_list = [config.fp]
    else:
        combined_setup_list = ['v_0.78_', 'v_0.79_', 'v_0.81_', 'v_0.83_', 'v_0.85_', 'v_0.87_',
                               'v_0.89_', 'v_0.91_', 'v_0.93_', 'v_0.95_','v_0.97_', 'v_0.99_',
                               'v_1.01_', 'v_1.03_', 'v_1.05_', 'all_datasets_', 'v_0.78_v_1.05_',
                               'v_0.78_v_0.91_v_1.05_', 'v_0.78_v_0.83_v_0.91_v_0.97_v_1.05_',
                               'v_0.78_v_0.83_v_0.87_v_0.91_v_0.93_v_0.97_v_1.01_v_1.05_']

        exp_name_list = ['v_0.78', 'v_0.79', 'v_0.81', 'v_0.83', 'v_0.85', 'v_0.87',
                               'v_0.89', 'v_0.91', 'v_0.93', 'v_0.95','v_0.97', 'v_0.99',
                               'v_1.01', 'v_1.03', 'v_1.05','expAll', 'exp2set', 'exp3set', 'exp5set', 'exp8set']
        predict_from_chkpnt_cust()
    for i in range(len(exp_name_list)):
        print('Running for {}'.format(exp_name_list[i]))
        chkpoint_filename = config.model_chkpnt_path + combined_setup_list[i] + 'model_min_loss.ckpt'
        std_scalar_filename = config.model_chkpnt_path + combined_setup_list[i] + 'std_scaler_train.pk'
        std_y_filename = config.model_chkpnt_path + combined_setup_list[i] + 'sc_y_train.pk'
        training_time_filename = config.model_chkpnt_path + combined_setup_list[i] + 'training_time.pk'
        exp_name=exp_name_list[i]
        predict_from_chkpnt(chkpoint_filename, std_scalar_filename, std_y_filename, exp_name, training_time_filename)


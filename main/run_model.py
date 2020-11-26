import pickle
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import glob
import time
import configure as config
import data_collector
import model

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
    print("Number of GPU {}".format(torch.cuda.device_count()))
    print("GPU type {}".format(torch.cuda.get_device_name(0)))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')
    torch.backends.cudnn.benchmark = False
    # torch.cuda.memory_summary(device=None, abbreviated=False)
    # print(torch.backends.cudnn.version())
else:
    device = torch.device("cpu")
    print("Running on the CPU")
# device = torch.device("cpu")

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def main(X_train, X_valid, fp):  # do not need X, Y if we used data_collector.read_data() inside main function.
    print('Running Main for {}'.format(fp))
    seq_tensor_l_train, seq_tensor_d_train, seq_tensor_c_train, seq_tensor_vs_train, seq_lens_d_train, X_sub_train, target_train, std_scaler_train, sc_y_train = data_collector.tensor_maker(
        X_train, None, None)
    seq_tensor_l_valid, seq_tensor_d_valid, seq_tensor_c_valid, seq_tensor_vs_valid, seq_lens_d_valid, X_sub_valid, target_valid, std_scaler_valid, sc_y_valid = data_collector.tensor_maker(
        X_valid, std_scaler_train, sc_y_train)

    pickle.dump(std_scaler_train, open(config.model_chkpnt_path + fp + 'std_scaler_train.pk', 'wb'))
    pickle.dump(sc_y_train, open(config.model_chkpnt_path + fp + 'sc_y_train.pk', 'wb'))
    # data loader
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_sub_train, seq_tensor_l_train, seq_tensor_d_train, seq_tensor_c_train,
                                       seq_tensor_vs_train,
                                       seq_lens_d_train, target_train),
        batch_size=config.batch_size, shuffle=True, drop_last=True)  # 3 separate dataset, not merged
    # loading valid set by all
    valid_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_sub_valid, seq_tensor_l_valid, seq_tensor_d_valid, seq_tensor_c_valid, seq_tensor_vs_valid,
                                       seq_lens_d_valid, target_valid),
        batch_size=len(target_valid), shuffle=True, drop_last=True)

    # Define Model
    rapta = model.RAPTA(config.input_dim, config.hidden_dim, config.layer_dim, config.output_dim, config.dropout)
    rapta.to(device)

    rapta.apply(init_weights)
    # print("Model Summary \n---------------------------------")
    # print(rapta)

    error = nn.MSELoss()  # mean Square root error; ((input-target)**2).mean()
    optimizer = torch.optim.Adam(rapta.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    count = 0
    min_loss = None  # the lower the better

    train_loss_list = []
    valid_loss_list = []
    batch_epoch_time = 0
    training_time_tic = time.perf_counter()
    for epoch in range(config.number_of_epoch):

        for i, (all_sub, images_l, images_d, images_c, images_vs, cur_len, labels) in enumerate(train_loader):
            # print('batch {}'.format(i))

            batch_epoch_tic = time.perf_counter()

            # Clear gradients
            '''Since the backward() function accumulates gradients, and 
            you don’t want to mix up gradients between minibatches, 
            you have to zero them out at the start of a new minibatch. 
            This is exactly like how a general (additive) accumulator variable is 
            initialized to 0 in code.'''
            optimizer.zero_grad()

            all_sub = all_sub.to(device)
            images_l = images_l.to(device)
            images_d = images_d.to(device)
            images_c = images_c.to(device)
            images_vs = images_vs.to(device)
            cur_len = cur_len.to(device)
            labels = labels.to(device)
            # del all_sub, images_l, images_d, images_c, images_vs, cur_len, labels

            '''Usually we don’t need gradients in our input. 
            However, gradients in the input might be needed for some special use cases 
            e.g. creating adversarial samples.'''
            all_sub = Variable(all_sub, requires_grad=False)
            images_l = Variable(images_l, requires_grad=False)
            images_d = Variable(images_d, requires_grad=False)
            images_c = Variable(images_c, requires_grad=False)
            images_vs = Variable(images_vs, requires_grad=False)

            # Forward propagation
            all_outputs = rapta(images_l, images_d, images_c, images_vs)  # label, sub_label_l, sub_label_d, sub_label_c,
            outputs = torch.flatten(all_outputs[0])  # label
            outputs_l = torch.flatten(all_outputs[1])  # label
            outputs_d = torch.flatten(all_outputs[2])  # label
            outputs_c = torch.flatten(all_outputs[3])  # label

            # Calculate softmax and ross entropy loss
            loss_out = error(outputs, labels)  # labels in ns
            loss_l = error(outputs_l, all_sub[:, 0].view(all_sub.size()[0]))
            loss_d = error(outputs_d, all_sub[:, 1].view(all_sub.size()[0]))
            loss_c = error(outputs_c, all_sub[:, 2].view(all_sub.size()[0]))

            # Calculating auto gradients
            loss_seq = []
            loss_seq.append(loss_l)
            loss_seq.append(loss_d)
            loss_seq.append(loss_c)
            loss_seq.append(loss_out)
            train_loss_list.append(loss_out.tolist())
            torch.autograd.backward(loss_seq)

            # Update parameters
            optimizer.step()

            batch_epoch_toc = time.perf_counter()
            batch_epoch_time += (batch_epoch_toc - batch_epoch_tic)

            count += 1
            if count % 10 == 0:
                del cur_len, all_sub, images_l, images_d, images_c, images_vs, labels, loss_seq, loss_out, loss_l, loss_d, loss_c
                torch.cuda.empty_cache()
                correct = 0
                total = 0
                # Iterate through valid dataset
                for j, (all_sub, images_l, images_d, images_c, images_vs, _, labels) in enumerate(valid_loader):
                    _ = _.to(device)
                    all_sub = all_sub.to(device)
                    images_l = images_l.to(device)
                    images_d = images_d.to(device)
                    images_c = images_c.to(device)
                    images_vs = images_vs.to(device)
                    labels = labels.to(device)

                    all_sub = Variable(all_sub, requires_grad=False)
                    images_l = Variable(images_l, requires_grad=False)
                    images_d = Variable(images_d, requires_grad=False)
                    images_c = Variable(images_c, requires_grad=False)
                    images_vs = Variable(images_vs, requires_grad=False)
                    # Forward propagation
                    outputs_all = rapta(images_l, images_d, images_c, images_vs)
                    outputs = torch.flatten(outputs_all[0])

                    # Total number of labels
                    total += 1 # number of iteration/batch of valid set
                    correct += error(outputs, labels).item()
                    del _, all_sub, images_l, images_d, images_c, images_vs
                    torch.cuda.empty_cache()

                valid_loss = correct / total  # average for whole valid set

                # Print Loss
                print('Iteration: {}/{}  Current Loss: {} and previously recorded Min Loss {}'.format(epoch, count,
                                                                                                      valid_loss,
                                                                                                      min_loss))
                valid_loss_list.append(valid_loss)
                if min_loss == None:
                    min_loss = valid_loss
                    optimum_count = count
                    training_time_without_validset = (batch_epoch_time / count) * optimum_count
                    pickle.dump(training_time_without_validset,
                                open(config.model_chkpnt_path + fp + 'training_time_without_validset.pk', 'wb'))
                elif valid_loss < min_loss:
                    min_loss = valid_loss
                    optimum_count = count
                    training_time_without_validset = (batch_epoch_time / count) * optimum_count
                    pickle.dump(training_time_without_validset,
                                open(config.model_chkpnt_path + fp + 'training_time_without_validset.pk', 'wb'))

                    # Save the model checkpoint
                    training_time_toc = time.perf_counter()
                    training_time = training_time_toc-training_time_tic
                    pickle.dump(training_time, open(config.model_chkpnt_path + fp + 'training_time.pk', 'wb'))
                    torch.save(rapta.state_dict(), config.model_chkpnt_path + fp + 'model_min_loss.ckpt')
                    print(' Model saved')


    avg_lr_time = (batch_epoch_time / config.number_of_epoch)
    pickle.dump(avg_lr_time, open(config.model_chkpnt_path + fp + 'avg_lr_time.pk', 'wb'))

    print('Minimum loss {}'.format(min_loss))
    print('Total Learning time {}, Learning time without validation {}, Average Learning time {}'.format(training_time, training_time_without_validset, avg_lr_time))
    pickle.dump(train_loss_list, open(config.model_chkpnt_path + fp + 'train_loss.pk', 'wb'))
    pickle.dump(valid_loss_list, open(config.model_chkpnt_path + fp + 'valid_loss.pk', 'wb'))


def multiF_sepR():  # multiple files separate run
    filelist_X_train = sorted(glob.glob(config.X_train_path_multi))
    print(len(filelist_X_train))
    for i in range(len(filelist_X_train)):
        # reading processed data
        X_train_org = pickle.load(open(filelist_X_train[i], 'rb'))
        X_train, X_valid = train_test_split(X_train_org, test_size=0.2, random_state=42,
                                                          shuffle=True)  # Split the set
        fp = filelist_X_train[i].split('\\')[-1][0:6] + '_'
        # running model
        print('Starting files set {}'.format(fp))
        print('Train file {}'.format(filelist_X_train[i].split('\\')[-1]))
        tic = time.perf_counter()
        main(X_train, X_valid, fp)
        toc = time.perf_counter()
        print('Finished files set {}'.format(fp))
        model_time = (toc - tic)
        print("Model took {} seconds".format(model_time))
        pickle.dump(model_time, open(config.model_chkpnt_path + fp + 'model_time.pk', 'wb'))

def multiF_singR():  # multiple files single run
    filelist_X_train = sorted(glob.glob(config.X_train_path_multi))
    X_train = []
    X_valid = []
    for i in range(len(filelist_X_train)):
        # reading processed data
        X_train_org = pickle.load(open(filelist_X_train[i], 'rb'))
        X_train_single, X_valid_single = train_test_split(X_train_org, test_size=0.2, random_state=42,
                                                          shuffle=True)  # Split the set
        X_train += list(X_train_single)
        X_valid += list(X_valid_single)
    print(len(X_train))
    print(len(X_valid))
    # running model
    tic = time.perf_counter()
    main(np.array(X_train), np.array(X_valid), 'all_datasets_')
    toc = time.perf_counter()
    model_time = (toc - tic)
    print("Model took {} seconds".format(model_time))
    pickle.dump(model_time, open(config.model_chkpnt_path + 'all_datasets' + '_model_time.pk', 'wb'))

def multiF_custR():
    for i in range(len(config.filepaths_train)):
        voltage_lists = config.filepaths_train[i]
        X_train = []
        X_valid = []
        for j in range(len(voltage_lists)):
            X_train_org = pickle.load(open(config.exp_data_path + voltage_lists[j] + '_train.pk', 'rb'))
            X_train_single, X_valid_single = train_test_split(X_train_org, test_size=0.2, random_state=42,
                                                shuffle=True)  # Split the set
            X_train += list(X_train_single)
            X_valid += list(X_valid_single)
        print(len(X_train))
        print(len(X_valid))
        # running model
        tic = time.perf_counter()
        main(np.array(X_train), np.array(X_valid), '_'.join(voltage_lists) + '_')
        toc = time.perf_counter()
        model_time = (toc - tic)
        print("Model took {} seconds".format(model_time))
        pickle.dump(model_time, open(config.model_chkpnt_path + '_'.join(voltage_lists) + '_model_time.pk', 'wb'))


if __name__ == "__main__":
    if config.running_option == 1:  # single file single run
        # reading processed data
        X_train_org = pickle.load(open(config.X_train_path_single, 'rb'))
        X_train, X_valid = train_test_split(X_train_org, test_size=0.2, random_state=42, shuffle=True)  # Split the set
        print(len(X_train))
        # running model
        tic = time.perf_counter()
        main(np.array(X_train), np.array(X_valid), config.fp)
        toc = time.perf_counter()
        print("Model took {} seconds".format((toc - tic)))
    elif config.running_option == 2:  # multiple files separate run
        multiF_sepR()
    elif config.running_option == 3:  # multiple files single run
        multiF_singR()
    elif config.running_option == 4:  # multiple files single run customized
        multiF_custR()
    elif config.running_option == 5:  # from experiment_setup
        print('Running Experiment 0~15')
        multiF_sepR()
        print('Running Experiment 16')
        multiF_singR()
        print('Running Experiment 17~20')
        multiF_custR()





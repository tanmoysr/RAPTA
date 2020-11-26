
appl_bench = 'PT_Ethernet' # PT_S38417, PT_AES128, PT_Ethernet, PD_S38417, PD_AES128, PD_Ethernet
appl_bench_model = appl_bench+'_RAPTA'

processed_saved_path = '../data/original_data/'+appl_bench+'/'
exp_data_path = '../data/model_purpose/'+appl_bench+'/'
gate_type_list = ['0','AND', 'AO', 'AOI', 'DELLN', 'DFF', 'DFFAR', 'DFFAS',
                  'IBUFF', 'INV', 'LATCH', 'MUX', 'NAND', 'NBUFF', 'NOR', 'OA',
                  'OAI', 'OR', 'SDFF', 'SDFFAR', 'SDFFAS', 'TNBUFF', 'XNOR', 'XOR']
'''Running Options
1 = single file
2 = multiple files separate run
3 = multiple files single run
4 = choosing manually multiple files
5 = run 2, 3, 4
6 = creating data for bilstm from processed file
7 = creating data for base model and bilstm'''
running_option = 8

fp = 'v_1.05_' # processed input voltage
X_train_path_single = '../data/model_purpose/'+appl_bench+'/v_1.05_train.pk'
Y_test_path_single = '../data/single_file/v_0.78_train.pk'
X_sub_data_path_single = '../data/single_file/X_sublabels.pk'
X_train_path_single_base = '../data/model_purpose/'+appl_bench+'/v_0.78_train_base.pk'

X_train_path_multi = '../data/model_purpose/'+appl_bench+'/*_train.pk'
X_test_path_multi = '../data/model_purpose/'+appl_bench+'/*_test.pk'
X_train_path_multi_base = '../data/model_purpose/'+appl_bench+'/*_train_base.pk'
X_test_path_multi_base = '../data/model_purpose/'+appl_bench+'/*_test_base.pk'
X_data_path = '../data/model_purpose/'+appl_bench+'/'
X_data_path_multi = '../data/model_purpose/'+appl_bench+'/*_X_gate.pk'
Y_data_path_multi = '../data/model_purpose/'+appl_bench+'/*_Y_gate.pk'
X_sub_data_path_multi = '../data/model_purpose/'+appl_bench+'/*_X_sublabels.pk'

filepaths = ['v_0.78', 'v_0.79', 'v_0.81', 'v_0.83', 'v_0.85', 'v_0.87', 'v_0.89', 'v_0.91', 'v_0.93', 'v_0.95',
             'v_0.97', 'v_0.99', 'v_1.01', 'v_1.03', 'v_1.05']
filepaths_train = [['v_0.78', 'v_1.05'],['v_0.78', 'v_0.91', 'v_1.05'],['v_0.78', 'v_0.83', 'v_0.91', 'v_0.97', 'v_1.05'],['v_0.78', 'v_0.83', 'v_0.87', 'v_0.91', 'v_0.93', 'v_0.97', 'v_1.01','v_1.05']]

model_chkpnt_path = '../saved_model/'+appl_bench_model+'/'
model_chkpnt_path_multi = '../saved_model/'+appl_bench_model+'/*model_min_loss.ckpt'
timing_option = 1 # 0 = inactive, 1 = active; used in utility file

# for utility file configuration
label_file = '../saved_model/'+appl_bench_model+'/'+fp+'labels.pk'
output_file = '../saved_model/'+appl_bench_model+'/'+fp+'output.pk'
label_file_multi = '../saved_model/'+appl_bench_model+'/*_labels.pk'
output_file_multi = '../saved_model/'+appl_bench_model+'/*_output.pk'
label_file_multi = '../saved_model/'+appl_bench_model+'/*_labels.pk'
output_file_multi = '../saved_model/'+appl_bench_model+'/*_output.pk'
label_file_multi = '../saved_model/'+appl_bench_model+'/*_labels.pk'
output_file_multi = '../saved_model/'+appl_bench_model+'/*_output.pk'
label_file_multi = '../saved_model/'+appl_bench_model+'/*_labels.pk'
output_file_multi = '../saved_model/'+appl_bench_model+'/*_output.pk'

learning_time_multi = '../saved_model/'+appl_bench_model+'/*_avg_lr_time.pk'
training_time_multi = '../saved_model/'+appl_bench_model+'/*training_time2.pk'
testing_time_multi = '../saved_model/'+appl_bench_model+'/*_testing_time.pk'


# Model Parameter
total_features=46
original_input_dim = 46
input_dim = 65  # input dimension 24, 43, 44, modelV5:45, mlp:138, 65
hidden_dim = 69  # hidden layer dimension 32, modelV5:32, mlp:69
layer_dim = 3  # number of hidden layers 2, modelV5: 2, hot_encoded: 3
output_dim = 1  # output dimension
activation1 = 'selu' #'selu', 'relu', 'rrelu', 'tanh', 'tanhshrink', 'softmax'
activation2 = 'selu'
batch_size = 128 # 128
learning_rate = 0.001 #0.001
weight_decay=0.00001 #  0.00001. Weight decay applies L2 regularization to the learned parameters
dropout = 0.0001 # 0.0001. 0 means no output, all dropped; 1 means no dropout at all
number_of_epoch = 300 # 300
stop_after_no_change = 60 #30



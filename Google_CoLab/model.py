
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
else:
    device = torch.device("cpu")

# device = torch.device("cpu")

class RAPTA(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout=0.1):
        super(RAPTA, self).__init__()
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim
        self.dropout = dropout

        # RNN
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=True)
        # tensor containing the output features (h_t) from the last layer of the LSTM

        # Readout layer
        self.fc = nn.Linear(hidden_dim * 4, hidden_dim)  # 2 for bidirection
        self.fc1 = nn.Linear((hidden_dim * 3)+1, hidden_dim * 2)  # 3 for concatenating 3 bilstm + voltage
        self.fc2 = nn.Linear(hidden_dim*2, output_dim)  # layer conversion
        self.fc3 = nn.Linear(hidden_dim + 1, output_dim)  # layer conversion

        # ----
        self.fc_l = nn.Linear(hidden_dim * 4*2, hidden_dim)  # 2 for bidirection
        self.fc_d = nn.Linear(hidden_dim * 28*2, hidden_dim)  # 2 for bidirection
        self.fc_c = nn.Linear(hidden_dim * 4*2, hidden_dim)  # 2 for bidirection

    def forward(self, x_l, x_d, x_c, x_vs):

        # Set initial states
        h0_l = torch.zeros(self.layer_dim * 2, x_l.size(0), self.hidden_dim).to(device)  # 2 for bidirection
        c0_l = torch.zeros(self.layer_dim * 2, x_l.size(0), self.hidden_dim).to(device)
        # print(h0_l.size()) #[layer*2, hidden_dimension, batch_size]
        h0_d = torch.zeros(self.layer_dim * 2, x_d.size(0), self.hidden_dim).to(device)  # 2 for bidirection
        c0_d = torch.zeros(self.layer_dim * 2, x_d.size(0), self.hidden_dim).to(device)

        h0_c = torch.zeros(self.layer_dim * 2, x_c.size(0), self.hidden_dim).to(device)  # 2 for bidirection
        c0_c = torch.zeros(self.layer_dim * 2, x_c.size(0), self.hidden_dim).to(device)

        # Forward propagate LSTM
        out_l_a, _l = self.rnn(x_l, (h0_l, c0_l))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        out_d_a, _d = self.rnn(x_d, (h0_d, c0_d))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        out_c_a, _c = self.rnn(x_c, (h0_c, c0_c))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # merging two cell final stages of forward and reverse path
        # Activation layer 1
        #-----------selu-------------
        out_l = F.selu(self.fc(torch.cat((out_l_a[:, -1, :], out_l_a[:, 0, :]), dim=1)))
        out_d = F.selu(self.fc(torch.cat((out_d_a[:, -1, :], out_d_a[:, 0, :]), dim=1)))
        out_c = F.selu(self.fc(torch.cat((out_c_a[:, -1, :], out_c_a[:, 0, :]), dim=1)))

        sub_l = self.fc3(torch.cat((out_l, x_vs[:, 0].view(x_vs.size()[0], 1)), dim=1))  # voltage
        sub_d = self.fc3(torch.cat((out_d, x_vs[:, 0].view(x_vs.size()[0], 1)), dim=1))  # voltage
        sub_c = self.fc3(torch.cat((out_c, x_vs[:, 0].view(x_vs.size()[0], 1)), dim=1))  # voltage

        # concatenating without sublabels
        out_ld = torch.cat((out_l, out_d), dim=1)
        out_ldc = torch.cat((out_ld, out_c), dim=1)
        out_ldcvs = torch.cat((out_ldc, x_vs[:,0].view(x_vs.size()[0],1)), dim=1) # voltage

         # Activation layer 2
        out = F.selu(self.fc1(out_ldcvs))  # activation layer
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.fc2(out)  # output layer
        # print(out.size()) #[hidden, 1]

        del h0_l, c0_l, h0_d, c0_d, h0_c, c0_c, out_l_a, out_d_a, out_c_a, out_l, out_d, out_c, out_ld, out_ldc, out_ldcvs
        torch.cuda.empty_cache()
        # return out
        return out, sub_l, sub_d, sub_c
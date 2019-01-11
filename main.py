# Main file, on the which to set the parameters and call the model

import os
import csv
import time
import numpy as np 
import torch

from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from models import lstm_v1


# Parameters


# data 
dataset = "PTB"
data_path = "data/{}/".format(dataset)


# model architecture
num_layers = 2
n_emb_lstm = 200
nhid = 400
nhidlast = 200


# training
sequence_length = 35
batch_size = 12 # PTB: 12, WT2: 15
eval_batch_size = 5
test_batch_size = 1
epochs = 50
log_interval = 200
CUDA = True


# optimization
lr = 20 # PTB: 20, WT2: 15
clipping = 0.25


# Regularization tricks in Neural Language Modeling

# Trick 1: early stopping
# Note that we perform early stopping by default by only saving the model if the validation perplexity is improving.
stopping_criteria = "plateau"
plateau = 10

# Trick 2: Learning rate decay
# When validation perplexity augments by a certain threshold (lr_decay_thresh), we anneal the learning rate (by lr_decay), as long as it stays above a certain value (min_lr)
lr_decay_thres = 1
lr_decay = 1
min_lr = 0

# Trick 3: Dropout rate(s)
dropout = 0.0 # input layer dropout
var_rnn = False
rec_dropout = 0.6

# Trick 4: tying the encoder and decoder weights
tie_weights = False


# exporting
saving = True
saving_path = "../saved_models/language_modeling/lstm/{}_epochs_{}_plateau_{}_lr_decay_{}_dropout_{}_var_rnn_{}_tie_weights_{}".format(dataset, epochs, plateau, lr_decay, dropout, var_rnn, tie_weights)




if __name__ == '__main__':

    torch.manual_seed(1111)

    lstm_v1(data_path, saving, saving_path, num_layers, n_emb_lstm, nhid, nhidlast, 
        sequence_length, batch_size, eval_batch_size, test_batch_size, epochs, log_interval, CUDA, lr, clipping, 
        stopping_criteria, plateau, lr_decay_thres, lr_decay, min_lr, dropout, var_rnn, rec_dropout, tie_weights)
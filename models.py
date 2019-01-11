# Classical models (LSTM, RNN-relu, etc)
# inspired by:
# https://github.com/pytorch/examples/tree/master/word_language_model

import os
import csv
import time
import numpy as np 
import torch
import torch.utils.data

from sklearn.metrics import roc_auc_score as auc
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from data import Corpus
from utils import repackage_hidden, batchify, get_batch
from nets import SimpleLSTM


def lstm_v1(data_path, saving, saving_path, num_layers, n_emb_lstm, nhid, nhidlast, 
        sequence_length, batch_size, eval_batch_size, test_batch_size, epochs, log_interval, CUDA, lr, clipping, 
        stopping_criteria, plateau, lr_decay_thres, lr_decay, min_lr, dropout, var_rnn, rec_dropout, tie_weights):


    # Data preparation
    corpus = Corpus(data_path)
    train_data = batchify(corpus.train, batch_size, CUDA)
    print("******************")
    print(train_data.size(0))
    val_data = batchify(corpus.valid, eval_batch_size, CUDA)
    test_data = batchify(corpus.test, test_batch_size, CUDA)

    # Parameters
    num_words = len(corpus.dictionary)
    print("Number of words: {}".format(num_words))
    n_train_batches = int(train_data.size(0)/sequence_length)
    n_val_batches = int(val_data.size(0)/sequence_length)
    n_test_batches = int(test_data.size(0)/sequence_length)

    # Model
    model = SimpleLSTM(num_layers, n_emb_lstm, nhid, nhidlast, num_words, batch_size, CUDA, dropout, var_rnn, rec_dropout, tie_weights)
    if CUDA:
        model = model.cuda()
    total_params = sum(x.data.nelement() for x in model.parameters())
    print("Model total number of parameters: {}".format(total_params))

    # Loss function
    criterion = nn.CrossEntropyLoss()

    valid_perp = []
    test_perp = []


    # TRAINING LOOP

    for epoch in tqdm(range(epochs)):

        # Annealing the learning rate
        if (epoch >= 2):
            if not(valid_perp[-1] < lr_decay_thres * valid_perp[-2]):
                if (lr > min_lr):
                    print("Annealing the learning rate")
                    lr *= lr_decay
                    print("New LR: {}".format(lr))

        start = time.time()

        print("****New epoch: epoch {} out of {} *****".format(epoch+1,epochs))

        ## TRAINING ROUND

        model.train()
        all_train_loss = 0
        train_loss = 0
        idx = 0

        times = []

        hidden = model.init_hidden(batch_size)

        means = []
        stds = []

        for batch, i in enumerate(range(0, train_data.size(0) - 1, sequence_length)):

            t1 = time.time()

            data, targets = get_batch(train_data, sequence_length, i)

            hidden = repackage_hidden(hidden)
            model.zero_grad()

            output, raw_output, hidden, m, s = model.forward_fc(data, hidden)
            means.append(m.cpu().detach().numpy())
            stds.append(s.cpu().detach().numpy())

            pre_loss = criterion(output.view(-1, num_words), targets)
            loss = 1 * pre_loss

            train_loss += pre_loss.cpu().detach().numpy()
            all_train_loss += pre_loss.cpu().detach().numpy()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)
            for p in model.parameters():
                p.data.add_(-lr, p.grad.data)

            idx += 1
            t8 = time.time()
            times.append(t8-t1)

            if idx % log_interval == 0:

                train_loss /= log_interval
                print("LSTM training, batch: {} out of {}, training loss: {:0.3f}, training perplexity: {:0.3f}".format(idx, n_train_batches, train_loss, np.exp(train_loss)))
                train_loss = 0
                print("Average time per batch: {:0.4f}".format(np.mean(np.array(times))))
                times = []

        print("Average training loss in this epoch: {:0.3f}, average training perplexity: {:0.3f}".format(all_train_loss/n_train_batches, np.exp(all_train_loss/n_train_batches)))
        print("Average mean of last layer, training: {:0.4f}".format(np.mean(np.array(means))))
        print("Average std of last layer, training: {:0.4f}".format(np.mean(np.array(stds))))

        # VALIDATION ROUND

        print("*****Validation round*****")

        model.eval()
        all_val_loss = 0
        hidden = model.init_hidden(eval_batch_size)

        means = []
        stds = []

        with torch.no_grad():

            for i in range(0, val_data.size(0) - 1, sequence_length):

                data, targets = get_batch(val_data, sequence_length, i)

                output, raw_output, hidden, m, s = model.forward_fc(data, hidden)
                means.append(m.cpu().detach().numpy())
                stds.append(s.cpu().detach().numpy())
                output_flat = output.view(-1, num_words)
                loss = criterion(output_flat, targets)
                hidden = repackage_hidden(hidden)

                all_val_loss += loss.cpu().detach().numpy()

        all_val_loss /= (n_val_batches)
        perp = np.exp(all_val_loss)

        print("LSTM testing - full epoch, val loss: {:0.3f}, validation perplexity: {:0.3f}".format(all_val_loss, perp))
        print("Average mean of last layer, validation: {:0.4f}".format(np.mean(np.array(means))))
        print("Average std of last layer, validation: {:0.4f}".format(np.mean(np.array(stds))))

        # EARLY-STOPPING CHECK

        if epoch > (plateau + 1):
            if stopping_criteria == "plateau":
                if perp > min(valid_perp[:-plateau]):
                    break

        # EXPORTING

        if saving:
            if epoch > 0:
                if perp < np.min(np.array(valid_perp)):
                    torch.save(model, saving_path)
                    print("model saved !")

        valid_perp.append(perp)

        # TESTING round

        print("*****Testing round*****")

        model.eval()
        all_test_loss = 0
        hidden = model.init_hidden(test_batch_size)

        with torch.no_grad():

            for i in range(0, test_data.size(0) - 1, sequence_length):

                data, targets = get_batch(test_data, sequence_length, i)

                output, raw_output, hidden, m, s = model.forward_fc(data, hidden)
                output_flat = output.view(-1, num_words)
                loss = criterion(output_flat, targets)
                hidden = repackage_hidden(hidden)

                all_test_loss += loss.cpu().detach().numpy()

        all_test_loss /= (n_test_batches)
        t_perp = np.exp(all_test_loss)
        print("LSTM testing - full epoch, val loss: {:0.3f}, test perplexity: {:0.3f}".format(all_test_loss, t_perp))

        test_perp.append(t_perp)


    # TRAINING SUMMARY

    best_idx = np.argmin(np.array(valid_perp))
    print("Best validation perplexity after: {} epochs".format(best_idx + 1))
    print("Best validation perplexity: {}".format(np.array(valid_perp)[best_idx]))
    print("Corresponding test perplexity: {}".format(np.array(test_perp)[best_idx]))

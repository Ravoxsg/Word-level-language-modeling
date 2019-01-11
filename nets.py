# The neural nets

import os
import torch
import torch.utils.data

from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F



class SimpleLSTM(nn.Module):


    def __init__(self, num_layers, n_emb, nhid, nhidlast, num_words, batch_size, CUDA, dropout, var_rnn = False, rec_dropout = 0.2, tie_weights = False):

        super(SimpleLSTM, self).__init__()

        # Architecture
        self.num_layers = num_layers
        self.n_emb = n_emb
        self.nhid = nhid
        self.nhidlast = nhidlast
        self.num_words = num_words

        self.bs = batch_size
        self.cud = CUDA

        self.dropout = dropout
        self.var_rnn = var_rnn
        self.rec_dropout = rec_dropout

        # Activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim = 1)

        # LAYERS
        self.emb = nn.Embedding(self.num_words, self.n_emb)
        self.d1 = nn.Dropout(p = self.dropout)
        #self.l1 = nn.LSTM(input_size = self.latent_size, hidden_size = self.latent_size, num_layers = self.num_layers, bidirectional = self.bi, dropout = self.dropout)
        self.rnns = [torch.nn.LSTM(self.n_emb if l == 0 else self.nhid, self.nhid if l != self.num_layers - 1 else self.nhidlast, 1, dropout=0) for l in range(self.num_layers)]
        self.rnns = torch.nn.ModuleList(self.rnns)

        self.fc1 = nn.Linear(self.nhidlast, self.num_words)

        if tie_weights:
            self.fc1.weight = self.emb.weight

        if self.var_rnn:
            self.sample_mask()

        self.init_weights()


    def init_weights(self):

        initrange = 0.1
        self.emb.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc1.weight.data.uniform_(-initrange, initrange)


    def init_hidden(self, bs):

        weight = next(self.parameters())

        # return (weight.new_zeros(self.num_layers, bs, self.latent_size),
        #             weight.new_zeros(self.num_layers, bs, self.latent_size))

        return [(Variable(weight.new(1, bs, self.nhid if l != self.num_layers - 1 else self.nhidlast).zero_()),
                 Variable(weight.new(1, bs, self.nhid if l != self.num_layers - 1 else self.nhidlast).zero_()))
                for l in range(self.num_layers)]


    def sample_mask(self):

        keep = 1.0 - self.rec_dropout
        self.mask = [Variable(torch.bernoulli(torch.Tensor(self.nhid).fill_(keep)))
                    for l in range(self.num_layers)]
        if self.cud:
            self.mask = [Variable(torch.bernoulli(torch.Tensor(self.nhid).fill_(keep))).cuda()
                            for l in range(self.num_layers)]


    def forward_fc(self, x, hidden):

        emb = self.d1(self.emb(x))

        #output, hidden = self.l1(emb, hidden)

        input = emb
        new_hidden = []
        for l, rnn in enumerate(self.rnns):
            output, new_h = rnn(input, hidden[l])
            new_hidden.append(new_h)
            input = output
        hidden = new_hidden

        if self.var_rnn:
            for l in range(self.num_layers):
                hidden[0][l].data.set_(torch.mul(hidden[0][l], self.mask[l]).data)
                hidden[0][l].data *= 1.0/(1.0 - self.rec_dropout)

        m = torch.mean(output)
        s = torch.std(output)
        decoded = self.fc1(output.view(output.size(0) * output.size(1), output.size(2)))

        return decoded.view(output.size(0), output.size(1), decoded.size(1)), output, hidden, m, s


    def forward_dot(self, x):

        h1, (hn, cn) = self.l1(x)

        return hn[-1]




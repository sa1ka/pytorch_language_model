#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
from data import DataIter, Dictionary
from torch.autograd import Variable
import pickle
import numpy as np
import pdb
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

def length2mask(length):
    mask = torch.ByteTensor(len(length), max(length)).zero_().cuda()
    for i, l in enumerate(length):
        mask[i][:l].fill_(1)
    return Variable(mask)

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout, batch_first=True)
        self.decoder = Decoder(nhid, ntoken)
        self.rnn_type = 'LSTM'

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.init_weights()

    def forward(self, input, length=None):
        emb = self.drop(self.encoder(input))
        if length is not None:
            emb = pack_padded_sequence(emb, length, batch_first=True)
        output, hidden = self.rnn(emb)
        if isinstance(output, PackedSequence):
            output, _  = pad_packed_sequence(output, batch_first=True)
        output = self.drop(output)
        return output

    def loss(self, data):
        input, target, length = data
        output = self(input, length)
        data = [output] + data[1:]
        return self.decoder.forward_with_loss(*data)

class Decoder(nn.Module):
    def __init__(self, nhid, ntoken):
        super(Decoder, self).__init__()
        self.nhid = nhid
        self.decoder = nn.Linear(nhid, ntoken)
        self.criterion = nn.CrossEntropyLoss()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        return self.decoder(input)

    def forward_all(self, input):
        input = input.view(-1, self.nhid)
        return self(input)

    def forward_with_loss(self, input, target, length):
        mask = length2mask(length)
        input = input.masked_select(
            mask.unsqueeze(dim=2).expand_as(input)
        )
        output = self(input.view(-1, self.nhid))
        target = target.masked_select(mask)
        return self.criterion(output, target)



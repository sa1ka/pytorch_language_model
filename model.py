#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable

import pickle
import pdb
from utils import length2mask

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, decoder, dropout=0.5):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout, batch_first=True)

        self.decoder = decoder

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.init_weights()

    def forward_rnn(self, input, length=None):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb)
        output = self.drop(output)
        return output

    def forward(self, data):

        # forward rnn
        input, target, length = data
        rnn_output = self.forward_rnn(input, length)

        # discard the pad
        mask = length2mask(length)
        rnn_output = rnn_output.masked_select(
            mask.unsqueeze(dim=2).expand_as(rnn_output)
        ).view(-1, self.nhid)
        target = target.masked_select(mask)

        # forward decoder and calculate loss
        decoder_loss = self.decoder.forward_with_loss(rnn_output, target)

        return decoder_loss

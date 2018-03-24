#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable

import pdb
from collections import defaultdict
from utils import list2longtensor, map_dict_value
from alias_multinomial import AliasMethod

class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

class SMDecoder(nn.Module):
    def __init__(self, nhid, ntoken):
        super(SMDecoder, self).__init__()
        self.nhid = nhid
        self.decoder = nn.Linear(nhid, ntoken)
        self.CE = nn.CrossEntropyLoss()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        return self.decoder(input)

    def forward_with_loss(self, rnn_output, target):
        output = self(rnn_output)
        return self.CE(output, target)

class ClassBasedSMDecoder(nn.Module):
    def __init__(self, nhid, ncls, word2cls, class_chunks):
        super(ClassBasedSMDecoder, self).__init__()
        self.nhid = nhid
        self.cls_decoder = nn.Linear(nhid, ncls)

        words_decoders = []
        for c in class_chunks:
            words_decoders.append(nn.Linear(nhid, c))
        self.words_decoders = ListModule(*words_decoders)

        self.CELoss = nn.CrossEntropyLoss(size_average=False)

        # collect word in the same class
        cls_cluster = defaultdict(lambda: [])

        # the within index of each words in their word cluster
        within_cls_idx = []
        for i, c in enumerate(word2cls):
            within_cls_idx.append(len(cls_cluster[c]))
            cls_cluster[c].append(i)

        self.word2cls = list2longtensor(word2cls)
        self.within_cls_idx = list2longtensor(within_cls_idx)
        self.cls_cluster = map_dict_value(list2longtensor, cls_cluster)

    def init_weights(self):
        r = .1
        self.cls_decoder.weight.data.uniform_(-r, r)
        self.cls_decoder.bias.data.fill_(0)
        for word_decoder in self.words_decoders:
            word_decoder.weight.data.uniform_(-r, r)
            word_decoder.bias.data.fill_(0)

    def build_labels(self, target):
        #TODO: too much time is wasted in this function

        # cls idx of each word
        cls_idx = self.word2cls.index_select(0, target)
        # word within class idx of each word
        within_cls_idx = self.within_cls_idx.index_select(0, target)

        cls_idx_ = cls_idx.data.cpu()
        wci = within_cls_idx.data.cpu()

        # collect the batch index of words in the same class
        within_batch_idx_dic = defaultdict(lambda: [])
        # collect the within index of words in the same class
        within_cls_idx_dic = defaultdict(lambda: [])

        for i, (c, w) in enumerate(zip(cls_idx_, wci)):
            within_batch_idx_dic[c].append(i)
            within_cls_idx_dic[c].append(w)

        within_batch_idx_dic = map_dict_value(list2longtensor, within_batch_idx_dic)
        within_cls_idx_dic = map_dict_value(list2longtensor, within_cls_idx_dic)

        return cls_idx, within_cls_idx_dic, within_batch_idx_dic

    def forward(self, input, within_batch_idx):
        p_class = self.cls_decoder(input)
        p_words = {}

        for c in within_batch_idx:
            d = input.index_select(0, within_batch_idx[c])
            p_words[c] = self.words_decoders[c](d)

        return p_class, p_words

    def forward_with_loss(self, rnn_output, target):

        cls_idx, within_cls_idx, within_batch_idx = self.build_labels(target)

        p_class, p_words = self(rnn_output, within_batch_idx)

        # by applying log function, the product of class prob and word prob can be break down,
        # hence we can calculate the class and word CE loss respectively.

        closs = self.CELoss(p_class, cls_idx)
        wloss = []
        for c in p_words:
            wloss.append(self.CELoss(p_words[c], within_cls_idx[c]))

        return (closs + sum(wloss)) / len(cls_idx)

class NCEDecoder(nn.Module):
    def __init__(self, nhid, ntoken, noise_dist, nsample=10):
        super(NCEDecoder, self).__init__()
        self.nhid = nhid
        self.word_embeddings = nn.Embedding(ntoken, nhid)
        self.word_bias = nn.Embedding(ntoken, 1)

        noise_dist = noise_dist / noise_dist.sum()
        self.noise_dist = noise_dist.cuda()
        self.alias = AliasMethod(self.noise_dist)
        self.nsample = nsample
        self.norm = 9

        self.CE = nn.CrossEntropyLoss()
        self.valid = False

    def init_weights(self):
        initrange = 0.1
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)
        self.word_bias.weight.data.fill_(0)

    def _get_noise_prob(self, indices):
        return Variable(self.noise_dist[indices.data.view(-1)].view_as(indices))

    def forward(self, input, target):
        #model prob for target and sample words

        sample = Variable(self.alias.draw(input.size(0), self.nsample).cuda())
        indices = torch.cat([target.unsqueeze(1), sample], dim=1)

        embed = self.word_embeddings(indices)
        bias = self.word_bias(indices)

        score = torch.baddbmm(1, bias, 1, embed, input.unsqueeze(2)).squeeze()
        score = score.sub(self.norm).exp()
        target_prob, sample_prob = score[:, 0], score[:, 1:]

        return target_prob, sample_prob, sample

    def nce_loss(self, target_prob, sample_prob, target, sample):
        target_noise_prob = self._get_noise_prob(target)
        sample_noise_prob = self._get_noise_prob(sample)

        def log(tensor):
            EPSILON = 1e-10
            return torch.log(EPSILON + tensor)

        target_loss = log(
            target_prob / (target_prob + self.nsample * target_noise_prob)
        )

        sample_loss = log(
            self.nsample * sample_noise_prob / (sample_prob + self.nsample * sample_noise_prob)
        )

        return - (target_loss + torch.sum(sample_loss, -1).squeeze())

    def forward_with_loss(self, rnn_output, target):

        if self.training:
            target_prob, sample_prob, sample = self(rnn_output, target)
            loss = self.nce_loss(target_prob, sample_prob, target, sample)
            return loss.mean()
        else:
            output = torch.addmm(
                1, self.word_bias.weight.view(-1), 1, rnn_output, self.word_embeddings.weight.t()
            )
            return self.CE(output, target)

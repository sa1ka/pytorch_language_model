import os
import torch
from torch.autograd import Variable
from tree.gauss_tree import GaussClassTree
from collections import defaultdict
import pdb
import numpy as np
import codecs


UNK = '<unk>'
PAD = '<pad>'
BOS = '<bos>'
EOS = '<eos>'

class Dictionary(object):
    def __init__(self, vocab_path):
        self.word2idx = {}
        self.idx2word = []
        self.add_word(UNK)
        self.add_word(PAD)
        self.add_word(BOS)
        self.add_word(EOS)
        with codecs.open(vocab_path, 'r', 'utf8') as f:
            for line in f:
                self.add_word(line.strip().split('\t')[0])

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __getitem__(self, key):
        return self.word2idx.get(key, self.word2idx[UNK])

    def __len__(self):
        return len(self.idx2word)

    def indices2sent(self, indices):
        return map(self.idx2word.__getitem__, indices)

    def sent2indices(self, sent):
        return map(self.__getitem__, sent)

def map_dict_value(func, d):
    for k in d:
        d[k] = func(d[k])

class DataIter(object):
    def __init__(self, corpus_path, batch_size, dictionary, cuda=False):
        self.corpus_path = corpus_path
        self.batch_size = batch_size
        self.dictionary = dictionary
        self.cuda = cuda
        self.bos = dictionary[BOS]
        self.eos = dictionary[EOS]
        self.pad = dictionary[PAD]
        self.unk = dictionary[UNK]

        self.build_data()

    def build_data(self):
        self.lines = []
        with codecs.open(self.corpus_path, 'r', 'utf8') as f:
            for line in f:
                words = line.strip().split()
                self.lines.append([BOS] + words + [EOS])

    def __iter__(self):

        def wrapper(d):
            return Variable(d.cuda()) if self.cuda else Variable(d)

        for idx in range(len(self)):
            lines = self.lines[idx * self.batch_size: (idx+1) * self.batch_size]
            lines.sort(key=lambda x: len(x), reverse=True)
            length = map(len, lines)
            max_len = length[0]
            data = torch.LongTensor(len(lines), max_len).fill_(self.pad)
            for i, l in enumerate(lines):
                data[i][:len(l)] = torch.LongTensor(self.dictionary.sent2indices(l))
            data = wrapper(data)

            yield [data[:, :-1], data[:, 1:], map(lambda x: x-1, length)]

    def __len__(self):
        return len(self.lines) // self.batch_size


if __name__ == '__main__':
    data_path = './data/sms/'
    np.random.seed(1)

    dictionary = Dictionary(data_path + 'vocab.txt')
    batch_size = 20
    cuda = False
    train_iter = DataIter(
        corpus_path = data_path + 'valid.txt',
        batch_size = batch_size,
        dictionary = dictionary,
        cuda = cuda,
    )
    for d in train_iter:
        break

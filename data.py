import os
import torch
from torch.autograd import Variable
from collections import defaultdict
from random import shuffle
import pdb
import numpy as np
import codecs
import utils

UNK = '<unk>'
PAD = '<pad>'
BOS = '<bos>'
EOS = '<eos>'

class Dictionary(object):
    def __init__(self, vocab_path):
        self.word2idx = {}
        self.idx2word = []
        self.word2cls = []
        self.cls_set = set()
        self.add_word(UNK)
        self.add_word(PAD)
        self.add_word(BOS)
        self.add_word(EOS)
        with codecs.open(vocab_path, 'r', 'utf8') as f:
            for line in f:
                self.add_word(*line.strip().split())
        self.ncls = len(self.cls_set)

    def add_word(self, word, cls=0):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.word2cls.append(int(cls))
            self.cls_set.add(int(cls))
        return self.word2idx[word]

    def __getitem__(self, key):
        return self.word2idx.get(key, self.word2idx[UNK])

    def __len__(self):
        return len(self.idx2word)

    def indices2sent(self, indices):
        return list(map(self.idx2word.__getitem__, indices))

    def sent2indices(self, sent):
        return list(map(self.__getitem__, sent))

    def get_class_chunks(self):
        def ascent_check(nums):
            for i in range(1, len(nums)):
                if nums[i] < nums[i-1]:
                    return False
            return True
        assert(ascent_check(self.word2cls))
        cls_chunk_size = 1
        for i in range(1, len(self.word2cls)):
            if self.word2cls[i] != self.word2cls[i-1]:
                yield cls_chunk_size
                cls_chunk_size = 0
            cls_chunk_size += 1
        yield cls_chunk_size

class DataIter(object):
    def __init__(self, corpus_path, batch_size, dictionary, cuda=False, training=False):
        self.corpus_path = corpus_path
        self.batch_size = batch_size
        self.dictionary = dictionary
        self.cuda = cuda
        self.bos = dictionary[BOS]
        self.eos = dictionary[EOS]
        self.pad = dictionary[PAD]
        self.unk = dictionary[UNK]
        self.training = training

        self.build_data()

    def build_data(self):
        self.lines = []
        with codecs.open(self.corpus_path, 'r', 'utf8') as f:
            for line in f:
                words = line.strip().split()
                self.lines.append([BOS] + words + [EOS])

    def get_unigram_dist(self):
        dist = [0] * len(self.dictionary)
        for l in self.lines:
            # skip BOS
            for w in l[1:]:
                dist[self.dictionary[w]] += 1
        return torch.Tensor(dist)

    def __iter__(self):

        def wrapper(d):
            return Variable(d.cuda()) if self.cuda else Variable(d)

        indices = list(range(len(self.lines)))
        if self.training:
            shuffle(indices)

        for idx in range(len(self)):
            batch_indices = indices[idx * self.batch_size: (idx+1) * self.batch_size]
            lines = list(map(self.lines.__getitem__, batch_indices))
            #lines = self.lines[idx * self.batch_size: (idx+1) * self.batch_size]
            lines.sort(key=lambda x: len(x), reverse=True)
            length = list(map(len, lines))
            max_len = length[0]
            data = torch.LongTensor(len(lines), max_len).fill_(self.pad)
            for i, l in enumerate(lines):
                data[i][:len(l)] = torch.LongTensor(self.dictionary.sent2indices(l))
            data = wrapper(data)

            yield [data[:, :-1], data[:, 1:], list(map(lambda x: x-1, length))]

    def __len__(self):
        return len(self.lines) // self.batch_size


if __name__ == '__main__':
    data_path = './data/penn/'
    np.random.seed(1)

    dictionary = Dictionary(data_path + 'vocab.c.txt')
    batch_size = 20
    cuda = False
    train_iter = DataIter(
        corpus_path = data_path + 'valid.txt',
        batch_size = batch_size,
        dictionary = dictionary,
        cuda = cuda,
    )
    chunks = list(dictionary.get_class_chunks())
    pdb.set_trace()
    #for d in train_iter:
    #    break

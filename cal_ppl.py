#!/usr/bin/env python
# encoding: utf-8

import torch
from data import Dictionary, DataIter
from model import length2mask
import numpy as np


corpus_path = './data/sms/'
data_source = 'test'
#num_unk = 871 if data_source == 'valid' else 100
num_unk = 100
model = torch.load('./params/cut3000/model.pt')
model.eval()

dic = Dictionary(corpus_path + 'vocab.cut3000.txt')
data_iter = DataIter(
    corpus_path + data_source + '.txt',
    10,
    dic,
    True
)

softmax = torch.nn.Softmax()
softmax.cuda()

ppl = 0
count = 0

if True:
    for (input, target, length) in data_iter:
        output = softmax(model.forward_all(input, length))
        mask = length2mask(length)
        target = target.masked_select(mask)
        for i, t in enumerate(target):
            p = output[i][t.data[0]]
            if t.data[0] == data_iter.unk:
                p /= num_unk
            ppl += -torch.log(p).data[0]
            count += 1

print 'norm unk: '
print np.exp(ppl / count)

ppl = 0
count = 0

if True:
    for d in data_iter:
        loss = model.loss(d)
        ppl += loss.data[0]
        count += 1

print 'origin: '
print np.exp(ppl / count)

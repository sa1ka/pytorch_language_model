#!/usr/bin/env python
# encoding: utf-8
import torch
from data import Dictionary, DataIter
import numpy as np
from model import length2mask
import pdb

corpus_path = './data/sms/'
model = torch.load('./params/sms-allv/model.pt')
model.eval()

dic = Dictionary(corpus_path + 'all.vocab.txt')
data_iter = DataIter(
    corpus_path + 'rescore.txt',
    1,
    dic,
    True
)

loss = []
for i, d in enumerate(data_iter):
    if i % 1000 == 0:
        print i
    _, _, l = d
    p = model.loss(d).data[0] * l[0]
    loss.append(p)

with open('score.allv.txt', 'w') as f:
    for l in loss:
        f.write(str(l * 2) + '\n')

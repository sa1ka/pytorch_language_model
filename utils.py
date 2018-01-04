#!/usr/bin/env python
# encoding: utf-8
import torch
import time
from torch.autograd import Variable

def list2longtensor(x, cuda=True):
    t = torch.LongTensor(x)
    if cuda:
        t = t.cuda()
    return Variable(t)

def length2mask(length):
    mask = torch.ByteTensor(len(length), max(length)).zero_().cuda()
    for i, l in enumerate(length):
        mask[i][:l].fill_(1)
    return Variable(mask)

def map_dict_value(func, d):
    d_ = {}
    for k in d:
        d_[k] = func(d[k])
    return d_

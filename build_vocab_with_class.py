#!/usr/bin/env python
# encoding: utf-8

from collections import Counter
import argparse

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/penn')
    parser.add_argument('--ncls', type=int, default=30, help='number of class')
    parser.add_argument('--min_count', type=int, default=0, help='the min count of word that appear in corpus')
    args = parser.parse_args()
    return args

def freq_counter(corpus):
    counter = Counter()
    with open(corpus) as f:
        for l in f.read().splitlines():
            for w in l.split():
                counter[w] += 1
    return counter

def class_assign(counter, num_class, mcount):
    words_with_freq = list(counter.items())
    words_with_freq.sort(key=lambda x: x[1], reverse=True)

    words_with_freq = list(filter(lambda x: x[1]>=mcount, words_with_freq))

    words_sorted = list(map(lambda x:x[0], words_with_freq))
    class_list = []
    chunk_size = len(words_sorted) // num_class + 1
    for i in range(0, len(words_sorted)):
        class_list.append(i // chunk_size)
    return zip(words_sorted, class_list)

if __name__ == '__main__':
    args = arg_parse()
    counter = freq_counter(args.data + '/train.txt')
    words_with_cls = class_assign(counter, args.ncls, args.min_count)
    with open(args.data + '/vocab.c.txt', 'w') as f:
        for w, c in words_with_cls:
            f.write(w + ' ' + str(c) + '\n')




import os
import sys
import time
import math
import pickle
import argparse
from multiprocessing import Process

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.autograd import Variable

from model import RNNModel
from data import Dictionary, DataIter
from decoder import SMDecoder, ClassBasedSMDecoder, NCEDecoder

def arg_parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/ptb',
                        help='location of the data corpus')
    parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=300,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=1,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=50,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str,  default='params/tmp/model.pt',
                        help='path to save the final model')
    parser.add_argument('--cont', action='store_true',
                        help='if continue with the pretrained model')
    parser.add_argument('--decoder', type=str, default='sm',)
    parser.add_argument('--nce_nsample', type=int, default=10)

    parser.add_argument('--dist', action='store_true')
    parser.add_argument('--world_size', type=int, default=0)
    parser.add_argument('--devices', type=int, nargs='*')
    args = parser.parse_args()

    print('{:=^30}'.format('all args'))
    for arg in vars(args):
        print(' '.join(map(str, (arg, getattr(args, arg)))))

    return args

###############################################################################
# Training code
###############################################################################

class Trainer(object):
    def __init__(self, model, args,
                 train_iter, valid_iter, test_iter=None,
                 max_epochs=50,):
        self.model = model
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.test_iter = test_iter
        self.max_epochs = max_epochs
        self.args = args

    def check_dist(self):
        return not self.args.dist or dist.get_rank() == 0

    def __train(self, lr, epoch):
        self.model.train()
        total_loss = 0
        start_time = time.time()
        optim = torch.optim.SGD(self.model.parameters(), lr = lr, momentum=.9)
        for batch, data in enumerate(self.train_iter):
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            optim.zero_grad()

            loss = self.model(data)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.args.clip)
            optim.step()

            total_loss += loss.data

            if self.check_dist():
                if batch % self.args.log_interval == 0 and batch > 0:
                    cur_loss = total_loss[0] / self.args.log_interval
                    elapsed = time.time() - start_time
                    print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                            'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, batch, self.train_iter.nbatchs(), lr,
                        elapsed * 1000 / self.args.log_interval, cur_loss, math.exp(cur_loss)))
                    total_loss = 0
                    start_time = time.time()

            sys.stdout.flush()

    def train(self):
        # Loop over epochs.
        lr = self.args.lr
        best_val_loss = None

        if self.args.cont:
            with open(self.args.save, 'rb') as f:
                self.model = torch.load(f)

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            for epoch in range(1, self.max_epochs+1):
                epoch_start_time = time.time()
                self.__train(lr, epoch)
                val_loss = self.evaluate(self.valid_iter)
                if self.check_dist():
                    print('-' * 89)
                    print('| end of epoch {:3d} | time: {:5.2f}s '.format(epoch, (time.time() - epoch_start_time),))
                    print('-' * 89)
                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_loss or val_loss < best_val_loss:
                    if self.check_dist():
                        with open(self.args.save, 'wb') as f:
                            torch.save(self.model.module, f) if self.args.dist else torch.save(self.model, f)
                    best_val_loss = val_loss
                else:
                    # Anneal the learning rate if no improvement has been seen in the validation dataset.
                    lr /= 4.0
                    if lr < 0.01:
                        break

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

        if self.check_dist():
            # Load the best saved model.
            with open(self.args.save, 'rb') as f:
                if self.args.dist:
                    self.model.module = torch.load(f)
                else:
                    self.model = torch.load(f)
            if not self.test_iter is None:
                self.evaluate(self.test_iter, 'test')

    def evaluate(self, data_source, prefix='valid'):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        total_loss = 0
        for data in data_source:
            loss = self.model(data)
            total_loss +=  loss.data
        ave_loss = total_loss[0] / data_source.nbatchs()
        print('| {0} loss {1:5.2f} | {0} ppl {2:8.2f}'.format(prefix, ave_loss, math.exp(ave_loss)))
        return ave_loss

def main(args):
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    corpus_path = args.data + '/'
    dictionary = Dictionary(corpus_path + 'vocab.c.txt')

    eval_batch_size = 10

    train_iter = DataIter(
        corpus_path + 'train.txt',
        args.batch_size,
        dictionary = dictionary,
        cuda = args.cuda,
        dist = args.dist,
    )
    valid_iter = DataIter(
        corpus_path + 'valid.txt',
        eval_batch_size,
        dictionary = dictionary,
        cuda = args.cuda,
    )
    test_iter = DataIter(
        corpus_path + 'test.txt',
        eval_batch_size,
        dictionary = dictionary,
        cuda = args.cuda,
    )

    ###############################################################################
    # Build the model
    ###############################################################################

    ntoken = len(dictionary)

    if args.decoder == 'sm':
        decoder = SMDecoder(
            nhid = args.nhid,
            ntoken = ntoken
        )
    elif args.decoder == 'cls':
        decoder = ClassBasedSMDecoder(
            nhid = args.nhid,
            ncls = dictionary.ncls,
            word2cls = dictionary.word2cls,
            class_chunks = list(dictionary.get_class_chunks()),
        )
    elif args.decoder == 'nce':
        decoder = NCEDecoder(
            nhid = args.nhid,
            ntoken = ntoken,
            noise_dist = train_iter.get_unigram_dist(),
            nsample = args.nce_nsample,
        )

    model = RNNModel(
        ntoken = ntoken,
        ninp = args.emsize,
        nhid = args.nhid,
        nlayers = args.nlayers,
        decoder = decoder,
        dropout = args.dropout,
    )

    if args.cuda:
        model.cuda()

    if args.dist:
        model = nn.parallel.DistributedDataParallel(model)

    trainer = Trainer(
        model = model,
        train_iter = train_iter,
        valid_iter = valid_iter,
        test_iter = test_iter,
        max_epochs = args.epochs,
        args = args
    )

    trainer.train()

def init_processes(args, rank, device, fn, backend='gloo'):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    share_file = "file:///mnt/lustre/sjtu/users/rnc00/workspace/pytorch_lm/shared_file"
    dist.init_process_group(backend, rank=rank, world_size=args.world_size, init_method=share_file)
    main(args)

if __name__ == '__main__':
    args = arg_parse()
    if args.dist:
        if args.devices is not None:
            assert(len(args.devices) == args.world_size)
        else:
            args.devices = list(range(args.world_size))

        processes = []
        for rank in range(args.world_size):
            p = Process(target=init_processes, args=(args, rank, args.devices[rank], main))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        main(args)

import argparse
import sys
import time
import math
import torch
import torch.nn as nn
import model
import pickle
import torch.optim as optim
from torch.autograd import Variable, profiler
from data import Dictionary, DataIter

def arg_parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/penn',
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
    parser.add_argument('--cont', action='store_true')
    parser.add_argument('--cls', action='store_true')
    args = parser.parse_args()

    print('{:=^30}'.format('all args'))
    for arg in vars(args):
        print(' '.join(map(str, (arg, getattr(args, arg)))))

    return args

###############################################################################
# Training code
###############################################################################

class Trainer(object):
    def __init__(self, model, ntokens, args,
                 train_iter, valid_iter, test_iter=None,
                 max_epochs=50,):
        self.model = model
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.test_iter = test_iter
        self.ntokens = ntokens
        self.max_epochs = max_epochs
        self.args = args

    def __train(self, lr, epoch):
        self.model.train()
        total_loss = 0
        start_time = time.time()
        optim = torch.optim.SGD(self.model.parameters(), lr = lr, momentum=.9)
        for batch, data in enumerate(self.train_iter):
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            optim.zero_grad()

            loss = self.model.loss(data)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.args.clip)
            optim.step()

            total_loss += loss.data

            if batch % self.args.log_interval == 0 and batch > 0:
                cur_loss = total_loss[0] / self.args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(self.train_iter), lr,
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
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s '.format(epoch, (time.time() - epoch_start_time),))
                print('-' * 89)
                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_loss or val_loss < best_val_loss:
                    with open(self.args.save, 'wb') as f:
                        torch.save(self.model, f)
                    best_val_loss = val_loss
                else:
                    # Anneal the learning rate if no improvement has been seen in the validation dataset.
                    with open(self.args.save, 'rb') as f:
                        self.model = torch.load(f)
                    lr /= 4.0

                    if lr < 0.01:
                        break

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

        # Load the best saved model.
        with open(self.args.save, 'rb') as f:
            self.model = torch.load(f)
        if not self.test_iter is None:
            self.evaluate(self.test_iter, 'test')

    def evaluate(self, data_source, prefix='valid'):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        total_loss = 0
        for data in data_source:
            loss = self.model.loss(data)
            total_loss +=  loss.data
        ave_loss = total_loss[0] / len(data_source)
        print('| {0} loss {1:5.2f} | {0} ppl {2:8.2f}'.format(prefix, ave_loss, math.exp(ave_loss)))
        return ave_loss

if __name__ == '__main__':
    # Set the random seed manually for reproducibility.
    args = arg_parse()
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

    ntokens = len(dictionary)

    model = model.RNNModel(
        ntoken = ntokens,
        ninp = args.emsize,
        nhid = args.nhid,
        ncls = dictionary.ncls,
        word2cls = dictionary.word2cls,
        cls_based = args.cls,
        nlayers = args.nlayers,
        class_chunks = list(dictionary.get_class_chunks()),
        dropout = args.dropout,
    )

    if args.cuda:
        model.cuda()

    trainer = Trainer(
        model = model,
        ntokens = ntokens,
        train_iter = train_iter,
        valid_iter = valid_iter,
        test_iter = test_iter,
        max_epochs = args.epochs,
        args = args
    )

    trainer.train()

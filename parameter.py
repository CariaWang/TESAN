# -*- coding: utf-8 -*-
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Topic-Enhanced Self-Attention Network for Social Emotion Classification')

    # word2vec
    parser.add_argument('--WORD2VEC_DIR', default='word2vec.pickle')

    # dataset
    parser.add_argument('--train_size', default=1000,  type=int,   help='training set size')
    parser.add_argument('--test_size',  default=246,   type=int,   help='testing set size')
    parser.add_argument('--num_class',  default=6,     type=int,   help='number of classes')

    # model arguments
    parser.add_argument('--in_dim',     default=300,   type=int,   help='size of input word vector')
    parser.add_argument('--h_dim',      default=300,   type=int,   help='size of hidden unit')
    parser.add_argument('--num_topic',  default=30,    type=int,   help='number of topics')
    parser.add_argument('--en1_units',  default=100,   type=int,   help='size of encoder1 in NTM')
    parser.add_argument('--en2_units',  default=100,   type=int,   help='size of encoder2 in NTM')
    parser.add_argument('--variance',   default=0.995, type=float, help='default variance in prior normal')

    # training arguments
    parser.add_argument('--L',          default=0.03,  type=float, help='the lambda in the loss function')
    parser.add_argument('--num_epoch',  default=60,    type=int,   help='number of total epochs to run')
    parser.add_argument('--batch_size', default=20,    type=int,   help='batchsize for optimizer updates')
    parser.add_argument('--lr',         default=0.003, type=float, help='initial learning rate')
    parser.add_argument('--wd',         default=5e-5,  type=float, help='weight decay')
    parser.add_argument('--momentum',   default=0.99,  type=float)

    args = parser.parse_args()
    return args

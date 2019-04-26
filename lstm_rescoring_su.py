# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import model_surnn

import glob, os
import math
import numpy as np
import cPickle
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=400, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=60,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model_su.pt',
                    help='path to save the final model')
parser.add_argument('--dict', type=str,  default='',
                    help='path of dictionary')
parser.add_argument('--nbest', type=str, default='',
                    help='nbest path to rescore')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data, args.bptt, args.dict)


def get_batch(source, evaluation=False):
    seq_len = len(source)
    data = Variable(source, volatile=evaluation)
    # zero_padding = torch.cuda.LongTensor(seq_len,args.batch_size).zero_()
    zero_padding = torch.cat([source[2:seq_len],torch.cuda.LongTensor(2,args.batch_size).zero_()],0)
    futureword1 = Variable(zero_padding)
    zero_padding = torch.cat([source[3:seq_len],torch.cuda.LongTensor(3,args.batch_size).zero_()],0)
    futureword2 = Variable(zero_padding)
    return data, futureword1, futureword2

def predict(data_source):
    model.eval()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    data, futureword1, futureword2 = get_batch(data_source, evaluation=True)
    output, hidden = model(data, hidden, futureword1, futureword2)
    output_flat = output.view(-1, ntokens)
    softmax = nn.Softmax()
    output_pro = softmax(output_flat)
    # import pdb; pdb.set_trace()
    # hidden = repackage_hidden(hidden)
    return output_pro




# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
lstmP = dict()
os.chdir(args.nbest)
for job in range(1,9):
    filenames = []
    for file in glob.glob(str(job)+"/*.onlytext"):
        filenames.append(file)
    for fn in filenames:
        print fn
        fr = open(fn,'r')
        # fw = open(fn+'.lmresult.lstmscore','w')
        sentP = []
        sents = []
        for i in fr:
            sents.append([])
            line = i.split(' ')
            line.pop()
            sents[-1].append(corpus.dictionary.word2idx['<s>'])
            for k in line:
                sents[-1].append(corpus.dictionary.word2idx[k])
            sents[-1].append(corpus.dictionary.word2idx['<eos>'])
            # sents[-1] += (args.bptt-2-len(sents[-1]))*[0]
        sents_length = len(sents)
        # for i in range(args.batch_size-sents_length%args.batch_size):
        #     sents.append([0]*args.bptt)
        batch_num = int(math.ceil(len(sents)/float(args.batch_size)))
        for i in range(batch_num):
            batch_data = np.zeros((args.batch_size,args.bptt))
            if i==batch_num-1:
                batch_sents = sents[i*args.batch_size:]
            else:
                batch_sents = sents[i*args.batch_size:(i+1)*args.batch_size]
            for j in range(len(batch_sents)):
                for k in range(len(batch_sents[j])):
                    batch_data[j][k] = batch_sents[j][k]
            batch_data = np.transpose(batch_data).astype(int).tolist()
            probability = predict(torch.cuda.LongTensor(batch_data))
            for j in range(len(batch_sents)):
                probability_sent = probability[j:len(probability):args.batch_size]
                # score = 0
                sentP.append(0)
                for k in range(len(batch_sents[j])-1):
                    # score += math.log10(probability_sent[k][batch_sents[j][k+1]])
                    sentP[-1] += math.log10(float(probability_sent[k][batch_sents[j][k+1]]))
                # fw.write(str(score)+'\n')
        lstmP[fn.replace('.onlytext','')] = np.array(sentP)
# import pdb;pdb.set_trace()
with open('lstm.cpickle','w') as f:
    cPickle.dump(lstmP,f)

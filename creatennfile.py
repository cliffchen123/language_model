# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str,
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
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=45,
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
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--dict', type=str,  default='',
                    help='path of dictionary')
parser.add_argument('--text', type=str,  default='',
                    help='file to do nnfile')
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

sents = []
sentsidx = []
f = open(args.text,'r')
for i in f:
    sents.append(i.split())
    sentsidx.append([])
    for j in sents[-1]:
        sentsidx[-1].append(corpus.dictionary.word2idx[j])
import pdb; pdb.set_trace()
def predict(data_source):
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(1)
    # data, targets = get_batch(data_source, i, evaluation=True)
    # import pdb; pdb.set_trace()
    data = Variable(data_source, volatile=True)
    output, hidden = model(data, hidden)
    output_flat = output.view(-1, ntokens)
    softmax = nn.Softmax()
    output_pro = softmax(output_flat)
    # import pdb; pdb.set_trace()
    # hidden = repackage_hidden(hidden)
    return torch.max(output_pro,1)




# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.

# probability, word = predict(torch.cuda.LongTensor([[0],[1],[32966],[32967],[1],[0],[0],[32966],[32967],[13],[406],[23],[17],[6253],[19902],[310],[1444],[19902],[13],[26],[27],[2576],[16],[9],[19902],[115],[17],[4929],[4121],[9611],[13],[4854],[2429],[37],[651]]))
probability, word = predict(torch.cuda.LongTensor([]))
# print 'word: ',str(corpus.dictionary.idx2word[int(word)]), 'probability: ', float(probability)
import pdb; pdb.set_trace()
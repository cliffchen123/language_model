# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import data
import model
import model_mixer

import cPickle
import glob, os
import math

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
parser.add_argument('--model_mixer', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--model_expert', type=str,  default='model.pt',
                    help='folder to save the final model')
parser.add_argument('--dict', type=str,  default='',
                    help='path of dictionary')
parser.add_argument('--nbest', type=str, default='',
                    help='nbest path to rescore')
parser.add_argument('--job', type=int, default=8,
                    help='number of job')
parser.add_argument('--splitdata', type=str,  default='',
                    help='the file of split data')

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

with open(args.splitdata,'r') as f:
    splitdata = cPickle.load(f)
    splitdata.pop('01')
    splitdata.pop('03')
    splitdata.pop('04')
    splitdata.pop('05')
    splitdata.pop('06')
    splitdata.pop('07')
    splitdata.pop('08')
    splitdata.pop('09')
    splitdata.pop('11')
    splitdata.pop('12')
    splitdata.pop('13')
    splitdata.pop('21')
    splitdata.pop('22')
    splitdata.pop('24')
    splitdata.pop('25')
    splitdata.pop('28')
    splitdata.pop('29')
    splitdata.pop('30')
    splitdata.pop('31')
    splitdata.pop('38')
    splitdata.pop('39')
    splitdata.pop('40')
    splitdata.pop('42')
    splitdata.pop('43')
    splitdata.pop('53')
    splitdata.pop('54')
    splitdata.pop('56')


def extract_hidden(data_source, model_):
    model_.eval()
    hidden = model_.init_hidden(args.batch_size)
    emb = model_.drop(model_.encoder(data_source))
    output, hidden = model_.rnn(emb, hidden)
    
    # data, targets = get_batch(data_source, i, evaluation=True)

    # import pdb; pdb.set_trace()
    return output

def predict(data_source):
    model.eval()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    data = Variable(data_source, volatile=True)
    output_expert = []
    for j in model_expert:
        output_expert.append(extract_hidden(data,model_expert[j]))
    output, hidden = model(data, hidden, output_expert)
    output_flat = output.view(-1, ntokens)
    softmax = nn.Softmax()
    output_pro = softmax(output_flat)
    # import pdb; pdb.set_trace()
    return output_pro


# Load the mixer model. 
with open(args.model_mixer, 'rb') as f:
    model = torch.load(f)
# Load the expert model.
model_expert = dict()
for i in splitdata:
    with open(args.model_expert+'/'+i+'_model.pt', 'rb') as f:
        model_expert[i] = torch.load(f)

# Run on test data.
lstmP = dict()
os.chdir(args.nbest)
for job in range(1,args.job+1):
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





    # fr = open(i,'r')
    # fw = open(i+'.lmresult.lstmscore','w')
    # for j in fr:
    #     line = j.split(' ')
    #     line.pop()
    #     sents = [[corpus.dictionary.word2idx['<s>']]]
    #     for k in line:
    #         sents.append([corpus.dictionary.word2idx[k]])
    #     sents.append([corpus.dictionary.word2idx['<eos>']])
    #     # print sents
    #     if sents!=[]:
    #         probability = predict(torch.cuda.LongTensor(sents))
    #         score = 0
    #         for k in range(len(sents)-1):
    #             score += math.log10(probability[k][sents[k+1]])
    #     else:
    #         score=-99999
    #     fw.write(str(score)+'\n')

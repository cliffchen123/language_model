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
import cPickle
import glob, os
import math
import read_graph as rg

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
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--dict', type=str,  default='',
                    help='path of dictionary')
parser.add_argument('--lattice', type=str, default='',
                    help='lattice path to rescore')
parser.add_argument('--job', type=int, default=8,
                    help='number of job')
parser.add_argument('--scale', type=int, default=10,
                    help='scale of lm score')
parser.add_argument('--beam', type=int, default=100,
                    help='beam')
parser.add_argument('--approximation', type=int, default=10,
                    help='approximation of ngram')
parser.add_argument('--alpha', type=float, default=0.4,
                    help='coefficient of combine nnlm score and ngram score')
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


def predict(data_source,hidden=None):
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    softmax = nn.Softmax()
    # outputs = [None]*(len(data_source)-1)
    # hiddens = [None]*(len(data_source)-1)
    if type(hidden)==type(None):
        hidden = model.init_hidden(len(data_source))
    # for i in range(len(data_source)-1):
    data = Variable(torch.cuda.LongTensor([data_source]), volatile=True)
    output, hidden = model(data, hidden)

    output_flat = output.view(-1, ntokens)
    output_pro = softmax(output_flat)
    # outputs[i] = output_pro
    # hiddens[i] = hidden
    # import pdb; pdb.set_trace()
    return output_pro, hidden


filenames = []
f = open(args.data+'lattice_uni/lattice/nbest/text.filt','r')
for i in f:
    line = i.split()
    filenames.append(line[0])


# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
sents = dict()
os.chdir(args.lattice)
for job in range(1,args.job+1):
    for file in glob.glob(str(job)+"/*.lat"):
        print file
        nodes, arcs = rg.read_graph(file)
        # nsents, nodes = rg.pruning(file,1000,10,50)
        states = ['un']*len(nodes)
        hasdone = [0]*len(nodes)
        stack = []
        time = []
        for i in nodes:
            time.append(i.time)
            hasdone[i.id] = len(i.next_arc)
        time[-1]+=0.1 # the last node must be the first in the stack
        nodes_idx = sorted(range(len(time)), key=lambda x:time[x], reverse=True)
        for i in nodes_idx:
            stack.append(nodes[i])

        while stack!=[]:
            cur_node = stack[-1]
            if cur_node.pre_arc==[]: # the initial node
                states[cur_node.id] = [[0,[corpus.dictionary.word2idx['<s>']],model.init_hidden(1)]]
                stack.pop()

            else:
                states[cur_node.id] = []
                data = []
                target = []
                hiddens = None
                scores = []
                rescore_arc = []
                all_branch_hiddens = []
                button_hidden = []
                button_scores = []
                button_target = []
                check_arc = cur_node.pre_arc[:]
                for arc in check_arc:
                    # if arc.freeze:
                    #     continue
                    if arc.word=='!NULL':
                        for state in states[arc.start.id]:
                            score = state[0]+arc.acscore+args.scale*(1-args.alpha)*arc.lmscore
                            # score = state[0]+arc.acscore+args.scale*arc.lmscore
                            button_scores.append(score)
                            button_target.append(state[1])
                            button_hidden.append(state[2])
                            # states[cur_node.id]+=[[score,state[1],state[2]]]
                    else:
                        # min_score = states[arc.start.id][0][0]-args.beam
                        # beam_size = 0
                        # for j in range(len(states[arc.start.id])):
                        #     if min_score<states[arc.start.id][j][0]:
                        #         beam_size += 1
                        beam_size = min(args.beam,len(states[arc.start.id]))
                        # used_ngram = dict()
                        for j in range(beam_size):
                            # seq = str(states[arc.start.id][j][1][-args.approximation:])
                            # if seq in used_ngram and used_ngram[seq]>10:
                            #     continue
                            # elif seq in used_ngram:
                            #     used_ngram[seq] += 1
                            # else:
                            #     used_ngram[seq] = 0
                            target.append(states[arc.start.id][j][1]+[corpus.dictionary.word2idx[arc.word]])
                            scores.append(states[arc.start.id][j][0])
                            data.append(states[arc.start.id][j][1][-1])
                            if hiddens==None:
                                hiddens = states[arc.start.id][j][2]
                            else:
                                hiddens = (torch.cat((hiddens[0],states[arc.start.id][j][2][0]),1), torch.cat((hiddens[1],states[arc.start.id][j][2][1]),1))
                        rescore_arc.append([arc,beam_size])

                        # hasdone[arc.start.id]-=1
                        # if hasdone[arc.start.id]==0:
                        #     for j in range(len(states[arc.start.id])):
                        #         states[arc.start.id][j][2] = None

                if data!=[]:

                    probabilitys = []
                    index = 0
                    for i in range(0,len(data),args.batch_size):
                        batch_end = min(len(data),i+args.batch_size)
                        target_batch = target[i:batch_end]
                        # import pdb;pdb.set_trace()
                        indices = Variable(torch.cuda.LongTensor(range(i,batch_end)))
                        hiddens_split_temp0 = torch.index_select(hiddens[0],1,indices)
                        hiddens_split_temp1 = torch.index_select(hiddens[1],1,indices)
                        hiddens_split = (hiddens_split_temp0,hiddens_split_temp1)
                        new_probabilitys, new_hiddens = predict(data[i:batch_end],hiddens_split)

                        for j in range(len(new_probabilitys)):
                            probabilitys.append(new_probabilitys[j][target[index][-1]].data[0])
                            index+=1
                        split_num = new_hiddens[0].shape[1]
                        temp1=torch.chunk(new_hiddens[0],split_num,1)
                        temp2=torch.chunk(new_hiddens[1],split_num,1)
                        
                        for i in range(split_num):
                            all_branch_hiddens.append((temp1[i],temp2[i]))
                    

                    # probabilitys, new_hiddens = predict(data,hiddens)
                    # split_num = new_hiddens[0].shape[1]
                    # temp1=torch.chunk(new_hiddens[0],split_num,1)
                    # temp2=torch.chunk(new_hiddens[1],split_num,1)
                    
                    # for i in range(split_num):
                    #     all_branch_hiddens.append((temp1[i],temp2[i]))

                    index = 0
                    for i in rescore_arc:
                        arc = i[0]
                        beam_size = i[1]
                        max_score = -float('inf')
                        max_lmscore = -float('inf')
                        # used_ngram = dict()
                        for j in range(beam_size):
                            # seq = str(states[arc.start.id][j][1][-args.approximation:])
                            # if seq in used_ngram and used_ngram[seq]>10:
                            #     continue
                            # elif seq in used_ngram:
                            #     used_ngram[seq] += 1
                            # else:
                            #     used_ngram[seq] = 0
                            
                            # print(np.log10(probabilitys[index][target[index]].data[0]))
                            # print(arc.lmscore)
                            # import pdb;pdb.set_trace()
                            rescore_lmscore = args.alpha*np.log10(probabilitys[index]) + (1-args.alpha)*arc.lmscore
                            scores[index] += arc.acscore+args.scale*(rescore_lmscore)
                            # scores[index] += arc.acscore+args.scale*arc.lmscore
                            if max_score<scores[index]:
                                max_score = scores[index]
                                max_lmscore = rescore_lmscore
                            index+=1
                        arc.lmscore = max_lmscore

                scores += button_scores
                target += button_target
                all_branch_hiddens += button_hidden
                sort_idx = sorted(range(len(scores)), key=lambda x:scores[x], reverse=True)
                
                for i in sort_idx:
                    states[cur_node.id].append([scores[i],target[i],all_branch_hiddens[i]])

                stack.pop()

            if cur_node.next_arc==[]:
                pre_node = cur_node.pre_arc[0].start
                sent = states[pre_node.id][0][1]
                
                sent.pop(0)
                for i in range(len(sent)):
                    sent[i] = corpus.dictionary.idx2word[sent[i]]
                sent = ' '.join(sent)
                sents[file[file.find('/')+1:file.find('.lat')]] = sent
                # print sent+'\n'

        # rg.write_graph(file.replace('.lat','')+'.rescore',nodes,arcs)

with open('nbest/onebest.txt','w') as fw:
    for i in filenames:
        fw.write(i+' '+sents[i]+'\n')
        # sents = rg.nbest(1000,nodes,10,100)
        # with open(file+'.nbest','w') as fw:
        #     for i in sents:
        #         fw.write(i[0].replace('<s> ','').replace(' </s>','')+'\n')




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
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--lattice', type=str, default='',
                    help='lattice path to rescore')
parser.add_argument('--nbest', type=str, default='',
                    help='nbest path to rescore')
parser.add_argument('--job', type=int, default=30,
                    help='number of job')
parser.add_argument('--scale', type=int, default=10,
                    help='scale of lm score')
parser.add_argument('--beam', type=int, default=100,
                    help='beam')
parser.add_argument('--output', type=str,
                    help='output file')
args = parser.parse_args()


fw = open(args.output,'w')

cover_recall = []
for job in range(1,args.job+1):
    for file in glob.glob(args.lattice+'/'+str(job)+"/*.lat"):
        filename = file.replace(args.lattice+'/'+str(job)+'/','').replace('.lat','')
        # print file
        nodes, arcs = rg.read_graph(file)
        lattice_sents = rg.nbest_BFS(1000,nodes,args.scale,args.beam)
        

        nbest_sents=[]
        fr = open(args.nbest+'/'+str(job)+'/'+filename+'.onlytext')
        for i in fr:
            line = i.split()
            nbest_sents.append(line)
            # lattice_sent = sents[index][0]
            # lattice_sent = sents[index][0].split()
            # lattice_sent.pop(0)
            # lattice_sent.pop()

        cover_recall.append(0)
        for i in nbest_sents:
            for j in lattice_sents:
                if i==j[0]:
                    cover_recall[-1]+=1
                    break
        cover_recall[-1] = cover_recall[-1]/float(len(nbest_sents))
        fw.write(str(cover_recall[-1])+'\n')
        if len(cover_recall)%100==0:
            print(str(len(cover_recall))+'\tfinished')
fw.write('average: '+str(sum(cover_recall)/len(cover_recall))+'\n')
        # print cover_recall
        # print len(nbest_sents)
        # import pdb;pdb.set_trace()





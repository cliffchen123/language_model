# coding: utf-8
import argparse
import time
import math

import cPickle

parser = argparse.ArgumentParser(description='Create dictionary cPickle')
parser.add_argument('--dict', type=str,  default='',
                    help='path of dictionary text')
parser.add_argument('--save', type=str,  default='dictionary.cpickle',
                    help='path of dictionary text')
args = parser.parse_args()

dictionary = []
f = open(args.dict,'r')
for i in f:
     line = i.split()
     dictionary.append(line[0])
with open(args.save,'w') as fw:
     cPickle.dump(dictionary,fw)
import os
import argparse
import time
import math
import glob
import cPickle

parser = argparse.ArgumentParser(description='transfer utt2spk file to train file')
parser.add_argument('--path', type=str,  default='',
                    help='path of text file folder')
args = parser.parse_args()

os.chdir(args.path)

with open('dictionary.cpickle','r') as f:
	dictionary = cPickle.load(f)

fw = open('test.txt','w')
f = open('spk2utt_test')
for i in f:
	line = i.split()
	aaa=line.pop(0)
	for j in range(len(line)):
		if line[j] not in dictionary:
			line[j] = '<unk>'
	fw.write(' '.join(line))
	fw.write(' \n')

fw = open('valid.txt','w')
f = open('spk2utt_valid')
for i in f:
	line = i.split()
	aaa=line.pop(0)
	for j in range(len(line)):
		if line[j] not in dictionary:
			line[j] = '<unk>'
	fw.write(' '.join(line))
	fw.write(' \n')

fw = open('train.txt','w')
f = open('spk2utt_train')
for i in f:
	line = i.split()
	aaa=line.pop(0)
	for j in range(len(line)):
		if line[j] not in dictionary:
			line[j] = '<unk>'
	fw.write(' '.join(line))
	fw.write(' \n')
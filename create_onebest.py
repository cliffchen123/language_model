import os
import argparse
import time
import math
import glob
import cPickle

parser = argparse.ArgumentParser(description='create one best on onlytext files')
parser.add_argument('--path', type=str,  default='data/ami-s5b/lattice_uni_test/nbest/',
                    help='path of text file folder')
parser.add_argument('--job', type=int, default=30,
                    help='number of job')
args = parser.parse_args()

os.chdir(args.path)

filenames = []
f = open('text.filt','r')
for i in f:
	line = i.split()
	filenames.append(line[0])

text = dict()
for folder in range(1,args.job+1):
	for file in glob.glob(str(folder)+"/*.onlytext"):
		with open(file,'r') as f:
			for i in f:
				text[file.replace(str(folder)+"/",'').replace('.onlytext','')] = i
				break

fw = open('onebest.txt','w')
for i in filenames:
	fw.write(i+' '+text[i])
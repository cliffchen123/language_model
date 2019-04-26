import os
import glob
import cPickle
import math
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='create one best on onlytext files')
parser.add_argument('--path', type=str,  default='data/delta_data/lattice_uni/nbest/',
                    help='path of text file folder')
parser.add_argument('--job', type=int,  default=8,
                    help='number of job')
args = parser.parse_args()

filenames = []
os.chdir(args.path)
# read ngram file
acs = dict()
for i in range(1,args.job+1):
	for file in glob.glob(str(i)+"/*.onlytext"):
		f = open(file.replace('.onlytext',''),'r')
		score = []
		for i in f:
			line = i.split(' ')
			score.append(float(line[0]))
		acs[file.replace('.onlytext','')] = np.array(score)

with open('../../lattice_ngram/nbest/acscore.cpickle','w') as file:
	cPickle.dump(acs,file)
# import pdb;pdb.set_trace()
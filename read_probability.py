import os
import glob
import cPickle
import math
import numpy as np
filenames = []
os.chdir('data/delta_data/lattice_ngram/nbest/')
# read ngram file
ngP = dict()
for i in range(1,9):
	for file in glob.glob(str(i)+"/*.ngram"):
		f = open(file,'r')
		sentP = []
		start = False
		for i in f:
			line = i.split()
			if len(line)==0:
				continue
			line = [x for x in line if x != '']
			if not start and i.find('p(')!=-1:
				start = True
				sentP.append(0)
			if start and i.find('p(')==-1:
				start = False

			if start:
				sentP[-1] += math.log10( float(i[i.find('] ')+1:i.find('[ ')]) )
		ngP[file.replace('.onlytext.ngram','')] = np.array(sentP)

with open('ngram.cpickle','w') as ngfile:
	cPickle.dump(ngP,ngfile)
# import pdb;pdb.set_trace()
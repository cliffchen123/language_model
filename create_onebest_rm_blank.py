import os
import argparse
import time
import math
import glob
import cPickle

parser = argparse.ArgumentParser(description='create one best on onlytext files')
parser.add_argument('--path', type=str,  default='data/delta_new/lattice_uni/nbest/',
                    help='path of text file folder')
parser.add_argument('--job', type=int, default=6,
                    help='number of job')
args = parser.parse_args()

os.chdir(args.path)

blank = []
filenames = []
f = open('text.filt','r')
for i in f:
	line = i.split()
	filenames.append(line[0].replace('-',''))
	if len(line)==1:
		blank.append(True)
	else:
		blank.append(False)

text = dict()
for folder in range(1,args.job+1):
	for file in glob.glob(str(folder)+"/*.onlytext"):
		with open(file,'r') as f:
			for i in f:
				text[file.replace(str(folder)+"/",'').replace('.onlytext','').replace('-','')] = i
				break

ft = open('onebest.text','w')
fc = open('onebest.char','w')
index = 0
for i in filenames:
	if blank[index]==False:
		ft.write(i+' '+text[i])
		fc.write(i+' ')
		line = text[i].split(' ')
		for j in line:
			if j[0]<='z' and j[0]>='a':
				fc.write(j+' ')
			else:
				temp2 = ''
				index = 1
				for k in j:
					temp2+=k
					if index%3==0:
						temp2+=' '
					index+=1
				fc.write(temp2)
	index+=1
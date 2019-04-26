import math
import numpy as np
from sklearn.cluster import KMeans
import time
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import cPickle
import data
import argparse
parser = argparse.ArgumentParser(description='PLSA')
parser.add_argument('--data', type=str, default='./data/delta_data',
                    help='location of the data corpus')
parser.add_argument('--iter_num', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--clustersNum', type=int, default=64,
                    help='number of cluster')
parser.add_argument('--bptt', type=int, default=60,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--save', type=str,  default='./plsa.cpickle',
                    help='path to save the final model')
parser.add_argument('--dict', type=str,  default='./data/delta_data/dictionary.cpickle',
                    help='path of dictionary')
parser.add_argument('--spk2utt', type=str, default='./data/delta_data/spk2utt',
                    help='speaker to utterences')
parser.add_argument('--spk_doc', type=str, default='./data/delta_data/spk_doc',
                    help='speaker to document')
parser.add_argument('--preprocess', type=str,  default='delta',
                    help='the category of preprocess')
args = parser.parse_args()

corpus = data.Corpus(args.data, args.bptt, args.dict)

f = open(args.spk2utt,'r')
spk_doc = dict()
for i in f:
    if args.preprocess=='delta':
        spk = i[:i.find('-')]
    elif args.preprocess=='ami':
        spk = i.split('_')[3]
    elif args.preprocess=='delta_new':
        spk = ''
        half_line = i.split(' ')[0]
        hasFound = 0
        half_line = half_line.replace('-',' ').replace('.',' ').replace('!',' ! ')
        for j in half_line.split(' '):
            if 'A'<=j[0]<='Z':
                spk += j
                hasFound += 1
            elif spk!='':
                break
    if spk not in spk_doc:
    	spk_doc[spk] = []
    
    line = i.replace('\n','').split(' ')
    line.pop(0)
    for j in line:
        if j not in corpus.dictionary.word2idx:
        	spk_doc[spk].append(corpus.dictionary.word2idx['<unk>'])
        else:
        	spk_doc[spk].append(corpus.dictionary.word2idx[j])


fw = open(args.spk_doc,'w')
for i in spk_doc.values():
	for j in i:
		fw.write(str(j)+' ')
	fw.write('\n')

# attribute
datasetName = args.preprocess
fileName = args.spk_doc
dictSize = len(corpus.dictionary.idx2word)
docSize = len(spk_doc)
clustersNum = args.clustersNum
iter_num = args.iter_num


termVector = np.zeros((dictSize,docSize))

index=0
for i in spk_doc.values():
	for j in i:
		termVector[int(j)][index]=1
	index+=1
kmeans = KMeans(n_clusters=clustersNum, random_state=0).fit(termVector)

termVector = np.zeros((dictSize,docSize))
document = []
document_nr = []
document_word_num = np.zeros((docSize,dictSize))
term_in_doc = [None]*dictSize
for i in range(dictSize):
	term_in_doc[i]=[]



index=0
for i in spk_doc.values():
	document_i = []
	document_i_nr = []
	for j in i:
		termVector[int(j)][index]=1
		document_i.append(int(j))
		document_word_num[index,int(j)]+=1
		if document_i_nr.count(int(j))==0:
			document_i_nr.append(int(j))
			term_in_doc[int(j)].append(index)
	document.append(document_i)
	document_nr.append(document_i_nr)
	index+=1


pzd = np.random.rand(docSize,clustersNum)
for i in range(docSize):
	pzd[i]/=sum(pzd[i])

pwz = np.zeros((clustersNum,dictSize))
kmeansTransform = kmeans.transform(termVector[0:dictSize])
for i in range(clustersNum):
	temp = kmeansTransform[0:dictSize,i]
	temp = max(temp)-temp+1 #avoid zero
	temp /= sum(temp)
	pwz[i] = temp

# E steps
def Estep(pwz,pzd):
	# s = time.time()
	pzdw = [None]*docSize
	for d in range(docSize):
		pzdw[d] = dict()
	for d in range(docSize):
		# if d%100==0:
		# 	print(str(time.time()-s))
		# 	print(str(d))
		# 	s = time.time()
		for w in document_nr[d]:
			numerator = []
			denominator = 0
			numerator = pwz[:,w]*pzd[d,:]
			denominator = sum(numerator)
			# for z in range(clustersNum):
			# 	temp = pwz[z][w]*pzd[d][z]
			# 	numerator.append(temp)
			# 	denominator += temp
			pzdw[d][w] = numerator/denominator
	return pzdw

# M step
def Mstep(pzdw):

	# update pwz
	new_pwz = np.ones((clustersNum,dictSize))
	for z in range(clustersNum):
		# print(str(z))
		numerator = []
		denominator = 0
		for w in range(dictSize):
			# print(str(w))
			# temp = document_word_num[:,w]*pzdw[:,w,z]
			temp = 0
			for d in term_in_doc[w]:
				temp += document_word_num[d][w]*pzdw[d][w][z]
			numerator.append(temp)
			denominator += temp
		new_pwz[z] = np.array(numerator)/denominator

	# update pzd
	new_pzd = np.ones((docSize,clustersNum))
	for d in range(docSize):
		numerator = []
		denominator = len(document[d])
		for z in range(clustersNum):
			temp = 0
			for w in document_nr[d]:
				temp += document_word_num[d][w]*pzdw[d][w][z]
			numerator.append(temp)
		new_pzd[d] = np.array(numerator)/denominator

	return new_pwz,new_pzd

# Obj function
def Obj(pwz,pzd,pzdw):
	result = 0
	for d in range(docSize):
		for w in document_nr[d]:
			temp = pwz[:,w]*pzd[d,:]
			for i in range(len(temp)):
				if temp[i]==0:
					temp[i]=1
			temp = sum(pzdw[d][w][:]*np.log(temp))
			# temp = 0
			# for z in range(clustersNum):
			# 	if pwz[z,w]*pzd[d,z]==0:
			# 		print "d:"+str(d)+" w:"+str(w)+" z:"+str(z)
			# 		return
			# 	else:
			# 		temp += (pzdw[d][w][z]*math.log(pwz[z,w]*pzd[d,z]))
			result += document_word_num[d][w]*temp
	return result

pzdw = []
print("### " + datasetName + " training start")
for i in range(iter_num):
	
	print("\n---Iter "+str(i+1)+"-----")

	print("Estep start")
	startTime = time.time()
	pzdw = Estep(pwz,pzd)
	print("Estep finish time:"+str(time.time()-startTime))

	print("Mstep start")
	startTime = time.time()
	pwz,pzd = Mstep(pzdw)
	print("Mstep finish time:"+str(time.time()-startTime))

	print("Obj start")
	startTime = time.time()
	print("Obj function: "+str(Obj(pwz,pzd,pzdw)))
	print("Obj finish time:"+str(time.time()-startTime))	

with open(args.save, 'w') as f:
	cPickle.dump([pwz,pzd], f)

# with open(datasetName + '_plsa.pickle') as f:
# 	pwz,pzd = cPickle.load(f)
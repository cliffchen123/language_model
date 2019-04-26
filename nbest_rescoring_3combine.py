import sys
import os
import glob
import operator
import pickle
import cPickle
import numpy as np
if len(sys.argv) < 8:
	print 'no argument'
	sys.exit()
lstmpath1 = sys.argv[1]
lstmpath2 = sys.argv[2]
ngrampath = sys.argv[3]
nnlambda_start = float(sys.argv[4])
nnlambda_end = float(sys.argv[5])
nnlambda_start2 = float(sys.argv[6])
nnlambda_end2 = float(sys.argv[7])
ngramlambda_start = float(sys.argv[8])
ngramlambda_end = float(sys.argv[9])
lmscale_start = int(sys.argv[10])
lmscale_end = int(sys.argv[11])
save_path = sys.argv[12]
# if os.path.isfile(lstmpath)==False:
# 	print('No file')
# 	sys.exit()

with open(lstmpath1+'lstm.cpickle','r') as fr:
	lstmP = cPickle.load(fr)

with open(lstmpath2+'lstm.cpickle','r') as fr:
	lstmP2 = cPickle.load(fr)

with open(ngrampath+'lmscore.cpickle','r') as fr:
	ngramP = cPickle.load(fr)

with open(ngrampath+'acscore.cpickle','r') as fr:
	acscore = cPickle.load(fr)

with open('data/delta_data/lattice_uni/nbest/lstm.cpickle','r') as fr:
	uni_lstmP = cPickle.load(fr)

filenames = []
with open('data/delta_data/lattice_uni/nbest/text.filt','r') as fr:
	for i in fr:
		line = i.split(' ')
		filenames.append(line[0])

os.chdir(lstmpath1)
text = dict()
for i in range(1,9):
	for file in glob.glob(str(i)+"/*.onlytext"):
		fn = file.replace('.onlytext','')
		filenames[filenames.index(fn[2:])] = fn
		with open(file,'r') as f:
			text[fn] = []
			for i in f:
				text[fn].append(i)

nnlambda_range = [x/float(10) for x in range(int(nnlambda_start*10),int((nnlambda_end)*10+1))]
nnlambda_range2 = [x/float(10) for x in range(int(nnlambda_start2*10),int((nnlambda_end2)*10+1))]
ngramlambda_range = [x/float(10) for x in range(int(ngramlambda_start*10),int((ngramlambda_end)*10+1))]
for scale in range(lmscale_start,lmscale_end+1,3):
	for ngl in ngramlambda_range:
		for nnl in nnlambda_range:
			for nnl2 in nnlambda_range2:
				score_list = dict()
				rescore_sent = dict()
				print 'scale: '+str(scale)+', ngram lambda: '+str(ngl)+', nn lambda: '+str(nnl)
				ft = open('../../../../'+save_path+'rescore/'+'text.'+str(scale)+'.'+str(ngl)+'.'+str(nnl)+'.'+str(nnl2)+'.nbest','w')
				fc = open('../../../../'+save_path+'rescore/'+'char.'+str(scale)+'.'+str(ngl)+'.'+str(nnl)+'.'+str(nnl2)+'.nbest','w')
				for i in filenames:
					try:
						combine2adalstm = nnl2*lstmP[i]+(1-nnl2)*lstmP2[i]
						combine_orilstm_adalstm = nnl*uni_lstmP[i]+(1-nnl)*combine2adalstm
						combine_ngram_lstm = ngl*ngramP[i]+(1-ngl)*combine_orilstm_adalstm
						score_list[i] = acscore[i] + scale*combine_ngram_lstm
					except:
						import pdb;pdb.set_trace()

					max_sent_idx = score_list[i].tolist().index(max(score_list[i]))
					max_sent = text[i][max_sent_idx]
					ft.write(i[2:]+' '+max_sent)

					fc.write(i[2:]+' ')
					line = max_sent.split(' ')
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

				with open('../../../../'+save_path+'rescore/'+'score.'+str(scale)+'.'+str(ngl)+'.'+str(nnl)+'.'+str(nnl2)+'.cpickle','w') as fw:
					cPickle.dump(score_list,fw)
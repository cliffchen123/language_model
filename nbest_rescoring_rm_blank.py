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
lstmpath = sys.argv[1]
ngrampath = sys.argv[2]
unilstmpath = sys.argv[3]
nnlambda_start = float(sys.argv[4])
nnlambda_end = float(sys.argv[5])
ngramlambda_start = float(sys.argv[6])
ngramlambda_end = float(sys.argv[7])
lmscale_start = int(sys.argv[8])
lmscale_end = int(sys.argv[9])
job = int(sys.argv[10])
# if os.path.isfile(lstmpath)==False:
# 	print('No file')
# 	sys.exit()

with open(lstmpath+'lstm.cpickle','r') as fr:
	lstmP = cPickle.load(fr)
	new_lstmP = dict()
	for i in lstmP:
		new_lstmP[i.replace('-','')] = lstmP[i]
	lstmP = new_lstmP
with open(ngrampath+'lmscore.cpickle','r') as fr:
	ngramP = cPickle.load(fr)
	new_ngramP = dict()
	for i in ngramP:
		new_ngramP[i.replace('-','')] = ngramP[i]
	ngramP = new_ngramP
with open(ngrampath+'acscore.cpickle','r') as fr:
	acscore = cPickle.load(fr)
	new_acscore = dict()
	for i in acscore:
		new_acscore[i.replace('-','')] = acscore[i]
	acscore = new_acscore
with open(unilstmpath+'lstm.cpickle','r') as fr:
	uni_lstmP = cPickle.load(fr)
	new_uni_lstmP = dict()
	for i in uni_lstmP:
		new_uni_lstmP[i.replace('-','')] = uni_lstmP[i]
	uni_lstmP = new_uni_lstmP

filenames = []
blank = []
fw = open(unilstmpath+'text.filt.rm','w')
with open(unilstmpath+'text.filt','r') as fr:
	for i in fr:
		line = i.split()
		filenames.append(line[0].replace('-',''))
		if len(line)==1:
			blank.append(True)
		else:
			blank.append(False)
			fw.write(i)

fw = open(unilstmpath+'char.filt.rm','w')
with open(unilstmpath+'char.filt','r') as fr:
	index=0
	for i in fr:
		if blank[index]==False:
			fw.write(i)
		index+=1

fw.close()
os.chdir(lstmpath)
text = dict()
for i in range(1,job+1):
	for file in glob.glob(str(i)+"/*.onlytext"):
		fn = file.replace('.onlytext','').replace('-','')
		filenames[filenames.index(fn[(fn.find('/')+1):])] = fn
		with open(file,'r') as f:
			text[fn] = []
			for i in f:
				text[fn].append(i)

nnlambda_range = [x/float(10) for x in range(int(nnlambda_start*10),int((nnlambda_end)*10+1))]
ngramlambda_range = [x/float(10) for x in range(int(ngramlambda_start*10),int((ngramlambda_end)*10+1))]
for scale in range(lmscale_start,lmscale_end+1,1):
	for ngl in ngramlambda_range:
		for nnl in nnlambda_range:
			score_list = dict()
			rescore_sent = dict()
			print 'scale: '+str(scale)+', ngram lambda: '+str(ngl)+', nn lambda: '+str(nnl)
			ft = open('rescore/'+'text.'+str(scale)+'.'+str(ngl)+'.'+str(nnl)+'.nbest','w')
			fc = open('rescore/'+'char.'+str(scale)+'.'+str(ngl)+'.'+str(nnl)+'.nbest','w')
			idx = 0
			for i in filenames:
				if blank[idx]:
					idx+=1
					continue
				try:
					score_list[i] = acscore[i] + scale*(ngl*ngramP[i]+(1-ngl)*(nnl*uni_lstmP[i]+(1-nnl)*lstmP[i]))
				except:
					import pdb;pdb.set_trace()

				max_sent_idx = score_list[i].tolist().index(max(score_list[i]))
				max_sent = text[i][max_sent_idx]
				ft.write(i[(i.find('/')+1):]+' '+max_sent)

				fc.write(i[(i.find('/')+1):]+' ')
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
				idx+=1
			with open('rescore/'+'score.'+str(scale)+'.'+str(ngl)+'.'+str(nnl)+'.cpickle','w') as fw:
				cPickle.dump(score_list,fw)
			
				



		# # reduce lm scale by word frequency
		# with open('data/voc_num.pickle', 'rb') as handle:
		#     voc_num = pickle.load(handle)
		# newlmscale = lmscale
		# with open(path.replace('.lmresult',''),'r') as text:
		# 	index = 0
		# 	for i in text:
		# 		if index==20:
		# 			break
		# 		line = i.split(' ')
		# 		line.pop()
		# 		for w in line:
		# 			if (w not in voc_num.keys()) or (voc_num[w]<=10):
		# 				newlmscale = lmscale - 1
		# 		index+=1
		# lmscale = newlmscale


		# re_nbest = []
		# with open(path.replace('.onlytext.lmresult.lstmscore',''),'r') as ori_best:
		# 	index = 0
		# 	for i in ori_best:
		# 		line = i.split(' ')
		# 		score_list[index] = float(line[0]) + (score_list[index]*nnlambda + float(line[1])*(1-nnlambda))*lmscale
		# 		index+=1
		# with open(path.replace('.lmresult.lstmscore',''),'r') as text:
		# 	index = 0
		# 	for i in text:
		# 		re_nbest.append((i,score_list[index]))
		# 		index+=1
		# 	sorted_renbest = sorted(re_nbest, key=operator.itemgetter(1), reverse = True)
		# fw = open(path+'.renbest','w')
		# for i in sorted_renbest:
		# 	fw.write(i[0])
		# fw.close()

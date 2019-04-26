import math
import numpy as np
import os

def mySigmoid(x):
	return (1 / (1 + math.exp(-x))-0.5)*2

class smmClass:
	def __init__(self):
		self.allPtList = None
		self.lambdaList = None

class collection:
	def __init__(self, background):
		# self.dictSize = dictSize
		# self.termFrequency = termFrequency
		# self.sumOfTerm = 0
		# for i in self.termFrequency:
		# 	self.sumOfTerm+=i
		self.background = background
	def pg(self,term):
		# return float(self.termFrequency[term])/self.sumOfTerm
		return self.background[term]

	# def pd(self,term,doc):
	# 	termNum = len(doc)
	# 	return float(doc.count(term))/termNum

	def EStep(self,terms,allPtList,lambdaList):
		result = []
		for i in range(len(terms)):
			numerator = [0]*2
			denominator = 0
			for j in range(2):
				temp = lambdaList[j]*allPtList[i][j]
				numerator[j] = temp
				denominator += temp
			result.append(np.array(numerator)/float(denominator))
		return result

	def MStep(self,e,feedback,terms,allPtList,lambdaList):
		
		#swlm
		numerator = []
		denominator = 0
		for t in range(len(terms)):
			temp = feedback[terms[t]]*e[t][1]
			denominator += temp
			numerator.append(temp)
		new_sm = np.array(numerator)/float(denominator)
		for i in range(len(allPtList)):
			allPtList[i][1] = new_sm[i]

		

		return allPtList,lambdaList

	def training(self,iteration,feedback):
		# feedback1d = reduce(lambda x,y :x+y ,feedback)
		terms = []
		for i in range(len(feedback)):
			if feedback[i]!=0:
				terms.append(i)
		allPtList = []
		index = 0
		lambdaList = [0.5,0.5]
		sw_init = float(1)/len(terms)
		for t in terms:
			allPtList.append([self.pg(t),sw_init])
			index+=1
		for it in range(iteration):
			e = self.EStep(terms,allPtList,lambdaList)
			allPtList,lambdaList = self.MStep(e,feedback,terms,allPtList,lambdaList)
		print(str(iteration)+' em is finished')
		# 	objectFun=0
		# 	for d in feedback:
		# 		for t in d:
		# 			objectFun+=np.log(allPtList[terms.index(t)][0]*lambdaList[0]+allPtList[terms.index(t)][1]*lambdaList[1]+allPtList[terms.index(t)][2]*lambdaList[2])
		# 	if 'nextObjectFun' in locals() and objectFun<nextObjectFun:
		# 		break
		# 	nextObjectFun=objectFun
		# 	print objectFun
		# print "-------------------"

		return allPtList,lambdaList

# tdorsd = 'td'
# if tdorsd=='sd':
# 	docPath = 'Spoken_Doc'
# elif tdorsd=='td':
# 	docPath = 'SPLIT_DOC_WDID_NEW'
# print('dataset: '+tdorsd)	
# queryType = 'short'
# dictSize = 51253
# filenameDir = []
# termFrequency = [0]*dictSize
# path = docPath
# fw = open('allofdoc.txt','w')
# for filename in os.listdir(path):
# 	filenameDir.append(filename)
# 	f = open(path+'/'+filename,'r')
# 	index=-1
# 	for i in f:
# 		index+=1
# 		if index < 3:
# 			continue
# 		fw.write(i)
# 		i = i.split(' ')
# 		i.pop()
# 		for j in i:
# 			if j=='' or j=='-1':
# 				continue
# 			termFrequency[int(j)] += 1
# 	f.close()
# fw.close()


# #read Query
# filenameQueryDir = [0]*2265
# queryDoc = []
# path = 'QUERY_WDID_NEW_middle'
# fileIndex = 0
# for filename in os.listdir(path):
# 	filenameQueryDir[fileIndex] = filename
# 	temp = []
# 	f = open(path+'/'+filename,'r')
# 	for i in f:
# 		line = i.split(' ')
# 		line.pop()
# 		line = map(int,line)
# 		temp+=line
# 	queryDoc.append(temp)
# 	f.close()

# #read search result
# f = open('one_search_result_'+queryType+'_'+tdorsd+'.txt','r')
# searchResult = []
# for k in range(16):
# 	searchResult.append([])
# index = -1
# for i in f:
# 	if i[0]=='Q':
# 		index += 1
# 	elif i[0] == 'V':
# 		searchResult[index].append(i[:-2])
# f.close()

# print '### SimpleMixtureModel ###'
# #read feedback && training
# c = collection(dictSize,termFrequency)
# swlm = [0]*16
# expansion_num = 15
# print 'expansion_document_num = '+str(expansion_num)
# feedbackList = []
# for q in range(16):
# 	feedback = []
# 	for i in range(expansion_num):
# 		f = open(docPath+'/'+searchResult[q][i],'r')
# 		index=-1
# 		temp = []
# 		for j in f:
# 			index+=1
# 			if index < 3:
# 				continue
# 			line = j.split(' ')
# 			line.pop()
# 			for k in line:
# 				if k=='' or k=='-1':
# 					continue
# 				temp += [int(k)]
# 		feedback.append(temp)
# 	feedbackList.append(feedback)
# 	swlm[q] = smmClass()
# 	swlm[q].allPtList,swlm[q].lambdaList = c.training(100,queryDoc[q],feedback)
# print("EM training lamda:"+str(swlm[0].lambdaList))

# allTermsP = []
# newFeedbackList = []
# feedback_unigram = [None]*16
# index = 0
# for i in feedbackList:
# 	feedback1d = reduce(lambda x,y :x+y ,i)
# 	terms = []
# 	for t in feedback1d:
# 		if terms.count(t)==0:
# 			terms.append(t)

# 	#feedback unigram
# 	feedback_unigram[index] = [0]*51253
# 	for j in terms:
# 		feedback_unigram[index][j] = feedback1d.count(j)
# 	feedback_unigram[index] = np.array(feedback_unigram[index]) / float(sum(feedback_unigram[index]))

# 	termsP = []
# 	swlm[index].completeP = [[0]*51253,[0]*51253]
# 	for j in range(51253):
# 		swlm[index].completeP[0][j] = float(termFrequency[j])/c.sumOfTerm
# 	for j in range(len(terms)):
# 		swlm[index].completeP[1][terms[j]] = swlm[index].allPtList[j][1]
# 	index += 1
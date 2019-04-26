class Node:
	def __init__(self, nid, time):
		self.id = nid
		self.time = time
		self.next_arc = []
		self.pre_arc = []
		self.isStart = False
		self.state = None
		self.freeze = True

class Arc:
	def __init__(self, start, end, word, acscore, lmscore):
		self.start = start
		self.end = end
		self.word = word
		self.acscore = acscore
		self.lmscore = lmscore
		self.score = None
		self.freeze = True

def read_graph(filename):
	mysterious_number = 2.302585
	nodes = []
	arcs = []
	f = open(filename,'r')
	index = 0
	for i in f:
		if index < 2:
			index += 1
			continue
		elif index == 2:
			line = i.split()
			node_num = int(line[0].replace('N=',''))
			ark_num = int(line[1].replace('L=',''))
		elif index < 3+node_num:
			line = i.split()
			time = float(line[1].replace('t=',''))
			nid = int(line[0].replace('I=',''))
			nodes.append(Node(nid,time))
		elif index < 3+node_num+ark_num:
			line = i.split()
			start = nodes[int(line[1].replace('S=',''))]
			end = nodes[int(line[2].replace('E=',''))]
			word = str(line[3].replace('W=',''))
			acscore = float(line[5].replace('a=',''))/mysterious_number
			lmscore = float(line[6].replace('l=',''))/mysterious_number
			arc = Arc(start,end,word,acscore,lmscore)
			arcs.append(arc)
			start.next_arc.append(arc)
			end.pre_arc.append(arc)
			if start.pre_arc == []:
				start.isStart = True
		index += 1
	f.close()
	return nodes,arcs

def write_graph(filename,nodes,arcs):
	with open(filename,'w') as fw:
		fw.write('VERSION=1.1\n')
		fw.write('UTTERANCE='+filename[filename.find('/')+1:filename.find('.')]+'\n')
		fw.write('N='+str(len(nodes))+'\tL='+str(len(arcs))+'\n')
		for i in range(len(nodes)):
			fw.write('I='+str(i)+'\tt='+str(nodes[i].time)+'\n')
		for j in range(len(arcs)):
			arc = arcs[j]
			fw.write('J='+str(j)+'\tS='+str(arc.start.id)+'\tE='+str(arc.end.id)+'\tW='+arc.word+'\tv=0.000000\ta='+str(arc.acscore)+'\tl='+str(arc.lmscore)+'\n')


def nbest_save_node(N,nodes,scale,beam):
	sents = []
	stack = []
	for i in nodes:
		if len(i.next_arc)!=0:
			i.next_arc = sorted(i.next_arc, key=lambda x:(x.acscore+scale*x.lmscore), reverse=True)
		if i.isStart:
			stack.append([i,'<s>',[],[],False])

	while stack!=[]:
		cur_node = stack[-1][0]
		cur_str = stack[-1][1]
		cur_acscore = stack[-1][2]
		cur_lmscore = stack[-1][3]
		pre_word_is_null =  stack[-1][4]
		stack.pop()
		if cur_node.next_arc==[]:
			sents.append([cur_str+' </s>',cur_acscore,cur_lmscore,sum(cur_acscore)+scale*sum(cur_lmscore)])
		else:
			for i in cur_node.next_arc[:beam]:
				if i.word=='!NULL':
					if pre_word_is_null:
						temp_acscore = cur_acscore[:]
						temp_lmscore = cur_lmscore[:]
						temp_acscore[-1]+=i.acscore
						temp_lmscore[-1]+=i.lmscore
						stack.append([i.end, cur_str, temp_acscore, temp_lmscore,True])
					else:
						stack.append([i.end, cur_str, cur_acscore+[i.acscore], cur_lmscore+[i.lmscore],True])
				else:
					if pre_word_is_null:
						temp_acscore = cur_acscore[:]
						temp_lmscore = cur_lmscore[:]
						temp_acscore[-1]+=i.acscore
						temp_lmscore[-1]+=i.lmscore
						stack.append([i.end, cur_str+' '+i.word, temp_acscore, temp_lmscore,False])
					else:
						stack.append([i.end, cur_str+' '+i.word, cur_acscore+[i.acscore], cur_lmscore+[i.lmscore],False])
	sents = sorted(sents, key=lambda x:x[3], reverse=True)

	return sents[:N]

def nbest(N,nodes,scale,beam):
	sents = []
	stack = []
	for i in nodes:
		if len(i.next_arc)!=0:
			for j in i.next_arc:
				j.score = j.acscore+scale*j.lmscore
			i.next_arc = sorted(i.next_arc, key=lambda x:x.score, reverse=True)
		if i.isStart:
			stack.append([i,'<s>',0,0])

	while stack!=[]:
		cur_node = stack[-1][0]
		cur_str = stack[-1][1]
		cur_acscore = stack[-1][2]
		cur_lmscore = stack[-1][3]
		stack.pop()
		if cur_node.next_arc==[]:
			sents.append([cur_str+' </s>',cur_acscore,cur_lmscore,cur_acscore+scale*cur_lmscore])
		else:
			# max_score = -float('inf')
			# for i in cur_node.next_arc:
			# 	if i.score>max_score:
			# 		max_score = i.score
			# min_score = max_score-beam
			for i in cur_node.next_arc[:beam]:
				# if min_score>i.score:
				# 	break
				if i.word=='!NULL':
					stack.append([i.end, cur_str, cur_acscore+i.acscore, cur_lmscore+i.lmscore])
				else:
					stack.append([i.end, cur_str+' '+i.word, cur_acscore+i.acscore, cur_lmscore+i.lmscore])
		# print cur_node.id
	sents = sorted(sents, key=lambda x:x[3], reverse=True)

	return sents[:N]

def nbest_BFS_approx(N,nodes,scale,beam,approximation=10,beam_approximation=20):
	sents = []
	stack = []
	for i in nodes:
		sents.append([])
		if len(i.next_arc)!=0:
			for j in i.next_arc:
				j.score = j.acscore+scale*j.lmscore
			i.next_arc = sorted(i.next_arc, key=lambda x:x.score, reverse=True)

	time = []
	for i in nodes:
		time.append(i.time)
	time[-1]+=0.1 # the last node must be the first in the stack
	nodes_idx = sorted(range(len(time)), key=lambda x:time[x], reverse=True)
	for i in nodes_idx:
		stack.append(nodes[i])

	while stack!=[]:
		cur_node = stack[-1]
		stack.pop()
		if cur_node.pre_arc==[]:
			sents[cur_node.id]=[[[],0,0,0]]
		else:
			for i in cur_node.pre_arc[:beam]:
				pre_node = i.start

				max_score = -float('inf')
				for j in sents[pre_node.id]:
					if j[3]>max_score:
						max_score = j[3]
				min_score = max_score-beam				
				
				used_gram = dict()
				if i.word=='!NULL':
					for j in sents[pre_node.id]:
						if min_score <= j[3]:
							sents[cur_node.id].append([j[0],j[1]+i.acscore,j[2]+i.lmscore,j[3]+i.score])
						else:
							break
				else:
					for j in sents[pre_node.id]:
						if min_score <= j[3]:
							seq = ' '.join(j[0][-approximation:])
							if seq not in used_gram.keys():
								used_gram[seq]=0
								sents[cur_node.id].append([j[0]+[i.word],j[1]+i.acscore,j[2]+i.lmscore,j[3]+i.score])
							elif used_gram[seq]<10:
								used_gram[seq]+=1
								sents[cur_node.id].append([j[0]+[i.word],j[1]+i.acscore,j[2]+i.lmscore,j[3]+i.score])
						else:
							break
			sents[cur_node.id] = sorted(sents[cur_node.id], key=lambda x:x[3], reverse=True)

		last_node_id = cur_node.id

	sents = sorted(sents[last_node_id], key=lambda x:x[3], reverse=True)
	return sents[:N]

def nbest_BFS(N,nodes,scale,beam):
	sents = []
	stack = []
	for i in nodes:
		sents.append([])
		if len(i.next_arc)!=0:
			for j in i.next_arc:
				j.score = j.acscore+scale*j.lmscore
			i.next_arc = sorted(i.next_arc, key=lambda x:x.score, reverse=True)

	time = []
	for i in nodes:
		time.append(i.time)
	time[-1]+=0.1 # the last node must be the first in the stack
	nodes_idx = sorted(range(len(time)), key=lambda x:time[x], reverse=True)
	for i in nodes_idx:
		stack.append(nodes[i])

	while stack!=[]:
		cur_node = stack[-1]
		stack.pop()
		if cur_node.pre_arc==[]:
			sents[cur_node.id]=[[[],0,0,0]]
		else:
			for i in cur_node.pre_arc[:]:
				pre_node = i.start

				# max_score = -float('inf')
				# for j in sents[pre_node.id]:
				# 	if j[3]>max_score:
				# 		max_score = j[3]
				# min_score = max_score-beam				
				
				used_gram = dict()
				if i.word=='!NULL':
					for j in sents[pre_node.id]:
						sents[cur_node.id].append([j[0],j[1]+i.acscore,j[2]+i.lmscore,j[3]+i.score])
				else:
					for j in sents[pre_node.id][:beam]:
						sents[cur_node.id].append([j[0]+[i.word],j[1]+i.acscore,j[2]+i.lmscore,j[3]+i.score])

			sents[cur_node.id] = sorted(sents[cur_node.id], key=lambda x:x[3], reverse=True)

		last_node_id = cur_node.id

	sents = sorted(sents[last_node_id], key=lambda x:x[3], reverse=True)
	return sents

def pruning(filename,N,scale,beam):
	nodes, arcs = read_graph(filename)
	sents = []
	stack = []
	for i in nodes:
		sents.append([])
		if len(i.next_arc)!=0:
			for j in i.next_arc:
				j.score = j.acscore+scale*j.lmscore
			i.next_arc = sorted(i.next_arc, key=lambda x:x.score, reverse=True)

	time = []
	for i in nodes:
		time.append(i.time)
	time[-1]+=0.1 # the last node must be the first in the stack
	nodes_idx = sorted(range(len(time)), key=lambda x:time[x], reverse=True)
	for i in nodes_idx:
		stack.append(nodes[i])

	index = 0
	while stack!=[]:
		print(str(len(nodes))+': '+str(index))
		index+=1
		cur_node = stack[-1]
		stack.pop()
		if cur_node.pre_arc==[]:
			sents[cur_node.id]=[[[],0,0,0,[]]]
		else:
			for i in cur_node.pre_arc[:beam]:
				pre_node = i.start			

				max_score = -float('inf')
				for j in sents[pre_node.id]:
					if j[3]>max_score:
						max_score = j[3]
				min_score = max_score-beam
				
				for j in sents[pre_node.id]:
					if i.word=='!NULL':
						if min_score <= j[3]:
							sents[cur_node.id].append([j[0],j[1]+i.acscore,j[2]+i.lmscore,j[3]+i.score,j[4]+[i]])
					else:
						if min_score <= j[3]:
							sents[cur_node.id].append([j[0]+[i.word],j[1]+i.acscore,j[2]+i.lmscore,j[3]+i.score,j[4]+[i]])
			# sents[cur_node.id] = sorted(sents[cur_node.id], key=lambda x:x[3], reverse=True)

		last_node_id = cur_node.id

	sents = sorted(sents[last_node_id], key=lambda x:x[3], reverse=True)

	#pruning
	for i in sents[:N]:
		used_arc = i[4]
		for j in used_arc:
			j.freeze=False

	freeze_num = 0
	for i in arcs:
		if i.freeze==True:
			freeze_num+=1
	print len(arcs),freeze_num
	# import pdb;pdb.set_trace()
	return sents[:N], nodes

# test
# nodes, arcs = read_graph('data/ami-s5b/lattice_uni/1/AMI_ES2011a_H00_FEE041_0012427_0013579.lat')
# nodes, arcs = read_graph('data/ami-s5b/lattice_uni/1/AMI_ES2011a_H00_FEE041_0003714_0003915.lat')
# sents = nbest_BFS(1000,nodes,10,100)
# sents, nodes = pruning('data/ami-s5b/lattice_uni/1/AMI_ES2011a_H02_FEE043_0031442_0033864.lat',1000,10,20)
# for i in sents[980:1000]:
# 	print(str(i[1])+' '+str(i[2]))
# import pdb;pdb.set_trace()
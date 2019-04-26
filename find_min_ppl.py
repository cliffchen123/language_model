import sys
f = open(sys.argv[1],'r')
index = 1
min_ppl = float('inf')
next_element = False
for i in f:
	if index%3==0:
		Lambda = (index/3)/float(100)
		line = i.split(' ')
		for j in line:
			if next_element and float(j)<min_ppl:
				min_ppl = float(j)
				min_lambda = Lambda
			if j=='ppl=':
				next_element = True
			else:
				next_element = False

	index+=1
print(sys.argv[1]+' min ppl lambda: '+str(min_lambda))
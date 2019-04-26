import sys
from tempfile import mkstemp
from shutil import move
from os import remove, close
import os
if len(sys.argv) < 2:
	print 'no argument'
	sys.exit()
path = sys.argv[1]
ft = open(path+'/nbest.text','w')
fc = open(path+'/nbest.char','w')
fr = open(path+'/ambiguous','r')
ambiguousWord = []
replaceWord = []
for i in fr:
	line = i.split(' ')
	line.pop()
	ambiguousWord.append(line[0])
	replaceWord+=line[1:]
for person in range(1,9):
	filenameList = sorted(os.listdir(path+'/'+str(person)))
	temp = []
	for i in filenameList:
		if i.find('.renbest')!=-1:
			temp.append(i)
	filenameList = temp
	for filename in filenameList:
		file_path = path+'/'+str(person)+'/'+filename
		ft.write(filename.replace('.onlytext.lmresult.lstmscore.renbest','')+' ')
		fc.write(filename.replace('.onlytext.lmresult.lstmscore.renbest','')+' ')
		with open(file_path) as file:
			sentNum = 0
			sents = []
			max_replace = 0
			selectedSent = ''
			first = True
			first_ambiguous = 0
			first_replace = 0
			for i in file:
				line_ambiguous = 0
				line_replace = 0
				line = i.split(' ')
				for j in line:
					if j in ambiguousWord:
						line_ambiguous += 1
					if j in replaceWord:
						line_replace += 1
				if first:
					first_ambiguous = line_ambiguous
					first_replace = line_replace
					selectedSent = i
					first = False
					if line_ambiguous==0:
						break
				else:
					ambiguous_dec = (first_ambiguous-line_ambiguous)
					if ambiguous_dec == (line_replace-first_replace):
						if ambiguous_dec > max_replace:
							selectedSent = i
							max_replace = ambiguous_dec
				
				sentNum += 1
				if sentNum == 50:  # 1:origin
					break

			ft.write(selectedSent)
			line = selectedSent.split(' ')
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

		# ft.write('\n')
		# fc.write('\n')
import sys
from tempfile import mkstemp
from shutil import move
from os import remove, close
import os
if len(sys.argv) < 2:
	print 'no argument'
	sys.exit()
path = sys.argv[1]
# ft = open(path+'/nbest.text','w')
# fc = open(path+'/nbest.char','w')
person = 1
while os.path.isdir(path+'/'+str(person)):
	filenameList = sorted(os.listdir(path+'/'+str(person)))

	for filename in filenameList:
		if filename.find('.onlytext')!=-1:
			continue
		file_path = path+'/'+str(person)+'/'+filename
		ft = open(file_path+'.onlytext','w')
		with open(file_path) as file:
			for i in file:
				line = i.split(' ')
				index=0
				# hassomething = False
				for j in line:
					index+=1
					if index>4 and j!='!NULL' and j!='<sb>' and j!='{sil}' and j!='<unk>' and j!='{noise}' and j!='{japanese}' and j!='{sing}' and j!='{french}' and j!='</s>' and j!='\n' and '{' not in j:
						ft.write(j+' ')
				# 		hassomething = True
				# if hassomething==False :
				# 	ft.write('<unk> ')
				ft.write('\n')
		ft.close()
	person+=1
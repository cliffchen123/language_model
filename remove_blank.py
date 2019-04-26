import cPickle
import argparse
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--spk2utt', type=str, default='',
                    help='spk2utt file')
parser.add_argument('--save', type=str, default='',
                    help='save file')
parser.add_argument('--dict', type=str, default='',
                    help='dictionary')
args = parser.parse_args()

dictionary = []
if args.dict!='':
	with open(args.dict,'r') as fr:
		dictionary = cPickle.load(fr)
fr = open(args.spk2utt,'r')
fw = open(args.save,'w')
for i in fr:
	line = i.split()
	speaker = line.pop(0)
	temp = speaker+' '
	writable = False
	for j in line:
		if j!='!NULL' and j!='<unk>' and j!='<sb>' and j!='{sil}' and j!='{noise}' and j!='{japanese}' and j!='{sing}' and j!='{french}' and j!='</s>' and j!='\n' and '{' not in j:
			if args.dict=='' or j in dictionary:
				temp += (j+' ')
				writable = True
			else:
				temp += ('<unk> ')
	temp += ('\n')
	if writable:
		fw.write(temp)
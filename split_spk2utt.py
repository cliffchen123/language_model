import argparse
import re
import cPickle
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--spk2utt', type=str, default='./data/delta_data/spk2utt',
                    help='speaker to utterences')
parser.add_argument('--preprocess', type=str,  default='delta',
                    help='the category of preprocess')
parser.add_argument('--save', type=str,  default='splitdata.cpickle',
                    help='save file')
parser.add_argument('--save_folder',
                    help='save folder')
args = parser.parse_args()

# extract topic feature
f = open(args.spk2utt,'r')
topic = dict()
speaker_doc = dict()
utt2spk_train = []
for i in f:
    if args.preprocess=='delta':
        topic_idx = i[:i.find('-')]
    elif args.preprocess=='ami':
        topic_idx = i.split('_')[3]
    elif args.preprocess=='delta_new':
        topic_idx = ''
        half_line = i.split(' ')[0]
        hasFound = 0
        half_line = half_line.replace('-',' ').replace('.',' ').replace('!',' ! ')
        for j in half_line.split(' '):
            if 'A'<=j[0]<='Z':
                topic_idx += j
                hasFound += 1
            elif topic_idx!='':
                break
    line = i.split(' ')
    line.pop(0)
    if topic_idx not in topic:
        topic[topic_idx] = []
    topic[topic_idx].append(' '.join(line))
with open(args.save+'.cpickle','w') as fw:
    cPickle.dump(topic,fw)

fat = open(args.save_folder+'/all.train','w')
fav = open(args.save_folder+'/all.valid','w')
for i in topic:
    fw = open(args.save_folder+'/'+str(i)+'.txt','w')
    words = 0
    for j in topic[i]:
        fw.write(j)
        words+=len(j.split())
    print(i+' : '+str(words))
        
    ft = open(args.save_folder+'/'+str(i)+'.train','w')
    fv = open(args.save_folder+'/'+str(i)+'.valid','w')
    for j in range(len(topic[i])):
        if j%10==0:
            fv.write(topic[i][j])
            fav.write(topic[i][j])
        else:
            ft.write(topic[i][j])
            fat.write(topic[i][j])


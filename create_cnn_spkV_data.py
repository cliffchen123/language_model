import argparse
import cPickle
import numpy as np
import data
import sklearn.cluster as sk
import scipy.sparse as sp
import re
parser = argparse.ArgumentParser(description='create dataset of speker vector extractor')
parser.add_argument('--data', type=str, default='./data/delta_data',
                    help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=60,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--save_train', type=str,  default='./cnn_spkV_traindata.cpickle',
                    help='path to save the final model')
parser.add_argument('--save_valid', type=str,  default='./cnn_spkV_validdata.cpickle',
                    help='path to save the final model')
parser.add_argument('--save_test', type=str,  default='./cnn_spkV_testdata.cpickle',
                    help='path to save the final model')
parser.add_argument('--dict', type=str,  default='./data/delta_data/dictionary.cpickle',
                    help='path of dictionary')
parser.add_argument('--spk2utt', type=str, default='./data/delta_data/spk2utt',
                    help='speaker to utterences')
parser.add_argument('--valid_spk', type=str, default='./data/delta_data/valid_spk2utt',
                    help='speaker to utterences of valid file')
parser.add_argument('--test_spk', type=str, default='./data/delta_data/test_spk2utt',
                    help='speaker to utterences of test file')
parser.add_argument('--preprocess', type=str,  default='delta',
                    help='the category of preprocess')
args = parser.parse_args()

np.random.seed(args.seed) 

def qle_score(queryV, spkV):
     return np.dot(queryV,np.log(spkV))


corpus = data.Corpus(args.data, args.bptt, args.dict)
with open('dictionary.cpickle', 'wb') as f:
    cPickle.dump(corpus.dictionary.idx2word, f, cPickle.HIGHEST_PROTOCOL)


# extract speaker feature
f = open(args.spk2utt,'r')
speaker_unigram = dict()
bg_unigram = np.array([1]*len(corpus.dictionary))
bg_unigram = bg_unigram/float(sum(bg_unigram))
utt2spk_train = []
sents = []
spk_count = []
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
     if spk not in spk_count:
          spk_count.append(spk)
     utt2spk_train.append(spk)
     line = i.replace('\n','').split(' ')
     line.pop(0)
     sents.append([])
     if spk not in speaker_unigram:
          speaker_unigram[spk] = [0.0]*len(corpus.dictionary)

     for j in line:
          if j not in corpus.dictionary.word2idx:
               speaker_unigram[spk][corpus.dictionary.word2idx['<unk>']] += 1
               sents[-1].append(corpus.dictionary.word2idx['<unk>'])
          else:
               speaker_unigram[spk][corpus.dictionary.word2idx[j]] += 1
               sents[-1].append(corpus.dictionary.word2idx[j])
print('train speaker number: '+str(len(spk_count)))
print(spk_count)
utt_each_spk_start = []
last = ''
for i in range(len(utt2spk_train)):
     if last!=utt2spk_train[i]:
         utt_each_spk_start.append(i)
     last = utt2spk_train[i]
utt_each_spk_start.append(len(utt2spk_train))
# kmeans = sk.KMeans(n_clusters=5, random_state=args.seed, algorithm="full").fit(speaker_unigram.values())
# print kmeans.labels_

alpha = 0.05
for i in speaker_unigram:
     speaker_unigram[i] = np.log(alpha*(np.array(speaker_unigram[i])/float(sum(speaker_unigram[i]))) + (1-alpha)*(bg_unigram))

# kmeans_unigram = [None]*4
# for i in range(len(kmeans_unigram)):
#      kmeans_unigram[i] = np.array([0]*len(corpus.dictionary))
#      for j in speaker_unigram:
#           if kmeans.labels_[speaker_unigram.keys().index(j)]==i:
#                kmeans_unigram[i] = kmeans_unigram[i] + speaker_unigram[j]

# sent_matrix = sp.csr_matrix([])
sent_termf = np.array([0.0]*len(corpus.dictionary))
for j in sents[0]:
     sent_termf[j] = sents[0].count(j)/float(len(sents[0]))
sent_matrix = sp.csr_matrix(sent_termf)
for i in range(1,len(sents)):
     sent_termf = np.array([0.0]*len(corpus.dictionary))
     for j in sents[i]:
          sent_termf[j] = sents[i].count(j)/float(len(sents[i]))
     sent_matrix = sp.vstack([sent_matrix,sp.csr_matrix(sent_termf)])
     # sent_matrix.append(sp.csr_matrix(sent_termf))
# import pdb;pdb.set_trace()
score_matrix = (sent_matrix*sp.csr_matrix(np.transpose(np.matrix(speaker_unigram.values())))).todense()
# score_matrix = np.matrix(sent_matrix)*np.transpose(np.matrix(speaker_unigram.values()))


sents_train_X = []
sents_train_Y = []
speakers = speaker_unigram.keys()
for i in range(len(sents)):
     sents_train_X.append(sents[i])
     sents_train_Y.append(utt2spk_train[i])
     score_list = score_matrix[i].tolist()[0]

     spk_idx = speakers.index(utt2spk_train[i])
     score_list[spk_idx] = 0
     min_score = min(score_list)
     min_score_idx = score_list.index(min_score)
     # print min_score_idx

     utt_num = utt_each_spk_start[min_score_idx+1] - utt_each_spk_start[min_score_idx]
     rand = int(np.random.rand()*utt_num) + utt_each_spk_start[min_score_idx]
     # print rand
     sents_train_X.append(sents[rand])
     sents_train_Y.append('-)'+utt2spk_train[i])

with open(args.save_train,'w') as fw:
     cPickle.dump([sents_train_X,sents_train_Y],fw)
# import pdb;pdb.set_trace()

f = open(args.valid_spk,'r')
utt2spk_valid = []
sents = []
spk_count = []
spk_notin_train = []
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
     if spk not in spk_count:
          spk_count.append(spk)
     if spk not in utt2spk_train and spk not in spk_notin_train:
          spk_notin_train.append(spk)
          continue
     utt2spk_valid.append(spk)
     line = i.replace('\n','').split(' ')
     line.pop(0)
     sents.append([])
     # import pdb;pdb.set_trace()
     for j in line:
          if j not in corpus.dictionary.word2idx:
               sents[-1].append(corpus.dictionary.word2idx['<unk>'])
          else:
               sents[-1].append(corpus.dictionary.word2idx[j])
print('valid speaker number: '+str(len(spk_count)))
print('valid not in train speaker: '+str(len(spk_notin_train)))
print(spk_count)
sents_valid_X = []
sents_valid_Y = []
for i in range(len(sents)):
     sents_valid_X.append(sents[i])
     sents_valid_Y.append(utt2spk_valid[i])

with open(args.save_valid,'w') as fw:
     cPickle.dump([sents_valid_X,sents_valid_Y],fw)





f = open(args.test_spk,'r')
utt2spk_test = []
sents = []
spk_count = []
spk_notin_train = []
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
     if spk not in spk_count:
          spk_count.append(spk)
     if spk not in utt2spk_train and spk not in spk_notin_train:
          spk_notin_train.append(spk)
          continue
     utt2spk_test.append(spk)
     line = i.replace('\n','').split(' ')
     line.pop(0)
     sents.append([])
     # import pdb;pdb.set_trace()
     for j in line:
          if j not in corpus.dictionary.word2idx:
               sents[-1].append(corpus.dictionary.word2idx['<unk>'])
          else:
               sents[-1].append(corpus.dictionary.word2idx[j])
print('test speaker number: '+str(len(spk_count)))
print('test not in train speaker: '+str(len(spk_notin_train)))
print(spk_count)
sents_test_X = []
sents_test_Y = []
for i in range(len(sents)):
     sents_test_X.append(sents[i])
     sents_test_Y.append(utt2spk_test[i])

with open(args.save_test,'w') as fw:
     cPickle.dump([sents_test_X,sents_test_Y],fw)

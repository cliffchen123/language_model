import argparse
import cPickle
import numpy as np
import data
import sklearn.cluster as sk
import scipy.sparse as sp
parser = argparse.ArgumentParser(description='create dataset of speker vector extractor')
parser.add_argument('--data', type=str, default='./data/delta_data',
                    help='location of the data corpus')
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

utt_each_spk_start = []
last = ''
for i in range(len(utt2spk_train)):
     if last!=utt2spk_train[i]:
         utt_each_spk_start.append(i)
     last = utt2spk_train[i]
utt_each_spk_start.append(len(utt2spk_train))
# kmeans = sk.KMeans(n_clusters=5, random_state=args.seed, algorithm="full").fit(speaker_unigram.values())
# print kmeans.labels_

speaker_unigram_log = dict()
alpha = 0.99
for i in speaker_unigram:
    speaker_unigram[i] = alpha*(np.array(speaker_unigram[i])/float(sum(speaker_unigram[i]))) + (1-alpha)*(bg_unigram)
    speaker_unigram_log[i] = np.log(speaker_unigram[i])

kl_score = np.matrix(speaker_unigram.values())*np.matrix(speaker_unigram_log.values()).T
print(kl_score)
import pdb;pdb.set_trace()

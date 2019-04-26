import os
import torch
import cPickle

class Dictionary(object):
    def __init__(self, dict_path=''):
        self.word2idx = {}
        self.idx2word = []
        if dict_path!='':
            with open(dict_path, 'rb') as f:
                self.idx2word = cPickle.load(f)
                for i in range(len(self.idx2word)):
                    self.word2idx[self.idx2word[i]] = i
    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, bptt=45, dict_path=''):
        self.dictionary = Dictionary(dict_path)
        self.train = self.tokenize(os.path.join(path, 'train.txt'), bptt, dict_path)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'), bptt, dict_path)
        self.test = self.tokenize(os.path.join(path, 'test.txt'), bptt, dict_path)

    def tokenize(self, path, bptt, dict_path):
        """Tokenizes a text file."""
        assert os.path.exists(path)

        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = ['<s>'] + line.split() + ['<eos>'] + (bptt-2-len(line.split()))*['<no>']
                tokens += len(words)
                if dict_path == '':
                    for word in words:
                        self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = ['<s>'] + line.split() + ['<eos>'] + (bptt-2-len(line.split()))*['<no>']
                for word in words:
                    if word in self.dictionary.word2idx:
                        ids[token] = self.dictionary.word2idx[word]
                    else:
                        ids[token] = self.dictionary.word2idx['<unk>']
                    token += 1

        return ids

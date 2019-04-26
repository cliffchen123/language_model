import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, spk_nhid, nlayers, bptt, spk_num, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        real_spk_nhid = spk_nhid/2*2
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp, padding_idx=0)
        # self.spk_dense = nn.Linear(nhid*(bptt/2-1),nhid)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp+real_spk_nhid, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp+real_spk_nhid, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        self.tanh = nn.Tanh()

        # extractor
        self.conv1 = nn.Conv1d(ninp, spk_nhid/2, kernel_size=1)
        self.conv2 = nn.Conv1d(ninp, spk_nhid/2, kernel_size=2)
        # self.conv3 = nn.Conv1d(ninp, spk_nhid/3, kernel_size=3)
        self.max_pool1 = nn.MaxPool1d(bptt)
        self.max_pool2 = nn.MaxPool1d(bptt-1)
        # self.max_pool3 = nn.MaxPool1d(bptt-2)
        self.extractor_output = [None]*spk_num
        for i in range(spk_num):
            self.extractor_output[i] = nn.Linear(real_spk_nhid, 1)
        self.sigmoid = F.sigmoid

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        # self.init_weights()
        # self.spk_dense.bias.data.fill_(0)

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, spk, pos_sent, neg_sent):

        pos_emb = self.drop(self.encoder(pos_sent))
        pos_emb = pos_emb.permute(0,2,1)
        pos_c1 = F.relu(self.conv1(pos_emb))
        pos_c2 = F.relu(self.conv2(pos_emb))
        # pos_c3 = F.relu(self.conv3(pos_emb))
        pos_p1 = self.max_pool1(pos_c1).view(pos_c1.size(0),pos_c1.size(1))
        pos_p2 = self.max_pool2(pos_c2).view(pos_c2.size(0),pos_c1.size(1))
        # pos_p3 = self.max_pool3(pos_c3).view(pos_c3.size(0),pos_c1.size(1))
        speaker_vec = torch.cat([pos_p1,pos_p2],1)
        pos_spk_output = self.extractor_output[spk](speaker_vec)
        pos_spk_output = self.sigmoid(pos_spk_output)

        neg_emb = self.drop(self.encoder(neg_sent))
        neg_emb = neg_emb.permute(0,2,1)
        neg_c1 = F.relu(self.conv1(neg_emb))
        neg_c2 = F.relu(self.conv2(neg_emb))
        # neg_c3 = F.relu(self.conv3(neg_emb))
        neg_p1 = self.max_pool1(neg_c1).view(neg_c1.size(0),neg_c1.size(1))
        neg_p2 = self.max_pool2(neg_c2).view(neg_c2.size(0),neg_c1.size(1))
        # neg_p3 = self.max_pool3(neg_c3).view(neg_c3.size(0),neg_c1.size(1))
        neg_speaker_hidden = torch.cat([neg_p1,neg_p2],1)
        neg_spk_output = self.extractor_output[spk](neg_speaker_hidden)
        neg_spk_output = self.sigmoid(neg_spk_output)


        # speaker_hidden = self.tanh(self.spk_dense(spkfeature))

        word_emb = self.drop(self.encoder(input))
        word_emb = torch.cat([word_emb,speaker_vec.expand(word_emb.size(0),speaker_vec.size(0),speaker_vec.size(1))],2)
        output, hidden = self.rnn(word_emb, hidden)
        output = self.drop(output)
        
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        # import pdb; pdb.set_trace()
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden, pos_spk_output, neg_spk_output

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

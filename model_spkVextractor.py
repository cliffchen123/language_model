import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
class CNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, bptt, spk_num, dropout=0.5):
        super(CNNModel, self).__init__()
        self.nhid = nhid
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp, padding_idx=0)
        self.conv1 = nn.Conv1d(ninp, nhid, kernel_size=3)
        self.conv2 = nn.Conv1d(ninp, nhid, kernel_size=3)
        self.conv3 = nn.Conv1d(ninp, nhid, kernel_size=4)
        self.max_pool1 = nn.MaxPool1d(bptt-2)
        self.max_pool2 = nn.MaxPool1d(bptt-2)
        self.max_pool3 = nn.MaxPool1d(bptt-3)
        self.output = [None]*spk_num
        for i in range(spk_num):
            self.output[i] = nn.Linear(nhid, 1)
        self.sigmoid = F.sigmoid
        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        # if tie_weights:
        #     if nhid != ninp:
        #         raise ValueError('When using the tied flag, nhid must be equal to emsize')
        #     self.decoder.weight = self.encoder.weight

        # self.init_weights()

        # self.rnn_type = rnn_type
        # self.nhid = nhid
        # self.nlayers = nlayers

    # def init_weights(self):
    #     initrange = 0.1
    #     self.encoder.weight.data.uniform_(-initrange, initrange)
    #     self.decoder.bias.data.fill_(0)
    #     self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, spk):
        emb = self.drop(self.encoder(input))
        emb = emb.permute(0,2,1)
        # c1 = F.relu(self.conv1(emb))
        c2 = F.relu(self.conv2(emb))
        # c3 = F.relu(self.conv3(emb))
        
        # p1 = self.max_pool1(c1).view(c1.size(0),self.nhid)
        p2 = self.max_pool2(c2).view(c2.size(0),self.nhid)
        # p3 = self.max_pool3(c3).view(c3.size(0),self.nhid)
        
        # vec = torch.cat([p1,p2,p3],1)
        output = self.output[spk](p2)
        output = self.sigmoid(output)
        # import pdb;pdb.set_trace()
        return output

    def extract(self, input):
        # import pdb;pdb.set_trace()
        emb = self.drop(self.encoder(input))
        emb = emb.permute(0,2,1)
        # c1 = F.relu(self.conv1(emb))
        c2 = F.relu(self.conv2(emb))
        # c3 = F.relu(self.conv3(emb))
        
        # p1 = self.max_pool1(c1).view(c1.size(0),self.nhid)
        p2 = self.max_pool2(c2).view(c2.size(0),self.nhid)
        # p3 = self.max_pool3(c3).view(c3.size(0),self.nhid)
        
        # vec = torch.cat([p1,p2,p3],1)
        return p2

    # def init_hidden(self, bsz):
    #     weight = next(self.parameters()).data
    #     if self.rnn_type == 'LSTM':
    #         return (Variable(weight.new(self.nlayers*self.directional_num, bsz, self.nhid).zero_()),
    #                 Variable(weight.new(self.nlayers*self.directional_num, bsz, self.nhid).zero_()))
    #     else:
    #         return Variable(weight.new(self.nlayers*self.directional_num, bsz, self.nhid).zero_())

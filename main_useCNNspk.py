# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import data
import model_useCNNspk
import model_spkVextractor
import cPickle

parser = argparse.ArgumentParser(description='PyTorch Language Model adapation use CNN speaker vector ')
parser.add_argument('--data', type=str, default='./data/delta_data',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=300,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=30,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=60,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model_useCNNspk.pt',
                    help='path to save the final model')
parser.add_argument('--dict', type=str,  default='./data/delta_data/dictionary.cpickle',
                    help='path of dictionary')
parser.add_argument('--only_ada', action='store_true',
                    help='only adapation')
parser.add_argument('--valid_file', type=str, default='./data/delta_data/valid.txt',
                    help='speaker to utterences of test file')
parser.add_argument('--test_file', type=str, default='./data/delta_data/test.txt',
                    help='speaker to utterences of test file')
parser.add_argument('--cnn_spkV_model', type=str, default='./data/delta_data/model_cnnspk.pt',
                    help='speaker to utterences of test file')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data, args.bptt, args.dict)
with open('dictionary.cpickle', 'wb') as f:
    cPickle.dump(corpus.dictionary.idx2word, f, cPickle.HIGHEST_PROTOCOL)

with open(args.data+'/cnn_spkV_traindata.cpickle', 'rb') as f:
    cnn_spkV_traindata = cPickle.load(f)

def batchify_spkV(feature, bsz, bptt):
    padzero = bsz - len(feature)%bsz
    for j in range(padzero):
        feature.append([0])
    temp = np.zeros((len(feature)/bsz,bsz)).tolist()
    row = 0
    col = 0
    for j in range(len(feature)):
        for k in range(bptt-len(feature[j])):
            feature[j].append(0)
        temp[row][col] = feature[j]
        col+=1
        if col==bsz:
            col=0
            row+=1
    # import pdb; pdb.set_trace()
    data = torch.LongTensor(temp)
    if args.cuda:
        data = data.cuda()
    return data

# extract speaker feature
with open(args.cnn_spkV_model, 'rb') as f:
    model = torch.load(f)
embedding_weight = model.encoder.weight
# training set
train_sents = cnn_spkV_traindata[0][0:len(cnn_spkV_traindata[0]):2]
batch_spkV = batchify_spkV(train_sents,args.batch_size,args.bptt)
cnn_spkV = torch.cuda.FloatTensor([])
for i in range(batch_spkV.size(0)):
    data = Variable(batch_spkV[i], volatile=True)
    cnn_spkV = torch.cat([cnn_spkV, model.extract(data).data],0)

# validation set
valid_sents = []
with open(args.valid_file,'r') as f:
    for i in f:
        line = i.replace('\n','')
        line = line.split()
        valid_sents.append([])
        for j in line:
            if j in corpus.dictionary.word2idx:
                valid_sents[-1].append(corpus.dictionary.word2idx[j])
            else:
                valid_sents[-1].append(corpus.dictionary.word2idx['<unk>'])
batch_spkV = batchify_spkV(valid_sents,args.batch_size,args.bptt)
cnn_spkV_valid = torch.cuda.FloatTensor([])
for i in range(batch_spkV.size(0)):
    data = Variable(batch_spkV[i], volatile=True)
    cnn_spkV_valid = torch.cat([cnn_spkV_valid, model.extract(data).data],0)

# test set
test_sents = []
with open(args.test_file,'r') as f:
    for i in f:
        line = i.replace('\n','')
        line = line.split()
        test_sents.append([])
        for j in line:
            if j in corpus.dictionary.word2idx:
                test_sents[-1].append(corpus.dictionary.word2idx[j])
            else:
                test_sents[-1].append(corpus.dictionary.word2idx['<unk>'])
batch_spkV = batchify_spkV(test_sents,args.batch_size,args.bptt)
cnn_spkV_test = torch.cuda.FloatTensor([])
for i in range(batch_spkV.size(0)):
    data = Variable(batch_spkV[i], volatile=True)
    cnn_spkV_test = torch.cat([cnn_spkV_test, model.extract(data).data],0)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz, bptt):
    padzero = bsz - data.size(0)/bptt%bsz
    data = torch.cat([data,torch.LongTensor(padzero*bptt).zero_()],0)
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

eval_batch_size = args.batch_size
train_data = batchify(corpus.train, args.batch_size, args.bptt)
val_data = batchify(corpus.valid, eval_batch_size, args.bptt)
test_data = batchify(corpus.test, eval_batch_size, args.bptt)
# import pdb; pdb.set_trace()
###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model_useCNNspk.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, cnn_spkV.size(1), args.dropout, args.tied)
model.encoder.weight = embedding_weight
if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss(ignore_index=0)

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i, spkV, adapation=False, evaluation=False):
    # if evaluation:
    #     utt2spk = utt2spk_test
    # else:
    #     utt2spk = utt2spk_train
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    batch = i/args.bptt
    zero_padding = torch.cuda.FloatTensor(seq_len,args.batch_size,spkV.size(1)).zero_()
    if adapation:
        batch_step = source.size(0)/args.bptt
        for a in range(args.batch_size):
            for b in range(seq_len):
                if int(data[b][a])!=0:
                    try:
                        zero_padding[b][a] = spkV[batch_step*a+i/args.bptt]
                    except:
                        import pdb;pdb.set_trace()
    spkfeature = Variable(zero_padding)
    zero_padding = torch.cat([source[i+1:i+seq_len].view(-1),torch.cuda.LongTensor(args.batch_size).zero_()],0)
    target = Variable(zero_padding)
    # import pdb;pdb.set_trace()
    return data, target, spkfeature


def evaluate(data_source, spkV, adapation):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets, spkfeature = get_batch(data_source, i, spkV, adapation=adapation, evaluation=True)
        output, hidden = model(data, hidden, spkfeature)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        # import pdb; pdb.set_trace()
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)



def train():
    # Turn on training mode which enables dropout.
    model.train()
    for p in model.spk_dense.parameters():
        p.requires_grad=False
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets, spkfeature = get_batch(train_data, i, cnn_spkV, adapation=False)
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden, spkfeature)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            if p.grad is not None:
                p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def adapation():
    # Turn on training mode which enables dropout.
    model.train()
    for p in model.encoder.parameters():
        p.requires_grad=False
    # for p in model.spk_dense.parameters():
    #     p.requires_grad=True
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)    
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets, spkfeature = get_batch(train_data, i, cnn_spkV, adapation=True)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden, spkfeature)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            if p.grad is not None:
                p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
            # import pdb;pdb.set_trace()

# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    if not args.only_ada:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            adapation()
            val_loss = evaluate(val_data, cnn_spkV_valid, True)
            test_loss = evaluate(test_data, cnn_spkV_test, True)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f} | test ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss), math.exp(test_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save+'.pre', 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0

    # with open(args.save+'.pre', 'rb') as f:
    #     model = torch.load(f)
    # lr = args.lr / 4.0
    # best_val_loss = None
    # for epoch in range(1, args.epochs+1):
    #     epoch_start_time = time.time()
    #     adapation()
    #     val_loss = evaluate(val_data, cnn_spkV_valid, True)
    #     test_loss = evaluate(test_data, cnn_spkV_test, True)
    #     print('-' * 89)
    #     print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
    #             'valid ppl {:8.2f} | test ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
    #                                        val_loss, math.exp(val_loss), math.exp(test_loss)))
    #     print('-' * 89)
    #     # Save the model if the validation loss is the best we've seen so far.
    #     if not best_val_loss or val_loss < best_val_loss:
    #         with open(args.save, 'wb') as f:
    #             torch.save(model, f)
    #         best_val_loss = val_loss
    #     else:
    #         # Anneal the learning rate if no improvement has been seen in the validation dataset.
    #         lr /= 4.0

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(val_data, cnn_spkV_valid, True)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

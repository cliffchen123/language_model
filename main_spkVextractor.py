# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

import data
import model_spkVextractor

import cPickle

parser = argparse.ArgumentParser(description='PyTorch speaker vector extractor')
parser.add_argument('--data', type=str, default='./data/delta_data',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=300,
                    help='number of hidden units per layer')
parser.add_argument('--lr', type=float, default=0.1,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=60,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='./model_cnnspk.pt',
                    help='path to save the final model')
parser.add_argument('--dict', type=str,  default='./data/delta_data/dictionary.cpickle',
                    help='path of dictionary')
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
    train_cnn_spkV = cPickle.load(f)

with open(args.data+'/cnn_spkV_validdata.cpickle', 'rb') as f:
    val_cnn_spkV = cPickle.load(f)

with open(args.data+'/cnn_spkV_testdata.cpickle', 'rb') as f:
    test_cnn_spkV = cPickle.load(f)

# import pdb;pdb.set_trace()
train_spk_num = 0
last = None
for i in range(len(train_cnn_spkV[1])):
    temp = train_cnn_spkV[1][i].replace('-)','')
    if last!=temp:
        train_spk_num += 1
    last = temp

def batchify(data, bsz, bptt):

    spk_start = []
    last = None
    for i in range(len(data[1])):
        temp = data[1][i].replace('-)','')
        if last!=temp:
            spk_start.append(i)
        last = temp
    spk_start.append(len(data[1]))

    feature = data[0]
    label = data[1]
    data = torch.LongTensor([])
    target = torch.FloatTensor([])
    target_spk = []
    spk=0
    for i in range(len(spk_start)-1):
        feature_t = feature[spk_start[i]:spk_start[i+1]]
        label_t = label[spk_start[i]:spk_start[i+1]]
        padzero = bsz - len(feature_t)%bsz
        for j in range(padzero):
            feature_t.append([0])
            label_t.append([0])
        target_spk+=[spk]*(len(feature_t)/bsz)
        spk+=1
        temp = np.zeros((len(feature_t)/bsz,bsz)).tolist()
        row = 0
        col = 0
        for j in range(len(feature_t)):
            for k in range(bptt-len(feature_t[j])):
                feature_t[j].append(0)
            temp[row][col] = feature_t[j]
            col+=1
            if col==bsz:
                col=0
                row+=1
        feature_t = torch.LongTensor(temp)

        temp = np.zeros((len(label_t)/bsz,bsz)).tolist()
        row = 0
        col = 0
        for j in range(len(label_t)):
            if label_t[j][0:2]=='-)':
                temp[row][col]=0
            else:
                temp[row][col]=1
            col+=1
            if col==bsz:
                col=0
                row+=1
        label_t = torch.FloatTensor(temp)
        # import pdb; pdb.set_trace()
        data = torch.cat([data,feature_t],0)
        target = torch.cat([target,label_t],0)
    if args.cuda:
        data = data.cuda()
        target = target.cuda()
    return data,target,target_spk

eval_batch_size = args.batch_size
train_data, train_label, train_batch_spk = batchify(train_cnn_spkV, args.batch_size, args.bptt)
val_data, val_label, val_batch_spk = batchify(val_cnn_spkV, eval_batch_size, args.bptt)
test_data, test_label, test_batch_spk = batchify(test_cnn_spkV, eval_batch_size, args.bptt)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model_spkVextractor.CNNModel(ntokens, args.emsize, args.nhid, args.bptt, train_spk_num, args.dropout)
if args.cuda:
    model.cuda()
    for i in range(train_spk_num):
        model.output[i].cuda()

criterion = nn.BCELoss()

###############################################################################
# Training code
###############################################################################


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, label, i, evaluation=False):
    data = Variable(source[i], volatile=evaluation)
    target = Variable(label[i].view(-1))
    # import pdb;pdb.set_trace()
    return data, target


def evaluate(data_source, label_source, batch_spk, use_accuracy=False):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    accuracy = 0
    for i in range(0, data_source.size(0) - 1):
        data, target = get_batch(data_source, label_source, i, evaluation=True)
        output = model(data,batch_spk[i])
        output_flat = output.view(-1)
        total_loss += len(data) * criterion(output_flat, target).data
        accuracy += np.dot(np.round(output_flat.data.tolist()),np.array(target.data.tolist()))
    if not use_accuracy:
        return total_loss[0] / (data_source.size(0)*data_source.size(1))
    else:
        return accuracy / (data_source.size(0)*data_source.size(1))



def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    for batch, i in enumerate(range(0, train_data.size(0) - 1)):
        data, targets = get_batch(train_data, train_label, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        output = model(data, train_batch_spk[i])
        loss = criterion(output.view(-1), targets)
        loss.backward()
        optimizer.step()
        
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        # for p in model.parameters():
        #     if p.grad is not None:
        #         p.data.add_(-lr, p.grad.data)
        

        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()

# Loop over epochs.
lr = args.lr
best_val_loss = None
optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
# import pdb;pdb.set_trace()
# At any point you can hit Ctrl + C to break out of training early.
try:
    # lr = args.lr / 4.0
    best_val_loss = None
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data,val_label,val_batch_spk)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid accuracy {:5.2f} |'
            .format(epoch, (time.time() - epoch_start_time),val_loss,evaluate(val_data,val_label,val_batch_spk,True)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on valid data.
valid_loss = evaluate(val_data,val_label,val_batch_spk)
valid_accuracy = evaluate(val_data,val_label,val_batch_spk,True)
print('=' * 89)
print('| End of training | valid loss {:5.2f} | valid accuracy {:3.4f}'.format(
    valid_loss, valid_accuracy))
print('=' * 89)


# Run on test data.
test_loss = evaluate(test_data,test_label,test_batch_spk)
test_accuracy = evaluate(test_data,test_label,test_batch_spk,True)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test accuracy {:3.4f}'.format(
    test_loss, test_accuracy))
print('=' * 89)

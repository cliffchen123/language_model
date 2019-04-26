# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import data
import model_jointCNN

import cPickle

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/delta_data',
					help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
					help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
					help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=300,
					help='number of hidden units per layer')
parser.add_argument('--spk_nhid', type=int, default=150,
					help='number of speaker feature hidden units')
parser.add_argument('--nlayers', type=int, default=1,
					help='number of layers')
parser.add_argument('--lr', type=float, default=20,
					help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
					help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
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
parser.add_argument('--save', type=str,  default='model_jointCNN.pt',
					help='path to save the final model')
parser.add_argument('--dict', type=str,  default='data/delta_data/dictionary.cpickle',
					help='path of dictionary')
parser.add_argument('--only_ada', action='store_true',
					help='only adapation')
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


speaker = []
for i in range(len(train_cnn_spkV[1])):
	temp = train_cnn_spkV[1][i].replace('-)','')
	if temp not in speaker:
		speaker.append(temp)

train_spk_start = []
last = None
for i in range(len(train_cnn_spkV[1])):
	temp = train_cnn_spkV[1][i].replace('-)','')
	if last!=temp:
		train_spk_start.append(i)
	last = temp
train_spk_start.append(len(train_cnn_spkV[1]))

val_spk_start = []
last = None
for i in range(len(val_cnn_spkV[1])):
	temp = val_cnn_spkV[1][i].replace('-)','')
	if last!=temp:
		val_spk_start.append(i)
	last = temp
val_spk_start.append(len(val_cnn_spkV[1]))
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

def batchify(data, bsz, bptt, spk_start):
	spk_start = list(np.array(spk_start)/2*args.bptt)
	data = list(data)
	result = None
	for i in range(len(spk_start)-1):
		temp = torch.LongTensor(data[spk_start[i]:spk_start[i+1]])
		try:
			padzero = bsz - temp.size(0)/bptt%bsz
		except:
			import pdb;pdb.set_trace()
		temp = torch.cat([temp,torch.LongTensor(padzero*bptt).zero_()],0)
		nbatch = temp.size(0) // bsz
		temp = temp.view(bsz, -1).t().contiguous()
		if type(result)==type(None):
			result = temp
		else:
			result = torch.cat([result,temp],0)
	if args.cuda:
		result = result.cuda()			
	# import pdb;pdb.set_trace()
	return result


	# padzero = bsz - data.size(0)/bptt%bsz
	# data = torch.cat([data,torch.LongTensor(padzero*bptt).zero_()],0)
	# # Work out how cleanly we can divide the dataset into bsz parts.
	# nbatch = data.size(0) // bsz
	# # Trim off any extra elements that wouldn't cleanly fit (remainders).
	# data = data.narrow(0, 0, nbatch * bsz)
	# # Evenly divide the data across the bsz batches.
	# data = data.view(bsz, -1).t().contiguous()
	# if args.cuda:
	# 	data = data.cuda()
	# return data

def sub_batchify(data, bsz, bptt, spk_start):

	spk_start = list(np.array(spk_start)/2)
	pos_feature = []
	neg_feature = []
	for i in range(0,len(data[0]),2):
		pos_sent = data[0][i]+(bptt - len(data[0][i]))*[0]
		neg_sent = data[0][i+1]+(bptt - len(data[0][i+1]))*[0]
		pos_feature.append(pos_sent)
		neg_feature.append(neg_sent)

	new_pos_feature=None
	new_neg_feature=None
	label = []
	for i in range(len(spk_start)-1):
		pos_temp = pos_feature[spk_start[i]:spk_start[i+1]]
		neg_temp = neg_feature[spk_start[i]:spk_start[i+1]]
		spk_idx = speaker.index(data[1][spk_start[i]*2])
		padzero = bsz - len(pos_temp)%bsz
		pos_temp += [[0]*bptt]*padzero
		neg_temp += [[0]*bptt]*padzero
		nbatch = len(pos_temp) // bsz
		label+=[spk_idx]*nbatch
		pos_temp2 = []
		neg_temp2 = []
		for j in range(bsz):
			pos_temp2.append(pos_temp[j*nbatch:(j+1)*nbatch])
			neg_temp2.append(neg_temp[j*nbatch:(j+1)*nbatch])
		pos_temp2 = np.transpose(pos_temp2,[1,0,2])
		neg_temp2 = np.transpose(neg_temp2,[1,0,2])
		if type(new_pos_feature)==type(None):
			new_pos_feature = pos_temp2
			new_neg_feature = neg_temp2
		else:
			new_pos_feature = np.concatenate((new_pos_feature,pos_temp2),axis=0)
			new_neg_feature = np.concatenate((new_neg_feature,neg_temp2),axis=0)
	# import pdb;pdb.set_trace()
	return new_pos_feature,new_neg_feature,label


eval_batch_size = args.batch_size
train_data = batchify(corpus.train, args.batch_size, args.bptt, train_spk_start)
val_data = batchify(corpus.valid, eval_batch_size, args.bptt, val_spk_start)
test_data = val_data #batchify(corpus.test, eval_batch_size, args.bptt)
train_sub_pos_data, train_sub_neg_data, train_sub_label = sub_batchify(train_cnn_spkV, args.batch_size, args.bptt, train_spk_start)
val_sub_pos_data, val_sub_neg_data, val_sub_label = sub_batchify(val_cnn_spkV, eval_batch_size, args.bptt, val_spk_start)
test_sub_pos_data, test_sub_neg_data, test_sub_label = val_sub_pos_data, val_sub_neg_data, val_sub_label #batchify(corpus.test, eval_batch_size, args.bptt)

# import pdb; pdb.set_trace()
###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model_jointCNN.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.spk_nhid, args.nlayers, args.bptt, len(speaker), args.dropout, args.tied)
if args.cuda:
	model.cuda()
	for i in range(len(speaker)):
		model.extractor_output[i].cuda()

criterion_class = nn.CrossEntropyLoss(ignore_index=0)
criterion_binary = nn.BCELoss()

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

def get_batch(source, sub_pos_input, sub_neg_input, label, i, adapation=False, evaluation=False):	
	seq_len = min(args.bptt, len(source) - 1 - i)
	data = Variable(source[i:i+seq_len], volatile=evaluation)
	zero_padding = torch.cat([source[i+1:i+seq_len].view(-1),torch.cuda.LongTensor(args.batch_size).zero_()],0)
	target = Variable(zero_padding)

	batch = i/args.bptt

	sub_pos_data = Variable(torch.LongTensor(sub_pos_input[batch]).cuda(), volatile=evaluation)
	sub_neg_data = Variable(torch.LongTensor(sub_neg_input[batch]).cuda(), volatile=evaluation)
	return data, target, sub_pos_data, sub_neg_data, label[batch]


def evaluate(data_source, sub_pos_data, sub_neg_data, sub_label):
	# Turn on evaluation mode which disables dropout.
	model.eval()
	total_loss = 0
	ntokens = len(corpus.dictionary)
	hidden = model.init_hidden(eval_batch_size)
	for i in range(0, data_source.size(0) - 1, args.bptt):

		data, targets, pos_sent, neg_sent, spk = get_batch(data_source, sub_pos_data, sub_neg_data, sub_label, i, evaluation=True)
		output, hidden, pos_spk_output, neg_spk_output = model(data, hidden, spk, pos_sent, neg_sent)
		output_flat = output.view(-1, ntokens)
		total_loss += len(data) * criterion_class(output_flat, targets).data
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
		data, targets, pos_sent, neg_sent, spk = get_batch(train_data, train_sub_pos_data, train_sub_neg_data, train_sub_label, i, adapation=False)
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

def adapation():
	# Turn on training mode which enables dropout.
	# model.train()
	# for p in model.spk_dense.parameters():
	#	 p.requires_grad=True
	# for p in model.encoder.parameters():
	#	 p.requires_grad=False
	# for p in model.decoder.parameters():
	#	 p.requires_grad=False
	# for p in model.rnn.parameters():
	#	 p.requires_grad=False
	total_loss = 0
	word_loss = 0
	pos_loss = 0
	neg_loss = 0
	start_time = time.time()
	ntokens = len(corpus.dictionary)
	hidden = model.init_hidden(args.batch_size)	
	for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
		data, targets, pos_sent, neg_sent, spk = get_batch(train_data, train_sub_pos_data, train_sub_neg_data, train_sub_label, i, adapation=True)
		# Starting each batch, we detach the hidden state from how it was previously produced.
		# If we didn't, the model would try backpropagating all the way to start of the dataset.
		hidden = repackage_hidden(hidden)
		model.zero_grad()
		output, hidden, pos_spk_output, neg_spk_output = model(data, hidden, spk, pos_sent, neg_sent)
		
		loss1 = criterion_class(output.view(-1, ntokens), targets)
		loss2 = criterion_binary(pos_spk_output, Variable(torch.cuda.FloatTensor([1]*args.batch_size)))
		loss3 = criterion_binary(neg_spk_output, Variable(torch.cuda.FloatTensor([0]*args.batch_size)))
		# import pdb;pdb.set_trace()
		loss_sum = sum([0.2*loss1, 0.2*loss2, 0.6*loss3])
		loss_sum.backward()

		# `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
		torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
		for p in model.parameters():
			if p.grad is not None:
				p.data.add_(-lr, p.grad.data)
		# import pdb;pdb.set_trace()
		total_loss += loss_sum.data
		word_loss += loss1.data
		pos_loss += loss2.data
		neg_loss += loss3.data
		if batch % args.log_interval == 0 and batch > 0:
			cur_loss = total_loss[0] / args.log_interval
			cur_loss_word = word_loss[0] / args.log_interval
			cur_loss_pos = pos_loss[0] / args.log_interval
			cur_loss_neg = neg_loss[0] / args.log_interval
			elapsed = time.time() - start_time
			print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
					'sum_loss {:5.2f} | word_loss {:5.2f} | pos_loss {:5.2f} | neg_loss {:5.2f} | ppl {:8.2f}'.format(
				epoch, batch, len(train_data) // args.bptt, lr,
				elapsed * 1000 / args.log_interval, cur_loss, cur_loss_word, cur_loss_pos, cur_loss_neg, math.exp(cur_loss_word)))
			total_loss = 0
			word_loss = 0
			pos_loss = 0
			neg_loss = 0
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
			val_loss = evaluate(val_data, val_sub_pos_data, val_sub_neg_data, val_sub_label)
			print('-' * 89)
			print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
					'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
											   val_loss, math.exp(val_loss)))
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
	#	 model = torch.load(f)
	# lr = args.lr / 4.0
	# best_val_loss = None
	# for epoch in range(1, args.epochs+1):
	#	 epoch_start_time = time.time()
	#	 adapation()
	#	 val_loss = evaluate(val_data)
	#	 print('-' * 89)
	#	 print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
	#			 'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
	#										val_loss, math.exp(val_loss)))
	#	 print('-' * 89)
	#	 # Save the model if the validation loss is the best we've seen so far.
	#	 if not best_val_loss or val_loss < best_val_loss:
	#		 with open(args.save, 'wb') as f:
	#			 torch.save(model, f)
	#		 best_val_loss = val_loss
	#	 else:
	#		 # Anneal the learning rate if no improvement has been seen in the validation dataset.
	#		 lr /= 4.0

except KeyboardInterrupt:
	print('-' * 89)
	print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
	model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
	test_loss, math.exp(test_loss)))
print('=' * 89)

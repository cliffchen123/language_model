# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import model
import model_mixer

import cPickle

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
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
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=60, #ami:110
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
parser.add_argument('--splitdata', type=str,  default='',
                    help='the file of split data')
parser.add_argument('--save_folder', type=str,  default='',
                    help='path to save the final model')
parser.add_argument('--dict', type=str,  default='',
                    help='path of dictionary')
parser.add_argument('--bi', action='store_true',
                    help='use bidirational rnn')
parser.add_argument('--spk2utt', type=str, default='./data/delta_data/spk2utt',
                    help='speaker to utterences')
parser.add_argument('--premodel', type=str, default='',
                    help='pretrain model')
parser.add_argument('--stage', type=int, default=1,
                    help='pretrain model')
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


with open(args.splitdata,'r') as f:
    splitdata = cPickle.load(f)

for i in splitdata:
    tokens = 0
    for line in splitdata[i]:
        words = ['<s>'] + line.split() + ['<eos>'] + (args.bptt-2-len(line.split()))*['<no>']
        tokens += len(words) 
    ids = torch.LongTensor(tokens)
    token = 0
    for line in splitdata[i]:
        words = ['<s>'] + line.split() + ['<eos>'] + (args.bptt-2-len(line.split()))*['<no>']
        for word in words:
            if word in corpus.dictionary.word2idx:
                ids[token] = corpus.dictionary.word2idx[word]
            else:
                ids[token] = corpus.dictionary.word2idx['<unk>']
            token += 1
    splitdata[i] = ids

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


# import pdb; pdb.set_trace()
###############################################################################
# Build the model
###############################################################################


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

def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    try:
        zero_padding = torch.cat([source[i+1:i+seq_len].view(-1),torch.cuda.LongTensor(args.batch_size).zero_()],0)
    except:
        import pdb;pdb.set_trace()
    target = Variable(zero_padding)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)

def evaluate_mixer(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output_expert = []
        for j in splitdata:
            output_expert.append(extract_hidden(data,model_expert[j]))
        output, hidden = model(data, hidden, output_expert)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)

def predict(data_source):
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(1)
    # data, targets = get_batch(data_source, i, evaluation=True)
    output, hidden = model(data_source, hidden)
    output_flat = output.view(-1, ntokens)
    softmax = nn.Softmax()
    output_pro = softmax(output_flat)
    # import pdb; pdb.set_trace()
    # hidden = repackage_hidden(hidden)
    return torch.max(output_pro,0)

def extract_hidden(data_source, model_):
    model_.eval()
    hidden = model_.init_hidden(args.batch_size)
    emb = model_.drop(model_.encoder(data_source))
    output, hidden = model_.rnn(emb, hidden)
    
    # data, targets = get_batch(data_source, i, evaluation=True)

    # import pdb; pdb.set_trace()
    return output

def train_expert():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        # import pdb;pdb.set_trace()
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

def train_mixer():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output_expert = []
        for j in splitdata:
            output_expert.append(extract_hidden(data,model_expert[j]))
        output, hidden = model(data, hidden, output_expert)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        # import pdb;pdb.set_trace()
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

# train experts model
if args.stage<=1:
    for i in splitdata:

        eval_batch_size = args.batch_size
        train_data = batchify(splitdata[i], args.batch_size, args.bptt)
        val_data = train_data
        test_data = train_data

        ntokens = len(corpus.dictionary)
        with open(args.premodel, 'rb') as f:
            model = torch.load(f)
        if args.cuda:
            model.cuda()
        for p in model.encoder.parameters():
            p.requires_grad=False

        criterion = nn.CrossEntropyLoss(ignore_index=0)

        # Loop over epochs.
        lr = args.lr
        best_val_loss = None

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            for epoch in range(1, args.epochs+1):
                epoch_start_time = time.time()
                train_expert()
                val_loss = evaluate(val_data)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                   val_loss, math.exp(val_loss)))
                print('-' * 89)
                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_loss or val_loss < best_val_loss:
                    with open(args.save_folder+'/'+i+'_model.pt', 'wb') as f:
                        torch.save(model, f)
                    best_val_loss = val_loss
                else:
                    # Anneal the learning rate if no improvement has been seen in the validation dataset.
                    lr /= 4.0
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

        # Load the best saved model.
        with open(args.save_folder+'/'+i+'_model.pt', 'rb') as f:
            model = torch.load(f)

        # Run on test data.
        test_loss = evaluate(test_data)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
        print('=' * 89)




# train Mixer model
if args.stage<=2:
    splitdata.pop('01')
    splitdata.pop('03')
    splitdata.pop('04')
    splitdata.pop('05')
    splitdata.pop('06')
    splitdata.pop('07')
    splitdata.pop('08')
    splitdata.pop('09')
    splitdata.pop('11')
    splitdata.pop('12')
    splitdata.pop('13')
    splitdata.pop('21')
    splitdata.pop('22')
    splitdata.pop('24')
    splitdata.pop('25')
    splitdata.pop('28')
    splitdata.pop('29')
    splitdata.pop('30')
    splitdata.pop('31')
    splitdata.pop('38')
    splitdata.pop('39')
    splitdata.pop('40')
    splitdata.pop('42')
    splitdata.pop('43')
    splitdata.pop('53')
    splitdata.pop('54')
    splitdata.pop('56')

    eval_batch_size = args.batch_size
    train_data = batchify(corpus.train, args.batch_size, args.bptt)
    val_data = batchify(corpus.valid, eval_batch_size, args.bptt)
    test_data = batchify(corpus.test, eval_batch_size, args.bptt)
    ntokens = len(corpus.dictionary)

    # Load the expert model.
    model_expert = dict()
    for i in splitdata:
        with open(args.save_folder+'/'+i+'_model.pt', 'rb') as f:
            model_expert[i] = torch.load(f)

    model = model_mixer.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, len(splitdata), args.dropout, args.tied)
    model.encoder.weight = model_expert[splitdata.keys()[0]].encoder.weight

    if args.cuda:
        model.cuda()
    for p in model.encoder.parameters():
        p.requires_grad=False
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Loop over epochs.
    lr = args.lr
    best_val_loss = None

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train_mixer()
            val_loss = evaluate_mixer(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save_folder+'/mixer_model.pt', 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save_folder+'/mixer_model.pt', 'rb') as f:
        model = torch.load(f)

    # Run on test data.
    test_loss = evaluate_mixer(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import ipdb
from datetime import datetime
import pickle
use_cuda = torch.cuda.is_available()

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    # s = re.sub(r"([.!?])", r" \1", s)
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs1(lang1, lang2, reverse=False):
    print("Reading lines...")

    lines = open('./data/%s-%s_w2w.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs1 = [[normalizeString(s) for s in l.split('@')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs1 = [list(reversed(p)) for p in pairs1]
        input_lang1 = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang1 = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang1, output_lang, pairs1

def readLangs2(lang1, lang2, reverse=False):
    print("Reading lines...")

    lines = open('./data/%s-%s_w2c.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs2 = [[normalizeString(s) for s in l.split('@')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs2 = [list(reversed(p)) for p in pairs2]
        input_lang2 = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang2 = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang2, output_lang, pairs2

def readLangs3(lang1, lang2, reverse=False):
    print("Reading lines...")

    lines = open('./data/%s-%s_w2p.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs3 = [[normalizeString(s) for s in l.split('@')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs3 = [list(reversed(p)) for p in pairs3]
        input_lang3 = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang3 = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang3, output_lang, pairs3


MAX_LENGTH = 50
print_every = 100

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH 

# def filterPair(p):
    # return len(p[0].split(' ')) < MAX_LENGTH and \
        # len(p[1].split(' ')) < MAX_LENGTH and \
        # p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


######################################################################
# The full process for preparing the data is:
#
# -  Read text file and split into lines, split lines into pairs
# -  Normalize text, filter by length and content
# -  Make word lists from sentences in pairs
#

def prepareData1(lang1, lang2, reverse=False):
    input_lang1, output_lang, pairs1 = readLangs1(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs1))
    pairs1 = filterPairs(pairs1)
    print("Trimmed to %s sentence pairs" % len(pairs1))
    print("Counting words...")
   
    for pair in pairs1:
        input_lang1.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang1.name, input_lang1.n_words)
    print(output_lang.name, output_lang.n_words)
    print("add <unk> word:")
    input_lang1.addWord('<unk>')
    print(input_lang1.name, input_lang1.n_words)
    print(output_lang.name, output_lang.n_words)

    return input_lang1, output_lang, pairs1

def prepareData2(lang1, lang2, reverse=False):
    input_lang2, output_lang, pairs2 = readLangs2(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs2))
    pairs2 = filterPairs(pairs2)
    print("Trimmed to %s sentence pairs" % len(pairs2))
    print("Counting words...")
   
    for pair in pairs2:
        input_lang2.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang2.name, input_lang2.n_words)
    print(output_lang.name, output_lang.n_words)
    print("add <unk> word:")
    input_lang2.addWord('<unk>')
    print(input_lang2.name, input_lang2.n_words)
    print(output_lang.name, output_lang.n_words)

    return input_lang2, output_lang, pairs2

def prepareData3(lang1, lang2, reverse=False):
    input_lang3, output_lang, pairs3 = readLangs3(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs3))
    pairs3 = filterPairs(pairs3)
    print("Trimmed to %s sentence pairs" % len(pairs3))
    print("Counting words...")
   
    for pair in pairs3:
        input_lang3.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang3.name, input_lang3.n_words)
    print(output_lang.name, output_lang.n_words)
    print("add <unk> word:")
    input_lang3.addWord('<unk>')
    print(input_lang3.name, input_lang3.n_words)
    print(output_lang.name, output_lang.n_words)

    return input_lang3, output_lang, pairs3

input_lang1, output_lang, pairs1 = prepareData1('com', 'eng', True)
input_lang2, output_lang, pairs2 = prepareData2('com', 'eng', True)
input_lang3, output_lang, pairs3 = prepareData3('com', 'eng', True)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden_list, encoder_output_list, encoder_outputs_list):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        
        encoder1_outputs, encoder2_outputs, encoder3_outputs = encoder_outputs_list 

        hidden1, hidden2, hidden3 = hidden_list           
        attn_weights1 = F.softmax(self.attn(torch.cat((embedded[0], hidden1[0]), 1)))
        attn_weights2 = F.softmax(self.attn(torch.cat((embedded[0], hidden2[0]), 1)))
        attn_weights3 = F.softmax(self.attn(torch.cat((embedded[0], hidden3[0]), 1)))

        attn_weights_list = [attn_weights1, attn_weights2, attn_weights3]

        attn_applied1 = torch.bmm(attn_weights1.unsqueeze(0), encoder1_outputs.unsqueeze(0))
        attn_applied2 = torch.bmm(attn_weights2.unsqueeze(0), encoder2_outputs.unsqueeze(0))
        attn_applied3 = torch.bmm(attn_weights3.unsqueeze(0), encoder3_outputs.unsqueeze(0))

        buff = torch.add(attn_applied1, attn_applied2)
        attn_applied_sum = torch.add(buff, attn_applied3)

        output = torch.cat((embedded[0], attn_applied_sum[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        hidden = torch.add(hidden1, hidden2)
        hidden = torch.add(hidden, hidden3)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights_list

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result



def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variablesFromPair(input_lang_vp, out_lang, pair):
    input_variable = variableFromSentence(input_lang_vp, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)


teacher_forcing_ratio = 1.0


def train(input_variable_list, target_variable, encoder_list, decoder, encoder_optimizer_list, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    
    encoder1, encoder2, encoder3 = encoder_list 
    
    encoder1_hidden = encoder1.initHidden()
    encoder2_hidden = encoder2.initHidden()
    encoder3_hidden = encoder3.initHidden()

    [encoder1_optimizer, encoder2_optimizer, encoder3_optimizer] = encoder_optimizer_list

    encoder1_optimizer.zero_grad()
    encoder2_optimizer.zero_grad()
    encoder3_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable_list[0].size()[0]
    target_length = target_variable.size()[0]
   

    encoder1_outputs = Variable(torch.zeros(max_length, encoder1.hidden_size))
    encoder1_outputs = encoder1_outputs.cuda() if use_cuda else encoder1_outputs
    
    encoder2_outputs = Variable(torch.zeros(max_length, encoder2.hidden_size))
    encoder2_outputs = encoder2_outputs.cuda() if use_cuda else encoder2_outputs
    
    encoder3_outputs = Variable(torch.zeros(max_length, encoder3.hidden_size))
    encoder3_outputs = encoder3_outputs.cuda() if use_cuda else encoder3_outputs
   
    loss = 0

    encoder_output_list = []
    for ei in range(input_length):
        # print(np.shape(input_variable_list[0])[0], np.shape(input_variable_list[0])[0], np.shape(input_variable_list[0])[0])
        # ipdb.set_trace()
        encoder1_output, encoder1_hidden = encoder1(input_variable_list[0][ei], encoder1_hidden)
        encoder2_output, encoder2_hidden = encoder2(input_variable_list[1][ei], encoder2_hidden)
        encoder3_output, encoder3_hidden = encoder3(input_variable_list[2][ei], encoder3_hidden)
        
        # encoder_outputs_list.append([encoder1_output, encoder2_output, encoder3_output])  
        
        encoder1_outputs[ei] = encoder1_output[0][0]
        encoder2_outputs[ei] = encoder2_output[0][0]
        encoder3_outputs[ei] = encoder3_output[0][0]
        
        # encoder_outputs[ei] = encoder_output[0][0]
    
    encoder_output_list = [encoder1_output, encoder2_output, encoder3_output]
    encoder_outputs_list = [encoder1_outputs, encoder2_outputs, encoder3_outputs]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
   
    # encoder_hidden_buff = torch.add(encoder1_hidden, encoder2_hidden)
    # encoder_hidden_sum = torch.add(encoder_hidden_buff, encoder3_hidden)

    decoder_hidden_list = [encoder1_hidden, encoder2_hidden, encoder3_hidden]

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention_list = decoder(
                decoder_input, decoder_hidden_list, encoder_output_list, encoder_outputs_list)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention_list = decoder(
                decoder_input, decoder_hidden_list, encoder_output_list, encoder_outputs_list)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            
            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            
            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()

    encoder1_optimizer.step()
    encoder2_optimizer.step()
    encoder3_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


######################################################################
# The whole training process looks like this:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
#
# Then we call ``train`` many times and occasionally print the progress (%
# of examples, time so far, estimated time) and average loss.
#

def trainIters(encoder_list, decoder, repeat, print_every=50, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    
    encoder1, encoder2, encoder3 = encoder_list 
    encoder1_optimizer = optim.SGD(encoder1.parameters(), lr=learning_rate)
    encoder2_optimizer = optim.SGD(encoder2.parameters(), lr=learning_rate)
    encoder3_optimizer = optim.SGD(encoder3.parameters(), lr=learning_rate)
    encoder_optimizer_list = [encoder1_optimizer, encoder2_optimizer, encoder3_optimizer] 

    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # training_pairs = [variablesFromPair(random.choice(pairs))
                      # for i in range(n_iters)]

    #TJ: Don't randomize it. we have very little dataset.
    # repeat = 10
    training_pairs = []
    
    pairs_shuffle = []
    for r in range(repeat):

        # random.shuffle(pairs) 
        # pairs_shuffle = [variablesFromPair(pairs[k]) for k in range(len(pairs))]
        # training_pairs.extend(pairs_shuffle)
        
        np.random.seed(0)
        random_index = np.random.permutation(list(range(len(pairs1))))
        # random_index = random_index[:10]
        for k, idx in enumerate(random_index):

            pairs_list = [variablesFromPair(input_lang1, output_lang, pairs1[idx]), variablesFromPair(input_lang2, output_lang, pairs2[idx]), variablesFromPair(input_lang3, output_lang, pairs3[idx])]
            pairs_shuffle.append(pairs_list) 
        training_pairs.extend(pairs_shuffle)


    n_iters = len(random_index) * repeat
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable1 = training_pair[0][0]
        input_variable2 = training_pair[1][0]
        input_variable3 = training_pair[2][0]
       
        target_variable = training_pair[0][1]

        input_variable_list = [input_variable1, input_variable2, input_variable3]
        loss = train(input_variable_list, target_variable, encoder_list,
                     decoder, encoder_optimizer_list, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    # showPlot(plot_losses)


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def evaluate(encoder_list, decoder, sentence_list, max_length=MAX_LENGTH):
    encoder1, encoder2, encoder3 = encoder_list 
    sentence1, sentence2, sentence3 = sentence_list
    
    input_variable1 = variableFromSentence(input_lang1, sentence1)
    input_variable2 = variableFromSentence(input_lang2, sentence2)
    input_variable3 = variableFromSentence(input_lang3, sentence3)

    input_length = input_variable1.size()[0]
    # encoder_hidden = encoder.initHidden()
    encoder1_hidden = encoder1.initHidden()
    encoder2_hidden = encoder2.initHidden()
    encoder3_hidden = encoder3.initHidden()


    encoder1_outputs = Variable(torch.zeros(max_length, encoder1.hidden_size))
    encoder1_outputs = encoder1_outputs.cuda() if use_cuda else encoder1_outputs
    
    encoder2_outputs = Variable(torch.zeros(max_length, encoder2.hidden_size))
    encoder2_outputs = encoder2_outputs.cuda() if use_cuda else encoder2_outputs
    
    encoder3_outputs = Variable(torch.zeros(max_length, encoder3.hidden_size))
    encoder3_outputs = encoder3_outputs.cuda() if use_cuda else encoder3_outputs

    for ei in range(input_length):
        encoder1_output, encoder_hidden = encoder1(input_variable1[ei], encoder1_hidden)
        encoder2_output, encoder_hidden = encoder2(input_variable2[ei], encoder2_hidden)
        encoder3_output, encoder_hidden = encoder3(input_variable3[ei], encoder3_hidden)
        
        encoder1_outputs[ei] = encoder1_outputs[ei] + encoder1_output[0][0]
        encoder2_outputs[ei] = encoder2_outputs[ei] + encoder2_output[0][0]
        encoder3_outputs[ei] = encoder3_outputs[ei] + encoder3_output[0][0]

    encoder_output_list = [encoder1_output, encoder2_output, encoder3_output]
    encoder_outputs_list = [encoder1_outputs, encoder2_outputs, encoder3_outputs]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden_list = [encoder1_hidden, encoder2_hidden, encoder3_hidden]

    decoded_words = []
    decoder_attentions1 = torch.zeros(max_length, max_length)
    decoder_attentions2 = torch.zeros(max_length, max_length)
    decoder_attentions3 = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention_list = decoder(
            decoder_input, decoder_hidden_list, encoder_output_list, encoder_outputs_list)
        decoder_attentions1[di] = decoder_attention_list[0].data
        decoder_attentions2[di] = decoder_attention_list[1].data
        decoder_attentions3[di] = decoder_attention_list[2].data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])
        
        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_attention_out_list = [decoder_attentions1[:di + 1], decoder_attentions2[:di + 1], decoder_attentions3[:di + 1]]
    return decoded_words, decoder_attention_list


def evaluateRandomly(encoder_list, decoder, n=10):
     
    encoder1, encoder2, encoder3 = encoder_list 

    for i in range(n):
        
        random_index = np.random.permutation(list(range(len(pairs1))))
        
        # pair = random.choice(pairs)
        pair1 = pairs1[random_index[i]]
        pair2 = pairs2[random_index[i]]
        pair3 = pairs3[random_index[i]]
        pair0_list = [pair1[0], pair2[0], pair3[0]]
        print('>', pair2[0])
        print('=', pair2[1])
        output_words, attentions = evaluate(encoder_list, decoder, pair0_list)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

# hidden_size = 128
# encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
# attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words,
                               # 1, dropout_p=0.1)
def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(110)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
		       ['<EOS>'], rotation=89)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0))

    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
	encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    # showAttention(input_sentence, output_words, attentions)




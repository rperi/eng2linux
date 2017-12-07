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
import pickle
from datetime import datetime
from eng2linux import *
from nltk.corpus import wordnet
import itertools as IT
from nltk.metrics import *
import ipdb
from word2word import *
from word2pos import *
from word2char import *


use_cuda = torch.cuda.is_available()

def read_and_store(list_path):
    with open(list_path) as f:
        content = []
        for line in f:
            line = line.strip()
            content.append(line)
    return content

def f(word1, word2):
    if wordnet.synsets(word1) != []:
        wordFromList1 = wordnet.synsets(word1)[0]
    else:
        s = 0

    if wordnet.synsets(word2) != []: 
        wordFromList2 = wordnet.synsets(word2)[0] 
    else:
        s = 0
    
    if wordnet.synsets(word1) == [] or wordnet.synsets(word2) == []:  
        s = 0
    else: 
        s = wordFromList1.wup_similarity(wordFromList2)
    return s

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def evaluate_randomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    # s = re.sub(r"([.!?])", r" \1", s)
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def length_filter(sentence):
    sentence_list = sentence.split(' ')
    if len(sentence_list) >= (MAX_LENGTH -1) :
        sent_out = sentence_list[:(MAX_LENGTH -1)]
    else:
        sent_out = sentence_list
    return ' '.join(sent_out)

def ev(input_sentence):
    

    sentence1 = word2word(input_sentence)
    sentence2 = word2char_test(input_sentence)
    sentence3 = word2pos(input_sentence)
   
    sentence1 = length_filter(sentence1)
    sentence2 = length_filter(sentence2)
    sentence3 = length_filter(sentence3)

    sentence3_list = sentence3.split(' ')
    sentence3_list = [normalizeString(s) for s in sentence3_list]
    sentence3 = ' '.join(sentence3_list)

    input_sentence_list = [sentence1, sentence2, sentence3]
    print('input_sentence_list:', input_sentence_list)
    output_words, attentions = evaluate(encoder_list, attn_decoder1, input_sentence_list)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    # showAttention(input_sentence, output_words, attentions)
    return ''.join(output_words)[:-5]

# Missing Word Replacer
def mwr(input_sent, input_lang):
    input_sent = input_sent.split(' ')
    word_list = input_lang.index2word.values()
    wn_word_list = [ x for x in word_list if len(wordnet.synsets(x)) > 0 ]
    replace_mat = []
    
    # for k, word in enumerate(input_sent):
        # dist_buff = []
        # if word not in word_list:
            # replace_mat.append('<unk>')
        # else:
            # replace_mat.append(word)

    for k, word in enumerate(input_sent):
        dist_buff = []
        if word not in word_list:
            for ct, tr_word in enumerate(wn_word_list):
                dist = f(word, tr_word)
                # print(' %s %s %s'%(word, tr_word, dist) )
                if dist == None:
                    dist_val = 0
                # elif dist == 0:
                    # dist_val = 0
                else:
                    dist_val = dist
                dist_buff.append(dist_val)
            
            if set(dist_buff) == {0}:
                replace_mat.append('<unk>')
                # replace_mat.append('')
            
            else:
                replace_mat.append(wn_word_list[np.argmax(dist_buff)])
        else:
            replace_mat.append(word)

    # print('Replaced Mat:', ' '.join(replace_mat))
    replaced_string = ' '.join(replace_mat)
    return replaced_string


def get_nearest(dec_comm, comm_list_train):
    dist_array = np.zeros((len(comm_list_train), len(dec_comm)))
    
    print('Searching the command dictionary...')
    comm_list_train_join = [ ''.join(val.split(' ')) for val in comm_list_train  ]
    for ct, des_train in enumerate(comm_list_train_join):
        buff = [] 
        for k, val in enumerate(dec_comm):
            buff.append(edit_distance(val, des_train))

        dist_array[ct, :] = buff 
        # print(distances_list)
   
    est_comm = []
    for kd, dist_vals in enumerate(dec_comm):
        print('Estimating: ' , dist_vals)
        estimated_command = comm_list_train_join[np.argmin(dist_array[:, kd])]
        # ipdb.set_trace()
         
        est_comm.append(estimated_command)

    return est_comm


def eval_est(est_comm_list, des_list_test, comm_list_test):
    
    grade_list = []
    for k, val in enumerate(comm_list_test):
        val_join = ''.join(val.split(' '))
        grade_list.append(val_join == est_comm_list[k])
        print('description:', des_list_test[k])
        print('EST:', est_comm_list[k], 'ACTUAL:', val_join, '\n')
    score = sum(grade_list)/len(comm_list_test)
    print('Score:', score)
    return grade_list, score


def ev_test_set(des_list_test, input_lang):
    dec_comm = []
    for k, des in enumerate(des_list_test):
        print(k+1, '/', len(des_list_test), ' description:', des)
        dec_comm.append( ev(mwr(des, input_lang1) ) )

    return dec_comm

if __name__ == '__main__':
    
    hidden_size = 128
    encoder1 = EncoderRNN(input_lang1.n_words, hidden_size)
    encoder2 = EncoderRNN(input_lang2.n_words, hidden_size)
    encoder3 = EncoderRNN(input_lang3.n_words, hidden_size)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words,
                                       1, dropout_p=0.1)
    
    if use_cuda:
        encoder1 = encoder1.cuda()
        encoder2 = encoder2.cuda()
        encoder3 = encoder3.cuda()
        attn_decoder1 = attn_decoder1.cuda()
    

    enc_dec_filename = 'enc_dec_12_38PM_November_29_2017.pickle'
    enc_dec_filename = '3enc_dec_04_15PM_December_06_2017'
    enc_dec_filename = '3enc_dec_05_25PM_December_06_2017.pickle'


    print('Loading saved encoder and decoder...')
    with open(enc_dec_filename, 'rb') as handle:
        encoder_list, attn_decoder1, input_lang_list, output_lang = pickle.load(handle)
   
    encoder1, encoder2, encoder3 = encoder_list 
    input_lang1, input_lang2, input_lang3 = input_lang_list  

    # evaluate_randomly(encoder1, attn_decoder1) 
    
    # replace_mat = mwr('gummy bear sisters', input_lang)
    # replace_mat = mwr('Where are all the people ? we have been waiting here for like 19 hours', input_lang)

    train_data_list = read_and_store('./data/com-eng_train_stop.txt')
    test_data_list = read_and_store('./data/com-eng_test_stop.txt')
    # train_data_list = read_and_store('../data/com-eng_train_org.txt')
    # test_data_list_org = read_and_store('../data/com-eng_test_org.txt')
    # train_data_list = read_and_store('../data/com-eng_train_unclean.txt')
    # test_data_list = read_and_store('../data/com-eng_test_unclean.txt')
    # test_data_list = read_and_store('../data/com-eng_stoflw_test_short.txt')
    # test_data_list = read_and_store('../data/com-eng_stoflw_test.txt')

    # Description and Command from test set
    des_list_test = [ line.split('@')[1] for k, line in enumerate(test_data_list) ]
    comm_list_test = [ line.split('@')[0] for k, line in enumerate(test_data_list) ]
    # comm_list_test_org = [ line.split('@')[0] for k, line in enumerate(test_data_list_org) ]
    
    # Description and Command from train set
    des_list_train = [ line.split('@')[1] for k, line in enumerate(train_data_list) ]
    comm_list_train = [ line.split('@')[0] for k, line in enumerate(train_data_list) ]
   
    tr_tst = 1

    if tr_tst == 1 :
        subs = len(test_data_list)
       
        # Description and Command from train set
        dec_comm_list = ev_test_set(des_list_test[:subs], input_lang1)
        est_comm = get_nearest(dec_comm_list, comm_list_train)
        print(est_comm)
        
        # Evaluate the system - TEST set
        eval_est(est_comm, des_list_test[:subs], comm_list_test[:subs])
       
    elif tr_tst == 2:
        subs = len(train_data_list)
        
        # Description and Command from train set
        dec_comm_list_train = ev_test_set(des_list_train[:subs], input_lang)
        est_comm = get_nearest(dec_comm_list_train, comm_list_train)
        # print(est_comm)
        
        # Evaluate the system - TRAIN set
        eval_est(est_comm, des_list_train[:subs], comm_list_train[:subs])

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

use_cuda = torch.cuda.is_available()
torch.backends.cudnn.enabled = False
def read_and_store(list_path):
    with open(list_path, encoding='utf-8', errors='ignore') as f:
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


def ev(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    # showAttention(input_sentence, output_words, attentions)
    return ''.join(output_words)[:-5]

# Missing Word Replacer
def mwr(input_sent, input_lang):
    input_sent = input_sent.split(' ')
    word_list = list(input_lang.index2word.values())
    word_list.remove('EOS')
    word_list.remove('SOS')
    wn_word_list = [ x for x in word_list if len(wordnet.synsets(x)) > 0 ]
    replace_mat = []
   
    ### No MWR
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
                print("<unk>")
                replace_mat.append('<unk>')
                # replace_mat.append('')
                break 
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
        # print('Description Train Set:', ct, '/ ', len(comm_list_train_join), flush=True)
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
        dec_comm.append( ev(mwr(des, input_lang) ) )

    return dec_comm


def ev_test_set_no_mwr(des_list_test, input_lang):
    dec_comm = []
    for k, des in enumerate(des_list_test):
        print(k+1, '/', len(des_list_test), ' description:', des)
        dec_comm.append( ev(des) )

    return dec_comm

if __name__ == '__main__':
    
    # hidden_size = 128
    # encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
    # attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words,
                                       # 1, dropout_p=0.1)
    
    # if use_cuda:
        # encoder1 = encoder1.cuda()
        # attn_decoder1 = attn_decoder1.cuda()
    

    # enc_dec_filename = 'enc_dec_07_29PM_November_22_2017.pickle'
    # enc_dec_filename = 'enc_dec_03_52PM_November_27_2017.pickle'
    # enc_dec_filename = 'enc_dec_08_16PM_November_27_2017.pickle'
    # enc_dec_filename = 'enc_dec_10_38PM_November_27_2017.pickle'
    # enc_dec_filename = 'enc_dec_11_37PM_November_27_2017.pickle'
    # enc_dec_filename = 'enc_dec_03_03PM_November_28_2017_clean.pickle'
    # enc_dec_filename = 'enc_dec_03_54PM_November_28_2017_clean_inlang.pickle'
    # enc_dec_filename = 'enc_dec_03_03PM_November_28_2017_clean.pickle'
    # enc_dec_filename = 'enc_dec_12_38PM_November_29_2017_clean_withlangs_t1.pickle'
    # enc_dec_filename =  'enc_dec_04_58PM_December_07_2017_word_stop.pickle'
    # enc_dec_filename =  'enc_dec_05_07PM_December_07_2017_word_stop_augmented_full.pickle'
    # enc_dec_filename = 'enc_dec_11_37PM_December_07_2017_hidden1024_word_stop.pickle'
    #enc_dec_filename = 'enc_dec_04_18PM_December_08_2017_hidden1024_char_stop.pickle' 
    #enc_dec_filename = 'enc_dec_10_32PM_November_29_2017_character_iter80_teacherLoading1_stpwrd.pickle'
    #enc_dec_filename = 'enc_dec_10_21PM_December_08_2017_char_iter80_teacherLoading1_stpwrd.pickle'
    #enc_dec_filename = 'enc_dec_02_04AM_December_09_2017_char_iter80_teacherLoading1_stpwrd_aug.pickle'
    enc_dec_filename = 'enc_dec_04_50PM_November_29_2017_character_iter80_teacherLoading1.pickle'
    print('Loading saved encoder and decoder...')
    with open(enc_dec_filename, 'rb') as handle:
        encoder1, attn_decoder1, input_lang, output_lang = pickle.load(handle)
    
    # evaluate_randomly(encoder1, attn_decoder1) 
    
    # replace_mat = mwr('gummy bear sisters', input_lang)
    # replace_mat = mwr('Where are all the people ? we have been waiting here for like 19 hours', input_lang)

    #train_data_list = read_and_store('../data/com-eng_train_stop.txt')
    train_data_list = read_and_store('../data/com-eng_train_character.txt')
    # train_data_list = read_and_store('../data/com-eng_train_stop_augmented_full.txt')
    # test_data_list = read_and_store('../data/com-eng_stoflw_test_stop.txt')
    #test_data_list = read_and_store('../data/com-eng_stoflw_test_stop_character.txt')
    #test_data_list = read_and_store('../data/com-eng_test_character.txt')
    test_data_list = read_and_store('../data/com-eng_stoflow_test_character.txt')
    # Description and Command from test set
    des_list_test = [ line.split('@')[1] for k, line in enumerate(test_data_list) ]
    comm_list_test = [ line.split('@')[0] for k, line in enumerate(test_data_list) ]
    # comm_list_test_org = [ line.split('@')[0] for k, line in enumerate(test_data_list_org) ]
    
    # Description and Command from train set
    des_list_train = [ line.split('@')[1] for k, line in enumerate(train_data_list) ]
    comm_list_train = [ line.split('@')[0] for k, line in enumerate(train_data_list) ]
   
    tr_tst = 0

    if tr_tst == 0 : # NO MWR for char level
        subs = len(test_data_list)
       
        # Description and Command from train set
        dec_comm_list = ev_test_set_no_mwr(des_list_test[:subs], input_lang)
        est_comm = get_nearest(dec_comm_list, comm_list_train)
        print(est_comm)

        
        # Evaluate the system - TEST set
        eval_est(est_comm, des_list_test[:subs], comm_list_test[:subs])
    
    if tr_tst == 1 :
        subs = len(test_data_list)
       
        # Description and Command from train set
        dec_comm_list = ev_test_set(des_list_test[:subs], input_lang)
        est_comm = get_nearest(dec_comm_list, comm_list_train)
        print(est_comm)

        
        # Evaluate the system - TEST set
        grade_list, score = eval_est(est_comm, des_list_test[:subs], comm_list_test[:subs])
       
    elif tr_tst == 2:
        subs = len(train_data_list)
        
        # Description and Command from train set
        dec_comm_list_train = ev_test_set(des_list_train[:subs], input_lang)
        grade_list, score = est_comm = get_nearest(dec_comm_list_train, comm_list_train)
        # print(est_comm)
        
        # Evaluate the system - TRAIN set
        eval_est(est_comm, des_list_train[:subs], comm_list_train[:subs])


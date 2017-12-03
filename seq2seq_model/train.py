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
import eng2linux
from eng2linux import *
use_cuda = torch.cuda.is_available()


if __name__ == '__main__':
    
    hidden_size = 128
    encoder1 = EncoderRNN(input_lang_word.n_words, input_lang_char.n_words, hidden_size)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words,
                                   1, dropout_p=0.1)
    if use_cuda:
        encoder1 = encoder1.cuda()
        attn_decoder1 = attn_decoder1.cuda()

    # Start the training process.
    trainIters(encoder1, attn_decoder1,  repeat=50, print_every=100)

    enc_dec = [encoder1, attn_decoder1]
    evaluateRandomly(encoder1, attn_decoder1)
    
    dtime = datetime.now().strftime("%I_%M%p_%B_%d_%Y")
    with open('enc_dec_'+ dtime + '_iter50_TeacherForcing1_stop_merged.pickle', 'wb') as handle:
        # pickle.dump([encoder1, attn_decoder1], handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump([encoder1, attn_decoder1, input_lang_word, input_lang_char, output_lang], handle, protocol=pickle.HIGHEST_PROTOCOL)


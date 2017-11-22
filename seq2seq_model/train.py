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
    # Start the training process.
    trainIters(eng2linux.encoder1, attn_decoder1, repeat=30, print_every=100)

    enc_dec = [encoder1, attn_decoder1]
    evaluateRandomly(encoder1, attn_decoder1)
    
    dtime = datetime.now().strftime("%I_%M%p_%B_%d_%Y")
    with open('enc_dec_'+ dtime + '.pickle', 'wb') as handle:
        pickle.dump([encoder1, attn_decoder1], handle, protocol=pickle.HIGHEST_PROTOCOL)


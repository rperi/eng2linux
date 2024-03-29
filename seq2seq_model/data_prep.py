import numpy as np
import json
import random
import string
from augment_data import *

def read_and_store(list_path):
    with open(list_path) as f:
        content = []
        for line in f:
            line = line.strip().split(' ')
            content.append(line)
    return content

def read_and_store2(list_path):
    with open(list_path) as f:
        content = []
        for line in f:
            line = line.strip()
            content.append(line)
    return content

def read_and_write(w_path, list_to_wr):
    with open(w_path, "w") as output:
        for k, val in enumerate(list_to_wr):
            output.write(val + '\n')
    return None


def read_json(json_fn):
    with open(json_fn) as data_file:
        data_loaded = json.load(data_file)
    return data_loaded


def erase_punc(desc):
    punc_list = string.punctuation

    new_desc = []
    for k, letter in enumerate(desc):
        if letter in punc_list:
            new_desc.append(' ') 
        else: 
            new_desc.append(letter)
    new_desc = ''.join(new_desc)
    new_desc = new_desc.replace('  ', ' ')            
    new_desc = new_desc.replace('   ', ' ')            
    new_desc = new_desc.replace('    ', ' ')            
    return new_desc 

if __name__ == '__main__':
   
    json_dict = read_json('../data/com2des_TJ.json')
    
    data_out = []
    for key, val in json_dict.items():
        sep_let = list(key)
        for desc in val:
            COM = ' '.join(sep_let).strip() 
            # DES = erase_punc(desc)
            DES = desc 
            data_out.append(COM + '@'+ DES)
    random.shuffle(data_out)

    random.seed(0)
    split_ratio = 0.9
    train_set = data_out[:int(split_ratio*len(data_out))]
    dev_set = data_out[int(split_ratio*len(data_out)):]
    
    read_and_write('../data/com-eng_train.txt', train_set)
    read_and_write('../data/com-eng_test.txt', dev_set)
    # read_and_write('../data/com-eng_train_unclean.txt', train_set)
    # read_and_write('../data/com-eng_test_unclean.txt', dev_set)

    stoflw_test_set = read_and_store2('../data/com2des_RP_170pages_stackoverflow.txt')

    data_out = []
    for key, val in enumerate(stoflw_test_set):
        split_list = val.split('@')
        # print(split_list)
        sep_let = split_list[0]
        desc = split_list[1]
        # print(desc)
        COM = ' '.join(sep_let).strip() 
        DES = erase_punc(desc)
        data_out.append(COM + '@'+ DES)
    random.shuffle(data_out)

    random.seed(0)
    split_ratio = 0.9
    
    read_and_write('../data/com-eng_stoflw_test.txt', data_out)


    # Parameters
    # data_file = "../data/com-eng_train.txt"  # file containing the original command-description pairs
    # data_file = "../data/com-eng_train_toy.txt"  file containing the original command-description pairs
    # out_data_file = "../data/com-eng_train_aug.txt"  output file where the new augmented pairs are written

    # max_num_syn = 3  maximum number of synonyms per word to be used for augmentation

    # FLAG_POS_Stanf = 1  when set to 0, it uses the averaged perceptron tagger from nltk else uses stanford POS tagger
    # FLAG_Save_Verbs2Syns_dict = 0 flag to save the verbs-synonyms dictionary to file. Need to manually set path for dictionary

    # Function call
    # augment_eng2linux_Data(data_file, out_data_file, max_num_syn, FLAG_POS_Stanf, FLAG_Save_Verbs2Syns_dict)


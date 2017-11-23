import numpy as np
import json
import random

def read_and_store(list_path):
    with open(list_path) as f:
        content = []
        for line in f:
            line = line.strip().split(' ')
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

if __name__ == '__main__':
   
    json_dict = read_json('../data/com2des_TJ.json')
    
    data_out = []
    for key, val in json_dict.items():
        sep_let = list(key)
        for desc in val:
            COM = ' '.join(sep_let).strip() 
            DES = desc
            data_out.append(COM + '@'+ DES)
    random.shuffle(data_out)

    random.seed(0)
    split_ratio = 0.9
    train_set = data_out[:int(split_ratio*len(data_out))]
    dev_set = data_out[int(split_ratio*len(data_out)):]
    
    read_and_write('../data/com-eng_train.txt', train_set)
    read_and_write('../data/com-eng_test.txt', dev_set)



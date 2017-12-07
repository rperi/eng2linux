from word2word import *
from word2pos import *
from word2char import *

def read_and_store(list_path):
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

def out_three_data(in_list):
   
    sent1_list = []
    sent2_list = []
    sent3_list = []

    for k, sent in enumerate(in_list):
        command = sent.split("@")[0].strip()
        sentence = sent.split("@")[1].strip()
        
        # sentence = ' '.join(sent)
        # print('count:', k, sentence)
        if len(sentence) != 0:
            enc1data = word2word(sentence)
            enc2data = word2char(sentence)
            enc3data = word2pos(sentence)
            len_list = [len(enc1data.split(' ')), len(enc2data.split(' ')), len(enc3data.split(' '))]
            if len(list(set(len_list))) > 1:
                print( 'length mismatch:', str(k+1), ' command')
            sent1_list.append(command + '@' + enc1data)
            sent2_list.append(command + '@' + enc2data)
            sent3_list.append(command + '@' + enc3data) 
        else:
            sent1_list.append(command + '@')
            sent2_list.append(command + '@')
            sent3_list.append(command + '@')


    return sent1_list, sent2_list, sent3_list


if __name__ == "__main__":

    org_list = read_and_store('./com-eng_train_stop.txt')
    sent1_list, sent2_list, sent3_list = out_three_data(org_list)
    read_and_write('./data/com-eng_w2w.txt', sent1_list)
    read_and_write('./data/com-eng_w2c.txt', sent2_list)
    read_and_write('./data/com-eng_w2p.txt', sent3_list)




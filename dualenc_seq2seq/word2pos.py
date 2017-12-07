import nltk
from word2char import *

def word2pos(in_string):

    text = nltk.word_tokenize(in_string)
    pos_tag = nltk.pos_tag(text)
    in_string_list = in_string.split(' ')
    pos_list = [ x[1] for x in pos_tag ] 

    if len(in_string) > len(pos_list):
        diff = len(in_string) - len(pos_list)
        for k in range(diff):
            pos_tag.append('NN')
    elif len(in_string) < len(pos_list):
        pos_list = pos_list[:len(in_string)]
    
    char_list = word2char(in_string)
    char_list = char_list.split(' ')
    pos_repeat = []
    for k, word in enumerate(in_string_list):
        word_len = len(list(word))
        pos_repeat.extend([pos_list[k]] * word_len)
        pos_repeat.extend(['_'])
    pos_repeat = pos_repeat[:-1]
    
    if len(pos_repeat) != len(char_list):
        print('WARNING: Length difference.')
    
    return ' '.join(pos_repeat)
         
if __name__ == "__main__":
    pass

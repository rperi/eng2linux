import word2char

def word2word(in_string):

    in_string_list = in_string.split(' ')
    
    word_repeat = []
    for k, word in enumerate(in_string_list):
        word_len = len(list(word))
        word_repeat.extend([word] * word_len)
        word_repeat.extend(['_'])
    word_repeat = word_repeat[:-1]
    return ' '.join(word_repeat)


if __name__ == "__main__":
    pass

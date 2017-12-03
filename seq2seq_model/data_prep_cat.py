# Given the word and character level splits, convert them into a format to be used for merging the two modalities

split = 'test'
inp_file_word = "../data/com-eng_%s_stop.txt" % split

out_file_word = "../data/com-eng_%s_stop_repeat.txt" % split


w = open(inp_file_word, 'r')
o = open(out_file_word, 'a+')
for lines in w.readlines():

    comm = lines.strip().split('@')[0].strip()
    desc = lines.strip().split('@')[1].strip()

    words_list = desc.split()

    word_list_new = []
    for idx, words in enumerate(words_list):
        char_len = len(list(words.strip()))
        word_list_new.extend([words]*char_len)
        if idx != len(words_list) - 1:
            word_list_new.append('_')

    out_string = comm + '@' + " ".join(word_list_new) + "\n"

    o.writelines(out_string)
o.close()


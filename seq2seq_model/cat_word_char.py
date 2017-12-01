import os.path

splits = ['train', 'test']
for split in splits:

    word_file = "../data/com-eng_%s_stop.txt" % split
    char_file = "../data/com-eng_%s_stop_character.txt" % split
    out_file = "../data/com-eng_%s_stop_cat.txt" % split
    if not os.path.isfile(out_file):
        w = open(word_file, 'r')
        c = open(char_file, 'r')
        o = open(out_file, 'a+')

        assert len(w.readlines()) == len(c.readlines())

        c = open(char_file, 'r')
        for idx, lines_c in enumerate(c.readlines()):
            comm = lines_c.strip().split('@')[0].strip()
            w = open(word_file, 'r')
            lines_w = w.readlines()[idx]
            desc_c = lines_c.strip().split('@')[1].strip()
            desc_w = lines_w.strip().split('@')[1].strip()

            desc_new = desc_c + " - " + desc_w

            output_string = comm + "@" + desc_new + '\n'
            o.writelines(output_string)

        o.close()
        c.close()
        w.close()
    else:
        print("output file already exists. Not over writing. Exiting ...")
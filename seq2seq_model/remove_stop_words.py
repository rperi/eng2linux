# Takes as input the com-eng_train.txt file and removes stop words from descriptions
import nltk
nltk.download('stopwords')

import os.path

from nltk.corpus import stopwords

splits = ["train", "test"]

for split in splits:

    input_file = "../data/com-eng_" + split + ".txt"
    output_file = "../data/com-eng_" + split + "_stop.txt"

    if not os.path.exists(output_file):
        f = open(input_file, 'r')
        o = open(output_file, 'a+')

        for lines in f.readlines():
            comm = lines.strip().split("@")[0].strip()
            desc = lines.strip().split("@")[1].strip()

            word_list = desc.split()
            word_list_stop = []

            for words in word_list:
                if words not in stopwords.words('english'):
                    word_list_stop.append(words)

            desc_stop = " ".join(word_list_stop)

            output = comm + "@" + desc_stop + "\n"

            o.writelines(output)
        f.close()
        o.close()
    else:
        print("Output file already exists. Not writing to it again. Exiting...")


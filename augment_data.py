
import time
import numpy as np
import nltk

from nltk.corpus import wordnet as wn
import unicodedata
from nltk.tokenize import RegexpTokenizer


class ContinueOuterLoop(Exception):
    pass


def augment_eng2linuxData(data_file, out_data_file, max_num_syn, FLAG_POS_Stanf, FLAG_Save_Verbs2Syns_dict, POS_Tagger_Stanford_dir):

    if FLAG_POS_Stanf == 1:
        from nltk.tag.stanford import StanfordPOSTagger as PT

    tokenizer = RegexpTokenizer(r'\w+')

    if FLAG_POS_Stanf == 0:

        nltk.download('averaged_perceptron_tagger')

    else:

        # Add appropriate paths to the Stanford POS tagger models and jar files here
        stanf_tagger = PT(POS_Tagger_Stanford_dir + "/models/english-bidirectional-distsim.tagger",
                          POS_Tagger_Stanford_dir + "/stanford-postagger.jar")

        # print(stanf_tagger.tag(nltk.word_tokenize("List all the files")))

    # This part of code finds "max_num_syn" synonyms for each verb for each description given in data file and stores in dictionary

    data = open(data_file, 'r')
    verbs2syns_dict = dict()

    start_time = time.time()
    for idx, lines in enumerate(data.readlines()):
        print(idx+1)
        command = lines.strip().split("@")[0]

        desc = lines.strip().split("@")[1]
        desc = tokenizer.tokenize(desc)
        desc = [d.lower() for d in desc]
        desc = [d for d in desc if d.find('\xe2') == -1] # to take care of special characters like 1. A different symbol for apostrophe(')  2. a weird symbol for period in line 688

        if FLAG_POS_Stanf == 0:
            tokens_st = nltk.pos_tag(desc)
        else:
            tokens_st = stanf_tagger.tag(desc)
            tokens_st = [tuple(map(str, eachTuple)) for eachTuple in tokens_st]

        tokens_VB = [item for item in tokens_st if item[1].find("VB") != -1]

        for verbs in [tup[0] for tup in tokens_VB]:
            try:
                if verbs not in verbs2syns_dict.keys():
                    verbs2syns_dict[verbs] = []

                    syns = wn.synsets(verbs, pos='v')
                    syns_list = []
                    for idx, syn in enumerate(syns):
                        syn_list = syn.lemma_names()
                        for synonyms in syn_list:
                            synonym = unicodedata.normalize('NFKD', synonyms).encode('ascii', 'ignore')
                            if len(syns_list) <= max_num_syn:
                                if synonym not in syns_list and synonym != verbs:
                                    syns_list.append(synonym)
                            else:
                                raise ContinueOuterLoop()
            except ContinueOuterLoop:
                verbs2syns_dict[verbs].extend(syns_list)
                continue

    print(time.time() - start_time)

    if FLAG_Save_Verbs2Syns_dict == 1:
        np.save('/home/raghuveer/coursework/CSCI662/final_project/data/verb2syns_dict', verbs2syns_dict)

    # This part of code does actual data augmentation

    data = open(data_file, 'r')
    out_file = open(out_data_file, 'a+')

    start_time = time.time()
    for idx, lines in enumerate(data.readlines()):
        print(idx + 1)
        command = lines.strip().split("@")[0]

        desc = lines.strip().split("@")[1]
        desc = tokenizer.tokenize(desc)
        desc = [d.lower() for d in desc]
        desc = [d for d in desc if d.find('\xe2') == -1]

        if FLAG_POS_Stanf == 0:
            tokens_st = nltk.pos_tag(desc)
        else:
            tokens_st = stanf_tagger.tag(desc)
            tokens_st = [tuple(map(str, eachTuple)) for eachTuple in tokens_st]

        tokens_VB = [item for item in tokens_st if item[1].find("VB") != -1]

        verbs_in_desc = [tup[0] for tup in tokens_VB]
        verbs_in_desc = verbs_in_desc[0:min(len(verbs_in_desc), 2)]

        if verbs_in_desc:
            for syns_verb0 in verbs2syns_dict[verbs_in_desc[0]]:
                desc_0 = [d.replace(verbs_in_desc[0], syns_verb0) for d in desc]
                if len(verbs_in_desc) > 1:
                    for syns_verb1 in verbs2syns_dict[verbs_in_desc[1]]:
                        desc_new = [d.replace(verbs_in_desc[1], syns_verb1) for d in desc_0]
                        string = command + "@" + " ".join(desc_new) + "\n"
                        out_file.writelines(string)
                else:
                    desc_new = desc_0
                    string = command + "@" + " ".join(desc_new) + "\n"
                    out_file.writelines(string)

    print(time.time() - start_time)


if __name__ == '__main__':

    # Parameters
    data_file = "/home/raghuveer/coursework/CSCI662/final_project/data/data_TJ_new.txt"  # file containing the original command-description pairs
    out_data_file = "/home/raghuveer/coursework/CSCI662/final_project/data/temp2.txt"  # output file where the new augmented pairs are written

    max_num_syn = 3  # maximum number of synonyms per word to be used for augmentation

    FLAG_POS_Stanf = 1  # when set to 0, it uses the averaged perceptron tagger from nltk else uses stanford POS tagger
    FLAG_Save_Verbs2Syns_dict = 0 # flag to save the verbs-synonyms dictionary to file. Need to manually set path for dictionary

    POS_Tagger_Stanford_dir = '/home/raghuveer/PycharmProjects/NLP/final_project/tagger' # directory where the stanford models and jar file have been unzipped

    # Function call
    augment_eng2linuxData(data_file, out_data_file, max_num_syn, FLAG_POS_Stanf, FLAG_Save_Verbs2Syns_dict, POS_Tagger_Stanford_dir)
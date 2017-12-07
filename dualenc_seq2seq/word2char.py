import os.path

def word2char_test(in_string):

    char_list = list(in_string)
    char_list = [ w.replace('<', '_') for w in char_list ] 
    char_list = [ w.replace('>', '_') for w in char_list ] 

    char_list = [ch.replace(" ","_") for ch in char_list]
    char_string = " ".join(char_list)
    return char_string

def word2char(in_string):

    char_list = list(in_string)
    char_list = [ch.replace(" ","_") for ch in char_list]
    char_string = " ".join(char_list)
    return char_string


if __name__ == "__main__":
    splits = ["train", "test"]

    for split in splits:

        input_file = "../data/com-eng_" + split + "_stop.txt"
        output_file = "../data/com-eng_" + split + "_stop_character.txt"

        if not os.path.exists(output_file):
            f = open(input_file, "r")
            o = open(output_file, "a+")

            for lines in f.readlines():
                #print(lines.strip())
                desc = lines.split("@")[1].strip()
                char_list = list(desc)
                char_list = [ch.replace(" ","_") for ch in char_list]
                char_string = " ".join(char_list)

                out_string = lines.split("@")[0].strip() + "@" + char_string
                o.writelines(out_string + "\n")
            f.close()
            o.close()
        else:
            print("Output file already exists. Not writing to it again. Exiting...")

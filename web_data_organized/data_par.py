import json
from subprocess import PIPE, Popen
from bs4 import BeautifulSoup
import numpy as np
import fileinput


def read_and_write_nonl(w_path, list_to_wr):
    with open(w_path, "w") as output:
        for k, val in enumerate(list_to_wr):
            output.write(val)
    return None


def read_and_write(w_path, list_to_wr):
    with open(w_path, "w") as output:
        for k, val in enumerate(list_to_wr):
            output.write(val+'\n')
    return None


def data1():

    fn = 'data1.html'

    data_list = []
    for k, line in enumerate(fileinput.input(fn)):
        # print(line)
        data_buff = []
        
        spl_line = line.replace('<','>').split('>')
        if '/td' in spl_line:
            # print('spl_line:', spl_line)
            data_list.append(spl_line[-3])
    
    out_list = []
    for k, val in enumerate(data_list):
        if k % 2 == 0:
            buff = data_list[k]+'@'
        elif k % 2 == 1:
            out_list.append(buff + data_list[k].strip())

    read_and_write('data1.txt', out_list)
    return out_list

def data3(): 
    fn = 'data3.html'
    data_list = []
    for k, line in enumerate(fileinput.input(fn)):
        data_buff = []
        if len(line) > 2:
            buff_list = line.strip().split(' ')
            outstring = buff_list[0] + '@'+ ' '.join(buff_list[1:]).strip()
            data_list.append(outstring)
    read_and_write('data3.txt', data_list)

    return data_list

def data9():
    fn = 'data9.html'
    data_list = []
    for k, line in enumerate(fileinput.input(fn)):
        if len(line) > 2:
            
            line = line.replace(': ', '@')
            line = line.strip()
            data_list.append(line)
    read_and_write_nonl('data9.txt', data_list)
    return data_list

def data10():

    fn = 'data10.html'
    markup = open(fn)
    soup = BeautifulSoup(markup.read(), 'html.parser')
    markup.close()
    f = open("data10_text.tmp", "w")
    f.write(soup.get_text())
    f.close()
    
    fn = 'data10_text.tmp'
    data_list = []
    for k, line in enumerate(fileinput.input(fn)):
        data_list.append(line)
    flag_st = 0
    flag_end = 0
    
    out_list = []
    sub_list = []
    count = 0
    for k in range(1, len(data_list)):
        cn = '\n'        
        if '@' in list(line):
            break
        
        elif data_list[k-1] == cn and data_list[k] != cn:
            flag_st = 1

        elif data_list[k-1] != cn and data_list[k] != cn:
            flag_st = 1

        elif data_list[k-1] != cn and data_list[k] == cn:
            flag_st = 0
            out_list.append(sub_list)
            count = 0
            sub_list = []
        
        if data_list[k] != cn:
            count += 1

        if flag_st ==1 and data_list[k] != cn:
            
            buff = data_list[k].strip()
            sub_list.append(buff)
    
    out_list2 = []
    for k, val in enumerate(out_list):
        if len(val) in [3,4]:
            buff = val[0].strip() + '@' + val[2].strip()
            out_list2.append(buff)
    out_list2 = out_list2[3:-4] 
    read_and_write('data10.txt', out_list2)
    return out_list2[3:-4]

def data11():

    fn = 'data11.html'
    markup = open(fn)
    soup = BeautifulSoup(markup.read(), 'html.parser')
    markup.close()
    f = open("data11_text.tmp", "w")
    f.write(soup.get_text())
    f.close()
    
    fn = 'data11_text.tmp'
    data_list = []
    for k, line in enumerate(fileinput.input(fn)):
        data_list.append(line)
    flag_st = 0
    flag_end = 0
    
    out_list = []
    sub_list = []
    count = 0
    for k in range(1, len(data_list)):
        cn = '\n'        
        if '@' in list(line):
            break
        
        elif data_list[k-1] == cn and data_list[k] != cn:
            flag_st = 1

        elif data_list[k-1] != cn and data_list[k] != cn:
            flag_st = 1

        elif data_list[k-1] != cn and data_list[k] == cn:
            flag_st = 0
            out_list.append(sub_list)
            count = 0
            sub_list = []
        
        if data_list[k] != cn:
            count += 1

        if flag_st ==1 and data_list[k] != cn:
            sub_list.append(data_list[k])
            # print('sub_list:', sub_list)
    
    out_list2 = []
    for k, val in enumerate(out_list):
        if len(val) in [3,4]:
            buff = val[0].strip() + '@' + val[2].strip()
            out_list2.append(buff)
    
    out_list2 = out_list2[2:-10] 
    read_and_write('data11.txt', out_list2)
    return out_list2

def data12():
    
    fn = 'data12.html'
    data_list = []
    for k, line in enumerate(fileinput.input(fn)):
        if len(line) > 2:
            
            cl = line.strip()
            buff = cl.split(':')
            buff = buff[0] + '@' + buff[1]
            buff = buff.strip()
            data_list.append(buff)

    read_and_write('data12.txt', data_list)
    return data_list


def data14():
    
    fn = 'data14.html'
    str_list = []
    for k, line in enumerate(fileinput.input(fn)):
        str_list.append(line)
    
    big_str = ''.join(str_list)
    br_list = big_str.split('<br />')

    data_list = []
    for k, val in enumerate(br_list):
        val = val.replace('\n','')
        val = val.replace(': ','@')
        val = val.replace(':','@')
        if '@' in val and not '<' in val:
            spl = val.split('@')
            des = spl[1].split('.')
            desf = des[0]
            val = spl[0] + '@'+ desf
            val = val.strip()
            data_list.append(val)
            
    read_and_write('data14.txt', data_list)
    return data_list

def data15():
    
    fn = 'data15.html'
    data_list = []
    for k, line in enumerate(fileinput.input(fn)):
        if len(line) > 2:
            
            cl = line.strip()
            buff = cl.split(':')
            buff = buff[0] + '@' + buff[1].strip()
            data_list.append(buff)

    read_and_write('data15.txt', data_list)
    return data_list

def data16():

    fn = 'data16.html'
    markup = open(fn)
    soup = BeautifulSoup(markup.read(), 'html.parser')
    markup.close()
    f = open("data16_text.tmp", "w")
    f.write(soup.get_text())
    f.close()
    
    fn = 'data16_text.tmp'
    data_list = []
    for k, line in enumerate(fileinput.input(fn)):
        data_list.append(line)
    flag_st = 0
    flag_end = 0
    
    # print(data_list) 
    out_list = []
    sub_list = []
    count = 0
    for k in range(1, len(data_list)):
        cn = '\n'        
        if '@' in list(line):
            break
        
        elif data_list[k-1] == cn and data_list[k] != cn:
            flag_st = 1

        elif data_list[k-1] != cn and data_list[k] != cn:
            flag_st = 1

        elif data_list[k-1] != cn and data_list[k] == cn:
            flag_st = 0
            out_list.append(sub_list)
            count = 0
            sub_list = []
        
        if data_list[k] != cn:
            count += 1

        if flag_st ==1 and data_list[k] != cn:
            sub_list.append(data_list[k])
    
    out_list2 = []
    for k, val in enumerate(out_list):
        if len(val)==2:
            buff = val[0].strip() + '@' + val[1].strip()
            out_list2.append(buff)
    
    out_list2 = out_list2[4:-15] 

    fn = 'data16_text2.tmp'
    c_bag = []
    for k, line in enumerate(fileinput.input(fn)):
        if '-' in line: 
            
            spl = line.split('-')
            des = spl[1].split('.')[0]
            comm = spl[0].strip() 
            buff = comm + '@' + des.strip()
            if comm not in c_bag and ' ' not in comm:
                c_bag.append(comm)
                out_list2.append(buff)
       
    out_list2 = out_list2[:-2]
    read_and_write('data16.txt', out_list2)
    return out_list2

def data17():
    fn = 'data17.html'
    data_list = []
    for k, line in enumerate(fileinput.input(fn)):
        if len(line) > 2:
            spl = line.split(' ')
            des = ' '.join(spl[1:])
            des = des.strip()
            comm = spl[0].strip() 
            buff = comm + '@' + des
            data_list.append(buff) 
    read_and_write('data17.txt', data_list)
            
    return data_list

def dict_add(dic, key, string):
    if key not in dic:
        dic[key] = [string]
    elif key in dic:
        if string not in dic[key]:
            dic[key].append(string)



def cal_set(total_data):
    
    com2des_dict = {}
    comms_list = []
    for line in total_data:
        line = line.lower()
        spl = line.split('@')
        comms_list.append(spl[0])
        dict_add(com2des_dict, spl[0], spl[1].strip('.')) 
    comms_set = set(comms_list)
     
    print('Commands:', comms_set)
    print('The length of the set:', len(list(comms_set)) )


    return com2des_dict
def cmdline(command):
    process = Popen(
        args=command,
        stdout=PIPE,
        shell=True
    )
    return process.communicate()[0]

def asc2char(asc):
    return ''.join(map(chr, out_sys))

def save_json(json_fn, dic):
    with open(json_fn, 'w') as outfile:
        json.dump(dic, outfile, indent=4)

def get_whatis(in_list):
    # fn = 'data1.txt'

    data_list = []
    # for k, line in enumerate(fileinput.input(fn)):
    for k, line in enumerate(in_list):
        spl = line.split('@')
        out_sys = cmdline('whatis -l ' + spl[0])
        strout = ''.join(map(chr, out_sys))
        print('sys message:', strout)
        splso = strout.split(' ')
        spldash = strout.split(' - ')
        if spl[0] == splso[0]:
            data_list.append(spl[0] + '@' + spldash[-1].strip().strip('.'))
      
    print('The length of the get_whatis:', len(data_list))
    read_and_write('data_whatis.txt', data_list)
    return data_list
if __name__ == '__main__':
    
    total_data = []
    total_data.extend(data1())
    total_data.extend(data3())
    total_data.extend(data9())
    total_data.extend(data10())
    total_data.extend(data11())
    total_data.extend(data12())
    total_data.extend(data14())
    total_data.extend(data15())
    total_data.extend(data16())
    total_data.extend(data17())
    total_data.extend(get_whatis(total_data))
    read_and_write('../data/com2des_TJ.txt', total_data)
    c2d_dict = cal_set(total_data)
    save_json('../data/com2des_TJ.json', c2d_dict)

    count_c2d_dict = {}
    count_list  = [] 
    for k, val in c2d_dict.items():
        count_c2d_dict['k'] = len(val)
        count_list.append(len(val))


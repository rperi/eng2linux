from subprocess import PIPE, Popen
import subprocess
import fileinput
import numpy as np 
import os 

def cmdline(command):
    process = Popen(
        args=command,
        stdout=PIPE,
        shell=True
    )
    return process.communicate()[0]

def asc2char(asc):
    return ''.join(map(chr, out_sys))
if __name__ == '__main__':

    fn = 'data1.txt'
    fn = 'data1.txt'

    data_list = []
    for k, line in enumerate(fileinput.input(fn)):
        spl = line.split('@')
        out_sys = cmdline('whatis ' + spl[0])
        strout = ''.join(map(chr, out_sys))
        print('sys message:', strout)
        splso = strout.split(' ')
        spldash = strout.split(' - ')
        if spl[0] == splso[0]:
            data_list.append(spldash[-1])

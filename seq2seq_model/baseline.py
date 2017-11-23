import numpy as np
from data_prep import *
from nltk import metrics, stem, tokenize
from nltk.metrics import *
import fileinput 
import ipdb
def read_and_store(list_path):
    with open(list_path) as f:
        content = []
        for line in f:
            line = line.strip()
            content.append(line)
    return content

def edit_dist_estimator(description, train_data_list):
    comm_list = [ line.split('@')[0] for k, line in enumerate(train_data_list) ]
    des_list = [ line.split('@')[1] for k, line in enumerate(train_data_list) ]

    
    print(comm_list[:10])
    
    distances_list = [[],] * len(description)
    dist_array = np.zeros((len(des_list), len(description)))
    for ct, des_train in enumerate(des_list):
        buff = [] 
        print('Description Train Set:', ct, '/ ', len(des_list), flush=True)
        for k, val in enumerate(description):
            buff.append(edit_distance(val, des_train))

        dist_array[ct, :] = buff 
        # print(distances_list)
   
    est_comm = []
    for kd, dist_vals in enumerate(description):
        print('Estimating: ' , dist_vals)
        estimated_command = comm_list[np.argmin(dist_array[:, kd])]
        # ipdb.set_trace()
         
        est_comm.append(comm_list[np.argmin(dist_array[:, kd])])
    return est_comm

def eval(est_comm_list, des_list_test, comm_list_test):
    
    grade_list = []
    for k, val in enumerate(comm_list_test):
        grade_list.append(val == est_comm_list[k])
        print('description:', des_list_test[k])
        print('EST:', est_comm_list[k], 'ACTUAL:', val, '\n')
    score = sum(grade_list)/len(comm_list_test)
    print('Score:', score)
    return grade_list, score

if __name__ == '__main__':
    train_data_list = read_and_store('../data/com-eng_train.txt') 
    test_data_list = read_and_store('../data/com-eng_test.txt') 
   
    des_list_test = [ line.split('@')[1] for k, line in enumerate(test_data_list) ]
    comm_list_test = [ line.split('@')[0] for k, line in enumerate(test_data_list) ]

    est_comm_list = edit_dist_estimator(des_list_test, train_data_list)

    grade_list, score = eval(est_comm_list, des_list_test, comm_list_test)
    

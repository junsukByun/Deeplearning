# -*- coding: utf-8 -*-
print('Read_data.py_loaded')
import glob
from keras.utils import np_utils, plot_model
import random

def read_datalist(train_path, test_path):
    
    train_test = ['train', 'test']
    ok_ng = ['Nonkiller','Killer']

    data_list = {}
    
    if train_path != test_path:
    
        if train_path.split('/')[-1][-3:] == 'txt':
            with open(train_path, 'r') as f1:
                data = f1.readlines()
            data_train_kl = []
            data_train_nk = []
            for i in data:
                if i.split(' ')[1][0] == '1':
                    data_train_nk.append(i.split(' ')[0])
                else:
                    data_train_kl.append(i.split(' ')[0])
            data_list['train', 'Killer'] = data_train_kl        
            data_list['train', 'Nonkiller'] = data_train_nk
        else:
            data_list['train', 'Killer']    = glob.glob(train_path + '/Killer/*99.JPG') + glob.glob(train_path + '/killer/*99.JPG')
            data_list['train', 'Nonkiller'] = (glob.glob(train_path + '/NonKiller/*99.JPG') + glob.glob(train_path + '/Nonkiller/*99.JPG') + glob.glob(train_path + '/nonKiller/*99.JPG') + glob.glob(train_path + '/nonkiller/*99.JPG'))

        if test_path.split('/')[-1][-3:] == 'txt':
            with open(test_path, 'r') as f2:
                data2 = f2.readlines()
            data_test_kl = []
            data_test_nk = []
            for i in data2:
                if i.split(' ')[1][0] == '1':
                    data_test_nk.append(i.split(' ')[0])
                else:
                    data_test_kl.append(i.split(' ')[0])
            data_list['test', 'Killer'] = data_test_kl        
            data_list['test', 'Nonkiller'] = data_test_nk    
        else:
            data_list['test', 'Killer']     = glob.glob(test_path + '/Killer/*99.JPG') + glob.glob(test_path + '/killer/*99.JPG')
            data_list['test', 'Nonkiller']  = (glob.glob(test_path + '/NonKiller/*99.JPG') + glob.glob(test_path + '/Nonkiller/*99.JPG') + glob.glob(train_path + '/nonKiller/*99.JPG') + glob.glob(test_path + '/nonkiller/*99.JPG'))
    
    elif train_path == test_path:
        print("Let's separate train_data to train_data:valid_data = 7:3")    
        if train_path.split('/')[-1][-3:] == 'txt':
            with open(train_path, 'r') as f1:
                data = f1.readlines()
            data_train_kl = []
            data_train_nk = []
            for i in data:
                if i.split(' ')[1][0] == '1':
                    data_train_nk.append(i.split(' ')[0])
                else:
                    data_train_kl.append(i.split(' ')[0])
            random.shuffle(data_train_nk)
            random.shuffle(data_train_kl)
            len_kl = len(data_train_kl)
            len_nk = len(data_train_nk)
            len_kl_train = int(len_kl * 0.7)
            len_nk_train = int(len_nk * 0.7)
            data_list['train', 'Killer'] = data_train_kl[0:len_kl_train]        
            data_list['train', 'Nonkiller'] = data_train_nk[0:len_nk_train]
            data_list['test', 'Killer'] = data_train_kl[len_kl_train:]        
            data_list['test', 'Nonkiller'] = data_train_nk[len_nk_train:]
        else:
            data_train_kl = glob.glob(train_path + '/Killer/*99.JPG') + glob.glob(train_path + '/killer/*99.JPG')
            data_train_nk = (glob.glob(train_path + '/NonKiller/*99.JPG') + glob.glob(train_path + '/Nonkiller/*99.JPG') + glob.glob(train_path + '/nonKiller/*99.JPG') + glob.glob(train_path + '/nonkiller/*99.JPG'))
            random.shuffle(data_train_nk)
            random.shuffle(data_train_kl)
            len_kl = len(data_train_kl)
            len_nk = len(data_train_nk)
            len_kl_train = int(len_kl * 0.7)
            len_nk_train = int(len_nk * 0.7)
            data_list['train', 'Killer'] = data_train_kl[0:len_kl_train]        
            data_list['train', 'Nonkiller'] = data_train_nk[0:len_nk_train]
            data_list['test', 'Killer'] = data_train_kl[len_kl_train:]        
            data_list['test', 'Nonkiller'] = data_train_nk[len_nk_train:]

    n_train = len(data_list[('train', 'Killer')]) + len(data_list[('train', 'Nonkiller')])
    n_test = len(data_list[('test', 'Killer')]) + len(data_list[('test', 'Nonkiller')])

    # length 확인
    for i in train_test:
        for j in ok_ng:
            print('%s %s %d' % (i, j, len(data_list[i,j])))
            
    return n_train, n_test, data_list

def data_processor(data_list):
    X_train = data_list[('train', 'Nonkiller')]+ data_list[('train', 'Killer')]
    Y_train = [1] * len(data_list[('train','Nonkiller')]) + [0] * len(data_list[('train','Killer')]) 
    X_test =  data_list[('test', 'Nonkiller')] + data_list[('test', 'Killer')]
    Y_test = [1] * len(data_list[('test','Nonkiller')]) + [0] * len(data_list[('test','Killer')]) 

    Y_train = np_utils.to_categorical(Y_train, 2)
    Y_test = np_utils.to_categorical(Y_test, 2)
    
    return X_train, Y_train, X_test, Y_test
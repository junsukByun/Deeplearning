# -*- coding: utf-8 -*-
from Helper.Library import *
from keras.utils import np_utils, plot_model
from PIL import Image
import numpy as np
np.random.seed(1111)
import glob
import time
from tqdm import trange, tqdm_notebook
import sys 
from tqdm import tqdm




# def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100): 
#     formatStr = "{0:." + str(decimals) + "f}" 
#     percent = formatStr.format(100 * (iteration / float(total))) 
#     filledLength = int(round(barLength * iteration / float(total))) 
#     bar = '#' * filledLength + '-' * (barLength - filledLength) 
#     sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)), 
#     if iteration == total: 
#         sys.stdout.write('\n') 
#     sys.stdout.flush()

def Data_preprocessing(test_path):
    test_list = {}
    if test_path.split('/')[-1][-3:] == 'txt':
        print('inference file is txt.file')
        with open(test_path, 'r') as f1:
            data = f1.readlines()
        data_train_kl = []
        data_train_nk = []
        for i in data:
            if i.split(' ')[1][0] == '1':
                data_train_nk.append(i.split(' ')[0])
            else:
                data_train_kl.append(i.split(' ')[0])
        test_list['Killer'] = data_train_kl        
        test_list['Nonkiller'] = data_train_nk
    else:
        test_list['Killer'] = glob.glob(test_path + '/Killer/*.JPG') + glob.glob(test_path + '/killer/*.JPG')
        test_list['Nonkiller']  = (glob.glob(test_path + '/NonKiller/*.JPG') + glob.glob(test_path + '/Nonkiller/*.JPG') + 
        glob.glob(test_path + '/nonKiller/*.JPG') + glob.glob(test_path + '/nonkiller/*.JPG'))

    for j in ['Killer', 'Nonkiller']:
        print('The number of images for %s:  %d' % (j, len(test_list[j])))
        
    original_kl = list(filter(lambda x: x.split('.')[-2][-2:]=='99', test_list['Killer']))    
    original_nk = list(filter(lambda x: x.split('.')[-2][-2:]=='99', test_list['Nonkiller'])) 
    
    print('The number of original images for %s:  %d' % ('Killer', len(original_kl)))
    print('The number of original images for %s:  %d' % ('Nonkiller', len(original_nk)))       
    print('Image, label loading started')
          
          
    X_test = np.zeros(((len(test_list['Nonkiller']) +len(test_list['Killer'])), 256, 256, 3)) 
    y_test = np.zeros((len(test_list['Nonkiller']) +len(test_list['Killer'])))
    n = 0    
    a = time.time()
    for tt in ['Nonkiller', 'Killer']:
        for i in tqdm(range(len(test_list[tt]))):
            X_test[n, :, :, :] = Image.open(test_list[tt][i])
            if tt == 'Nonkiller':
                y_test[n] = 1.0
            else:
                y_test[n] = 0.0
            n += 1
            time.sleep(0.000000000001)
    print( 'Image processing time : {} '.format(time.time() -a))
    print('normalization of x_test start')
    norm_start_time = time.time()
    for i in tqdm(range(len(X_test))):
        X_test[i] = normalization(X_test[i])
        time.sleep(0.00000000000001)
    print('normalization of x_test finished')
    print('normalization time : {} '.format(time.time() - norm_start_time))
    
    Y_test = np_utils.to_categorical(y_test, 2)
    
    print('Image, label loading finished')

    return test_list, X_test, Y_test

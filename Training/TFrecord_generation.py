# -*- coding: utf-8 -*-
print('TFrecord_generation.py_loaded')
import tensorflow as tf
from Helper.Library import _bytes_feature, to_float
from Training.Read_train_data import *
from PIL import Image
import numpy as np
np.random.seed(1111)
import sys
import time
import numpy as np

def TFrecord_generation(data_list, tfrecord_filename, file_read_mode):
    start_time = time.time()
    X_train, Y_train, X_test, Y_test = data_processor(data_list)
    
    train_kl = Y_train.tolist().count([1,0])
    train_nk = Y_train.tolist().count([0,1])
    valid_kl = Y_test.tolist().count([1,0])
    valid_nk = Y_test.tolist().count([0,1])
    
    Y_train = to_float(Y_train)
    Y_test = to_float(Y_test)

    for file_type in ['train', 'test']:   
        
        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
        writer = tf.python_io.TFRecordWriter('./TFrecords/'+ tfrecord_filename + '_' + file_type, options=options)
                
        if file_type == 'train':            
            zip_list = list(zip(X_train, Y_train))
            
        else:
            zip_list = list(zip(X_test, Y_test))
                     
        i = 1
        for image_name, label_name in zip_list:            

#             if not i % 1000 or i == (len(zip_list)):

#                 print ('{} Data: {}/{}'.format(file_type, i, len(zip_list)))
#                 sys.stdout.flush()

            image = Image.open(image_name)
            image = np.array(image)
            image = image.astype('float32')
            image = image.tostring()     
            label = label_name.tostring()

            feature = {'label': _bytes_feature(label),
                       'image': _bytes_feature(image)}

            example = tf.train.Example(features=tf.train.Features(feature=feature))

           
            writer.write(example.SerializeToString())

            i += 1

        writer.close()
        sys.stdout.flush()
    with open('./TFrecords/' + tfrecord_filename + '_information.txt', 'w') as f:
        f.writelines(' number of train_killer:{}\n number of train_nonkiller:{}\n number of valid_killer:{}\n number of valid_nonkiller:{}\n  '.format(train_kl, train_nk, valid_kl, valid_nk))
        generation_time = time.time() - start_time
        f.writelines('TFrecord_generation time : {0} seconds \n'.format(generation_time))
    print('TFrecord_generation time: {0} seconds '.format(generation_time))
    return False
    
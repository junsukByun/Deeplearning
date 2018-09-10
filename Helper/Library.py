# -*- coding: utf-8 -*-
import tensorflow as tf
import os
print('Library.py_loaded')

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def safe_mkdir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass

def to_float(data):
    data = data.astype('float32')
    return data

def normalization(data):
    data = data / 255.0 - 0.5
    return data

def image_name_parser(test_list):
    total_filename_list = test_list['Nonkiller'] + test_list['Killer']
    total_filename_list = [filename.split('/')[-1] for filename in total_filename_list]
    
    return total_filename_list
## multicrop로 inference 하기 위해서 원장의 이름을 불러온다.
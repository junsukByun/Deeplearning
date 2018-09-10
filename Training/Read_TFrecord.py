# -*- coding: utf-8 -*-
import tensorflow as tf
from Training.TFrecord_generation import *
from Helper.Augmentation import *
from Helper.Library import *
import sys 


def read_from_tfrecord(example_proto, mode):

    tfrecord_features = tf.parse_single_example(example_proto,
                        features={
                            'label': tf.FixedLenFeature([], tf.string),
                            'image': tf.FixedLenFeature([], tf.string),
                        }, name='features')
    
    
    image = tf.decode_raw(tfrecord_features['image'], tf.float32)
    image = tf.reshape(image, [256,256,3])
    if mode == 'train':
        image = augmentation(image, 
                    prob_filp_up_down = 0.5,
                    prob_filp_left_right = 0.5,
                    prob_brightness = 0.5,
                    prob_random_noise = 0.5,
                    prob_contrast = 0.5,
                    prob_shift = 0.5,
                    prob_shear = 0.5,
                    prob_zoom = 0.5,
                    prob_rot = 0.5 )
        image = normalization(image)
    else:
        image = normalization(image)
    
    label = tf.decode_raw(tfrecord_features['label'], tf.float32)
    label = tf.reshape(label, [2])
    
    return label, image


def read_train_and_test(tf_filenm):

    train_dataset = tf.data.TFRecordDataset('./TFrecords/'+tf_filenm + '_train', compression_type='ZLIB')
    train_dataset = train_dataset.map(lambda x: read_from_tfrecord(x, mode = 'train'))
    
    test_dataset = tf.data.TFRecordDataset('./TFrecords/'+tf_filenm + '_test', compression_type='ZLIB')                       # test dataset은 nor만 수행
    test_dataset = test_dataset.map(lambda x: read_from_tfrecord(x, mode = 'test'))

    return train_dataset, test_dataset

def make_dataset(dataset, tf_filenm, file_read_mode):
    print('Generation of TFrecord starts. Please wait for a while. If data is big, it will take long.')
    TFrecord_generation(dataset, tf_filenm, file_read_mode)
        



        
print('Network.py_loaded')
from Helper.Augmentation import *
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.contrib.slim as slim
slim = tf.contrib.slim
from tensorflow.contrib import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from keras.models import Sequential, Model, load_model, Input
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, GaussianNoise
from keras.layers import Conv2D, MaxPooling2D, Input, Lambda, concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D

def conv_relu(inputs, filters, k_size, stride, padding, scope_name):
 
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_channels = inputs.shape[-1]
        kernel = tf.get_variable('kernel', 
                                [k_size, k_size, in_channels, filters], 
                                initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable('biases', 
                                [filters],
                                initializer=tf.random_normal_initializer())
        conv = tf.nn.conv2d(inputs, kernel, strides=[1, stride, stride, 1], padding=padding)
    return tf.nn.relu(conv + biases, name=scope.name)

def maxpool(inputs, ksize, stride, padding='VALID', scope_name='pool'):

    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        pool = tf.nn.max_pool(inputs, 
                            ksize=[1, ksize, ksize, 1], 
                            strides=[1, stride, stride, 1],
                            padding=padding)
    return pool


def inception_resnet_v2(inputs, is_training):
    
    m1 = hub.Module("https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/1", tags={"train"}, trainable=True)
    height, width = hub.get_expected_image_size(m1)
    resize_inputs = tf.image.resize_images(inputs,[height,width])
    net = resize_inputs
    net = m1(net)
    
    return net

def inception_v1(inputs, is_training):
    
    m2 = hub.Module("https://tfhub.dev/google/imagenet/inception_v1/classification/1", tags={"train"}, trainable=True)
    height, width = hub.get_expected_image_size(m2)
    resize_inputs = tf.image.resize_images(inputs,[height,width])
    net = resize_inputs
    net = m2(net)
    
    return net

def inception_v2(inputs, is_training):
    
    m3 = hub.Module("https://tfhub.dev/google/imagenet/inception_v2/classification/1", tags={"train"}, trainable=True)
    height, width = hub.get_expected_image_size(m3)
    resize_inputs = tf.image.resize_images(inputs,[height,width])
    net = resize_inputs
    net = m3(net)
    
    return net

def inception_v3(inputs, is_training):
    
    m4 = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/classification/1", tags={"train"}, trainable=True)
    height, width = hub.get_expected_image_size(m4)
    resize_inputs = tf.image.resize_images(inputs,[height,width])
    net = resize_inputs
    net = m4(net)
    
    return net

def fully_connected(inputs, out_dim, scope_name='fc'):

    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_dim = inputs.shape[-1]
        w = tf.get_variable('weights', [in_dim, out_dim],
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer(0.0))
        out = tf.matmul(inputs, w) + b
    return out

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

def fully_connected(inputs, out_dim, scope_name='fc'):

    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_dim = inputs.shape[-1]
        w = tf.get_variable('weights', [in_dim, out_dim],
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer(0.0))
        out = tf.matmul(inputs, w) + b
    return out

def simplenet(inputs, equip_no, keep_prob, n_classes):
    
    conv1 = conv_relu(inputs=inputs,
                    filters=32,
                    k_size=3,
                    stride=1,
                    padding='SAME',
                    scope_name='conv1')
    pool1 = maxpool(conv1, 2, 2, 'VALID', 'pool1')
    conv2 = conv_relu(inputs=pool1,
                    filters=64,
                    k_size=3,
                    stride=2,
                    padding='SAME',
                    scope_name='conv2')
    pool2 = maxpool(conv2, 2, 2, 'VALID', 'pool2')
    feature_dim = pool2.shape[1] * pool2.shape[2] * pool2.shape[3]
    pool2 = tf.reshape(pool2, [-1, feature_dim])
    fc = fully_connected(pool2, 1024, 'fc')
    dropout = tf.nn.dropout(tf.nn.relu(fc), keep_prob, name='relu_dropout')
    fc_2 = tf.concat([dropout, equip_nos], axis=1)
    logits = fully_connected(fc_2, n_classes, 'logits')     

    return logits

def vgg_19(inputs,
           equip_no,
           num_classes=2,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=False,
           scope='vgg_19',
           fc_conv_padding='SAME',
           global_pool=False):
  
  with tf.variable_scope(scope, 'vgg_19', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.flatten(inputs=net, scope='flatten')
        net = tf.concat([net, equip_no], axis=1)
        net = slim.fully_connected(inputs=net, num_outputs=2, scope='fc8')
                
    return net
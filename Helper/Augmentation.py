# -*- coding: utf-8 -*-
print('Augmentation.py_loaded')
# from TFrecord_generation import *
import tensorflow as tf

################# Augmentation ###########################################
def _get_image_depth(x):
    return x.get_shape()[2].value

def _get_image_size(x):
    return x.get_shape()[1].value


def random_rot(image,lower, upper):
    angle = tf.random_uniform([1], np.deg2rad(lower), np.deg2rad(upper))
    image = tf.contrib.image.rotate(image, angle)
     
    return image

def random_shift(left_right, up_down):
    shift_x = tf.random_uniform([1], -1 * left_right, left_right)
    shift_y = tf.random_uniform([1], -1 * up_down, up_down)
 
    shift_row1 = tf.concat([tf.ones([1]), tf.zeros([1]), shift_x], axis=0)
    shift_row2 = tf.concat([tf.zeros([1]), tf.ones([1]), shift_y], axis=0)
    shift_row3 = tf.concat([tf.zeros([2]), tf.ones([1])], axis=0)
    shift_matrix = tf.stack([shift_row1, shift_row2, shift_row3])
 
    return shift_matrix

def random_shear(lower=-5.0, upper=5.0):
    shear_angle = tf.random_uniform([1], np.deg2rad(lower), np.deg2rad(upper))
 
    shear_row1 = tf.concat([tf.ones([1]), tf.negative(tf.sin(shear_angle)), tf.zeros([1])], axis=0)
    shear_row2 = tf.concat([tf.zeros([1]), tf.cos(shear_angle), tf.zeros([1])], axis=0)
    shear_row3 = tf.concat([tf.zeros([2]), tf.ones([1])], axis=0)
    shear_matrix = tf.stack([shear_row1, shear_row2, shear_row3])
 
    return shear_matrix

def random_zoom(lower=0.95,upper=1.05):
    zoom = tf.random_uniform([1], 1/upper, 1/lower)
 
    zoom_row1 = tf.concat([zoom, tf.zeros([2])], axis=0)
    zoom_row2 = tf.concat([tf.zeros([1]), zoom, tf.zeros([1])], axis=0)
    zoom_row3 = tf.concat([tf.zeros([2]), tf.ones([1])], axis=0)
    zoom_matrix = tf.stack([zoom_row1, zoom_row2, zoom_row3])
 
    return zoom_matrix

def random_affine(
    image,
    prob_shift,
    prob_shear,
    prob_zoom,
    prob_rot,
    shift_left_right,
    shift_up_down,
    shear_lower,
    shear_upper,
    zoom_lower,
    zoom_upper,
    rot_lower,
    rot_upper,
    padding ):

    img_size = _get_image_size(image)
    pad_size = int(img_size/4)

    if (padding == True):
        image = tf.pad(image, [[pad_size, pad_size], [pad_size, pad_size], [0, 0]], "REFLECT")

    random_value = tf.random_uniform([4], 0, 1) 

    transform_matrix = tf.cond(random_value[0] < prob_shift,
                               lambda: random_shift(shift_left_right, shift_up_down),
                               lambda: tf.eye(3))
    transform_matrix = tf.cond(random_value[1] < prob_shear,
                               lambda: tf.matmul(transform_matrix, random_shear(shear_lower, shear_upper)),
                               lambda: transform_matrix)
    transform_matrix = tf.cond(random_value[2] < prob_zoom,
                               lambda: tf.matmul(transform_matrix, random_zoom(zoom_lower, zoom_upper)),
                               lambda: transform_matrix)

    image = tf.cond((random_value[0] < prob_shift) | (random_value[1] < prob_shear) | (random_value[2] < prob_zoom),
                    lambda: tf.contrib.image.transform(
                            image,
                            tf.gather_nd(transform_matrix, [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1]])),
                    lambda: image)

    image = tf.cond(random_value[3] < prob_rot,
                    lambda: random_rot(image,rot_lower,rot_upper),
                    lambda: image)

    if (padding == True):
        image = tf.slice(image, [pad_size, pad_size, 0], [img_size, img_size, _get_image_depth(image)])   

    return image

    
def augmentation(image, 
                prob_filp_up_down,
                prob_filp_left_right,
                prob_brightness,
                prob_random_noise,
                prob_contrast,
                prob_shift,
                prob_shear,
                prob_zoom,
                prob_rot):
    
    # Flip up down
    random_value = tf.random_uniform([1], 0, 1)
    image = tf.cond(random_value[0] < prob_filp_up_down,
                    lambda: tf.image.random_flip_up_down(image),
                    lambda: image)
    
    # Flip left right
    random_value = tf.random_uniform([1], 0, 1)
    image = tf.cond(random_value[0] < prob_filp_left_right,
                lambda: tf.image.random_flip_left_right(image),
                lambda: image)
    
    # Noise input
    random_value = tf.random_uniform([1], 0, 1)    
    noise = tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=1.0, dtype=tf.float32)
    image = tf.cond(random_value[0] < prob_random_noise,
                    lambda: tf.add(image, noise),
                    lambda: image)      

    # Brightness
    random_value = tf.random_uniform([1], 0, 1) 
    image = tf.cond(random_value[0] < prob_brightness,
                    lambda: tf.image.random_brightness(image, max_delta = 1),
                    lambda: image)        
    
    # Contrast
    random_value = tf.random_uniform([1], 0, 1) 
    image = tf.cond(random_value[0] < prob_contrast,
                    lambda: tf.image.random_contrast(image, lower = 0.9, upper = 1.1),
                    lambda: image)       
    
    # Affine transformation
#     image = random_affine(
#             image,
#             prob_shift= prob_shift,
#             prob_shear= prob_shear,
#             prob_zoom= prob_zoom,
#             prob_rot= prob_rot,
#             shift_left_right=10.0,
#             shift_up_down=10.0,
#             shear_lower=-5.0,
#             shear_upper=5.0,
#             zoom_lower=0.95,
#             zoom_upper=1.05,
#             rot_lower=-3,
#             rot_upper=3,
#             padding=False)

    return image
import tensorflow as tf
import numpy as np
import os, sys

FLAGS = tf.app.flags.FLAGS

def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def _glorot_initializer(prev_units, num_units, stddev_factor=1.0):
    stddev  = np.sqrt(stddev_factor / np.sqrt(prev_units*num_units))
    return tf.truncated_normal([prev_units, num_units], mean=0.0, stddev=stddev)

def _glorot_initializer_conv2d(prev_units, num_units, mapsize, stddev_factor=1.0):
    stddev  = np.sqrt(stddev_factor / (np.sqrt(prev_units*num_units)*mapsize*mapsize))
    return tf.truncated_normal([mapsize, mapsize, prev_units, num_units],
                                    mean=0.0, stddev=stddev)
                                    
def weight_variable(shape, stddev=0.02, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def leaky_relu(x, alpha=0.2, name=""):
    return tf.maximum(alpha * x, x, name)
    
def conv2d_strided(x, W, b):
    conv = tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)
                    
def upscale(x):
    old_size = x.get_shape()
    size = [2 * int(s) for s in old_size[1:3]]
    out  = tf.image.resize_nearest_neighbor(x, size)
    return out
    
def conv2d_transpose(x, num_units, mapsize=1, stride=1, stddev_factor=1.0):
    assert len(x.get_shape()) == 4 and "Previous layer must be 4-dimensional"

    with tf.variable_scope("conv2d_transpose"):
        prev_units = int(x.get_shape()[-1])
                
        initw  = _glorot_initializer_conv2d(prev_units, num_units,
                                                 mapsize,
                                                 stddev_factor=stddev_factor)
        weight = tf.get_variable('weight', initializer=initw)
        weight = tf.transpose(weight, perm=[0, 1, 3, 2])
        prev_output = x
        output_shape = [FLAGS.batch_size, #maine flag olarak batch size koy
                        int(prev_output.get_shape()[1]) * stride,
                        int(prev_output.get_shape()[2]) * stride,
                        num_units]
        out    = tf.nn.conv2d_transpose(x, weight,
                                        output_shape=output_shape,
                                        strides=[1, stride, stride, 1],
                                        padding='SAME')

        # Bias term
        initb  = tf.constant(0.0, shape=[num_units])
        bias   = tf.get_variable('bias', initializer=initb)
        out    = tf.nn.bias_add(out, bias)        
    return out
  

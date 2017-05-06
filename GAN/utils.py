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
            
def weight_variable(shape, stddev=0.02, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def leaky_relu(x, alpha=0.2, name=""):
    return tf.maximum(alpha * x, x, name)                

def add_activation_summary(var):
    tf.summary.histogram(var.op.name + "/activation", var)
    tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))

def add_gradient_summary(grad, var):
    if grad is not None:
        tf.summary.histogram(var.op.name + "/gradient", grad)     

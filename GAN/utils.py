import tensorflow as tf
import numpy as np

#bias initialization
def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

#xavier init
def _glorot_initializer(prev_units, num_units, stddev_factor=1.0):
    stddev  = np.sqrt(stddev_factor / np.sqrt(prev_units*num_units))
    return tf.truncated_normal([prev_units, num_units], mean=0.0, stddev=stddev)

#weight initialization            
def weight_variable(shape, stddev=0.02, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

#leaky relu implementation
def leaky_relu(x, alpha=0.2, name=""):
    return tf.maximum(alpha * x, x, name)                

#stores activations
def add_activation_summary(var):
    tf.summary.histogram(var.op.name + "/activation", var)
    tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))

#combines patches to get the whole image
def block2img(img_blocks,img_size):    
    row,col = img_size
    img=np.zeros((row,col,3)).astype(np.float32)
    n,k,l,c=img_blocks.shape                 
    for i in range(0,int(row/k)):
        for j in range(0,int(col/k)):
            img[i*k:(i+1)*k,j*l:(j+1)*l,:]=img_blocks[int(i*col/k+j),:,:,:]
    return img
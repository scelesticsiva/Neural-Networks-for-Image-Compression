from __future__ import print_function

__author__ = "kubra"
"""
Tensorflow implementation of Wasserstein GAN
"""
import numpy as np
import tensorflow as tf
import models
from models import *

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "64", "batch size for training")
tf.flags.DEFINE_float("learning_rate", "2e-4", "learning rate for adam")
tf.flags.DEFINE_float("optimizer_param", "0.5", "beta1 for adam-decay for RMSProp")
tf.flags.DEFINE_float("iterations", "10", "training iterations")
tf.flags.DEFINE_string("optimizer", "Adam", "Optimizer")
#add real and downsampled image sizes

def main(argv=None):
    generator_dims = [128, 96]
    discriminator_dims = [3, 64, 1]

    print("asama 1")
    model = models.WGAN('train', FLAGS.batch_size,
                               clip_values=(-0.01, 0.01), critic_iterations=5, num_data=1280)
    print("asama 2")
    model.create_model(generator_dims, discriminator_dims, FLAGS.optimizer, FLAGS.learning_rate,
                         FLAGS.optimizer_param)

    print("asama 3")
    model.train_model(FLAGS.batch_size, int(1 + FLAGS.iterations))

if __name__ == "__main__":
    tf.app.run()

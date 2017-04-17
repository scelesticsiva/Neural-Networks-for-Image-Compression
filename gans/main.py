from __future__ import print_function

import numpy as np
import tensorflow as tf
import models
from models import *

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "64", "batch size for training")
tf.flags.DEFINE_float("learning_rate", "2e-4", "learning rate for adam")
tf.flags.DEFINE_float("optimizer_param", "0.5", "beta1 for adam-decay for RMSProp")
tf.flags.DEFINE_float("iterations", "100000", "training iterations")
tf.flags.DEFINE_string("optimizer", "RMSProp", "Optimizer")
tf.flags.DEFINE_string("loss_type", "wasserstein", "wasserstein/imp_wasserstein")
#add real and downsampled image sizes

def main(argv=None):
    generator_dims = [128, 3]
    discriminator_dims = [3, 64, 1]

    print("stage 1")
    model = models.WGAN('train', FLAGS.batch_size,
                         clip_values=(-0.01, 0.01), disc_iterations=5, num_data=1280)
    print("stage 2")
    model.create_model(generator_dims, discriminator_dims, FLAGS.optimizer, FLAGS.learning_rate,
                         FLAGS.optimizer_param, FLAGS.loss_type)

    print("stage 3")
    model.train_model(FLAGS.batch_size, int(FLAGS.iterations))

if __name__ == "__main__":
    tf.app.run()

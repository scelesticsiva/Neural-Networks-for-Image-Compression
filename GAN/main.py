from __future__ import print_function

import numpy as np
import tensorflow as tf
import models
from models import *

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "64", "batch size for training")
tf.flags.DEFINE_float("learning_rate", "2e-5", "learning rate for optimizers")
tf.flags.DEFINE_float("optimizer_param", "0.5", "beta1 for adam-decay for RMSProp")
tf.flags.DEFINE_float("iterations", "500000", "training iterations")
tf.flags.DEFINE_string("optimizer", "RMSProp", "RMSProp/Adam")
tf.flags.DEFINE_string("loss_type", "wasserstein_l2_loss", "wasserstein/imp_wasserstein/wasserstein_l1_loss/wasserstein_l2_loss/imp_wasserstein_l2_loss")

def main(argv=None):
    discriminator_dims = [3, 16, 64, 1]
    kernel_encoder = [5,7,9]
    kernel_decoder = [9,7,5]
    encoder_dims = [64,16,3]
    decoder_dims = [16,32,3]

    print("stage 1")
    model = models.GAN_AE(FLAGS.batch_size,
                         clip_values=(-0.01, 0.01), disc_iterations=5, num_train_data=38400, num_test_data=6400, folder='wgan_l2')
    print("stage 2")
    model.create_model(discriminator_dims, kernel_encoder, kernel_decoder, encoder_dims, decoder_dims, "RMSProp", FLAGS.learning_rate,
                         FLAGS.optimizer_param, FLAGS.loss_type)

    print("stage 3")
    model.train_model(FLAGS.batch_size, int(FLAGS.iterations))

    discriminator_dims = [3, 16, 64, 1]

if __name__ == "__main__":
    tf.app.run()

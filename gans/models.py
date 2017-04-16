import tensorflow as tf
import numpy as np
import os, sys, inspect
import time

import utils as utils
import read_cifar10 as cf10


class WGAN(object):
        
    def __init__(self, data_name, batch_size, clip_values, critic_iterations, num_data):
        self.batch_size = batch_size
        self.input_images = self.cifar10_dataset(data_name,4, num_data) #tek fonk yap tek seferde yukle
        self.real_images = self.cifar10_dataset(data_name,1, num_data) #usttekiyle tek fonk yap birlestir
        self.critic_iterations = critic_iterations
        self.clip_values = clip_values
        self.z_dim = 8 #degistir flexible yap
        self.real_dim = 32
       
    def cifar10_dataset(self,name, downsample, total_num): 
        assert name in ['train', 'test']
        X, y = (cf10.load_training_data() if name == 'train'
                                 else cf10.load_test_data())    
        data = X[:total_num]
        return data[:,::downsample,::downsample,:]
        
    def cifar10_batches(self, name, downsample, batch_size, num, total_num=12800): #batch size -1 tum data olayini da yap
        assert name in ['train', 'test']
        assert total_num % batch_size == 0 or batch_size == -1 #otomatik handle yap sonra
        X0, y0 = (cf10.load_training_data() if name == 'train'
                                 else cf10.load_test_data())    
        #act_restrict_size = total_num if name == 'train' else int(1e10)
        X = X0[:total_num]
        y = y0[:total_num]
        #data_len = X.shape[0] 
        idx = num * batch_size
        X_batch = X[idx:idx + batch_size] # bugfix: thanks Zezhou Sun!
        X_batch = X_batch[:,::downsample,::downsample,:]
        y_batch = np.ravel(y[idx:idx + batch_size])
        return X_batch
              
    def generator(self, z, dims, activation=tf.nn.relu, scope_name="generator"): #upscale yapmiyor
        #N = len(dims)
        dims = [128, 3] #sillll
        with tf.variable_scope(scope_name):
            with tf.variable_scope("layer1"): #scopeu duzenle
                h1 = utils.conv2d_transpose(z, dims[0], mapsize=1, stride=1, stddev_factor=1.)
                h2 = utils.upscale(h1)
            with tf.variable_scope("layer2"):    
                h3 = utils.conv2d_transpose(h2, dims[1], mapsize=1, stride=1, stddev_factor=1.)
                pred_image = utils.upscale(h3)
  
        return pred_image
        
    def discriminator(self, input_images, dims, activation=tf.nn.relu, scope_name="discriminator",
                       scope_reuse=False):
        #N = len(dims)
        with tf.variable_scope(scope_name) as scope:
            if scope_reuse:
                scope.reuse_variables()
            h = input_images
            for index in range(1):
                W = utils.weight_variable([4, 4, dims[index], dims[index + 1]], name="W_%d" % index)
                b = utils.bias_variable([dims[index + 1]], name="b_%d" % index)
                h = utils.conv2d_strided(h, W, b)

            W_pred = utils.weight_variable([4, 4, dims[-2], dims[-1]], name="W_pred")
            b = tf.zeros([dims[-1]])
            h_pred = utils.conv2d_strided(h, W_pred, b)
        return h_pred
        
        
    def wgan_loss(self, logits_real, logits_fake):
        self.discriminator_loss = tf.reduce_mean(logits_real - logits_fake)
        self.gen_loss = tf.reduce_mean(logits_fake)

    def get_optimizer(self, optimizer_name, learning_rate, optimizer_param):
        self.learning_rate = learning_rate
        if optimizer_name == "Adam":
            return tf.train.AdamOptimizer(learning_rate, beta1=optimizer_param)
        elif optimizer_name == "RMSProp":
            return tf.train.RMSPropOptimizer(learning_rate, decay=optimizer_param)
        else:
            raise ValueError("Unknown optimizer %s" % optimizer_name)

    def optimizer_train(self, loss_val, var_list, optimizer):
        grads = optimizer.compute_gradients(loss_val, var_list=var_list)
        return optimizer.apply_gradients(grads)
            
    def create_model(self, gen_dims, disc_dims, optimizer="Adam", learning_rate=2e-4,
                       optimizer_param=0.9):
        print("create model")
        #placeholders
        self.input_batch = tf.placeholder(tf.float32, [self.batch_size, self.z_dim, self.z_dim, 3], name="z")
        self.real_batch = tf.placeholder(tf.float32, [self.batch_size, self.real_dim, self.real_dim, 3], name="z")
        self.train_phase = tf.placeholder(tf.bool)
        
        self.gen_images = self.generator(self.input_batch, gen_dims, scope_name="generator")
        #print(self.gen_images.get_shape())
        logits_real= self.discriminator(self.real_batch, disc_dims,
                                                    activation=utils.leaky_relu,
                                                    scope_name="discriminator",
                                                    scope_reuse=False)
        print(logits_real.get_shape())
        logits_fake = self.discriminator(self.gen_images, disc_dims,
                                                    activation=utils.leaky_relu,
                                                    scope_name="discriminator",
                                                    scope_reuse=True)

        self.wgan_loss(logits_real, logits_fake)
        train_variables = tf.trainable_variables()
        print(train_variables[0].name)
        self.generator_variables = [v for v in train_variables if v.name.startswith("generator")]
        print(self.generator_variables)
        self.discriminator_variables = [v for v in train_variables if v.name.startswith("discriminator")]

        optim = self.get_optimizer(optimizer, learning_rate, optimizer_param)

        self.generator_train_op = self.optimizer_train(self.gen_loss, self.generator_variables, optim)
        self.discriminator_train_op = self.optimizer_train(self.discriminator_loss, self.discriminator_variables, optim)

    def train_model(self, num_data, max_iterations):
            print("training is starting!!")
            #initialize
            sess = tf.InteractiveSession()
            sess.run(tf.global_variables_initializer())
            
            clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, self.clip_values[0], self.clip_values[1]))                                          for var in self.discriminator_variables]
         
            def get_feed_dict(train_phase=True):
                    batch_z = self.cifar10_dataset('train',4,num_data).astype(np.float32)
                    batch_real = self.cifar10_dataset('train',1,num_data).astype(np.float32) 
                    feed_dict = {self.input_batch: batch_z, self.real_batch: batch_real, self.train_phase: train_phase}
                    return feed_dict
                    
            for itr in xrange(1, max_iterations):
                   if itr < 25 or itr % 500 == 0:
                       critic_itrs = 25
                   else:
                       critic_itrs = self.critic_iterations
                       
                   for critic_itr in range(critic_itrs):
                        sess.run(self.discriminator_train_op, feed_dict=get_feed_dict(True))
                        sess.run(clip_discriminator_var_op)
                        
                   feed_dict = get_feed_dict(True)
                   sess.run(self.generator_train_op, feed_dict=feed_dict)
                         
                   if itr % 10 == 0:
                        g_loss_val, d_loss_val = sess.run([self.gen_loss, self.discriminator_loss],
                                                               feed_dict=feed_dict)
                        print("Step: %d, generator loss: %g, discriminator_loss: %g" % (
                            itr, g_loss_val, d_loss_val))
               
               
          
        



import tensorflow as tf
import numpy as np
import os, sys, inspect
import time

import utils as utils
import read_cifar10 as cf10
import read_data
import numpy as np
import time

FLAGS = tf.app.flags.FLAGS

class WGAN(object):
       
    def __init__(self, data_name, batch_size, clip_values, disc_iterations, num_data):
        self.batch_size = batch_size
        self.input_images = self.cifar10_dataset(data_name,4, num_data) #tek fonk yap tek seferde yukle
        self.real_images = self.cifar10_dataset(data_name,1, num_data) #usttekiyle tek fonk yap birlestir
        self.disc_iterations = disc_iterations
        self.clip_values = clip_values
        self.z_dim = 8 #change it make it flexible
        self.real_dim = 32
        self.logs_dir = "logs/wgan_logs/"

            
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
        X = X0[:total_num]
        X_batch = X[num * batch_size:(num+1) + batch_size] # bugfix: thanks Zezhou Sun!
        X_batch = X_batch[:,::downsample,::downsample,:]
        return X_batch
 
    
    def batch_sampler(self, x, batch_size):
        shape = x.shape
        assert len(shape) == 4
        data_num = shape[0]
        idx = np.random.randint(low=0, high=data_num-1, size=batch_size)
        batch = x[idx]
        return batch, idx


    def upscale(self,x):
        old_size = x.get_shape()
        size = [2 * int(s) for s in old_size[1:3]]
        out  = tf.image.resize_nearest_neighbor(x, size)
        return out


    def glorot_initializer_conv2d(self, prev_units, num_units, mapsize, stddev_factor=1.0):
        stddev  = np.sqrt(stddev_factor / (np.sqrt(prev_units*num_units)*mapsize*mapsize))
        return tf.truncated_normal([mapsize, mapsize, prev_units, num_units],
                                        mean=0.0, stddev=stddev)
    
    
    def conv2d_strided(self, x, W, b):
        conv = tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding="SAME")
        return tf.nn.bias_add(conv, b)
      
           
    def conv2d_transpose(self, x, num_units, mapsize=1, stride=1, stddev_factor=1.0):
        assert len(x.get_shape()) == 4 and "Previous layer must be 4-dimensional"
        
        with tf.variable_scope("conv2d_transpose"):
            prev_units = int(x.get_shape()[-1])                    
            initw = self.glorot_initializer_conv2d(prev_units, num_units,
                                                     mapsize, stddev_factor=stddev_factor)
            weight = tf.get_variable('weight', initializer=initw)
            weight = tf.transpose(weight, perm=[0, 1, 3, 2])
            output_shape = [FLAGS.batch_size, #maine flag olarak batch size koy
                            int(x.get_shape()[1]) * stride,
                            int(x.get_shape()[2]) * stride,
                            num_units]
            out = tf.nn.conv2d_transpose(x, weight,
                                            output_shape=output_shape,
                                            strides=[1, stride, stride, 1],
                                            padding='SAME')

            initb = tf.constant(0.0, shape=[num_units])
            bias = tf.get_variable('bias', initializer=initb)
            out = tf.nn.bias_add(out, bias)        
        return out
     
                 
    def generator(self, z, dims, activation=tf.nn.relu, scope_name="generator"): 
        with tf.variable_scope(scope_name):
            with tf.variable_scope("layer1"): #arrange scope
                h1 = self.conv2d_transpose(z, dims[0], mapsize=1, stride=1, stddev_factor=1.)
                h2 = self.upscale(h1)
            with tf.variable_scope("layer2"):    
                h3 = self.conv2d_transpose(h2, dims[1], mapsize=1, stride=1, stddev_factor=1.)
                pred_image = self.upscale(h3)
        utils.add_activation_summary(pred_image)
        return pred_image
      
        
    def discriminator(self, input_images, dims, activation=tf.nn.relu, scope_name="discriminator",
                       scope_reuse=False):
        with tf.variable_scope(scope_name) as scope:
            if scope_reuse:
                scope.reuse_variables()
            h = input_images
            for index in range(1):
                W = utils.weight_variable([4, 4, dims[index], dims[index + 1]], name="W_%d" % index)
                b = utils.bias_variable([dims[index + 1]], name="b_%d" % index)
                h = self.conv2d_strided(h, W, b)

            W_pred = utils.weight_variable([4, 4, dims[-2], dims[-1]], name="W_pred")
            b = tf.zeros([dims[-1]])
            h_pred = self.conv2d_strided(h, W_pred, b)
            print(h_pred.get_shape())
        return h_pred
       
                
    def wgan_loss(self, logits_real, logits_fake):
        self.discriminator_loss = tf.reduce_mean(logits_real- logits_fake) #isarete dikksttt 
        self.gen_loss = tf.reduce_mean(logits_fake) #isarete dikkatt eksi mi arti miiii
        tf.summary.scalar("Disc_loss", self.discriminator_loss)
        tf.summary.scalar("Gen_loss", self.gen_loss)


    def imp_wgan_loss(self, logits_real, logits_fake, real_images, fake_images):
        lmda = 10
        self.gen_loss = tf.reduce_mean(logits_fake)
        self.discriminator_loss = tf.reduce_mean(logits_real) - tf.reduce_mean(logits_fake)
        
        alpha = tf.random_uniform( shape=[self.batch_size,1], minval=0., maxval=1.)
        shape = fake_images.get_shape().as_list()
        out_dim = shape[1] * shape[1] * shape[3]
        
        real_data = tf.reshape(real_images, [self.batch_size, out_dim])
        fake_data = tf.reshape(fake_images, [self.batch_size, out_dim])   
          
        differences = fake_data - real_data
        interpolates = real_data + (alpha*differences)
        interpolates = tf.reshape(interpolates, [self.batch_size, shape[1], shape[1],shape[3]])
        gradients = tf.gradients(self.discriminator(interpolates, [3, 64,1], scope_reuse=True), [interpolates])[0] #D dimensionu moduler yap
        #print(gradients.get_shape())
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2,3]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        self.discriminator_loss += lmda*gradient_penalty


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
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
        return optimizer.apply_gradients(grads)
     
            
    def create_model(self, gen_dims, disc_dims, optimizer="Adam", learning_rate=2e-4,
                       optimizer_param=0.9, loss_type="wasserstein"):
        print("create model")
        #placeholders
        self.input_batch = tf.placeholder(tf.float32, [self.batch_size, self.z_dim, self.z_dim, 3], name="z")
        self.real_batch = tf.placeholder(tf.float32, [self.batch_size, self.real_dim, self.real_dim, 3], name="z")
        
        self.gen_images = self.generator(self.input_batch, gen_dims, scope_name="generator")
        #print(self.gen_images.get_shape())
        tf.summary.image("image_real", self.real_batch*255+127.5, max_outputs=2)
        tf.summary.image("image_generated", self.gen_images*255+127.5, max_outputs=2)
        logits_real= self.discriminator(self.real_batch, disc_dims,
                                                    activation=utils.leaky_relu,
                                                    scope_name="discriminator",
                                                    scope_reuse=False)
        print(logits_real.get_shape())
        logits_fake = self.discriminator(self.gen_images, disc_dims,
                                                    activation=utils.leaky_relu,
                                                    scope_name="discriminator",
                                                    scope_reuse=True)
        if loss_type == "wasserstein":
            self.wgan_loss(logits_real, logits_fake)
        elif loss_type == "imp_wasserstein":
            self.imp_wgan_loss(logits_real, logits_fake,self.real_batch, self.gen_images)
        else:
            raise ValueError("Unknown loss %s" % loss_type)
        
        train_variables = tf.trainable_variables()
        #print(train_variables[0].name)
        self.generator_variables = [v for v in train_variables if v.name.startswith("generator")]
        print(self.generator_variables)
        self.discriminator_variables = [v for v in train_variables if v.name.startswith("discriminator")]

        optim = self.get_optimizer(optimizer, learning_rate, optimizer_param)

        self.generator_train_op = self.optimizer_train(self.gen_loss, self.generator_variables, optim)
        self.discriminator_train_op = self.optimizer_train(self.discriminator_loss, self.discriminator_variables, optim)


    def train_model(self, num_data, max_iterations):
            print("training is starting!!")
            start_time = time.time()
            #initialize
            sess = tf.InteractiveSession()
            
            self.summary_op = tf.summary.merge_all()
            self.saver = tf.train.Saver()
            self.summary_writer = tf.summary.FileWriter(self.logs_dir, sess.graph) #logs dir i yazzzz
            
            sess.run(tf.global_variables_initializer())
            
            clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, self.clip_values[0], self.clip_values[1]))                                          for var in self.discriminator_variables]
                      
            def get_feed_dict():
                    #merge them into one module
                    batch_z, idx = self.batch_sampler(self.input_images, self.batch_size)
                    batch_real = self.real_images[idx,:,:,:]
                    feed_dict = {self.input_batch: batch_z, self.real_batch: batch_real}
                    return feed_dict
                    
            for itr in xrange(1, max_iterations):
                   if itr < 25 or itr % 500 == 0:
                       disc_itrs = 25
                   else:
                       disc_itrs = self.disc_iterations 
             
                   for disc_itr in range(disc_itrs):
                        sess.run(self.discriminator_train_op, feed_dict=get_feed_dict())
                        sess.run(clip_discriminator_var_op)
                        
                   feed_dict = get_feed_dict()
                   sess.run(self.generator_train_op, feed_dict=feed_dict)
                   
                   if itr % 100 == 0:
                        summary_str = sess.run(self.summary_op, feed_dict=feed_dict)
                        self.summary_writer.add_summary(summary_str, itr)
                           
                   if itr % 100 == 0:
                        stop_time = time.time()
                        duration = (stop_time - start_time) / 5.0
                        start_time = stop_time
                        g_loss_val, d_loss_val = sess.run([self.gen_loss, self.discriminator_loss],
                                                              feed_dict=feed_dict)
                        print("Time: %g/itr, Step: %d, generator loss: %g, discriminator_loss: %g" % (
                        duration, itr, g_loss_val, d_loss_val))
                  
                   if itr % 100 == 0:
                        self.saver.save(sess, self.logs_dir + "model.ckpt", global_step=itr)

               
          
        



import tensorflow as tf
import numpy as np

import utils as utils
import read_cifar10 as cf10
import read_data
import skimage.io
import time

FLAGS = tf.app.flags.FLAGS

class GAN_AE(object):      
    def __init__(self, batch_size, clip_values, disc_iterations, num_train_data, num_test_data, folder):
        self.batch_size = batch_size
        self.input_images = self.cifar10_dataset('train',1, num_train_data) 
        self.test_images =  self.cifar10_dataset('test',1, num_test_data)
        self.disc_iterations = disc_iterations
        self.clip_values = clip_values #clip of weights in wasserstein to approximate Lipschitz
        self.z_dim = 32 #input dim
        self.real_dim = 32 #real images dim
        self.logs_dir = "logs/wgan_logs/"
        self.folder = folder

    #returns original or downsampled cifar images            
    def cifar10_dataset(self,name, downsample, total_num): 
        assert name in ['train', 'test']
        X, y = (cf10.load_training_data() if name == 'train'
                                 else cf10.load_test_data())    
        data = X[:total_num]
        return data[:,::downsample,::downsample,:]

    #returns batch images    
    def cifar10_batches(self, name, downsample, batch_size, num, total_num=12800):
        assert name in ['train', 'test']
        assert total_num % batch_size == 0 or batch_size == -1 
        X0, y0 = (cf10.load_training_data() if name == 'train'
                                 else cf10.load_test_data())    
        X = X0[:total_num]
        X_batch = X[num * batch_size:(num+1) + batch_size]
        X_batch = X_batch[:,::downsample,::downsample,:]
        return X_batch
 
    #returns randomly sampled inputs and indices   
    def batch_sampler(self, x, batch_size):
        shape = x.shape
        assert len(shape) == 4
        data_num = shape[0]
        idx = np.random.randint(low=0, high=data_num-1, size=batch_size)
        batch = x[idx]
        return batch, idx

    #upsampling x2
    def upscale(self,x):
        old_size = x.get_shape()
        size = [2*int(k) for k in old_size[1:3]]
        out  = tf.image.resize_images(x, size)
        return out

    #xavier initialization for conv
    def glorot_initializer_conv2d(self, prev_units, num_units, mapsize, stddev_factor=1.0):
        stddev  = np.sqrt(stddev_factor / (np.sqrt(prev_units*num_units)*mapsize*mapsize))
        return tf.truncated_normal([mapsize, mapsize, prev_units, num_units],
                                        mean=0.0, stddev=stddev)
    
    #conv2d with stride number = 2
    def conv2d_strided(self, x, W, b):
        conv = tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding="SAME")
        return tf.nn.bias_add(conv, b)
      
    #transposed conv2d        
    def conv2d_transpose(self, x, num_units, mapsize=1, stride=1, stddev_factor=1.0):
        assert len(x.get_shape()) == 4        
        with tf.variable_scope("conv2d_transpose"):
            prev_units = int(x.get_shape()[-1])                    
            initw = self.glorot_initializer_conv2d(prev_units, num_units,
                                                     mapsize, stddev_factor=stddev_factor)
            weight = tf.get_variable('weight', initializer=initw)
            weight = tf.transpose(weight, perm=[0, 1, 3, 2])
            output_shape = [FLAGS.batch_size,
                            int(x.get_shape()[1]) * stride,
                            int(x.get_shape()[2]) * stride, num_units]
            out = tf.nn.conv2d_transpose(x, weight,
                                            output_shape=output_shape,
                                            strides=[1, stride, stride, 1],
                                            padding='SAME')
            initb = tf.constant(0.0, shape=[num_units])
            bias = tf.get_variable('bias', initializer=initb)
            out = tf.nn.bias_add(out, bias)        
        return out
    
    #autoencoder (generator) 
    def generator(self, x,kernels1=[5,7,9],kernels2=[9,7,5],filters1=[64,16,3],filters2=[16,32,3],pool_size=[1,2,2], scope_name="generator"):
        out=x
        with tf.variable_scope(scope_name):
            for k in range(len(kernels1)):
                conv = tf.layers.conv2d(inputs=out,
                                        filters=filters1[k],
                                        kernel_size=[kernels1[k],kernels1[k]],
                                        padding="same",
                                        activation=tf.nn.relu,
                                        name='conv'+str(k))
                pool_now=pool_size[k]
                if(pool_now==1):
                    out=conv
                else:
                    out = tf.layers.max_pooling2d(inputs=conv, 
                                                   pool_size=[pool_now,pool_now], 
                                                   strides=pool_now,
                                                   name = 'pool'+str(k))               
                out_quant=tf.round(out*255.)/255.
            for k in range(len(kernels2)):
                with tf.variable_scope("deconv") as var_scope:
                    pool_now=pool_size[-1-k]
                    if(pool_now==1):
                        x_up=out
                        out = tf.layers.conv2d(inputs=x_up,
                                                filters=filters2[k],
                                                kernel_size=[kernels2[k],kernels2[k]],
                                                padding="same",
                                                activation=tf.nn.relu,
                                                name='deconv'+str(k))
                    else:
                        shape = out.get_shape().as_list()
                        x_up = tf.image.resize_images(out,[shape[1]*pool_now,shape[2]*pool_now])
                        out = tf.layers.conv2d(inputs=x_up,
                                                filters=filters2[k],
                                                kernel_size=[kernels2[k],kernels2[k]],
                                                padding="same",
                                                activation=tf.nn.relu,
                                                name='deconv'+str(k))    
        return out
    
    #discriminator                     
    def discriminator(self, input_images, dims, activation=tf.nn.relu, scope_name="discriminator",
                       scope_reuse=False):
        with tf.variable_scope(scope_name) as scope:
            if scope_reuse:
                scope.reuse_variables()
            h = input_images
            for index in range(2):
                W = utils.weight_variable([4, 4, dims[index], dims[index + 1]], name="W_%d" % index)
                b = utils.bias_variable([dims[index + 1]], name="b_%d" % index)
                h = self.conv2d_strided(h, W, b)                
            W_pred = utils.weight_variable([4, 4, dims[-2], dims[-1]], name="W_pred")
            b = tf.zeros([dims[-1]])
            h_pred = self.conv2d_strided(h, W_pred, b)
        return h_pred
       
    #wasserstein loss            
    def wgan_loss(self, logits_real, logits_fake):
        self.discriminator_loss = tf.reduce_mean(logits_real- logits_fake) 
        self.gen_loss = tf.reduce_mean(logits_fake) 
        tf.summary.scalar("Disc_loss", self.discriminator_loss)
        tf.summary.scalar("Gen_loss", self.gen_loss)

    #weighted L1 loss 
    def add_l1_loss(self, real_image, fake_image, reg_var):
        self.gen_loss += reg_var * tf.reduce_mean(tf.abs(real_image - fake_image))
        self.mse_loss = tf.reduce_mean(tf.square(real_image - fake_image))
    
    #weighted L2 loss    
    def add_l2_loss(self, real_image, fake_image, reg_var):
        self.mse_loss = reg_var * tf.reduce_mean(tf.square(real_image - fake_image))
        self.gen_loss += self.mse_loss 

    #improved wasserstein loss / discriminator dimension needs to be given by hand  
    def imp_wgan_loss(self, logits_real, logits_fake, real_images, fake_images):
        lmda = 1000
        self.gen_loss = tf.reduce_mean(logits_fake)
        self.discriminator_loss = tf.reduce_mean(logits_real) - tf.reduce_mean(logits_fake)           
        alpha = tf.random_uniform( shape=[self.batch_size,1], minval=0., maxval=1.)
        shape = fake_images.get_shape().as_list()
        out_dim = shape[1] * shape[1] * shape[3]
        
        real_data = tf.reshape(real_images, [self.batch_size, out_dim])
        fake_data = tf.reshape(fake_images, [self.batch_size, out_dim])   
          
        error = fake_data - real_data
        new_var = real_data + (alpha*error)
        new_var = tf.reshape(new_var, [self.batch_size, shape[1], shape[1],shape[3]])
        gradients = tf.gradients(self.discriminator(new_var, [3, 16, 64,1], scope_reuse=True), [new_var])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2,3]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        self.discriminator_loss += lmda*gradient_penalty

    #choose optimizer for the network
    def get_optimizer(self, optimizer_name, learning_rate, optimizer_param):
        self.learning_rate = learning_rate
        if optimizer_name == "Adam":
            return tf.train.AdamOptimizer(learning_rate, beta1=optimizer_param)
        elif optimizer_name == "RMSProp":
            return tf.train.RMSPropOptimizer(learning_rate, decay=optimizer_param)
        else:
            raise ValueError("Unknown optimizer %s" % optimizer_name)

    #calculate and apply gradients of the specified variables
    def optimizer_train(self, loss_val, var_list, optimizer):
        grads = optimizer.compute_gradients(loss_val, var_list=var_list)
        return optimizer.apply_gradients(grads)
     
    #creates the placeholders and the model        
    def create_model(self, disc_dims, kernel_encoder, kernel_decoder, encoder_dims, decoder_dims, optimizer="Adam", learning_rate=2e-4,
                       optimizer_param=0.9, loss_type="wasserstein"):
        self.input_batch = tf.placeholder(tf.float32, [self.batch_size, self.z_dim, self.z_dim, 3], name="z")
        self.real_batch = tf.placeholder(tf.float32, [self.batch_size, self.real_dim, self.real_dim, 3], name="z")
        self.gen_images = self.generator(self.input_batch, kernel_encoder, kernel_decoder, encoder_dims, decoder_dims, scope_name="generator")
  
        tf.summary.image("image_real", self.real_batch*255+127.5, max_outputs=1)
        tf.summary.image("image_generated", self.gen_images*255+127.5, max_outputs=1)        
        logits_real= self.discriminator(self.real_batch, disc_dims,
                                                    activation=utils.leaky_relu,
                                                    scope_name="discriminator",
                                                    scope_reuse=False)
        logits_fake = self.discriminator(self.gen_images, disc_dims,
                                                    activation=utils.leaky_relu,
                                                    scope_name="discriminator",
                                                    scope_reuse=True)
        if loss_type == "wasserstein":
            self.wgan_loss(logits_real, logits_fake)
        elif loss_type == "imp_wasserstein":
            self.imp_wgan_loss(logits_real, logits_fake,self.real_batch, self.gen_images)
        elif loss_type == "imp_wasserstein_l2_loss":
            self.imp_wgan_loss(logits_real, logits_fake,self.real_batch, self.gen_images)
            self.add_l2_loss(self.real_batch, self.gen_images, 1)
        elif loss_type == "wasserstein_l2_loss":
            self.wgan_loss(logits_real, logits_fake)
            self.add_l2_loss(self.real_batch, self.gen_images, 10)
        elif loss_type == "wasserstein_l1_loss":
            self.wgan_loss(logits_real, logits_fake)
            self.add_l1_loss(self.real_batch, self.gen_images, 10)
       # elif loss_type =="dcgan":
       #     self.dcgan_loss()            
        else:
            raise ValueError("Unknown loss %s" % loss_type)
        
        train_variables = tf.trainable_variables()
        self.generator_variables = [v for v in train_variables if v.name.startswith("generator")]
        self.discriminator_variables = [v for v in train_variables if v.name.startswith("discriminator")]      
        optim = self.get_optimizer(optimizer, learning_rate, optimizer_param)

        self.generator_train_op = self.optimizer_train(self.gen_loss, self.generator_variables, optim)
        self.discriminator_train_op = self.optimizer_train(self.discriminator_loss, self.discriminator_variables, optim)

    #trains the model and prints outputs to txt file; saves images in some intermediate steps/ tests on Lena
    def train_model(self, num_data, max_iterations):
            start_time = time.time()
            sess = tf.InteractiveSession()            
            self.summary_op = tf.summary.merge_all()
            self.saver = tf.train.Saver()
            self.summary_writer = tf.summary.FileWriter(self.logs_dir, sess.graph) #logs dir i yazzzz
            
            sess.run(tf.global_variables_initializer())            
            mse = []
            clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, self.clip_values[0], self.clip_values[1]))                                          for var in self.discriminator_variables]
                      
            def get_feed_dict():
                    batch_z, idx = self.batch_sampler(self.input_images, self.batch_size)
                    batch_real = self.input_images[idx]
                    feed_dict = {self.input_batch: batch_z, self.real_batch: batch_real}
                    return feed_dict
                    
            f = open("./"+self.folder+"/mse.txt", "wb")   
            for itr in xrange(1, max_iterations):
                   if itr < 25 or itr % 100 == 0:
                       disc_itrs = 25
                   else:
                       disc_itrs = self.disc_iterations 
                                   
                   for disc_itr in range(disc_itrs):
                        sess.run(self.discriminator_train_op, feed_dict=get_feed_dict())
                        sess.run(clip_discriminator_var_op)                      
                   feed_dict = get_feed_dict()
                   sess.run(self.generator_train_op, feed_dict=feed_dict)
                   
                   if itr % 10000 ==0:
                        gen_out = sess.run(self.gen_images, feed_dict=feed_dict )
                        real_out = sess.run(self.real_batch, feed_dict=feed_dict )
                        print(np.mean(np.square(gen_out-real_out)))
                   
                   if itr % 100== 0:
                        summary_str = sess.run(self.summary_op, feed_dict=feed_dict)
                        self.summary_writer.add_summary(summary_str, itr)
                        
                        gen_out = sess.run(self.gen_images, feed_dict=feed_dict )
                        real_out = sess.run(self.real_batch, feed_dict=feed_dict )
                        
                        mse.append(sess.run(self.mse_loss, feed_dict=feed_dict )) 
                        cur_mse = sess.run(self.mse_loss, feed_dict=feed_dict )
                        f.write(str(cur_mse))
                        f.write("\n")                                               
                        print(np.mean(np.square(gen_out-real_out)))

                        def convert2uint8(img):
                            img[img>255]=255
                            img[img<0]=0
                            return img.astype(np.uint8)
                            
                        new_fake = utils.block2img(gen_out, (256,256))
                        new_real = utils.block2img(real_out, (256,256))                        
                        print_fake = convert2uint8(new_fake*255)

                        skimage.io.imsave('./'+self.folder+'/genful'+str(itr)+'.tiff', print_fake)
                        skimage.io.imsave('./'+self.folder+'/realful'+str(itr)+'.tiff', new_real)                      
                        
                        #lena test    
                        lena = skimage.io.imread('./lena.tiff')
                        lena = np.asarray(lena)
                        lena = lena.astype(np.float32) 
                        row,col,color = lena.shape
                        img_8x8=np.zeros((int(row*col/1024),32,32,3)).astype(np.float32) 
                        count =0
                        for i in range(0,row-row%32,32):
                            for j in range(0,col-col%32,32):
                                img_8x8[count,:,:,:]=lena[i:i+32,j:j+32,:]
                                count = count +1
                        test_img = img_8x8                                                                                             
                        test = test_img[:64]
                        feed_test = {self.input_batch: test, self.real_batch: test}
                        block1 = sess.run(self.gen_images, feed_dict=feed_test )
                                                    
                        for i in range(1,4):
                            test = test_img[64*i:64*(i+1)]
                            feed_test = {self.input_batch: test, self.real_batch: test}
                            block2 = sess.run(self.gen_images, feed_dict=feed_test )
                            block1 = np.concatenate( [block1, block2], axis=0 )
                             
                        test_out = np.asarray(block1)                                    
                        test_res = utils.block2img(test_out, (512,512))
                        test_res = convert2uint8(test_res)
                        skimage.io.imsave('./'+self.folder+'/test'+str(itr)+'.tiff', test_res)
                        #
                   if itr % 10000 == 0:
                        stop_time = time.time()
                        duration = (stop_time - start_time) / 5.0
                        start_time = stop_time
                        g_loss_val, d_loss_val = sess.run([self.gen_loss, self.discriminator_loss],
                                                              feed_dict=feed_dict)            
                        print("Time: %g/itr, Step: %d, generator loss: %g, discriminator_loss: %g" % (
                        duration, itr, g_loss_val, d_loss_val))
             
                   if itr % 10000 == 0:
                        self.saver.save(sess, self.logs_dir + "model.ckpt", global_step=itr)
            f.close()
           
               
          
        



import tensorflow as tf
import read_cifar10 as cf10
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

batch = 1
number_of_images = 512

def cifar10_dataset_generator(dataset_name, batch_size, restrict_size=1000):
    assert dataset_name in ['train', 'test']
    assert batch_size > 0 or batch_size == -1  # -1 for entire dataset
    
    X_all_unrestricted, y_all = (cf10.load_training_data() if dataset_name == 'train'
                                 else cf10.load_test_data())
    
    actual_restrict_size = restrict_size if dataset_name == 'train' else int(1e10)
    X_all = X_all_unrestricted[:actual_restrict_size]
    data_len = X_all.shape[0]
    batch_size = batch_size if batch_size > 0 else data_len
    
    X_all_padded = np.concatenate([X_all, X_all[:batch_size]], axis=0)
    y_all_padded = np.concatenate([y_all, y_all[:batch_size]], axis=0)
    
    for slice_i in range(math.ceil(data_len / batch_size)):
        idx = slice_i * batch_size
        X_batch = X_all_padded[idx:idx + batch_size]
        y_batch = np.ravel(y_all_padded[idx:idx + batch_size])
        yield X_batch.astype(np.float32), y_batch.astype(np.uint8)
        
        
def mlp(x, hidden_sizes, activation_fn=tf.nn.relu):
    if not isinstance(hidden_sizes, (list, tuple)):
        raise ValueError("hidden_sizes must be a list or a tuple")
    W = {}
    b = {}
    h = {}
    for i in range(len(hidden_sizes)):
        if(W == {}):
           W["W"+str(i)] = tf.get_variable("W"+str(i),[1024,hidden_sizes[i]],tf.float32,tf.random_normal_initializer(stddev = 0.01))
           b["b"+str(i)] = tf.get_variable("b"+str(i),[hidden_sizes[i]],tf.float32,tf.constant_initializer(0.0))
           h["h"+str(i)] = activation_fn(tf.matmul(x,W["W"+str(i)])+b["b"+str(i)])
        elif(i == len(hidden_sizes)-1):
           W["W"+str(i)] = tf.get_variable("W"+str(i),[hidden_sizes[i-1],hidden_sizes[i]],tf.float32,tf.random_normal_initializer(stddev = 0.01))
           b["b"+str(i)] = tf.get_variable("b"+str(i),[hidden_sizes[i]],tf.float32,tf.constant_initializer(0.0))
           return (tf.matmul(h["h"+str(i-1)],W["W"+str(i)])+b["b"+str(i)])
        else:
           W["W"+str(i)] = tf.get_variable("W"+str(i),[hidden_sizes[i-1],hidden_sizes[i]],tf.float32,tf.random_normal_initializer(stddev = 0.01))
           b["b"+str(i)] = tf.get_variable("b"+str(i),[hidden_sizes[i]],tf.float32,tf.constant_initializer(0.0))
           h["h"+str(i)] = activation_fn(tf.matmul(h["h"+str(i-1)],W["W"+str(i)])+b["b"+str(i)])
           
           
with tf.Graph().as_default():
    x = tf.placeholder(tf.float32,[None,32,32,3])
    y = tf.placeholder(tf.float32,[None,1024])
    cropped_x = x[:,:,:,0:1];
    reshaped_x = tf.reshape(cropped_x,[batch,1024])
    mlp_dct = mlp(reshaped_x,[1024,1024,1024])
    
    #cropped_y = y[:,:,:,0:1]
    #reshaped_y = tf.reshape(cropped_y,[batch,1024])
    
    
    binary = tf.equal(mlp_dct,y)
    
    #accuracy and loss of the predicted results
    accuracy = tf.reduce_mean(tf.cast(binary,tf.float32))
    loss = tf.reduce_sum(tf.square(mlp_dct-y))
    mse = loss/(32*32)
    r = tf.reduce_max(abs(mlp_dct-y))
    psnr = 10*tf.log((r**2)/mse)
    
    optimizer = tf.train.AdamOptimizer(0.2).minimize(loss)
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch_i in range(10):

            for iter_i, data_batch in enumerate(cifar10_dataset_generator('train', number_of_images)):
                loss_total = 0
                psnr_total = 0
                count = 0
                x_train_batch,y_train = data_batch
                for x_train in x_train_batch:
                    cropped_y = x_train[:,:,0:1]
                    dct_y = fftpack.dct(cropped_y)
                    reshaped_y = np.reshape(dct_y,[batch,1024])
                    x_train = np.reshape(x_train,[batch,32,32,3])
                    #train_feed_dict = dict(zip([x,y], data_batch))
                    _,loss_actual,psnr_actual,prediction = sess.run([optimizer,loss,psnr,mlp_dct], feed_dict={x:x_train,y:reshaped_y})
                    loss_total += loss_actual
                    psnr_total += psnr_actual
                    count += 1
                print("--------train image--------")
                print(reshaped_y)
                print("------prediction----------")
                print(prediction)
                print("-----------loss-------------")
                print(count,(loss_total/number_of_images),(psnr_total/number_of_images))
                #to_compute = [loss,psnr]
                #loss,psnr = sess.run(to_compute,feed_dict = {x:x_train,y:y_train})
                #print(loss,psnr)


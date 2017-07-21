import tensorflow as tf
import skimage.io
import numpy as np
import image_func as img
import os


def cnn_autoencoder(x_, kernels1=[5, 7], kernels2=[7, 5], filters1=[16, 128], filters2=[128, 3], pool_size=[1, 2, 2, 1],
                    name='autoencoder'):
    '''
    Autoencoder network

    Inputs:
    x_ (tf.placeholder) : Input tensor
    kernels1 (1D array) : Size of the encoder kernels (assumed square kernels)
    kernels2 (1D array) : Size of the decoder kernels (assumed square kernels)
    filters1 (1D array) : Number of filters in encoder layers
    filters2 (1D array) : Number of filters in decoder layers
    pool_size (1D array): Pooling size in each layer. Its length must be equal to len(kernels1)+len(kernels2)
                          First len(kernels1) terms will be used as pooling layers of encoder/
                          Remainin terms will be used as unpooling layers of decoder

    Returns:
    out_ (tf.placeholder)     : Output of the autoencoder without quantization in the middle
    out_quant (tf.placeholder): Output of the autoencoder with quantization in the middle
    '''
    with tf.variable_scope(name):
        out_ = x_
        for k in range(len(kernels1)):
            conv = tf.layers.conv2d(inputs=out_,
                                    filters=filters1[k],
                                    kernel_size=[kernels1[k], kernels1[k]],
                                    padding="same",
                                    activation=tf.nn.relu,
                                    name='conv' + str(k))
            pool_now = pool_size[k]
            if (pool_now == 1):
                out_ = conv
            else:
                out_ = tf.layers.max_pooling2d(inputs=conv,
                                               pool_size=[pool_now, pool_now],
                                               strides=pool_now,
                                               name='pool' + str(k))

            out_quant = tf.round(out_ * 255.) / 255.

        for k in range(len(kernels2)):
            with tf.variable_scope("deconv") as var_scope:
                pool_now = pool_size[k + len(kernels1)]
                if (pool_now == 1):
                    x_up = out_
                    out_ = tf.layers.conv2d(inputs=x_up,
                                            filters=filters2[k],
                                            kernel_size=[kernels2[k], kernels2[k]],
                                            padding="same",
                                            activation=tf.nn.relu,
                                            name='deconv' + str(k))
                    var_scope.reuse_variables()
                    x_quant_up = out_quant
                    out_quant = tf.layers.conv2d(inputs=x_quant_up,
                                                 filters=filters2[k],
                                                 kernel_size=[kernels2[k], kernels2[k]],
                                                 padding="same",
                                                 activation=tf.nn.relu,
                                                 name='deconv' + str(k))
                else:
                    sh = out_.get_shape().as_list()
                    x_up = tf.image.resize_images(out_, [sh[1] * pool_now, sh[2] * pool_now])
                    out_ = tf.layers.conv2d(inputs=x_up,
                                            filters=filters2[k],
                                            kernel_size=[kernels2[k], kernels2[k]],
                                            padding="same",
                                            activation=tf.nn.relu,
                                            name='deconv' + str(k))
                    var_scope.reuse_variables()
                    x_quant_up = tf.image.resize_images(out_quant, [sh[1] * pool_now, sh[2] * pool_now])
                    out_quant = tf.layers.conv2d(inputs=x_quant_up,
                                                 filters=filters2[k],
                                                 kernel_size=[kernels2[k], kernels2[k]],
                                                 padding="same",
                                                 activation=tf.nn.relu,
                                                 name='deconv' + str(k))
    return out_, out_quant


def apply_classification_loss_mse(kernels1=[5, 7], kernels2=[7, 5],
                                  filters1=[16, 128], filters2=[128, 3],
                                  pool_size=[1, 2, 2, 1], learning_rate=1., FT=False):
    '''
    MSE based autoencoder optimizer.

    Inputs:
    kernels1 (1D array) : Size of the encoder kernels (assumed square kernels)
    kernels2 (1D array) : Size of the decoder kernels (assumed square kernels)
    filters1 (1D array) : Number of filters in encoder layers
    filters2 (1D array) : Number of filters in decoder layers
    pool_size (1D array): Pooling size in each layer. Its length must be equal to len(kernels1)+len(kernels2)
                          First len(kernels1) terms will be used as pooling layers of encoder/
                          Remainin terms will be used as unpooling layers of decoder
    learning_rate(float): Learning rate of the optimizer
    FT (boolean)        : Boolean value for fine-tuning operation on decoder weights


    Returns:
    model_dict          : Dictionary of the required output files
    '''

    with tf.Graph().as_default() as g:
        with tf.device("/gpu:0"):  # use gpu:0 if on GPU
            x_ = tf.placeholder(tf.float32, [None, 32, 32, 3])
            (x_out, x_out_quant) = cnn_autoencoder(x_, pool_size=pool_size, kernels1=kernels1, filters1=filters1,
                                                   kernels2=kernels2, filters2=filters2)

            mse_loss1 = tf.reduce_mean(tf.subtract(x_, x_out) ** 2)
            mse_loss2 = tf.reduce_mean(tf.subtract(x_, x_out_quant) ** 2)

            trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            if (FT):
                with tf.variable_scope('autoencoder/deconv', reuse=True) as vs:
                    var_list = [v for v in tf.global_variables() if v.name.startswith(vs.name)]
                train_op = trainer.minimize(mse_loss1, var_list=var_list)
            else:
                train_op = trainer.minimize(mse_loss1)

    model_dict = {'graph': g, 'inputs': x_, 'outputs': x_out, 'train_op': train_op, 'loss1': mse_loss1,
                  'loss2': mse_loss2}

    return model_dict


# Working
def apply_classification_loss_mse_with_rnn(kernels1=[5, 7], kernels2=[7, 5],
                                           filters1=[16, 128], filters2=[128, 3],
                                           pool_size=[2, 2], learning_rate=1., FT=False, depth=3):
    '''
    MSE based autoencoder optimizer.

    Inputs:
    kernels1 (1D array) : Size of the encoder kernels (assumed square kernels)
    kernels2 (1D array) : Size of the decoder kernels (assumed square kernels)
    filters1 (1D array) : Number of filters in encoder layers
    filters2 (1D array) : Number of filters in decoder layers
    pool_size (1D array): Pooling size in each layer. Its length must be equal to len(kernels1)+len(kernels2)
                          First len(kernels1) terms will be used as pooling layers of encoder/
                          Remainin terms will be used as unpooling layers of decoder
    learning_rate(float): Learning rate of the optimizer
    FT (boolean)        : Boolean value for fine-tuning operation on decoder weights
    depth(integer)

    Returns:
    model_dict          : Dictionary of the required output files
    '''
    with tf.Graph().as_default() as g:
        with tf.device("/gpu:0"):  # use gpu:0 if on GPU
            x_ = tf.placeholder(tf.float32, [None, 32, 32, 3])

            (x_out2, x_out_quant) = cnn_autoencoder(x_, pool_size=pool_size, kernels1=kernels1, filters1=filters1,
                                                    kernels2=kernels2, filters2=filters2, name='filter0')
            x_out1 = x_
            mse_loss1 = tf.reduce_mean(tf.subtract(x_out1, x_out2) ** 2)
            for k in range(1, depth):
                (x_out3, x_out_quant) = cnn_autoencoder(x_out1 - x_out2, pool_size=pool_size, kernels1=kernels1,
                                                        filters1=filters1,
                                                        kernels2=kernels2, filters2=filters2, name='filter' + str(k))
                mse_loss1 = tf.add(mse_loss1, tf.reduce_mean(tf.subtract(x_out1, tf.add(x_out2, x_out3)) ** 2))
                x_out1 = x_out2
                x_out2 = x_out3
            x_out3 = x_out2
            # y_dict = dict(labels=y_, logits=y_logits)
            # losses = tf.nn.sparse_softmax_cross_entropy_with_logits(**y_dict)
            # cross_entropy_loss = tf.reduce_mean(losses)
            # mse_loss1=tf.reduce_mean(tf.subtract(x_,x_out)**2)
            # a=tf.pad(tf.subtract(x_,x_out),[[0,0],[16,16],[16,16],[0,0]],'CONSTANT')

            # mse_loss1=tf.reduce_mean(tf.nn.conv2d(a,h3,strides=[1,1,1,1],padding="VALID")**2)
            mse_loss2 = tf.reduce_mean(tf.subtract(x_, x_out3) ** 2)
            trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            if (FT):
                with tf.variable_scope('deconv', reuse=True) as vs:
                    var_list = [v for v in tf.global_variables() if v.name.startswith(vs.name)]
                train_op = trainer.minimize(mse_loss1, var_list=var_list)
            else:
                train_op = trainer.minimize(mse_loss1)

                # y_pred = tf.argmax(tf.nn.softmax(y_logits), dimension=1)
                # correct_prediction = tf.equal(tf.cast(y_pred, tf.int32), y_)
                # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    model_dict = {'graph': g, 'inputs': x_, 'outputs': x_out_quant, 'train_op': train_op, 'loss1': mse_loss1,
                  'loss2': mse_loss2}

    return model_dict



def train_model(model_dict, x_tr, x_test, img_32, train_every=100, test_every=200, max_iter=20001, load=False,
                fname='cifar10_recon', outname='/tmp/cnn_autoencoder', ftname='/tmp/cnn_autoencoder'):
    '''
    Inputs:
    model_dict: Output of apply_classification_loss_mse
    x_tr      : Training images
    x_test    : Test Images
    x_32      : 32x32 patches of a big image
    load      : Boolean for loading the weights from pre-trained network
    fname     : Directory to save outputs
    outname   : Directory to save (load=False) or load (load=True) weights
    ftname    : Directory to save new weights when load+True
    '''
    if not os.path.exists('..//'+fname):
        os.makedirs('..//'+fname)
    with model_dict['graph'].as_default(), tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if (load):
            saver.restore(sess, outname)
            print("Model loaded")
        else:
            sess.run(tf.global_variables_initializer())

        ids = [i for i in range(100)]
        for iter_i in range(max_iter):
            batch_xs = x_tr[ids, :, :, :]
            ids = [(ids[0] + 100 + i) % x_tr.shape[0] for i in range(100)]
            sess.run(model_dict['train_op'], feed_dict={model_dict['inputs']: batch_xs})

            # test trained model
            if iter_i % train_every == 0:
                tf_feed_dict = {model_dict['inputs']: batch_xs}
                loss_val = sess.run(model_dict['loss1'], feed_dict={model_dict['inputs']: batch_xs})
                print('iteration %d\t train mse: %.3E\t' % (iter_i, loss_val))
                if iter_i % test_every == 0:
                    # tf_feed_dict = {x_: x_test}
                    loss_val1 = sess.run(model_dict['loss1'], feed_dict={model_dict['inputs']: x_test})
                    loss_val2 = sess.run(model_dict['loss2'], feed_dict={model_dict['inputs']: x_test})
                    print(
                        'iteration %d\t TEST MSE: %.3E\t TEST MSE(Quantized): %.3E\t' % (iter_i, loss_val1, loss_val2))

                    img_block = sess.run(model_dict['outputs'],
                                         feed_dict={model_dict['inputs']: img_32})
                    x_from_test = sess.run(model_dict['outputs'],
                                           feed_dict={
                                               model_dict['inputs']: x_test[:5, :, :, :].reshape([-1, 32, 32, 3])})

                    img_recon = img.block2img(img_block, (512, 512))
                    img_recon = img.convert2uint8(img_recon * 255.)
                    skimage.io.imsave('../' + fname + '/img32_recon_' + str(int(iter_i / test_every)) + '.tiff',
                                      img_recon)

                    for i in range(5):
                        img_recon = img.convert2uint8((255 * x_from_test[i, :, :, :]).reshape([32, 32, 3])).astype(np.uint8)
                        skimage.io.imsave(
                            '../' + fname + '/test' + str(i) + '_' + str(int(iter_i / test_every)) + '.tiff', img_recon)

        saver = tf.train.Saver()
        if load:
            outname = ftname
        save_path = saver.save(sess, outname)
        print("Model saved in file: %s" % save_path)
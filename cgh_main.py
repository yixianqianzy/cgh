# -*- coding: utf-8 -*-
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import function
import matplotlib.pyplot as plt
import scipy.io as sio

"""This is a simple demonstration of the collaborative generative hashing algorithm 
with linear decoder and encoder on 2017 Recsys Challenge Dataset. 

Created by xxx 2019"""

def VAE_stoc_neuron(alpha, gammau, gammav, dim_inputu, dim_inputv, dim_hidden, batchu_size, batchv_size, learning_rate, max_iter, utrain,
                    vtrain, rtrain, num_user, num_item, xvaru, xmeanu, xvarv, xmeanv):
    g = tf.Graph()
    dtype = tf.float32
    with g.as_default():
        xu = tf.placeholder(dtype, [None, dim_inputu])
        xv = tf.placeholder(dtype, [None, dim_inputv])
        r = tf.placeholder(dtype, [None, None])
        nnc = tf.placeholder(tf.int32, [None, None])
        nnc = tf.cast(nnc, dtype)
        nnz = tf.placeholder(tf.int32)
        nnz = tf.cast(nnz, dtype)
        numu = tf.placeholder(tf.int32)
        numu = tf.cast(numu, dtype)
        numv = tf.placeholder(tf.int32)
        numv = tf.cast(numv, dtype)
        @function.Defun(dtype, dtype, dtype, dtype)
        def DoublySNGrad(logits, epsilon, dprev, dpout):
            prob = 1.0 / (1 + tf.exp(-logits))
            yout = (tf.sign(prob - epsilon) + 1.0) / 2.0
            dlogits = prob * (1 - prob) * (dprev + dpout)
            depsilon = dprev
            return dlogits, depsilon

        @function.Defun(dtype, dtype, grad_func=DoublySNGrad)
        def DoublySN(logits, epsilon):
            prob = 1.0 / (1 + tf.exp(-logits))
            yout = (tf.sign(prob - epsilon) + 1.0) / 2.0
            return yout, prob

        with tf.name_scope('encodeu'):
            wencodeu = tf.Variable(
                tf.random_normal([dim_inputu, dim_hidden], stddev=1.0 / tf.sqrt(float(dim_inputu)), dtype=dtype),
                name='wencodeu')
            bencodeu = tf.Variable(tf.random_normal([dim_hidden], dtype=dtype), name='bencodeu')
            hencodeu = tf.matmul(xu, wencodeu) + bencodeu
            hepsilonu = tf.ones(shape=tf.shape(hencodeu), dtype=dtype) * .5

        youtu, poutu = DoublySN(hencodeu, hepsilonu)

        with tf.name_scope('decodeu'):
             wdecodeu = tf.Variable(
                tf.random_normal([dim_hidden, dim_inputu], stddev=1.0 / tf.sqrt(float(dim_hidden)), dtype=dtype),
                name='wdecodeu')
        with tf.name_scope('scaleu'):
            scale_parau = tf.Variable(tf.constant(xvaru, dtype=dtype), name="scale_parau")  # xvar: 1*784
            shift_parau = tf.Variable(tf.constant(xmeanu, dtype=dtype), name="shift_parau")
        uout = tf.matmul(youtu, wdecodeu) * tf.abs(scale_parau) + shift_parau
        monitoru = tf.nn.l2_loss(uout - xu, name=None) / numu

        with tf.name_scope('encodev'):
            wencodev = tf.Variable(
                tf.random_normal([dim_inputv, dim_hidden], stddev=1.0 / tf.sqrt(float(dim_inputv)), dtype=dtype),
                name='wencodev')
            bencodev = tf.Variable(tf.random_normal([dim_hidden], dtype=dtype), name='bencodev')
            hencodev = tf.matmul(xv, wencodev) + bencodev
            hepsilonv = tf.ones(shape=tf.shape(hencodev), dtype=dtype) * .5
        youtv, poutv = DoublySN(hencodev, hepsilonv)
        with tf.name_scope('decodev'):
             wdecodev = tf.Variable(
                tf.random_normal([dim_hidden, dim_inputv], stddev=1.0 / tf.sqrt(float(dim_hidden)), dtype=dtype),
                name='wdecodev')
        with tf.name_scope('scalev'):
            scale_parav = tf.Variable(tf.constant(xvarv, dtype=dtype), name="scale_parav")  # xvar: 1*784
            shift_parav = tf.Variable(tf.constant(xmeanv, dtype=dtype), name="shift_parav")
        vout = tf.matmul(youtv, wdecodev) * tf.abs(scale_parav) + shift_parav
        monitorv = tf.nn.l2_loss(vout - xv, name=None) / numv
        hu = tf.where(youtu > 0.5, youtu, tf.zeros_like(youtu))
        hv = tf.where(youtv > 0.5, youtv, tf.zeros_like(youtv))
        real_rating_loss = tf.nn.l2_loss(tf.multiply(nnc, r - tf.ones(shape=tf.shape(r), dtype=dtype) * .5 -
                                                tf.matmul(hencodeu, tf.transpose(hencodev)) / (2 * dim_hidden))) / nnz
        rating_loss = tf.nn.l2_loss(tf.multiply(nnc, r - tf.ones(shape=tf.shape(r), dtype=dtype) * .5 -
                                tf.matmul(hu, tf.transpose(hv)) / (2 * dim_hidden))) / nnz
        loss = gammau*monitoru + gammav*monitorv + rating_loss + \
               alpha * tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=hencodeu, labels=youtu)) / batchu_size + \
               alpha * tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=hencodev, labels=youtv)) / batchv_size + \
               beta * tf.nn.l2_loss(wdecodeu, name=None) / batchu_size + \
               beta * tf.nn.l2_loss(wdecodev, name=None) / batchv_size

        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)
        sess = tf.Session(graph=g)
        sess.run(tf.global_variables_initializer())
        loss_total = []
        ratings_loss = []
        users_loss = []
        items_loss = []
        real_ratings_loss = []
        np.random.seed = 3
        for i in range(max_iter):
            indx = np.random.choice(utrain.shape[0], batchu_size)
            indxv = np.random.choice(vtrain.shape[0], batchv_size)
            ubatch = utrain[indx]
            vbatch = vtrain[indxv]
            rbatch0 = rtrain[indx].toarray()
            rbatch = rbatch0[:, indxv]
            nnzbatch = np.count_nonzero(rbatch)
            nncbatch0 = rbatch > 0
            nncbatch = nncbatch0.astype(int)
            # numbatch = rbatch.shape[0]
            numubatch = ubatch.shape[0]
            numvbatch = vbatch.shape[0]
            um = ubatch.mean(axis=0).astype('float64')  # axis=0代表列，axis=1代表行换句话说
            uv = np.clip(ubatch.var(axis=0), 1e-7, np.inf).astype('float64')

            vm = vbatch.mean(axis=0).astype('float64')  # axis=0代表列，axis=1代表行换句话说
            vv = np.clip(vbatch.var(axis=0), 1e-7, np.inf).astype('float64')

            _, monitoru_value, monitorv_value, rating_loss_value, loss_value, real_rating_loss_value = sess.run([train_op, monitoru, monitorv, rating_loss, loss, real_rating_loss],
                                                    feed_dict={xu: ubatch, xv: vbatch, r: rbatch, nnz: nnzbatch,
                                                               numu: numubatch, numv: numvbatch, nnc: nncbatch})  # feed_dict: replace x with xbatch

            if i % 20 == 0:
                print('Num iteration: %d Loss: %0.04f, Rating_loss: %0.04f, Users_Loss: %0.08f, '
                      'Items_Loss: %0.08f' % (i, loss_value, rating_loss_value, monitoru_value, monitorv_value))
                loss_total.append(loss_value)
                ratings_loss.append(rating_loss_value)
                users_loss.append(monitoru_value)
                items_loss.append(monitorv_value)
                real_ratings_loss.append(monitorv_value)

                # train_err.append(loss_value)
            if i % 300 == 0:
                learning_rate = 0.5 * learning_rate

        node_list = ['youtu', 'poutu', 'youtv', 'poutv', 'uout', 'vout',
                     'wencodeu', 'bencodeu', 'wdecodeu', 'scale_parau', 'shift_parau',
                     'wencodev', 'bencodev', 'wdecodev', 'scale_parav', 'shift_parav']
        # sio.savemat('train_loss.mat', {'train_loss': loss_total, 'rating_loss': ratings_loss,'user_loss': users_loss, 'items_loss': items_loss})
        # x = np.array(range(len(loss_total)))*20
        # lines = plt.plot(x, loss_total, 'r^-', x, ratings_loss, 'go-', x, users_loss, 'yd-', x, items_loss, 'bs-')
        # plt.setp(lines[0], linewidth=3)
        # plt.setp(lines[1], linewidth=3)
        # plt.setp(lines[2], markersize=3)
        # plt.setp(lines[3], markersize=3)
        #
        # plt.legend(('Total training loss', 'Ratings loss', 'Loss of users', 'Loss of items'),
        #            loc='upper right')
        # plt.title('Average training loss')
        # plt.show()
        t_vars = tf.trainable_variables()
        para_list = {}
        for var in t_vars:
            para_list[var.name] = sess.run(var)
    return g, node_list, para_list, loss_value, monitoru_value, monitorv_value


if __name__ == "__main__":

    v_content = sio.loadmat('./data/item_content.mat')
    u_content = sio.loadmat('./data/user_content.mat')
    ufactor = sio.loadmat('./data/u_factor.mat')
    vfactor= sio.loadmat('./data/v_factor.mat')
    traindata_r = sio.loadmat('./data/R_train.mat')
    rtrain = traindata_r['R_train']
    hutrain = np.sign(ufactor['u_factor'])  # weighted matrix factorization of training ratings 'rtrain'
    hvtrain = np.sign(vfactor['v_factor'])
    num_user = hutrain.shape[0]
    num_item = hvtrain.shape[0]
    utrain = u_content['user_train']
    umean = np.mean(utrain, axis=0).astype('float32')
    uvar = np.clip(utrain.var(axis=0), 1e-7, np.inf).astype(
        'float32')
    vtrain = v_content['item_train']
    vmean = vtrain.mean(axis=0).astype('float32')
    vvar = np.clip(vtrain.var(axis=0), 1e-7, np.inf).astype('float32')

    dim_input_uer = utrain.shape[1]
    dim_input_item = vtrain.shape[1]
    dim_hidden = hutrain.shape[1]
    print('dim of hidden variable is %d' % dim_hidden)
    batchu_size = 5000
    batchv_size = 2000
    learning_rate = 0.01
    gammau = 0.8
    gammav = 0.8
    max_iter = 1000
    alpha = 1e-4
    beta = 1e-4

    print('batchu_size: %d ,batchv_size: %d, gammau: %0.02f, gammav: %0.02f, learning_rate: %0.04f' % (batchu_size, batchv_size, gammau, gammav, learning_rate))
    start_time = time.time()
    g, node_list, para_list, loss, monitoru, monitorv = VAE_stoc_neuron(alpha, gammau, gammav, dim_input_uer,
                                                                             dim_input_item, dim_hidden,
                                                                             batchu_size, batchv_size,
                                                                             learning_rate,  max_iter, utrain,
                                                                             vtrain, rtrain, num_user, num_item,
                                                                             uvar, umean, vvar, vmean)
    end_time = (time.time() - start_time)
    print('Running time: %0.04f s' % end_time)

Wu = para_list['encodeu/wencodeu:0']
bu = para_list['encodeu/bencodeu:0']
Uu = para_list['decodeu/wdecodeu:0']  # the codebook
shiftu = para_list['scaleu/shift_parau:0']
scaleu = para_list['scaleu/scale_parau:0']
Wv = para_list['encodev/wencodev:0']  # ??
bv = para_list['encodev/bencodev:0']
Uv = para_list['decodev/wdecodev:0']  # the codebook
shiftv = para_list['scalev/shift_parav:0']
scalev = para_list['scalev/scale_parav:0']
filename = 'hash' + str(dim_hidden) + 'bit.mat'
sio.savemat(filename, {'Wu': Wu, 'bu': bu, 'Uu': Uu, 'scaleu': scaleu,
                       'shiftu': shiftu, 'Wv': Wv, 'bv': bv, 'Uv': Uv, 'scalev': scalev, 'shiftv': shiftv})

# # predict cold item
# cd_item0 = sio.loadmat('./cold_item/R_train_item.mat')
# cd_item_test0 = sio.loadmat('./cold_item/R_test_item.mat')
# cd_item_train0 = sio.loadmat('./cold_item/cd_item_train.mat')
# epsilonu = 0.5
# epsilonv = 0.5
# cd_item_user0 = sio.loadmat('./cold_item/cd_item_user.mat')
# cd_item_test = cd_item_test0['R_test_item']
# cd_item_train = cd_item_train0['cd_item_train']
# cd_item_user = cd_item_user0['cd_item_user']
#
# cdtrainlogitsu = np.dot(np.array(cd_item_user), Wu) + bu
# cdtrainpresu = 1.0 / (1 + np.exp(-cdtrainlogitsu))
# cdhutrain = (np.sign(cdtrainpresu - epsilonu) + 1.0) / 2.0
#
# cdtrainlogitsv = np.dot(np.array(cd_item_train), Wv) + bv
# cdtrainpresv = 1.0 / (1 + np.exp(-cdtrainlogitsv))
# cdhvtrain = (np.sign(cdtrainpresv - epsilonv) + 1.0) / 2.0
# cdhutrain[cdhutrain < 1] = -1
# cdhvtrain[cdhvtrain < 1] = -1
# hu = cdhutrain
# hv = cdhvtrain
#
# filename = './cold_item/cd_item_' + str(dim_hidden) + 'bit.mat'
# sio.savemat(filename, {'hu': hu, 'hv': hv})



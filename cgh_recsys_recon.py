# -*- coding: utf-8 -*-
import time
import numpy as np
# import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.python.framework import function
import matplotlib.pyplot as plt
import scipy.io as sio
# import scipy.sparse
import hdf5storage
# from mxnet import nd
# import sys

import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='test.log',
                    filemode='w')

logging.debug('debug message')
logging.info('info message')
logging.warning('warning message')
logging.error('error message')
logging.critical('critical message')

"""This is a simple demonstration of the stochastic generative hashing algorithm 
with linear decoder and encoder on MNIST dataset. 

Created by Bo Dai 2016"""

# VAE_stoc_neuron(alpha, dim_input, dim_hidden, batchu_size, batchv_size, learning_rate,
                                                         # max_iter, utrain, rtrain, hvtrain, uvar, umean)
# def VAE_stoc_neuron(alpha, dim_input, dim_hidden, batch_size, learning_rate, max_iter, xtrain, rtrain, hfix，xvar, xmean):
# VAE_stoc_neuron(alpha, dim_input, dim_hidden, batchu_size, learning_rate, max_iter, utrain, rtrain, hvtrain, uvar, umean)
def VAE_stoc_neuron(alpha, gamma, dim_inputu, dim_inputv, dim_hidden, batchu_size, batchv_size, learning_rate, max_iter, utrain,
                    vtrain, rtrain, num_user, num_item, xvaru, xmeanu, xvarv, xmeanv):
    g = tf.Graph()
    dtype = tf.float32
    # VAE_stoc_neuron(alpha, gamma, dim_input_uer,
    #                 dim_input_item, dim_hidden, batch_size,
    #                 learning_rate, max_iter, utrain, vtrain, rtrain,
    #                 hvtrain, uvar, umean, vvar, vmean)
    with g.as_default():
        xu = tf.placeholder(dtype, [None, dim_inputu])  # ：*dim_input
        xv = tf.placeholder(dtype, [None, dim_inputv])  # ：*dim_input
        r = tf.placeholder(dtype, [None, None])  # 形参，执行时候再具体赋值
        nnc = tf.placeholder(tf.int32, [None, None])  # 形参，执行时候再具体赋值
        nnc = tf.cast(nnc, dtype)
        nnz = tf.placeholder(tf.int32)
        nnz = tf.cast(nnz, dtype)
        numu = tf.placeholder(tf.int32)
        numu = tf.cast(numu, dtype)
        numv = tf.placeholder(tf.int32)
        numv = tf.cast(numv, dtype)
        # nnz = tf.constant(1, dtype=tf.float32)
        # nnz = 1
        # define doubly stochastic neuron with gradient by DeFun
        @function.Defun(dtype, dtype, dtype, dtype)
        def DoublySNGrad(logits, epsilon, dprev, dpout):
            prob = 1.0 / (1 + tf.exp(-logits))
            # prob = tf.nn.tanh(logits, name=None)
            yout = (tf.sign(prob - epsilon) + 1.0) / 2.0
            # {-1, 1} coding
            # yout = tf.sign(prob - epsilon)

            # biased
            dlogits = prob * (1 - prob) * (dprev + dpout)
            # dlogits = 1 - tf.nn.tanh(logits, name=None)**2
            depsilon = dprev
            # print('prob: %0.04f, yout: %0.04f, biased: %0.04f, depsilon: %0.04f' % (
            #     np.max(prob)/batch_size, np.max(yout) / batch_size, np.max(dlogits) / batch_size, np.max(depsilon) / batch_size))
            return dlogits, depsilon

        @function.Defun(dtype, dtype, grad_func=DoublySNGrad)
        def DoublySN(logits, epsilon):
            prob = 1.0 / (1 + tf.exp(-logits))
            # prob = tf.nn.tanh(logits, name=None)
            yout = (tf.sign(prob - epsilon) + 1.0) / 2.0
            # print('prob: %0.04f, yout: %0.04f' % (np.max(prob) / batch_size, np.max(yout) / batch_size))
            return yout, prob

        with tf.name_scope('encodeu'):
            wencodeu = tf.Variable(
                tf.random_normal([dim_inputu, dim_hidden], stddev=1.0 / tf.sqrt(float(dim_inputu)), dtype=dtype),
                name='wencodeu')  # stddev是正态分布的标准差（standard deviation）
            bencodeu = tf.Variable(tf.random_normal([dim_hidden], dtype=dtype), name='bencodeu')
            hencodeu = tf.matmul(xu, wencodeu) + bencodeu
            # determinastic output
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
                name='wencodev')  # stddev是正态分布的标准差（standard deviation）
            bencodev = tf.Variable(tf.random_normal([dim_hidden], dtype=dtype), name='bencodev')
            hencodev = tf.matmul(xv, wencodev) + bencodev
            # determinastic output
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

        # print(monitor)
        # nnz = np.count_nonzero(r)
        # sess0 = tf.Session()
        # # sess.run(tf.initialize_all_variables())
        # sess0.run(tf.global_variables_initializer())
        # hu = uout
        # hu = uout.toarray()
        # hv = vout
        hu = tf.where(youtu > 0.5, youtu, tf.zeros_like(youtu))
        hv = tf.where(youtv > 0.5, youtv, tf.zeros_like(youtv))
        # hu[hu == 0] = -1
        # hv[hv == 0] = -1
        rating_loss = tf.nn.l2_loss(tf.multiply(nnc, r - tf.ones(shape=tf.shape(r), dtype=dtype) * .5 -
                                tf.matmul(hu, tf.transpose(hv)) / (2 * dim_hidden))) / nnz  # tf.nn.l2_loss= sum(t ** 2) / 2: t is the standard
        # loss = monitor + alpha * tf.reduce_sum(tf.reduce_sum(yout * tf.log(pout) + (1 - yout) * tf.log(1 - pout))) + beta * tf.nn.l2_loss(wdecode, name=None)
        loss = monitoru + monitorv + gamma * rating_loss + \
               alpha * tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=hencodeu, labels=youtu)) / batchu_size + \
               alpha * tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=hencodev, labels=youtv)) / batchv_size + \
               beta * tf.nn.l2_loss(wdecodeu, name=None) / batchu_size + \
               beta * tf.nn.l2_loss(wdecodev, name=None) / batchv_size



        # print(r.shape)
        # print(x.shape)
        # print(dim_hidden)

        # print(dir(yout))
        # print(dir(hfix))
        # print(yout.nonzero)
        # print(hfix.nonzero)
        # cross entropy loss with sigmoid function:
        # y_pred = sigmoid(logits)
        # loss = -y*np.log(y_pred)-(1-y)*np.log(1-y_pred)
        optimizer = tf.train.AdamOptimizer(learning_rate)  # 算法根据损失函数对每个参数的梯度的一阶矩估计和二阶矩估计动态调整针对于每个参数的学习速率。
        # optimizer = tf.train.RMSPropOptimizer(learning_rate)
        # optimizer = tf.train.AdadeltaOptimizer(learning_rate)
        # optimizer = tf.train.AdagradOptimizer(learning_rate)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)  # 并使用tf.train.Optimizer.minimize 操作来更新系统的训练权重和增加全局训练步骤。按照惯例，这个操作被称为 train_op，是 TensorFlow  会话执行一整个训练步骤必须执行的操作。
        # merged = tf.summary.merge_all()
        sess = tf.Session(graph=g)
        # sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())  #  run 所有global Variable 的 assign op，Initialize all global variables
        loss_total = []
        ratings_loss = []
        users_loss = []
        items_loss = []
        for i in range(max_iter):
            indx = np.random.choice(utrain.shape[0], batchu_size)  # ?? 随机选择有问题？所以不能收敛？因为每一行代表的含义不同！和mnist区别？
            indxv = np.random.choice(vtrain.shape[0], batchv_size)
            # print(indx)
            ubatch = utrain[indx]
            vbatch = vtrain[indxv]
            # print(xbatch.shape)
            # print(rtrain.shape)
            # print(tf.shape(rtrain))
            rbatch0 = rtrain[indx].toarray()
            rbatch = rbatch0[:, indxv]
            nnzbatch = np.count_nonzero(rbatch)
            nncbatch0 = rbatch > 0
            nncbatch = nncbatch0.astype(int)
            # numbatch = rbatch.shape[0]
            numubatch = ubatch.shape[0]
            numvbatch = vbatch.shape[0]
            # print(numubatch)
            # print(numvbatch)

            # nnz = tf.cast(nnz, tf.float32)
            # print(type(nnz))
            # print(dir(nnz))
            # print('nnz: %d' % nnz)
            # print(rbatch.shape)
            um = ubatch.mean(axis=0).astype('float64')  # axis=0代表列，axis=1代表行换句话说
            uv = np.clip(ubatch.var(axis=0), 1e-7, np.inf).astype('float64')

            vm = vbatch.mean(axis=0).astype('float64')  # axis=0代表列，axis=1代表行换句话说
            vv = np.clip(vbatch.var(axis=0), 1e-7, np.inf).astype('float64')
            # scale_para = tf.Variable(tf.constant(xv, dtype=dtype), name="scale_para")   # xvar: 1*784
            # shift_para = tf.Variable(tf.constant(xm, dtype=dtype), name="shift_para")

            _, monitoru_value, monitorv_value, rating_loss_value, loss_value = sess.run([train_op, monitoru, monitorv, rating_loss, loss],
                                                    feed_dict={xu: ubatch, xv: vbatch, r: rbatch, nnz: nnzbatch,
                                                               numu: numubatch, numv: numvbatch, nnc: nncbatch})  # feed_dict: replace x with xbatch

            if i % 20 == 0:
                print('Num iteration: %d Loss: %0.04f, Rating_loss: %0.04f,  Users_Loss: %0.08f, '
                      'Items_Loss: %0.08f' % (i, loss_value, rating_loss_value, monitoru_value, monitorv_value))
                loss_total.append(loss_value)
                ratings_loss.append(rating_loss_value)
                users_loss.append(monitoru_value)
                items_loss.append(monitorv_value)

                # train_err.append(loss_value)
            if i % 300 == 0:
                learning_rate = 0.5 * learning_rate

        node_list = ['youtu', 'poutu', 'youtv', 'poutv', 'uout', 'vout',
                     'wencodeu', 'bencodeu', 'wdecodeu', 'scale_parau', 'shift_parau',
                     'wencodev', 'bencodev', 'wdecodev', 'scale_parav', 'shift_parav']
        sio.savemat('train_loss.mat', {'train_loss': loss_total, 'rating_loss': ratings_loss, 'user_loss': users_loss, 'items_loss': items_loss})

        # plt.plot(t, t, 'r--', t, t ** 2, 'bs', t, t ** 3, 'g^')
        x = np.array(range(len(loss_total)))*20
        lines = plt.plot(x, loss_total, 'r^-', x, ratings_loss, 'go-', x, users_loss, 'yd-', x, items_loss, 'bs-')
        # plt.plot(x, , linewidth=5, label = 'Loss_total', x, ratings_loss, 'g.>', linewidth=5, label='Rating_loss', x, loss_total, 'r-o', linewidth=5, label='Loss_total', )
        # plt.plot(range(len(loss_total)), loss_total, linewidth=3)
        # plt.plot(range(len(loss_total)), loss_total, linewidth=5)
        # plt.plot(range(len(loss_total)), loss_total, linewidth=5)
        # lines = plt.plot(x, y, x, ym1, x, ym2, 'o')
        plt.setp(lines[0], linewidth=3)
        plt.setp(lines[1], linewidth=3)
        plt.setp(lines[2], markersize=3)
        plt.setp(lines[3], markersize=3)

        plt.legend(('Total training loss', 'Ratings loss', 'Loss of users', 'Loss of items'),
                   loc='upper right')
        plt.title('Average training loss')
        plt.show()
        # plt.show()
        t_vars = tf.trainable_variables()
        para_list = {}
        for var in t_vars:
            para_list[var.name] = sess.run(var)
    return g, node_list, para_list, loss_value, monitoru_value, monitorv_value


if __name__ == "__main__":
    # prepare data
    # please replace the dataset with your own directory.
    # mat = hdf5storage.loadmat('test.mat')
    # C:\Users\13184888\Desktop\generative_hashing\SGH\dataset\Yelp\madison
    # trainudata = hdf5storage.loadmat('C:/Users/13184888/Desktop/generative_hashing/SGH/baselines/DropoutNet-master/u_content.mat')
    # traindata = hdf5storage.loadmat('C:/Users/13184888/Desktop/generative_hashing/SGH/baselines/DropoutNet-master/R_train.mat')
    # u_content = hdf5storage.loadmat(
        # 'C:/Users/13184888/Desktop/generative_hashing/SGH/baselines/DropoutNet-master/u_content.mat')
    # v_content = hdf5storage.loadmat(
    #     'C:/Users/13184888/Desktop/generative_hashing/SGH/baselines/DropoutNet-master/v_content.mat')
    # testdata = hdf5storage.loadmat(
        # 'C:/Users/13184888/Desktop/generative_hashing/SGH/baselines/DropoutNet-master/R_test.mat')
    # traindata1 = sio.loadmat('/home/yzhang12/SGH/baselines/DropoutNet-master/R_train_r.mat')

    v_content = sio.loadmat('C:/Users/13184888/Desktop/generative_hashing/SGH/baselines/DropoutNet-master/item_train.mat')
    # v_content = sio.loadmat('/home/yzhang12/SGH/gh_recsys/item_train.mat')
    # hashutrain = hdf5storage.loadmat('C:/Users/13184888/Desktop/generative_hashing/SGH/baselines/DropoutNet-master/B.mat')
    # hashvtrain = hdf5storage.loadmat('C:/Users/13184888/Desktop/generative_hashing/SGH/baselines/DropoutNet-master/D.mat')
    # trainvdata = hdf5storage.loadmat('C:/Users/13184888/Desktop/generative_hashing/SGH/dataset/Yelp/madison/freqv.mat')
    # rating = hdf5storage.loadmat('C:/Users/13184888/Desktop/generative_hashing/SGH/dataset/Yelp/madison/R.mat')
    # hashutrain = hdf5storage.loadmat('C:/Users/13184888/Desktop/generative_hashing/SGH/dataset/Yelp/madison/B.mat')
    # hashvtrain = hdf5storage.loadmat('C:/Users/13184888/Desktop/generative_hashing/SGH/dataset/Yelp/madison/D.mat')
    # traindata = hdf5storage.loadmat('/home/yzhang12/SGH/freq_u.mat')

    # traindata = hdf5storage.loadmat('C:/Users/13184888/Desktop/generative_hashing/SGH/dataset/100k1/Train.mat')
    # testdata = hdf5storage.loadmat('C:/Users/13184888/Desktop/generative_hashing/SGH\dataset/100k1/Test.mat')
    # utrain = np.array(traindata['u_content'])
    vtrain = v_content['item_train']
    del v_content
    vmean = vtrain.mean(axis=0).astype('float32')  # axis=0代表列，axis=1代表行换句话说
    vvar = np.clip(vtrain.var(axis=0), 1e-7, np.inf).astype('float32')
    # xtest = trainvdata['freqv']
    # print(dir(utrain))
    # print(type(utrain))
    # print(utrain.shape)
    traindata_r = sio.loadmat('C:/Users/13184888/Desktop/generative_hashing/SGH/baselines/DropoutNet-master/R_train_r.mat')
    # traindata_r = sio.loadmat('/home/yzhang12/SGH/baselines/DropoutNet-master/R_train_r.mat')
    rtrain = traindata_r['R_train']
    del traindata_r
    # rtrain = rtrain.toarray()
    # print(dir(rtrain))
    # print(type(rtrain))
    # print(rtrain.shape)
    # utrain = np.matrix(utrain)
    # vtrain = np.matrix(vtrain)

    # utrain = utrain.A
    # print(dir(utrain))
    # print(type(utrain))

    # vtrain = vtrain.A
    # rtrain = rtrain.A
    # rtrain = rating['R_test']
    # rtrain = tf.transpose(rtrain)
    # traindata = sio.loadmat('C:/Users/13184888/Desktop/generative_hashing/SGH/baselines/DropoutNet-master/R_train.mat')
    traindata = sio.loadmat('C:/Users/13184888/Desktop/generative_hashing/SGH/gh_recsys/pre_latent_40.mat')

    # traindata = sio.loadmat('/home/yzhang12/SGH/baselines/DropoutNet-master/R_train.mat')
    hutrain = np.sign(traindata['U_factor'])  # can generate from R_train for other dimension
    hvtrain = np.sign(traindata['V_factor'])
    num_user = hutrain.shape[0]
    num_item = hvtrain.shape[0]
    # print(hvtrain.shape)  #(62435, 200)
    # print(hutrain.shape)  #(124961, 200)
    # xtest = testdata['Test']
    del traindata
    # xtrain = traindata['Train'] N
    u_content = sio.loadmat('C:/Users/13184888/Desktop/generative_hashing/SGH/baselines/DropoutNet-master/user_train.mat')
    # u_content = sio.loadmat('/home/yzhang12/SGH/gh_recsys/user_train.mat')
    utrain = u_content['user_train']
    del u_content
    umean = np.mean(utrain, axis=0).astype('float32')  # axis=0代表列，axis=1代表行换句话说
    # tttttt = utrain * utrain.transpose
    # uvar = utrain.dot(utrain).mean - umean.dot(umean).astype('float64')
    uvar = np.clip(utrain.var(axis=0), 1e-7, np.inf).astype(
        'float32')  # type()：返回参数的数据类型 dtype()返回数组中元素的数据类型 astype()对数据类型进行转换. numpy中方差var、协方差cov求法numpy.var
    # clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min。

    # utrain = (utrain - umean) / uvar
    # vtrain = (vtrain - vmean) / vvar
    # algorithm parameters
    # dim_input = 28 * 28
    # dim_input = 937
    dim_input_uer = utrain.shape[1]
    dim_input_item = vtrain.shape[1]
    # dim_input = 35736
    # 38121*35736
    # length of bits
    # dim_hidden = sys.argv[1]
    # dim_hidden = 200
    dim_hidden = 40
    print('dim of hidden variable is %d' % dim_hidden)

    # batchu_size = 200

# for l in range(10):
#     batchv_size = 100 * l + 100
#     for j in range(5):
#         gamma = 0.1 * (j + 1)
#         for k in range(3):
#             learningu_rate = 10**(-k-1)
#             learningv_rate = 10**(-k-1)
    batchu_size = 5000
    batchv_size = 2000
    learning_rate = 0.01
    # beta1 = 0.99
    gamma = 1
    # learningv_rate = 1e-3
    max_iter = 2000
    # max_step = 5
    # gamma = 0.5
    alpha = 1e-4
    beta = 1e-4
    # start training

    print('batchu_size: %d ,batchv_size: %d, gamma: %0.02f, learning_rate: %0.04f' % (batchu_size, batchv_size, gamma, learning_rate))
    # total_err = []
    # for i in range(max_step):
    # update users' parameters
    start_time = time.time()
    g, node_list, para_list, loss, monitoru, monitorv = VAE_stoc_neuron(alpha, gamma, dim_input_uer,
                                                                             dim_input_item, dim_hidden,
                                                                             batchu_size, batchv_size,
                                                                             learning_rate,  max_iter, utrain,
                                                                             vtrain, rtrain, num_user, num_item,
                                                                             uvar, umean, vvar, vmean)
    end_time = (time.time() - start_time)
    # print('loss_total: %0.04f, user_loss: %0.04f, item_loss: %0.04f' % (
    #     train_err, monitoru, monitorv))

    # plt.plot(np.array(range(len(trainu_err))) * 100, np.array(trainu_err) / batchu_size, linewidth=5)
    # plt.show()
    print('Running time: %0.04f s' % end_time)

Wu = para_list['encodeu/wencodeu:0']  # ??
bu = para_list['encodeu/bencodeu:0']
Uu = para_list['decodeu/wdecodeu:0']  # the codebook
shiftu = para_list['scaleu/shift_parau:0']
scaleu = para_list['scaleu/scale_parau:0']
epsilonu = 0.5
trainlogitsu = np.dot(np.array(utrain), Wu) + bu
trainpresu = 1.0 / (1 + np.exp(-trainlogitsu))
hutrain = (np.sign(trainpresu - epsilonu) + 1.0) / 2.0
poten_item = np.matmul(hutrain, Uu) * np.abs(scaleu) + shiftu

Wv = para_list['encodev/wencodev:0']  # ??
bv = para_list['encodev/bencodev:0']
Uv = para_list['decodev/wdecodev:0']  # the codebook
shiftv = para_list['scalev/shift_parav:0']
scalev = para_list['scalev/scale_parav:0']
epsilonv = 0.5
trainlogitsv = np.dot(np.array(vtrain), Wv) + bv
trainpresv = 1.0 / (1 + np.exp(-trainlogitsv))
hvtrain = (np.sign(trainpresv - epsilonv) + 1.0) / 2.0
poten_user = np.matmul(hvtrain, Uv) * np.abs(scalev) + shiftv
hutrain[hutrain < 1] = -1
hvtrain[hvtrain < 1] = -1
hu = hutrain
hv = hvtrain
# hu = tf.where(hutrain > 0.5, hutrain, tf.zeros_like(hutrain))
# hv = tf.where(hvtrain > 0.5, hvtrain, tf.zeros_like(hvtrain))
# xout = tf.matmul(yout, wdecode) * tf.abs(scale_para) + shift_para
# monitoru = tf.nn.l2_loss(poten_item - utrain, name=None)
# loss_totalu = trainu_err + monitoru
# total_err.append(loss_totalu)
filename = './warm/hash' + str(dim_hidden) + 'bit.mat'
sio.savemat(filename, {'hu': hu, 'hv': hv, 'Wu': Wu, 'bu': bu, 'Uu': Uu, 'scaleu': scaleu,
                       'shiftu': shiftu, 'Wv': Wv, 'bv': bv, 'Uv': Uv, 'scalev': scalev, 'shiftv': shiftv})
sio.savemat('gen_user_item.mat', {'gen_item': poten_item, 'gen_user': poten_user})

# predict cold user
cd_user0 = sio.loadmat('./cold_user/R_train_user.mat')
cd_user_test0 = sio.loadmat('./cold_user/R_test_user.mat')
# cd_user_train = cd_user0['u_content_user']
# cd_user_item = cd_user0['v_content_user']
cd_user_train0 = sio.loadmat('./cold_user/cd_user_train.mat')
cd_user_item0 = sio.loadmat('./cold_user/cd_user_item.mat')
cd_user_test = cd_user_test0['R_test_user']
cd_user_train = cd_user_train0['cd_user_train']
cd_user_item = cd_user_item0['cd_user_item']

cdtrainlogitsu = np.dot(np.array(cd_user_train), Wu) + bu
cdtrainpresu = 1.0 / (1 + np.exp(-cdtrainlogitsu))
cdhutrain = (np.sign(cdtrainpresu - epsilonu) + 1.0) / 2.0

cdtrainlogitsv = np.dot(np.array(cd_user_item), Wv) + bv
cdtrainpresv = 1.0 / (1 + np.exp(-cdtrainlogitsv))
cdhvtrain = (np.sign(cdtrainpresv - epsilonv) + 1.0) / 2.0
cdhutrain[cdhutrain < 1] = -1
cdhvtrain[cdhvtrain < 1] = -1
hu = cdhutrain
hv = cdhvtrain
# hu = tf.where(cdhutrain > 0.5, cdhutrain, tf.zeros_like(cdhutrain))
# hv = tf.where(cdhvtrain > 0.5, cdhvtrain, tf.zeros_like(cdhvtrain))

filename = './cold_user/cd_user_' + str(dim_hidden) + 'bit.mat'
sio.savemat(filename, {'hu': hu, 'hv': hv})
# poten_user = np.matmul(hvtrain, Uv) * np.abs(scalev) + shiftv


# predict cold item
cd_item0 = sio.loadmat('./cold_item/R_train_item.mat')
cd_item_test0 = sio.loadmat('./cold_item/R_test_item.mat')
# cd_item_train = cd_item0['v_content_item']
# cd_item_user = cd_item0['u_content_item']
cd_item_train0 = sio.loadmat('./cold_item/cd_item_train.mat')
cd_item_user0 = sio.loadmat('./cold_item/cd_item_user.mat')
cd_item_test = cd_item_test0['R_test_item']
cd_item_train = cd_item_train0['cd_item_train']
cd_item_user = cd_item_user0['cd_item_user']

cdtrainlogitsu = np.dot(np.array(cd_item_user), Wu) + bu
cdtrainpresu = 1.0 / (1 + np.exp(-cdtrainlogitsu))
cdhutrain = (np.sign(cdtrainpresu - epsilonu) + 1.0) / 2.0

cdtrainlogitsv = np.dot(np.array(cd_item_train), Wv) + bv
cdtrainpresv = 1.0 / (1 + np.exp(-cdtrainlogitsv))
cdhvtrain = (np.sign(cdtrainpresv - epsilonv) + 1.0) / 2.0
cdhutrain[cdhutrain < 1] = -1
cdhvtrain[cdhvtrain < 1] = -1
hu = cdhutrain
hv = cdhvtrain
# hu = tf.where(cdhutrain > 0.5, cdhutrain, tf.zeros_like(cdhutrain))
# hv = tf.where(cdhvtrain > 0.5, cdhvtrain, tf.zeros_like(cdhvtrain))

filename = './cold_item/cd_item_' + str(dim_hidden) + 'bit.mat'
sio.savemat(filename, {'hu': hu, 'hv': hv})

# potential customer for cold item

#
# def EuclideanDistances(A, B):
#     BT = B.transpose()
#     vecProd = np.dot(A, BT)
#     SqA = A ** 2
#     sumSqA = np.matrix(np.sum(SqA, axis=1))
#     sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
#
#     SqB = B ** 2
#     sumSqB = np.sum(SqB, axis=1)
#     sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
#     SqED = sumSqBEx + sumSqAEx - 2 * vecProd
#     SqED[SqED < 0] = 0.0
#     ED = np.sqrt(SqED)
#     return ED

#
# poten_user = np.matmul(cdhvtrain, Uu) * np.abs(scaleu) + shiftu
# filename = './cold_item/poten_user.mat'
# sio.savemat(filename, {'poten_user': poten_user})
# # wm = EuclideanDistances(poten_user[1:1000], cd_item_user)  # find potential warm-start user
# wm = cdist(poten_user[range(1000)], cd_item_user, metric='euclidean')
# wm_cos = cosine_similarity(poten_user[range(1000)], cd_item_user)
# # cd = EuclideanDistances(poten_user, cd_user_train)  # find potential cold-start user
# idm = np.argmax(wm, axis=1)  # return the maximum user_id in cd_item_user
# # print(idm.shape)
# idm_cos = np.argmax(wm_cos, axis=1)
# # print(idm_cos.shape)
# # idc = np.argmax(cd, axis=1)  # return the maximum user_id in cd_user_train
# # x = np.array([[1,2,5,0],[5,7,2,3]])
# L = np.argsort(- wm, axis=1)
# id_topk = L[:, range(10)]
# mk = 0
# print(cd_item_test.shape)
# for i in range(1000):
#     # if cd_item_test[idm[i], i] > 0:
#     if np.sum(cd_item_test[id_topk[i, :], i]) > 0:
#         mk += 1
#     else:
#         mk = mk
# pre_pot = mk / 1000
# print('Precision of Potential customer is: %0.04f ' % pre_pot)
#





# import heapq
# nums = [1, 8, 2, 23, 7, -4, 18, 23, 24, 37, 2]
# # 最大的3个数的索引
# max_num_index_list = map(nums.index, heapq.nlargest(3, nums))
# # 最小的3个数的索引
# min_num_index_list = map(nums.index, heapq.nsmallest(3, nums))
# print(list(max_num_index_list))
# print(list(min_num_index_list))



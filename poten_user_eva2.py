import scipy.io as sio
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity

dim_hidden = 40
para = sio.loadmat('./warm/hash' + str(dim_hidden) + 'bit.mat')
# sio.savemat(f, {'hu': hu, 'hv': hv, 'Wu': Wu, 'bu': bu, 'Uu': Uu, 'scaleu': scaleu,
#                        'shiftu': shiftu, 'Wv': Wv, 'bv': bv, 'Uv': Uv, 'scalev': scalev, 'shiftv': shiftv})
Wu = para['Wu']  # ??
bu = para['bu']
Uu = para['Uu']  # the codebook
shiftu = para['shiftu']
scaleu = para['scaleu']
epsilonu = 0.5
# trainlogitsu = np.dot(np.array(utrain), Wu) + bu
# trainpresu = 1.0 / (1 + np.exp(-trainlogitsu))
# hutrain = (np.sign(trainpresu - epsilonu) + 1.0) / 2.0
# poten_item = np.matmul(hutrain, Uu) * np.abs(scaleu) + shiftu

Wv = para['Wv']  # ??
bv = para['bv']
Uv = para['Uv']  # the codebook
shiftv = para['shiftv']
scalev = para['scalev']
epsilonv = 0.5

# cd_user0 = sio.loadmat('./cold_user/R_train_user.mat')
# cd_user_test0 = sio.loadmat('./cold_user/R_test_user.mat')
# # cd_user_train = cd_user0['u_content_user']
# # cd_user_item = cd_user0['v_content_user']
# cd_user_train0 = sio.loadmat('./cold_user/cd_user_train.mat')
# cd_user_item0 = sio.loadmat('./cold_user/cd_user_item.mat')
# cd_user_test = cd_user_test0['R_test_user']
# cd_user_train = cd_user_train0['cd_user_train']
# cd_user_item = cd_user_item0['cd_user_item']

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



# hu = tf.where(cdhutrain > 0.5, cdhutrain, tf.zeros_like(cdhutrain))
# hv = tf.where(cdhvtrain > 0.5, cdhvtrain, tf.zeros_like(cdhvtrain))

# filename = './cold_item/cd_item_' + str(dim_hidden) + 'bit.mat'
# sio.savemat(filename, {'hu': hu, 'hv': hv})


def EuclideanDistances(A, B):
    BT = B.transpose()  #831*500
    vecProd = np.dot(A, BT)  #1000*500
    SqA = A ** 2  #1000*831
    Sn = BT > 0  #831*500
    Asq = np.zeros(SqA.shape[0])
    for i in range(vecProd.shape[1]):
        Asqi = np.dot(SqA, Sn[:, i])   #1000*1
        # print(Asqi.shape)
        Asq = np.c_[Asq, Asqi]
        # print(Asq.shape)
    Asq = np.delete(Asq, 0, axis=1)  # delete the first column
    # SnqA = np.dot(A, BT)
    # sumSqA = np.matrix(np.sum(SqA, axis=1))
    # sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    SqB = B ** 2  #500*831
    sumSqB = np.sum(SqB, axis=1)  #500*1
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))     #500*1000
    SqED = sumSqBEx + Asq - 2 * vecProd
    SqED[SqED < 0] = 0.0
    ED = np.sqrt(SqED)
    return ED

def EuclideanDistances1(A, B):
    BT = B.transpose()  #831*500
    vecProd = np.dot(A, BT)  #1000*500
    SqA = A ** 2  #1000*831
    Sn = BT > 0  #831*500
    Asq = 0
    # print(A.shape)
    # print(BT.shape)
    # print(vecProd.shape)
    # print(SqA.shape)
    # print(Sn.shape)
    # print(Asq.shape)
    for i in range(vecProd.shape[0]):
        Asqi = np.dot(SqA, Sn[:, i])   #1000*1
        # print(Asqi.shape)
        Asq = np.c_[Asq, Asqi]
        # print(Asq.shape)
    Asq = np.delete(Asq, 0, axis=1)  # delete the first column
    # print(Asq.shape)
    # SnqA = np.dot(A, BT)
    # sumSqA = np.matrix(np.sum(SqA, axis=1))
    # sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    SqB = B ** 2  #500*831
    sumSqB = np.sum(SqB, axis=1)  #500*1
    # sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))     #500*1000
    SqED = sumSqB + Asq - 2 * vecProd
    SqED[SqED < 0] = 0.0
    ED = np.sqrt(SqED)
    # print(SqB.shape)
    # print(sumSqB.shape)
    # print(sumSqBEx.shape)
    # print(SqED.shape)
    # print(ED.shape)
    return ED
# warm start items
v_content = sio.loadmat('C:/Users/13184888/Desktop/generative_hashing/SGH/baselines/DropoutNet-master/item_train.mat')
vtrain = v_content['item_train']
u_content = sio.loadmat('C:/Users/13184888/Desktop/generative_hashing/SGH/baselines/DropoutNet-master/user_train.mat')
utrain = u_content['user_train']
Rating_test = sio.loadmat('./R_test.mat')
R_test = Rating_test['R_test']
traindata_r = sio.loadmat('C:/Users/13184888/Desktop/generative_hashing/SGH/baselines/DropoutNet-master/R_train_r.mat')
# traindata_r = sio.loadmat('/home/yzhang12/SGH/baselines/DropoutNet-master/R_train_r.mat')
rtrain = traindata_r['R_train']
del traindata_r

del v_content
vmean = vtrain.mean(axis=0).astype('float32')  # axis=0代表列，axis=1代表行换句话说
vvar = np.clip(vtrain.var(axis=0), 1e-7, np.inf).astype('float32')
trainlogitsv = np.dot(np.array(vtrain), Wv) + bv
trainpresv = 1.0 / (1 + np.exp(-trainlogitsv))
hvtrain = (np.sign(trainpresv - epsilonv) + 1.0) / 2.0
poten_user = np.matmul(hvtrain, Uu) * np.abs(scaleu) + shiftu
# poten_user = np.matmul(cdtrainpresv, Uu) * np.abs(scaleu) + shiftu
# poten_user = np.matmul(cdhvtrain, Uu) * np.abs(scaleu) + shiftu
filename = './warm/poten_user.mat'
sio.savemat(filename, {'poten_user': poten_user})

def pred1_poten_users(tk, nc, R_test, item_id, users_id, poten_user, utrain):
    mi = 0
    nnz = 0
    R_rows = R_test[users_id, :]
    # print(R_test.shape)
    # print(R_rows.shape)
    R_cl = R_rows[:, item_id]
    # print(R_cl.shape)
    # print(R_cl)
    nnz0 = np.sum(R_cl > 0)
    # print(nnz0)
    # print(R_rows[:, item_id].shape)
    for i in range(item_id.shape[0]):
        pos_id0 = np.argwhere(R_test[users_id, item_id[i]] > 0)  # positive users id
        pos_id1 = pos_id0[:, 1]
        pos_id = users_id[pos_id1]
        neg_id = np.setdiff1d(users_id, pos_id)
        kj = 0
        for j in range(pos_id.shape[0]):
            rnd_id = np.random.choice(range(neg_id.shape[0]), nc)
            rnd_neg_id = neg_id[rnd_id]  # randomly choose nc negative users
            # print(rnd_neg_id)
            # print(rnd_neg_id.shape)
            # print(pos_id[j])
            rows = np.append(pos_id[j], rnd_neg_id)
            # print(poten_user[item_id[i]].shape)
            # print(poten_user[item_id[i].shape[1])
            # print(utrain[rows, :].shape)
            # print(type(poten_user[item_id[i]]))
            # print(type(utrain[rows, :]))
            wm = EuclideanDistances1(np.array(poten_user[item_id[i]]), utrain[rows, :])
            L = np.argsort(wm, axis=1)
            id_topk = L[:, range(tk)]
            if np.sum(R_test[rows[id_topk], item_id[i]]) > 0:
               kj += 1
        # print(kj)
        mi += kj
        nnz += pos_id.shape[0]
        # print(mi)
        # print(nnz)
    pre_pot = mi / nnz
    print('Precision of Potential customer is: %0.08f, The number of hit: %d, The number of positive ratings: %d,'
      ' the number of original positive ratings: %d' % (pre_pot, mi, nnz, nnz0))
    return pre_pot






# wm = cdist(poten_user[range(1000)], utrain[range(nu)], metric='euclidean')  # cold-start: utrain -> cold-item-user
# wm = cdist(poten_user[range(1000)], utrain, metric='correlation')
# wm = -cdist(poten_user[range(1000)], utrain, metric='cosine')
# wm = cdist(poten_user[range(1000)], utrain, metric='jaccard')

# wm_cos = cosine_similarity(poten_user[range(1000)], cd_item_user)
# cd = EuclideanDistances(poten_user, cd_user_train)  # find potential cold-start user
# idm = np.argmax(wm, axis=1)  # renurn the maximum user_id in cd_item_user
# print(idm.shape)
# idm_cos = np.argmax(wm_cos, axis=1)
# print(idm_cos.shape)
# idc = np.argmax(cd, axis=1)  # return the maximum user_id in cd_user_train
# x = np.array([[1,2,5,0],[5,7,2,3]])
# L = np.argsort(wm, axis=1)
# print(poten_user[item_id].shape)
# print(utrain[users_id].shape)
# print(wm.shape)
# print(L.shape)
# id_topk = L[:, range(tk)]
# print(id_topk[1, :])
# print(users_id[id_topk[1, :]])
# print(users_id)
# print(id_topk.shape)

# L1 = np.argsort(- wm_cos, axis=1)
# id_topk_cos = L1[:, range(10)]
# prec1 = pred_poten_users1(tk, nc, R_test, item_id, users_id, poten_user, utrain)
def pred2_poten_users(tk, nc, R_test, item_id, users_id, poten_user, utrain):
    wm = EuclideanDistances(poten_user[item_id], utrain[users_id])  # find potential warm-start user
    L = np.argsort(wm, axis=1)
    id_topk = L[:, range(tk)]
    # print(id_topk)
    nnz = 0
    mk = 0
    rat = 0
    for i in range(item_id.shape[0]):
        # if cd_item_test[idm[i], i] > 0:
        # if np.sum(cd_item_test[id_topk_cos[i, :], i]) > 0:
        # if np.sum(cd_item_test[id_topk_cos[i, :], i]) > 0:
        #
        # print(users_id[id_topk[i, :]])
        # print(item_id[i])
        if np.sum(R_test[users_id, item_id[i]]) > nr and np.sum(R_test[users_id[id_topk[i, :]], item_id[i]]) > 0:  # for cold-start: R_test -> cd_item_test
        # if np.sum(R_test[users_id, item_id[i]]) > nr and np.sum(R_test[id_topk[i, :], item_id[i]]) > 0:
            mk += 1
            nnz += 1
        elif np.sum(R_test[users_id, item_id[i]]) > nr:
            nnz += 1
        # print(mk)
        # print(nnz)
        # nz = np.sum(R_test[id_topk[i, :], item_id[i]])
        # nnz = nnz + (nz / tk)
        # mk += nz
        # print(mk)
        # print(nnz)
        # nnz += k
        # if np.sum(cd_item_test[:, i] > 0:
        #     nnz += 1
        # nnz = sum(n)
        # nnz = sum(sum(ratings != 0))w
    pre_pot = mk / nnz
    # pre_pot = nnz / (item_id.shape[0])
    print('Precision of Potential customer is: %0.08f, The number of hit: %d, The number of test items: %0.02f' % (pre_pot, mk, nnz))
    return pre_pot

# for


for nrr in range(5):
    nu = 5000  # test users: positive & negative users
    nr = 5 * (6 + nrr)  # original positive users
    for u in range(6):
        tk = 5 * (u + 1)  # potential users
        nc = 1000  # randomly choose negative users
        np.random.seed(2)
        users_id = np.random.choice(range(utrain.shape[0]), nu)
        num_ratings = np.sum(R_test[users_id], axis=0)
        # print(num_ratings.shape)
        item_id0 = np.argwhere(num_ratings > nr)
        # print(item_id0.shape[0])
        # print(item_id0[:, 1])
        item_id = item_id0[:, 1]
        # print(item_id[1])
        # print(num_ratings[:, item_id[1]])
        prec2 = pred2_poten_users(tk, nc, R_test, item_id, users_id, poten_user, utrain)
        prec1 = pred1_poten_users(tk, nc, R_test, item_id, users_id, poten_user, utrain)
        # print('Precision_method1: %0.04f, Precision_method2: %0.04f' % (prec1, prec2))
        filename = 'eva_poten_results.txt'
        with open(filename, 'a') as f:  # 'a'表示append,即在原来文件内容后继续写数据（不清楚原有数据）
            f.write('dim_hidden: %d, No. rgl users: %d, No. poten users: %d, precision: %0.04f, precision: %0.04f' % (dim_hidden, nr, tk, prec1, prec2))
            # f.write(prec1, prec2)
            f.write('\n')

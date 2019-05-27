import scipy.io as sio
import numpy as np


def EuclideanDistances(A, B):
    BT = B.transpose()  #831*500
    vecProd = np.dot(A, BT)  #1000*500
    SqA = A ** 2  #1000*831
    Sn = BT > 0  #831*500
    Asq = np.zeros(SqA.shape[0])
    for i in range(vecProd.shape[1]):
        Asqi = np.dot(SqA, Sn[:, i])   #1000*1
        Asq = np.c_[Asq, Asqi]
        # print(Asq.shape)
    Asq = np.delete(Asq, 0, axis=1)  # delete the first column
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
    for i in range(vecProd.shape[0]):
        Asqi = np.dot(SqA, Sn[:, i])   #1000*1
        Asq = np.c_[Asq, Asqi]
    Asq = np.delete(Asq, 0, axis=1)  # delete the first column
    SqB = B ** 2  #500*831
    sumSqB = np.sum(SqB, axis=1)  #500*1
    SqED = sumSqB + Asq - 2 * vecProd
    SqED[SqED < 0] = 0.0
    ED = np.sqrt(SqED)
    return ED


def pred1_poten_users(tk, nc, R_test, item_id, users_id, poten_user, utrain):
    mi = 0
    nnz = 0
    R_rows = R_test[users_id, :]
    R_cl = R_rows[:, item_id]
    nnz0 = np.sum(R_cl > 0)
    for i in range(item_id.shape[0]):
        pos_id0 = np.argwhere(R_test[users_id, item_id[i]] > 0)  # positive users id
        pos_id1 = pos_id0[:, 1]
        pos_id = users_id[pos_id1]
        neg_id = np.setdiff1d(users_id, pos_id)
        kj = 0
        for j in range(pos_id.shape[0]):
            rnd_id = np.random.choice(range(neg_id.shape[0]), nc)
            rnd_neg_id = neg_id[rnd_id]  # randomly choose nc negative users
            rows = np.append(pos_id[j], rnd_neg_id)
            wm = EuclideanDistances1(np.array(poten_user[item_id[i]]), utrain[rows, :])
            L = np.argsort(wm, axis=1)
            id_topk = L[:, range(tk)]
            if np.sum(R_test[rows[id_topk], item_id[i]]) > 0:
               kj += 1
        mi += kj
        nnz += pos_id.shape[0]
    pre_pot = mi / nnz
    print('Precision of Potential customer is: %0.08f, The number of hit: %d, The number of positive ratings: %d,'
      ' the number of original positive ratings: %d' % (pre_pot, mi, nnz, nnz0))
    return pre_pot


def pred2_poten_users(tk, nc, R_test, item_id, users_id, poten_user, utrain):
    wm = EuclideanDistances(poten_user[item_id], utrain[users_id])  # find potential warm-start user
    L = np.argsort(wm, axis=1)
    id_topk = L[:, range(tk)]
    nnz = 0
    mk = 0
    for i in range(item_id.shape[0]):
        if np.sum(R_test[users_id, item_id[i]]) > nr and np.sum(R_test[users_id[id_topk[i, :]], item_id[i]]) > 0:  # for cold-start: R_test -> cd_item_test
            mk += 1
            nnz += 1
        elif np.sum(R_test[users_id, item_id[i]]) > nr:
            nnz += 1
    pre_pot = mk / nnz
    print('Precision of Potential customer is: %0.08f, The number of hit: %d, The number of test items: %0.02f' % (pre_pot, mk, nnz))
    return pre_pot


dim_hidden = 200
para = sio.loadmat('./warm/hash' + str(dim_hidden) + 'bit.mat')
Wu = para['Wu']
bu = para['bu']
Uu = para['Uu']  # the codebook
shiftu = para['shiftu']
scaleu = para['scaleu']
epsilonu = 0.5

Wv = para['Wv']
bv = para['bv']
Uv = para['Uv']  # the codebook
shiftv = para['shiftv']
scalev = para['scalev']
epsilonv = 0.5

# # warm start items
# v_content = sio.loadmat('item_train.mat')
# vtrain = v_content['item_train']
# u_content = sio.loadmat('user_train.mat')
# utrain = u_content['user_train']
# Rating_test = sio.loadmat('R_test.mat')
# R_test = Rating_test['R_test']


# predict cold item
# cd_item0 = sio.loadmat('./cold_item/R_train_item.mat')
cd_item_test0 = sio.loadmat('./cold_item/R_test_item.mat')
# cd_item_train = cd_item0['v_content_item']
# cd_item_user = cd_item0['u_content_item']
cd_item_train0 = sio.loadmat('./cold_item/cd_item_train.mat')
cd_item_user0 = sio.loadmat('./cold_item/cd_item_user.mat')
cd_item_test = cd_item_test0['R_test_item']
cd_item_train = cd_item_train0['cd_item_train']
cd_item_user = cd_item_user0['cd_item_user']
R_test = cd_item_test
vtrain = cd_item_train  # for cold start
utrain = cd_item_user  # for cold start
trainlogitsv = np.dot(np.array(vtrain), Wv) + bv
trainpresv = 1.0 / (1 + np.exp(-trainlogitsv))
hvtrain = (np.sign(trainpresv - epsilonv) + 1.0) / 2.0
poten_user = np.matmul(hvtrain, Uu) * np.abs(scaleu) + shiftu
# poten_user = np.matmul(cdtrainpresv, Uu) * np.abs(scaleu) + shiftu
# poten_user = np.matmul(cdhvtrain, Uu) * np.abs(scaleu) + shiftu
filename = './warm/poten_user.mat'
sio.savemat(filename, {'poten_user': poten_user})

# for nrr in range(5):
nu = 5000  # test users: positive & negative users
nr = 5 * 4  # original positive users
# for u in range(6):
tk = 5 * (3 + 1)  # potential users
nc = 1000  # randomly choose negative users
users_id = np.random.choice(range(utrain.shape[0]), nu)
num_ratings = np.sum(R_test[users_id], axis=0)
item_id0 = np.argwhere(num_ratings > nr)
item_id = item_id0[:, 1]
# prec2 = pred2_poten_users(tk, nc, R_test, item_id, users_id, poten_user, utrain)
prec1 = pred1_poten_users(tk, nc, R_test, item_id, users_id, poten_user, utrain)
filename = 'eva_poten_results.txt'
with open(filename, 'a') as f:
    f.write('dim_hidden: %d, No. rgl users: %d, No. poten users: %d, precision: %0.04f' % (dim_hidden, nr, tk, prec1))
    f.write('\n')

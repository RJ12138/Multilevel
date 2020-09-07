import time
import numpy as np
import tensorflow as tf
import os

from network import MyNetwork
from utils import process
from graph_coarsen import graph_coarsen

checkpt_file = 'pre_trained/cora/mod_cora.ckpt'

dataset = 'cora'

# training params
batch_size = 1
nb_epochs = 100000
patience = 100
lr = 0.005  # learning rate
l2_coef = 0.0005  # weight decay
hid_units = [8] # numbers of hidden units per each attention head in each layer
n_heads = [8, 1] # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = MyNetwork()
clustering_size_firstlevel = 5

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# gpu_options.allow_growth = True

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(dataset)
features, spars = process.preprocess_features(features)

print(type(adj))
print(type(features))
print(type(y_train))
print(type(y_val))


nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = y_train.shape[1]

adj = adj.toarray()

features = features[np.newaxis]
adj = adj[np.newaxis]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)

out_sz = 64
n_head = 8

np.set_printoptions(threshold=np.inf)
# print(np.max(np.sum(adj[0], axis=1)))

# clustering ##
# clustering_mat_0, convergent_bias_0, btw_super_bias_0, divergent_bias_0, adj_1 = process.sklearn_clustering2(adj, features, clustering_size_firstlevel)
# n_cluster = clustering_mat_0.shape[1]
# super_feat_zero = np.zeros((batch_size, n_cluster, out_sz * n_head))

# clustering_mat_1, convergent_bias_1, btw_super_bias_1, divergent_bias_1, adj_2 = process.sklearn_clustering2(adj_1, None, clustering_size_firstlevel)
# n_cluster_1 = clustering_mat_1.shape[1]
# n_node_1 = n_cluster
# print("n_cluster_1, ", n_cluster_1)
# super_feat_1_zero = np.zeros((batch_size, n_cluster_1, out_sz * n_head))



# clustering_mat = [clustering_mat_0, clustering_mat_1]
# convergent_bias = [convergent_bias_0, convergent_bias_1]
# btw_super_bias = [btw_super_bias_0, btw_super_bias_1]
# divergent_bias = [divergent_bias_0, divergent_bias_1]
## ---------- ##

## coarsening ##
clustering_mat, convergent_bias, btw_super_bias, divergent_bias = graph_coarsen(adj[0], features[0].A)
n_cluster = clustering_mat[3].shape[1]
n_cluster_1 = clustering_mat[4].shape[1]
super_feat_zero = np.zeros((batch_size, n_cluster, out_sz * n_head))
super_feat_1_zero = np.zeros((batch_size, n_cluster_1, out_sz * n_head))
n_node_1 = n_cluster
## clustering ##

with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
        spft_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, n_cluster, out_sz * n_head))
        conv_bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, n_cluster, nb_nodes + n_cluster))
        btw_bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, n_cluster, n_cluster))
        div_bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, n_cluster + nb_nodes))
        cluster_mat_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, n_cluster, nb_nodes))

#         spft_1_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, n_cluster_1, out_sz * n_head))
#         conv_bias_1_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, n_cluster_1, n_node_1 + n_cluster_1))
#         btw_bias_1_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, n_cluster_1, n_cluster_1))
#         div_bias_1_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, n_node_1, n_cluster_1 + n_node_1))
#         cluster_mat_1_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, n_cluster_1, n_node_1))

        lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes))
        msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))
        attn_drop = tf.placeholder(dtype=tf.float32, shape=())
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
        is_train = tf.placeholder(dtype=tf.bool, shape=())

    # 注意 low_level_dim 暂时指的是整个的，而high_level_dim 暂时指的是单个attention head 的
    satlt_feat, super_feat = model.inference(ftr_in, spft_in, 0, n_head, nb_classes, ffd_drop, attn_drop, 
        cluster_mat_in, conv_bias_in, btw_bias_in, div_bias_in, 
#         spft_1_in, cluster_mat_1_in, conv_bias_1_in, btw_bias_1_in, div_bias_1_in, 
        lower_level_dim=128, level_dim=out_sz)
    log_resh = tf.reshape(satlt_feat, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])
    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)

    train_op = model.training(loss, lr, l2_coef)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)

        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0

        for v in tf.trainable_variables():
            print(v.name)
        print("trainable_variables names finished")

        for epoch in range(nb_epochs):
            # if epoch % 1000 == 0:
            # print(epoch)

#             for i in range(n_head):
#                 tensor_name = 'level0_attn' + str(i) + '_w2/kernel:0'
#                 varvar = sess.graph.get_tensor_by_name(tensor_name)
#                 print("tensor_name")

            tr_step = 0
            tr_size = features.shape[0]

            while tr_step * batch_size < tr_size:
                _, loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy],
                    feed_dict={
                        ftr_in: features[tr_step*batch_size:(tr_step+1)*batch_size],
                        spft_in: super_feat_zero[tr_step*batch_size:(tr_step+1)*batch_size],
                        cluster_mat_in: clustering_mat[3][tr_step*batch_size:(tr_step+1)*batch_size],
                        conv_bias_in: convergent_bias[3][tr_step*batch_size:(tr_step+1)*batch_size],
                        btw_bias_in: btw_super_bias[3][tr_step*batch_size:(tr_step+1)*batch_size],
                        div_bias_in: divergent_bias[3][tr_step*batch_size:(tr_step+1)*batch_size],

#                         spft_1_in: super_feat_1_zero[tr_step*batch_size:(tr_step+1)*batch_size],
# #                         cluster_mat_1_in: clustering_mat[4][tr_step*batch_size:(tr_step+1)*batch_size],
#                         conv_bias_1_in: convergent_bias[4][tr_step*batch_size:(tr_step+1)*batch_size],
#                         btw_bias_1_in: btw_super_bias[4][tr_step*batch_size:(tr_step+1)*batch_size],
#                         div_bias_1_in: divergent_bias[4][tr_step*batch_size:(tr_step+1)*batch_size],

                        lbl_in: y_train[tr_step*batch_size:(tr_step+1)*batch_size],
                        msk_in: train_mask[tr_step*batch_size:(tr_step+1)*batch_size],
                        is_train: True,
                        attn_drop: 0.6, ffd_drop: 0.6})
                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                tr_step += 1

            vl_step = 0
            vl_size = features.shape[0]

            while vl_step * batch_size < vl_size:
                loss_value_vl, acc_vl = sess.run([loss, accuracy],
                    feed_dict={
                        ftr_in: features[vl_step*batch_size:(vl_step+1)*batch_size],
                        spft_in: super_feat_zero[vl_step*batch_size:(vl_step+1)*batch_size],
                        cluster_mat_in: clustering_mat[3][vl_step*batch_size:(vl_step+1)*batch_size],
                        conv_bias_in: convergent_bias[3][vl_step*batch_size:(vl_step+1)*batch_size],
                        btw_bias_in: btw_super_bias[3][vl_step*batch_size:(vl_step+1)*batch_size],
                        div_bias_in: divergent_bias[3][vl_step*batch_size:(vl_step+1)*batch_size],

#                         spft_1_in: super_feat_1_zero[vl_step*batch_size:(vl_step+1)*batch_size],
# #                         cluster_mat_1_in: clustering_mat[4][vl_step*batch_size:(vl_step+1)*batch_size],
#                         conv_bias_1_in: convergent_bias[4][vl_step*batch_size:(vl_step+1)*batch_size],
#                         btw_bias_1_in: btw_super_bias[4][vl_step*batch_size:(vl_step+1)*batch_size],
#                         div_bias_1_in: divergent_bias[4][vl_step*batch_size:(vl_step+1)*batch_size],

                        # bias_in: biases[vl_step*batch_size:(vl_step+1)*batch_size],
                        lbl_in: y_val[vl_step*batch_size:(vl_step+1)*batch_size],
                        msk_in: val_mask[vl_step*batch_size:(vl_step+1)*batch_size],
                        is_train: False,
                        attn_drop: 0.0, ffd_drop: 0.0})
                val_loss_avg += loss_value_vl
                val_acc_avg += acc_vl
                vl_step += 1

            print('The %d epoch: Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                    (epoch, train_loss_avg/tr_step, train_acc_avg/tr_step,
                    val_loss_avg/vl_step, val_acc_avg/vl_step))

            if val_acc_avg/vl_step >= vacc_mx or val_loss_avg/vl_step <= vlss_mn:
                if val_acc_avg/vl_step >= vacc_mx and val_loss_avg/vl_step <= vlss_mn:
                    vacc_early_model = val_acc_avg/vl_step
                    vlss_early_model = val_loss_avg/vl_step
                    saver.save(sess, checkpt_file)
                vacc_mx = np.max((val_acc_avg/vl_step, vacc_mx))
                vlss_mn = np.min((val_loss_avg/vl_step, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == patience:
                    print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                    print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                    break

            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

        saver.restore(sess, checkpt_file)

        ts_size = features.shape[0]
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0

        while ts_step * batch_size < ts_size:
            loss_value_ts, acc_ts = sess.run([loss, accuracy],
                feed_dict={
                    ftr_in: features[ts_step*batch_size:(ts_step+1)*batch_size],
                    spft_in: super_feat_zero[ts_step*batch_size:(ts_step+1)*batch_size],
                    cluster_mat_in: clustering_mat[3][ts_step*batch_size:(ts_step+1)*batch_size],
                    conv_bias_in: convergent_bias[3][ts_step*batch_size:(ts_step+1)*batch_size],
                    btw_bias_in: btw_super_bias[3][ts_step*batch_size:(ts_step+1)*batch_size],
                    div_bias_in: divergent_bias[3][ts_step*batch_size:(ts_step+1)*batch_size],

#                     spft_1_in: super_feat_1_zero[ts_step*batch_size:(ts_step+1)*batch_size],
# #                     cluster_mat_1_in: clustering_mat[4][ts_step*batch_size:(ts_step+1)*batch_size],
#                     conv_bias_1_in: convergent_bias[4][ts_step*batch_size:(ts_step+1)*batch_size],
#                     btw_bias_1_in: btw_super_bias[4][ts_step*batch_size:(ts_step+1)*batch_size],
#                     div_bias_1_in: divergent_bias[4][ts_step*batch_size:(ts_step+1)*batch_size],

                    lbl_in: y_test[ts_step*batch_size:(ts_step+1)*batch_size],
                    msk_in: test_mask[ts_step*batch_size:(ts_step+1)*batch_size],
                    is_train: False,
                    attn_drop: 0.0, ffd_drop: 0.0})
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1

        print('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step)

        sess.close()

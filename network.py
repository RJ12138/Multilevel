import tensorflow as tf
import numpy as np

step_type_list = ['convergent', 'btw_super', 'divergent']

class MyNetwork():

    def __init__(self):
        print("MyNetwork created")

    def inference(self, inputs, super_inputs, level_idx, n_head, n_classes, attn_drop, ffd_drop, clustering_mat, convergent_bias, btw_super_bias, divergent_bias, super_inputs_1, clustering_mat_1, convergent_bias_1, btw_super_bias_1,   divergent_bias_1, lower_level_dim, level_dim=256, activation=tf.nn.elu):
        # this dim is the size of one attention super_inputs_1, clustering_mat_1, convergent_bias_1, btw_super_bias_1, divergent_bias_1,
        satlt_feat = inputs
        super_feat = super_inputs
        # super_feat = satlt_feat
        super_feat = self.convergent_update(satlt_feat, super_feat, level_idx, n_head, attn_drop, ffd_drop, convergent_bias, lower_level_dim, level_dim)
        super_feat = self.btw_super_update(satlt_feat, super_feat, level_idx+1, n_head, attn_drop, ffd_drop, btw_super_bias, lower_level_dim, level_dim)
        
#         super_feat_1 = self.convergent_update(super_feat, super_inputs_1, level_idx+1, n_head, attn_drop, ffd_drop, convergent_bias_1, lower_level_dim, level_dim)
#         super_feat_1 = self.btw_super_update(super_feat, super_feat_1, level_idx+1, n_head*2, attn_drop, ffd_drop, btw_super_bias_1, lower_level_dim, level_dim)
#         super_feat = self.divergent_update(super_feat, super_feat_1, level_idx+1, 1, attn_drop, ffd_drop, divergent_bias_1, lower_level_dim, level_dim)
        
        satlt_feat = self.divergent_update(satlt_feat, super_feat, level_idx+2, 1, attn_drop, ffd_drop, divergent_bias, lower_level_dim, level_dim)
        satlt_feat = tf.layers.conv1d(satlt_feat, n_classes, 1, use_bias=False)

        return satlt_feat, super_feat

    def convergent_update(self, satlt, supernode, level_idx, n_head, attn_drop, ffd_drop, bias_mat, lower_level_dim, level_dim=256, 
        activation=tf.nn.elu, iscoarsen=True):
        attns = []
        for j in range(n_head):
            attns.append(self.attn_head(satlt, supernode, 'convergent', lower_level_dim, level_dim, bias_mat, 
                activation=activation, name='level'+str(level_idx)+'_attn'+str(j), in_drop=ffd_drop, coef_drop=attn_drop))
        super_feat = tf.concat(attns, axis=-1)

        return super_feat

    def btw_super_update(self, satlt, supernode, level_idx, n_head, attn_drop, ffd_drop, bias_mat, lower_level_dim, level_dim=256, 
        activation=tf.nn.elu, iscoarsen=True):
        attns = []
        for j in range(n_head):
            attns.append(self.attn_head(None, supernode, 'btw_super', lower_level_dim, level_dim, bias_mat, 
                activation=activation, name='level'+str(level_idx)+'_attn'+str(j), in_drop=ffd_drop, coef_drop=attn_drop))
        super_feat = tf.concat(attns, axis=-1)
        if n_head == 1:
            super_feat = tf.add_n(attns)
#             print("super_feat:", super_feat.shape)

        return super_feat

    def divergent_update(self, satlt, supernode, level_idx, n_head, attn_drop, ffd_drop, bias_mat, lower_level_dim, level_dim=256, 
        activation=tf.nn.elu, iscoarsen=True):
        # this dim is the size of one attention
        attns = []
        for j in range(n_head):
            attns.append(self.attn_head(satlt, supernode, 'divergent', lower_level_dim, level_dim, bias_mat, 
                activation=activation, name='level'+str(level_idx)+'_attn'+str(j), in_drop=ffd_drop, coef_drop=attn_drop))
        satlt_feat = tf.concat(attns, axis=-1)
        # satlt_feat = tf.layers.conv1d(satlt_feat, lower_level_dim, 1, use_bias=False, name='level'+str(level_idx)+'_divg', reuse=tf.AUTO_REUSE)
        # satlt_feat = tf.layers.conv1d(satlt_feat, n_classes, 1, use_bias=False)

        return satlt_feat

    ## base functions
    def loss(self, logits, labels, nb_classes, class_weights):
        sample_wts = tf.reduce_sum(tf.multiply(tf.one_hot(labels, nb_classes), class_weights), axis=-1)
        xentropy = tf.multiply(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits), sample_wts)
        return tf.reduce_mean(xentropy, name='xentropy_mean')

    def training(self, loss, lr, l2_coef):
        # weight decay
        vars = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not
                           in ['bias', 'gamma', 'b', 'g', 'beta']]) * l2_coef

        # optimizer
        opt = tf.train.AdamOptimizer(learning_rate=lr)

        # training op
        train_op = opt.minimize(loss+lossL2)
        
        return train_op

    def preshape(self, logits, labels, nb_classes):
        new_sh_lab = [-1]
        new_sh_log = [-1, nb_classes]
        log_resh = tf.reshape(logits, new_sh_log)
        lab_resh = tf.reshape(labels, new_sh_lab)
        return log_resh, lab_resh

    def confmat(self, logits, labels):
        preds = tf.argmax(logits, axis=1)
        return tf.confusion_matrix(labels, preds)

    def masked_softmax_cross_entropy(self, logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)

    def masked_sigmoid_cross_entropy(self, logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        labels = tf.cast(labels, dtype=tf.float32)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        loss=tf.reduce_mean(loss,axis=1)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)

    def masked_accuracy(self, logits, labels, mask):
        """Accuracy with masking."""
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return tf.reduce_mean(accuracy_all)

    def micro_f1(self, logits, labels, mask):
        """Accuracy with masking."""
        predicted = tf.round(tf.nn.sigmoid(logits))

        # Use integers to avoid any nasty FP behaviour
        predicted = tf.cast(predicted, dtype=tf.int32)
        labels = tf.cast(labels, dtype=tf.int32)
        mask = tf.cast(mask, dtype=tf.int32)

        # expand the mask so that broadcasting works ([nb_nodes, 1])
        mask = tf.expand_dims(mask, -1)
        
        # Count true positives, true negatives, false positives and false negatives.
        tp = tf.count_nonzero(predicted * labels * mask)
        tn = tf.count_nonzero((predicted - 1) * (labels - 1) * mask)
        fp = tf.count_nonzero(predicted * (labels - 1) * mask)
        fn = tf.count_nonzero((predicted - 1) * labels * mask)

        # Calculate accuracy, precision, recall and F1 score.
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fmeasure = (2 * precision * recall) / (precision + recall)
        fmeasure = tf.cast(fmeasure, tf.float32)
        return fmeasure

    def attn_head(self, satlt, supernode, step_type, lower_level_sz, higher_level_sz, bias_mat, activation, name, in_drop=0.6, coef_drop=0.6, residual=False):
    
        if step_type == 'convergent':
            with tf.name_scope('my_convergent_attn'):

                lineartrans_name = [name+'_w1', name+'_w2']
                a_name = [name+'_a1', name+'_a2']
                bias_name = name+'_convergent_bias'
#                 print('type satlt: ', type(satlt))
                
                if in_drop != 0.0:
                    satlt = tf.nn.dropout(satlt, 1.0 - in_drop)
                    supernode = tf.nn.dropout(supernode, 1.0 - in_drop)

                print("supernode shape: ", supernode.shape)

                satlt_fts = tf.layers.conv1d(satlt, higher_level_sz, 1, use_bias=False, name=lineartrans_name[0], reuse=tf.AUTO_REUSE)
                supernode_fts = tf.layers.conv1d(supernode, higher_level_sz, 1, use_bias=False, name=lineartrans_name[1], reuse=tf.AUTO_REUSE)
                # simplest self-attention possible
                mix_fts = tf.concat([satlt_fts, supernode_fts], axis=1)
                f_1 = tf.layers.conv1d(satlt_fts, 1, 1, name=a_name[0], reuse=tf.AUTO_REUSE)
                f_2 = tf.layers.conv1d(supernode_fts, 1, 1, name=a_name[1], reuse=tf.AUTO_REUSE) # 这里的f_2 反正是0，不如直接去掉
                f_mix = tf.concat([f_1, f_2], axis=1)
                logits = f_2 + tf.transpose(f_mix, [0, 2, 1])
                # print(logits.shape, mix_fts.shape)
                coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

                if in_drop != 0.0:
                    mix_fts = tf.nn.dropout(mix_fts, 1.0 - in_drop)
                if coef_drop != 0.0:
                    coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)

                vals = tf.matmul(coefs, mix_fts)
                ret = tf.contrib.layers.bias_add(vals) # , scope=bias_name, reuse=tf.AUTO_REUSE)

        elif step_type == 'btw_super':
            with tf.name_scope('my_btw_super_attn'):

                lineartrans_name = [name+'_w2', name+'_w2']
                a_name = [name+'_a1', name+'_a2']
                bias_name = name+'_btw_bias'
                
                if in_drop != 0.0:
                    supernode = tf.nn.dropout(supernode, 1.0 - in_drop)
                print("supernode sjape: ", supernode.shape)
                supernode_fts = tf.layers.conv1d(supernode, higher_level_sz, 1, use_bias=False, name=lineartrans_name[1], reuse=tf.AUTO_REUSE)
                # simplest self-attention possible
                f_1 = tf.layers.conv1d(supernode_fts, 1, 1, name=a_name[0], reuse=tf.AUTO_REUSE)
                f_2 = tf.layers.conv1d(supernode_fts, 1, 1, name=a_name[1], reuse=tf.AUTO_REUSE)
                logits = f_1 + tf.transpose(f_2, [0, 2, 1])
                coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

                if in_drop != 0.0:
                    supernode_fts = tf.nn.dropout(supernode_fts, 1.0 - in_drop)
                if coef_drop != 0.0:
                    coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)

                vals = tf.matmul(coefs, supernode_fts)
                ret = tf.contrib.layers.bias_add(vals, scope=bias_name, reuse=tf.AUTO_REUSE)

        else:
            with tf.name_scope('my_divergent_attn'):

                lineartrans_name = [name+'_w1', name+'_w2']
                a_name = [name+'_a3', name+'_a2']
                bias_name = name+'_divergent_bias'
                
                if in_drop != 0.0:
                    satlt = tf.nn.dropout(satlt, 1.0 - in_drop)
                    supernode = tf.nn.dropout(supernode, 1.0 - in_drop)

                satlt_fts = tf.layers.conv1d(satlt, higher_level_sz, 1, use_bias=False, name=lineartrans_name[0], reuse=tf.AUTO_REUSE)
                supernode_fts = tf.layers.conv1d(supernode, higher_level_sz, 1, use_bias=False, name=lineartrans_name[1], reuse=tf.AUTO_REUSE)
                # simplest self-attention possible
                mix_fts = tf.concat([satlt_fts, supernode_fts], axis=1)
                f_1 = tf.layers.conv1d(satlt_fts, 1, 1, name=a_name[0], reuse=tf.AUTO_REUSE)
                f_2 = tf.layers.conv1d(supernode_fts, 1, 1, name=a_name[1], reuse=tf.AUTO_REUSE)
                f_mix = tf.concat([f_1, f_2], axis=1)
                logits = f_1 + tf.transpose(f_mix, [0, 2, 1])
                coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

                if in_drop != 0.0:
                    mix_fts = tf.nn.dropout(mix_fts, 1.0 - in_drop)
                if coef_drop != 0.0:
                    coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)

                vals = tf.matmul(coefs, mix_fts)
                ret = tf.contrib.layers.bias_add(vals, scope=bias_name, reuse=tf.AUTO_REUSE)

            # residual connection
            # if residual:
            #     if satlt.shape[-1] != ret.shape[-1]:
            #         ret = ret + tf.layers.conv1d(satlt, ret.shape[-1], 1) # activation
            #     else:
            #         ret = ret + satlt

        return activation(ret)  # activation
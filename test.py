# from sklearn.cluster import SpectralClustering
# import numpy as np
# X = np.array([[1, 1], [2, 1], [1, 0],[4, 7], [3, 5], [3, 6]])
# clustering = SpectralClustering(n_clusters=2,assign_labels="discretize",random_state=0).fit(X)
# print(clustering.labels_)

# for i in range(1, 1):
#     print("test", i)

# import tensorflow as tf
# import numpy as np

# def build(x, level):
# 	with tf.name_scope('test'):
# 		return tf.layers.conv2d(x, 3, [2, 2], reuse=tf.AUTO_REUSE, name='conv')


# def build2(x, level):
# 	with tf.variable_scope('test' + str(level)):
# 		return tf.layers.conv2d(x, 3, [2, 2], reuse=tf.AUTO_REUSE, name='conv')

# x = tf.random_normal(shape=[10, 32, 32, 3])

# # conv1 = tf.layers.conv2d(x, 3, [2, 2], reuse=tf.AUTO_REUSE, name='conv')

# # conv2 = tf.layers.conv2d(x, 3, [2, 2], reuse=tf.AUTO_REUSE, name='conv')

# conv1 = build2(x, 1)
# conv2 = build2(x, 2)

# print(conv1.name)
# print(conv2.name)

# print([x.name for x in tf.global_variables()])

import numpy as np

arr = np.array([[0, 1, 2],
                [3, 4, 5],
                [6, 7, 8]])

flat_array = arr.flatten()
print('1D Numpy Array:')
print(flat_array.shape)












# import numpy as np
# # import tensorflow as tf
# import os
# import networkx as nx
# import matplotlib.pyplot as plt

# from utils import process

# checkpt_file = 'pre_trained/cora/mod_cora.ckpt'

# dataset = 'cora'

# # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# # gpu_options.allow_growth = True


# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(dataset)

# G=nx.from_numpy_matrix(adj.todense())

# adj = adj.todense()

# # shared_neighbor = adj * (adj @ adj)

# # print("shared_neighbor finished")

# # # for i in range(10):

# # print(np.sum(shared_neighbor == 1, axis=1))

# # print(np.histogram(shared_neighbor))

# np.set_printoptions(threshold=np.inf)
# print(np.sort(np.sum(adj, axis = 1).transpose()))

# # nx.draw(G,pos = nx.spring_layout(G), width = 0.15, node_color = 'b',edge_color = 'r',with_labels = True, font_size =2,node_size =9)
# # plt.show()

# count = 0

# subgraphs = sorted(nx.connected_components(G),key=len,reverse=True)

# # for c in subgraphs:
# #     # print(c)      #看看返回来的是什么？结果是{0,1,2,3}
# #     count += 1
# #     print("第"+str(count) + "个子图的长度: ", len(c))

# adj_sub_idx = list((subgraphs[1]))
# print(adj_sub_idx)
# adj_sub = adj[adj_sub_idx,:]
# adj_sub = adj_sub[:, adj_sub_idx].todense()
# print(adj_sub)

# G=nx.from_numpy_matrix(adj_sub)

# np.set_printoptions(threshold=np.inf)
# print(np.sort(np.sum(adj, axis = 1).transpose()))

# nx.draw(G,pos = nx.spring_layout(G), width = 1, node_color = 'b',edge_color = 'r',with_labels = True, font_size =2,node_size =9)
# plt.show()











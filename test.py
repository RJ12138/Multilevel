# from sklearn.cluster import SpectralClustering
# import numpy as np
# X = np.array([[1, 1], [2, 1], [1, 0],[4, 7], [3, 5], [3, 6]])
# clustering = SpectralClustering(n_clusters=2,assign_labels="discretize",random_state=0).fit(X)
# print(clustering.labels_)

for i in range(1, 1):
    print("test", i)

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
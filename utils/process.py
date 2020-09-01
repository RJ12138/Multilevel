import numpy as np
import time
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
from sklearn.cluster import SpectralClustering

"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""
def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)

# Preparing the coarsening mapping matrix. These two bias functions are prepared for softmax.

def mapping_to_bias(clustering_mat, sizes):
    nb_graphs = clustering_mat.shape[0]
    mt = np.empty(clustering_mat.shape)
    for g in range(nb_graphs):
        mt[g] = np.zeros(clustering_mat.shape[1:])
        for i in range(clustering_mat.shape[1]):
            for j in range(clustering_mat.shape[2]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)

def sklearn_clustering(adj, feat, cluster_size):
    nb_graphs = feat.shape[0]
    n_node = feat.shape[1]
    n_clusters = int(n_node / cluster_size)

    clustering_mat = np.zeros((nb_graphs, n_node, n_clusters))

    print("clustering start")
    clustering_start = time.time()
    clustering = SpectralClustering(n_clusters=n_clusters,assign_labels="discretize",random_state=0).fit(feat[0])
    clustering_time = time.time() - clustering_start
    print("clustering finished. Time: ", clustering_time)
    for i in range(len(clustering.labels_)):
        clustering_mat[0, i, clustering.labels_[i]] = 1.0
    clustering_mat = clustering_mat.transpose(0, 2, 1)
    np.set_printoptions(threshold=np.inf)
    non_empty_cluster = np.sum(clustering_mat[0], axis=1).nonzero()
    clustering_mat = clustering_mat[0, non_empty_cluster, :]
    n_clusters = clustering_mat.shape[1]

    convergent_mat = np.empty((nb_graphs, n_clusters, n_node + n_clusters))
    btw_super_mat = np.empty((nb_graphs, n_clusters, n_clusters))
    divergent_mat = np.empty((nb_graphs, n_node, n_node + n_clusters))

    convergent_mat[0] = np.concatenate((clustering_mat[0], np.eye(n_clusters)), axis=1)
    btw_super_mat[0] = clustering_mat[0] @ np.around(clustering.affinity_matrix_ - 0.42) @ clustering_mat[0].transpose()
    divergent_mat[0] = np.concatenate((np.eye(n_node), clustering_mat[0].transpose()), axis=1)

    n_neighbors = np.sum(btw_super_mat[0] != 0, axis=0)
    print(np.max(np.sum(btw_super_mat[0], axis=0)))

    def getNeighbor(adj_vec):
        k_th = int(np.sum(adj_vec != 0) * 0.2 * np.sum(adj_vec)/14338)
        k_th = max(min(k_th, 50), 2)
        ordered_idx = np.argpartition(adj_vec, k_th)
        return ordered_idx[:k_th], ordered_idx[k_th:]

    for i in range(btw_super_mat[0].shape[0]):
        neighbor_idx, non_neighbor_idx = getNeighbor(btw_super_mat[0,i])
        # if i % 30 == 0:
        #     print(neighbor_idx)
        btw_super_mat[0,i,neighbor_idx] = 1
        btw_super_mat[0,i,non_neighbor_idx] = 0

    print("diag: ", np.diag(btw_super_mat[0]))
    btw_super_mat[0] = btw_super_mat[0] + np.eye(btw_super_mat[0].shape[1])

    # print(np.sum(btw_super_mat[0], axis=1))
    # print(np.sum(np.around(btw_super_mat[0]/np.max(btw_super_mat[0], axis=1)+0.2), axis=1))
    # print(np.max(np.sum(btw_super_mat[0] != 0, axis=1)), ' ', np.min(np.sum(btw_super_mat[0] != 0, axis=1)))
    # print(np.sum(np.around(clustering.affinity_matrix_ - 0.42), axis=0))

    return clustering_mat, -1e9 * (1.0 - convergent_mat), -1e9 * (1.0 - btw_super_mat), -1e9 * (1.0 - divergent_mat)

def sklearn_clustering2(adj, feat, cluster_size):
    test = True
    if test:
        nb_graphs = adj.shape[0]
        n_node = adj.shape[1]
        n_clusters = int(n_node / cluster_size)

        clustering_mat = np.zeros((nb_graphs, n_node, n_clusters))

        print("adj clustering start")
        clustering_start = time.time()
        clustering = SpectralClustering(n_clusters=n_clusters,affinity="precomputed", assign_labels="discretize",random_state=0).fit(adj[0] + np.eye(n_node))
        clustering_time = time.time() - clustering_start
        print("adj clustering finished. Time: ", clustering_time)
        for i in range(len(clustering.labels_)):
            clustering_mat[0, i, clustering.labels_[i]] = 1.0
        clustering_mat = clustering_mat.transpose(0, 2, 1)
        # np.set_printoptions(threshold=np.inf)
        non_empty_cluster = np.sum(clustering_mat[0], axis=1).nonzero()
        clustering_mat = clustering_mat[0, non_empty_cluster, :]
        n_clusters = clustering_mat.shape[1]

        print("n_clusters: ", np.sum(clustering_mat[0], axis=1).shape)

        convergent_mat = np.empty((nb_graphs, n_clusters, n_node + n_clusters))
        btw_super_mat = np.empty((nb_graphs, n_clusters, n_clusters))
        divergent_mat = np.empty((nb_graphs, n_node, n_node + n_clusters))

        adj_next_level = np.empty((nb_graphs, n_clusters, n_clusters))

        convergent_mat[0] = np.concatenate((clustering_mat[0], np.eye(n_clusters)), axis=1)
        btw_super_mat[0] = clustering_mat[0] @ clustering.affinity_matrix_ @ clustering_mat[0].transpose()
        divergent_mat[0] = np.concatenate((np.eye(n_node), clustering_mat[0].transpose()), axis=1)

        n_neighbors = np.sum(btw_super_mat[0] != 0, axis=0)
        # print((np.sum(btw_super_mat[0], axis=0)))
        # print("n_neighbors", n_neighbors)
        # print("diag: ", np.diag(adj[0]))

        for i in range(btw_super_mat[0].shape[0]):
            for j in range(btw_super_mat[0].shape[1]):
                if btw_super_mat[0, i, j] != 0:
                    btw_super_mat[0,i,j] = 1
                if i == j:
                    btw_super_mat[0,i,j] = 1

        # print(np.sum(btw_super_mat[0], axis=1))
        adj_next_level[0] = btw_super_mat[0] - np.eye(n_clusters)


    else:
        nb_graphs = feat.shape[0]
        n_node = feat.shape[1]
        clustering_mat = np.empty((nb_graphs, n_node, n_node))
        convergent_mat = np.empty((nb_graphs, n_node, n_node + n_node))
        btw_super_mat = np.empty((nb_graphs, n_node, n_node))
        divergent_mat = np.empty((nb_graphs, n_node, n_node + n_node))

        clustering_mat[0] = np.eye(n_node)
        convergent_mat[0] = np.concatenate((clustering_mat[0], np.eye(n_node)), axis=1)
        btw_super_mat[0] = adj[0] + np.eye(n_node)
        divergent_mat[0] = np.concatenate((np.eye(n_node), clustering_mat[0].transpose()), axis=1)


    # print(np.sum(np.around(btw_super_mat[0]/np.max(btw_super_mat[0], axis=1)+0.2), axis=1))
    # print(np.max(np.sum(btw_super_mat[0] != 0, axis=1)), ' ', np.min(np.sum(btw_super_mat[0] != 0, axis=1)))
    # print(np.sum(np.around(clustering.affinity_matrix_ - 0.42), axis=0))

    return clustering_mat, -1e9 * (1.0 - convergent_mat), -1e9 * (1.0 - btw_super_mat), -1e9 * (1.0 - divergent_mat), adj_next_level

###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    print(adj.shape)
    print(features.shape)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def load_random_data(size):

    adj = sp.random(size, size, density=0.002) # density similar to cora
    features = sp.random(size, 1000, density=0.015)
    int_labels = np.random.randint(7, size=(size))
    labels = np.zeros((size, 7)) # Nx7
    labels[np.arange(size), int_labels] = 1

    train_mask = np.zeros((size,)).astype(bool)
    train_mask[np.arange(size)[0:int(size/2)]] = 1

    val_mask = np.zeros((size,)).astype(bool)
    val_mask[np.arange(size)[int(size/2):]] = 1

    test_mask = np.zeros((size,)).astype(bool)
    test_mask[np.arange(size)[int(size/2):]] = 1

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
  
    # sparse NxN, sparse NxF, norm NxC, ..., norm Nx1, ...
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def preprocess_adj_bias(adj):
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()  # This is where I made a mistake, I used (adj.row, adj.col) instead
    # return tf.SparseTensor(indices=indices, values=adj.data, dense_shape=adj.shape)
    return indices, adj.data, adj.shape

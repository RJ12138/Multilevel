import numpy as np
import networkx as nx
import os
from scipy.sparse import identity
from scipy.io import mmwrite
import sys
from argparse import ArgumentParser
from sklearn.preprocessing import normalize
import time
import matplotlib.pyplot as plt

from networkx.linalg.laplacianmatrix import laplacian_matrix
from graphzoom_utils import *

def graph_fusion(laplacian, feature, num_neighs, mcr_dir, coarse, fusion_input_path, \
                 search_ratio, fusion_output_dir, mapping_path, dataset):

    # obtain mapping operator
    if coarse == "simple":
        mapping = sim_coarse_fusion(laplacian)
    else:
        raise NotImplementedError

    # construct feature graph
    feats_laplacian = feats2graph(feature, num_neighs, mapping)

    # fuse adj_graph with feat_graph
    fused_laplacian = laplacian + feats_laplacian

    return fused_laplacian

def refinement(levels, projections, coarse_laplacian, embeddings, lda, power):
    for i in reversed(range(levels)):
        embeddings = projections[i] @ embeddings
        filter_    = smooth_filter(coarse_laplacian[i], lda)

        ## power controls whether smoothing intermediate embeddings,
        ## preventing over-smoothing
        if power or i == 0:
            embeddings = filter_ @ (filter_ @ embeddings)
    return embeddings

def graph_coarsen(adj, feature):

    args_fusion = False
    args_coarse = 'simple'
    args_level = 5

    # dataset = args.dataset
    # feature_path = "dataset/{}/{}-feats.npy".format(dataset, dataset)
    fusion_input_path = None # "dataset/{}/{}.mtx".format(dataset, dataset)
    reduce_results = None # "reduction_results/" # this is for lamg
    mapping_path = None # "{}Mapping.mtx".format(reduce_results) # this is for lamg

    coarsen_input_path = None

######Load Data######
    print("%%%%%% Loading Graph Data %%%%%%")
    # laplacian, G_origin = json2mtx(dataset)

    G = nx.from_numpy_matrix(adj)
    laplacian = laplacian_matrix(G, nodelist=range(len(G.nodes)))

######Graph Fusion######
    if args_fusion:
        print("%%%%%% Starting Graph Fusion %%%%%%")
        mcr_dir = None
        coarse_method = 'simple'
        search_ratio = None
        dataset = None
        num_neighs = 10
        fusion_start = time.process_time()
        laplacian    = graph_fusion(laplacian, feature, num_neighs, mcr_dir, coarse_method,\
                       fusion_input_path, search_ratio, reduce_results, mapping_path, dataset)
        fusion_time  = time.process_time() - fusion_start

######Graph Reduction######
    print("%%%%%% Starting Graph Reduction %%%%%%")
    reduce_start = time.process_time()

    if args_coarse == "simple":
        G, projections, laplacians, level = sim_coarse(laplacian, args_level)
        reduce_time = time.process_time() - reduce_start
        G_small = G

    else:
        raise NotImplementedError
    
#     adjacency_matrices = []
#     adjacency_last_level = adj
#     for i in range(len(projections)):
#         mapping = projections[i].toarray()
#         adjacency_this_level = mapping.transpose() @ adjacency_last_level @ mapping
#         adjacency_last_level = adjacency_this_level
#         adjacency_matrices.append(adjacency_this_level)
    
    clustering_mat_list = []
    convergent_mat_list = []
    btw_super_mat_list = []
    divergent_mat_list = []
    
#     print((np.sum(adj, axis=0)))
#     print(np.diag(adj))

    for i in range(len(projections)):
        # print(projections[i].toarray().shape)

        nb_graphs = 1
        mapping = projections[i].toarray()
        laplacian = laplacians[i + 1].toarray()
        n_node = mapping.shape[0]
        n_clusters = mapping.shape[1]

        print("mapping.shape:", mapping.shape)

        clustering_mat = np.zeros((nb_graphs, n_clusters, n_node))
        clustering_mat[0] = mapping.transpose()
        # print(clustering_mat[0].shape)
        # print(np.eye(n_clusters).shape)
        # print("n_clusters: ", np.sum(clustering_mat[0], axis=1).shape)

        convergent_mat = np.empty((nb_graphs, n_clusters, n_node + n_clusters))
        btw_super_mat = np.empty((nb_graphs, n_clusters, n_clusters))
        divergent_mat = np.empty((nb_graphs, n_node, n_node + n_clusters))

        convergent_mat[0] = np.concatenate((clustering_mat[0], np.eye(n_clusters)), axis=1)
        btw_super_mat[0] = np.diag(np.diag(laplacian)) - laplacian
        divergent_mat[0] = np.concatenate((np.eye(n_node), clustering_mat[0].transpose()), axis=1)

        n_neighbors = np.sum(btw_super_mat[0] != 0, axis=0)
#         print((np.sum(btw_super_mat[0], axis=0)))
        # print("n_neighbors", n_neighbors)
        # print("diag: ", np.diag(adj[0]))

        for i in range(btw_super_mat[0].shape[0]):
            for j in range(btw_super_mat[0].shape[1]):
                if btw_super_mat[0, i, j] != 0:
                    btw_super_mat[0,i,j] = 1
                if i == j:
                    btw_super_mat[0,i,j] = 1
        print('btw_super_mat:', np.sort(np.sum(btw_super_mat,axis=1)))


# clustering_mat, -1e9 * (1.0 - convergent_mat), -1e9 * (1.0 - btw_super_mat), -1e9 * (1.0 - divergent_mat)
        clustering_mat_list.append(clustering_mat)
        convergent_mat_list.append(-1e9 * (1.0 - convergent_mat))
        btw_super_mat_list.append(-1e9 * (1.0 - btw_super_mat))
        divergent_mat_list.append(-1e9 * (1.0 - divergent_mat))

    return clustering_mat_list, convergent_mat_list, btw_super_mat_list, divergent_mat_list

    # for i in range(len(laplacians) + 1):
    #     if i != len(laplacians):
    #         laplacian = laplacians[i]
    #         adjacency = diags(laplacian.diagonal(), 0) - laplacian
    #         G = nx.from_scipy_sparse_matrix(adjacency, edge_attribute='wgt')
    #     else:
    #         G = G_small
    #         adjacency = nx.adjacency_matrix(G_small)
    #     # nx.draw(G,pos = nx.spring_layout(G), width = 0.3, node_color = 'b',edge_color = 'r',with_labels = True, font_size =2,node_size =9)
    #     # plt.show()
    #     subgraphs = sorted(nx.connected_components(G),key=len,reverse=True)
    #     count = 0
    #     for c in subgraphs:
    #         count += 1
    #         if count >=10:
    #             break
    #         print("第"+str(count) + "个子图的长度: ", len(c))
    #     adj_sub_idx = list((subgraphs[1]))
    #     # print(adj_sub_idx)
    #     adj_sub = adjacency[adj_sub_idx,:]
    #     adj_sub = adj_sub[:, adj_sub_idx].todense()
    #     # print(adj_sub)

    #     G = nx.from_numpy_matrix(adj_sub)

    #     np.set_printoptions(threshold=np.inf)
    #     # print(np.sort(np.sum(adjacency, axis = 1).transpose()))

    #     nx.draw(G,pos = nx.spring_layout(G), width = 1, node_color = 'b',edge_color = 'r',with_labels = True, font_size =2,node_size =9)
    #     # plt.show()
    #     plt.savefig('./test' + str(i)+ '.png')
    #     plt.close()
    #     # print(adj_sub)
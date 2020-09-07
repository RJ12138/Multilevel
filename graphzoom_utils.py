import numpy as np
from numpy import linalg as LA
import json
import networkx as nx
from networkx.readwrite import json_graph
from networkx.linalg.laplacianmatrix import laplacian_matrix
from scipy.io import mmwrite
from scipy.sparse import csr_matrix, diags, identity, triu, tril
from itertools import combinations
import matplotlib.pyplot as plt

def cosine_similarity(x, y):
    flat_array = x.flatten()
    dot_xy = abs(np.inner(x, y))
    norm_x = LA.norm(x)
    norm_y = LA.norm(y)
    if norm_x == 0 or norm_y == 0:
        if norm_x == 0 and norm_y == 0:
            similarity = 1
        else:
            similarity = 0
    else:
        similarity = dot_xy/(norm_x * norm_y)
    return similarity

def maximum (A, B):
    ## calculate max{A, B}
    BisBigger = A-B
    BisBigger.data = np.where(BisBigger.data < 0, 1, 0)
    return A - A.multiply(BisBigger) + B.multiply(BisBigger)

def feats2graph(feature, num_neighs, mapping):
    # number of nodes in fine graph
    fine_dim   = mapping.shape[1]
    # number of nodes in coarse graph
    coarse_dim = mapping.shape[0]

    # This mapping matrix is the map between the coarsest map and the finest map. 
    # So every cluster has much more nodes than a mapping between two connected levels.

    all_rows   = []
    all_cols   = []
    all_data   = []
    for i in range(coarse_dim):
        row  = []
        col  = []
        data = []
        node_list = ((mapping[i,:].nonzero())[1]).tolist()
        if len(node_list)-1 > num_neighs:
            for j in node_list:
                col_  = []
                data_ = []
                dist  = []
                feat1 = feature[j, :]
                for k in node_list:
                    if j != k:
                        feat2 = feature[k, :]
                        dist.append(LA.norm(feat1-feat2))
                        col_.append(k)
                ids_sort = np.argsort(np.asarray(dist))
                col_ind  = (np.asarray(col_)[ids_sort]).tolist()[:num_neighs]
                for ind in col_ind:
                    feat2 = feature[ind, :]
                    data_.append(cosine_similarity(feat1, feat2))
                row  += (np.repeat(j, num_neighs)).tolist()
                col  += col_ind
                data += data_
        else:
            for pair in combinations(node_list, 2):
                feat1 = feature[pair[0], :]
                feat2 = feature[pair[1], :]
                row.append(pair[0])
                col.append(pair[1])
                data.append(cosine_similarity(feat1, feat2))
        all_rows += row
        all_cols += col
        all_data += data

    adj_initial      = csr_matrix((all_data, (all_rows, all_cols)), shape=(fine_dim, fine_dim))
    adj_max          = maximum(triu(adj_initial), tril(adj_initial).transpose())
    adj_final        = adj_max + adj_max.transpose()
    degree_matrix    = diags(np.squeeze(np.asarray(adj_final.sum(axis=1))), 0)
    laplacian_matrix = degree_matrix - adj_final

    return laplacian_matrix

def json2mtx(dataset):
    G_data    = json.load(open("dataset/{}/{}-G.json".format(dataset, dataset)))
    G         = json_graph.node_link_graph(G_data)
    laplacian = laplacian_matrix(G, nodelist=range(len(G.nodes)))
    file = open("dataset/{}/{}.mtx".format(dataset, dataset), "wb")
    mmwrite("dataset/{}/{}.mtx".format(dataset, dataset), laplacian)
    file.close()

    return laplacian, G

def mtx2matrix(proj_name):
    data = []
    row  = []
    col  = []
    with open(proj_name) as ff:
        for i,line in enumerate(ff):
            info = line.split()
            if i == 0:
                NumReducedNodes = int(info[0])
                NumOriginNodes  = int(info[1])
            else:
                row.append(int(info[0])-1)
                col.append(int(info[1])-1)
                data.append(1)
    matrix = csr_matrix((data, (row, col)), shape=(NumReducedNodes, NumOriginNodes))
    return matrix


def mtx2graph(mtx_path):
    G = nx.Graph()
    with open(mtx_path) as ff:
        for i,line in enumerate(ff):
            info = line.split()
            if i == 0:
                num_nodes = int(info[0])
            elif int(info[0]) < int(info[1]):
                G.add_edge(int(info[0])-1, int(info[1])-1, wgt=abs(float(info[2])))

    ## add isolated nodes
    for i in range(num_nodes):
        G.add_node(i)
    return G

def read_levels(level_path):
    with open(level_path) as ff:
        levels = int(ff.readline()) - 1
    return levels

def read_time(cputime_path):
    with open(cputime_path) as ff:
        cpu_time = float(ff.readline())
    return cpu_time

def construct_proj_laplacian(laplacian, levels, proj_dir):
    coarse_laplacian = []
    projections      = []
    for i in range(levels):
        projection_name = "{}/Projection_{}.mtx".format(proj_dir, i+1)
        projection      = mtx2matrix(projection_name)
        projections.append(projection.transpose())
        coarse_laplacian.append(laplacian)
        if i != (levels-1):
            laplacian = projection @ (laplacian @ (projection.transpose()))
    return projections, coarse_laplacian

def affinity(x, y):
    dot_xy = (np.dot(x, y))**2
    norm_x = (LA.norm(x))**2
    norm_y = (LA.norm(y))**2
    return dot_xy/(norm_x*norm_y)

def smooth_filter(laplacian_matrix, lda):
    dim        = laplacian_matrix.shape[0]
    adj_matrix = diags(laplacian_matrix.diagonal(), 0) - laplacian_matrix + lda * identity(dim)
    degree_vec = adj_matrix.sum(axis=1)
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.squeeze(np.asarray(np.power(degree_vec, -0.5)))
    d_inv_sqrt[np.isinf(d_inv_sqrt)|np.isnan(d_inv_sqrt)] = 0
    degree_matrix  = diags(d_inv_sqrt, 0)
    norm_adj       = degree_matrix @ (adj_matrix @ degree_matrix)
    return norm_adj

def spec_coarsen(filter_, laplacian):
    np.random.seed(seed=1)

    ## power of low-pass filter
    power = 2
    ## number of testing vectors
    t = 7
    ## threshold for merging nodes
    thresh = 0.3

    adjacency = diags(laplacian.diagonal(), 0) - laplacian
    G = nx.from_scipy_sparse_matrix(adjacency)
    tv_list = []
    num_nodes = len(G.nodes())

    ## generate testing vectors in [-1,1], 
    ## and orthogonal to constant vector
    for _ in range(t):
        tv = -1 + 2 * np.random.rand(num_nodes)
        tv -= np.ones(num_nodes)*np.sum(tv)/num_nodes
        tv_list.append(tv)
    tv_feat = np.transpose(np.asarray(tv_list))

    ## smooth the testing vectors
    for _ in range(power):
        tv_feat = filter_ @ tv_feat
    matched = [False] * num_nodes
    degree_map = [0] * num_nodes

    ## hub nodes are more important than others,
    ## treat hub nodes as seeds
    for (node, val) in G.degree():
        degree_map[node] = val
    sorted_idx = np.argsort(np.asarray(degree_map))[::-1]
    row = []
    col = []
    data = []
    cnt = 0
    count_one_matched_neighbor = 0
    count_neighbor_no_to_matched = 0
    ncolor = ['g'] * num_nodes
    for idx_ in sorted_idx:
        idx = idx_
        if matched[idx]:
            continue
        matched[idx] = True
        cluster = [idx]
        neighbors = G.neighbors(idx)
        neighbor_have_to_match = False
        for n in neighbors:
            # print(len(list(neighbors)))
            thresh = min(degree_map[idx] * 0.1, 0.35)
            thresh = min(thresh * degree_map[n] * 0.07, 0.35)
#             thresh = max(0.3 - degree_map[idx] * 0.01, 0)
#             thresh = max(0.3 - thresh * degree_map[n] * 0.01, 0)
#             thresh = max(0.3 - degree_map[idx] * 0.01, 0.01)
#             thresh = min(thresh * degree_map[n] * 0.05, 0.3)
            # print(thresh)

            if affinity(tv_feat[idx], tv_feat[n]) > thresh and not matched[n]:
            # if not matched[n]:
                cluster.append(n)
                matched[n] = True

        if len(cluster) == 1:
            ncolor[idx] = 'k'
            count_neighbor_no_to_matched += 1
        # if not neighbor_have_to_match:
        #     count_neighbor_no_to_matched+=1
        #     ncolor[idx] = 'b'
        row += cluster
        col += [cnt] * len(cluster)
        data += [1] * len(cluster)
        cnt += 1
    print("count_one_matched_neighbor ", count_one_matched_neighbor)
    print("no neighbor to match ", count_neighbor_no_to_matched)

    # nx.draw(G,pos = nx.spring_layout(G), width = 0.2, node_color = ncolor,edge_color = 'r',with_labels = True, font_size =2,node_size =9)
    # plt.show()
    # input('press Enter to continue')
    
    mapping = csr_matrix((data, (row, col)), shape=(num_nodes, cnt))
#     print("mapping: ", np.sum(mapping, axis = 0))
    # print("mapping max: ", np.max(np.sum(mapping, axis = 0)))
    coarse_laplacian = mapping.transpose() @ laplacian @ mapping
#     print("coarse_laplacian: ", coarse_laplacian)
    return coarse_laplacian, mapping

def sim_coarse(laplacian, level):
    projections = []
    laplacians = []
    mapping = identity(laplacian.shape[0])
    for i in range(level):
        filter_ = smooth_filter(laplacian, 0.1)
        laplacians.append(laplacian)
        laplacian, map_ = spec_coarsen(filter_, laplacian)
        mapping = mapping @ map_
        projections.append(map_)

        np.set_printoptions(threshold=np.inf)
        print("max cluter: ", np.max(np.sum(mapping,axis=0)))
#         print(np.sum(mapping,axis=0))

        print("Coarsening Level:", i+1)
#         print("Num of nodes: ", laplacian.shape[0], "Num of edges: ", int((laplacian.nnz - laplacian.shape[0])/2))

    laplacians.append(laplacian)
    adjacency = diags(laplacian.diagonal(), 0) - laplacian
    G = nx.from_scipy_sparse_matrix(adjacency, edge_attribute='wgt')
    return G, projections, laplacians, level

def sim_coarse_fusion(laplacian):
    level = 5
    mapping = identity(laplacian.shape[0])
    for _ in range(level):
        filter_ = smooth_filter(laplacian, 0.1)
        laplacian, map_ = spec_coarsen(filter_, laplacian)
        mapping = mapping @ map_
    mapping = mapping.transpose()
    return mapping

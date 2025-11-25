import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
import time
import heapq
from sklearn import metrics
from pynndescent import NNDescent     ## https://github.com/lmcinnes/pynndescent

# Try importing the compiled core functions (Algo 2 in paper) from Cython
try:
    from stage2_core import merge_loop as _merge_loop_cython 
    HAS_CYTHON_BACKEND = True
except ImportError:
    _merge_loop_cython = None
    HAS_CYTHON_BACKEND = False

def clust_rank(M, NC, SSE, k = 5, return_knn_time = False):
    c = np.shape(M)[0]  ## The current number of samples or clusters
    if NC[0] == 1:  # Initialization: The first merge operation
        k = 2
        t1 = time.time()
        if c <= 100:
            orig_dist = metrics.pairwise.pairwise_distances(M, M)
            np.fill_diagonal(orig_dist, 1e12)
            initial_rank = np.argmin(orig_dist, axis=1)
            NN = initial_rank
        elif c <= 20000:        #### sklearn
            knn = NearestNeighbors(n_neighbors=k)
            knn.fit(M)
            KNN = knn.kneighbors(M,return_distance=False)
            NN = KNN[np.arange(c), 1]   
        else:
            #### NNDescent      # for large datasets, need install pynndescent
            nnd = NNDescent(M, n_neighbors=k)  
            result, orig_dist = nnd.neighbor_graph
            NN = result[np.arange(c), 1] 
        t2 = time.time()
    else:
        k = min(c, k + 1)
        t1 = time.time()
        if c <= 100:
            orig_dist = metrics.pairwise.pairwise_distances(M, M, squared=False)
            np.fill_diagonal(orig_dist, 1e12)
            KNN = np.argsort(orig_dist, axis=1)[:, :k-1]    
            KNND = np.take_along_axis(orig_dist, KNN, axis=1)  
            KNND = KNND ** 2     
        elif c<= 20000:       
            knn = NearestNeighbors(n_neighbors=k)
            knn.fit(M)
            KNND, KNN = knn.kneighbors(M)
            KNND = KNND[:, 1:] 
            KNN = KNN[:, 1:]   
            KNND = KNND ** 2
        else:
            #### NNDescent
            nnd = NNDescent(M, n_neighbors=k)
            KNN, KNND = nnd.neighbor_graph
            KNND = KNND[:, 1:] 
            KNN = KNN[:, 1:]   

        t2 = time.time()


        N_NN = NC[KNN]      
        cof = (NC[:, np.newaxis] * N_NN) / (NC[:, np.newaxis] + N_NN)
        delta_SSE = cof * KNND  # Calculate the change in SSE based on neighbor distances
        min_indices = np.argmin(delta_SSE, axis=1)  # Find the index of the neighbor that minimizes the SSE change

        NN = KNN[np.arange(c), min_indices]    

        '''
        Following this approach leads to the bias that the merge process is guided by small clusters. To eliminate this bias we provide two methods. Optional
        '''

        # method 1, suggested, min_increment < eta * (current_SSE + neighbor_SSE)，eta suggests 10 or 6 or 2
        # min_increment = delta_SSE[np.arange(c), min_indices]  # SSE increment for each cluster
        # current_SSE = SSE
        # neighbor_SSE = SSE[NN]
        # valid_merge = (min_increment < 10 * (current_SSE + neighbor_SSE)) | (current_SSE < 1e-6)  | (neighbor_SSE < 1e-6) 
        # row_indices = np.arange(c)
        # NN[~valid_merge] = row_indices[~valid_merge]  

        # method 2, Refer to paper 2023_Multi-View Adjacency-Constrained Hierarchical Clustering，This approach is also valid, but it is recommended to set k to a higher value, such as 20
        # current_SSE = SSE
        # neighbor_SSE = SSE[NN]
        # mask = neighbor_SSE >= current_SSE  # condition：neighbor SSE >= itself
        # row_indices = np.arange(c)
        # NN[~mask] = row_indices[~mask]    # If the condition is not met, it refers to itself (equivalent to refusing the merge).

    if return_knn_time:
        return NN, t2-t1
    else:
        return NN, None


def get_cluster(NN):
    c = len(NN)     
    data = np.ones(len(NN))  
    row_indices = np.arange(c)
    graph = sp.coo_matrix((data, (row_indices, NN)), shape=(c, c))

    num_clusters, u = sp.csgraph.connected_components(csgraph=graph, directed=True, connection='weak', return_labels=True)
    return u, num_clusters


def cool_mean(M, u):    
    s = M.shape[0]      
    un, nf = np.unique(u, return_counts=True)
    umat = sp.csr_matrix((np.ones(s, dtype='float32'), (np.arange(0, s), u)), shape=(s, len(un)))

    mat = (umat.T @ M) / nf[..., np.newaxis]

    sample_counts = nf
    mat_expanded = mat[u]  # Shape (s, d), where each row is the centroid of the corresponding sample's cluster

    # Calculate squared differences
    squared_errors = np.sum((M - mat_expanded) ** 2, axis=1)  # Shape (s,)
    
    # Sum squared errors per cluster
    # sse = np.array([squared_errors[u == label].sum() for label in un])  # SSE for each cluster
    sse = np.bincount(u, weights=squared_errors, minlength=len(un))     

    return mat, sample_counts, sse

def get_merge(c, u, data):    
    if len(c) != 0:   
        _, ig = np.unique(c, return_inverse=True)
        c = u[ig]   
    else:  
        c = u

    mat, NC, SSE = cool_mean(data, c)  
    return c, mat, NC, SSE     

def get_label(parent):   
    label = np.zeros_like(parent)
    for idx, cluster in np.ndenumerate(parent):
        while cluster != parent[cluster]:
            cluster = parent[cluster]
        label[idx] = cluster
    return label

def _merge_loop_python(moments_1, moments_2, c_true, inertia):
    """
    moments_1: shape (n_samples,)
    moments_2: shape (n_samples, d)
    inertia:   A heap that has already been heapify [(loss, i, j), ...]
    """
    n_samples = moments_1.shape[0]
    d = moments_2.shape[1]
    n_nodes = 2 * n_samples - c_true

    if n_samples - c_true >= 10000:
        print(f'The while loop needs to execute {n_samples - c_true} times, so it\'s recommended to use the cython function execution for faster execution')

    expanded_moments_1 = np.zeros(n_nodes)
    expanded_moments_1[:n_samples] = moments_1
    expanded_moments_2 = np.zeros((n_nodes, d))
    expanded_moments_2[:n_samples, :] = moments_2
    moments_1 = expanded_moments_1
    moments_2 = expanded_moments_2

    parent = np.arange(n_nodes, dtype=np.intp)
    c_now, k = n_samples, n_samples

    t1 = time.time()
    while c_now > c_true:
        loss, i, j = heapq.heappop(inertia)

        if parent[i] == i and parent[j] == j:   # active clusters
            parent[i], parent[j], parent[k] = k, k, k
            moments_1[k] = moments_1[i] + moments_1[j]
            moments_2[k] = moments_2[i] + moments_2[j]
            c_now -= 1
            k += 1
        else:
            # Find the respective root node
            while parent[i] != i:
                i = parent[i]
            while parent[j] != j:
                j = parent[j]
            if i == j:
                continue

            weight = (moments_1[i] * moments_1[j]) / (moments_1[i] + moments_1[j])
            diff = moments_2[i] / moments_1[i] - moments_2[j] / moments_1[j]
            distance = np.linalg.norm(diff)
            inertia_ij = weight * distance**2
            heapq.heappush(inertia, (inertia_ij, i, j))

    t2 = time.time()
    return parent, t2 - t1

def stage2(KNN_list, moments_1, moments_2, c_true = 1, backend='auto'):
    """
    backend:
        - "auto"   : Use cython if you have a cython backend; otherwise, fall back to python
        - "cython" : Force cython to be used (if not, you get an error)
        - "python" : Force python implementation
    """

    KNN_array = np.array(KNN_list)
    n_samples = len(KNN_list) 

    moments_1 = np.asarray(moments_1, dtype=np.float64)
    moments_2 = np.asarray(moments_2, dtype=np.float64)

    moments_1_i = moments_1[:, np.newaxis]  
    moments_1_j = moments_1[KNN_array]     
    weights = (moments_1_i * moments_1_j) / (moments_1_i + moments_1_j)

    mu = moments_2 / moments_1.reshape(-1,1)
    mu_neighbor = mu[KNN_array]
    mu_expend = mu[:, np.newaxis, :]
    distance = np.linalg.norm(mu_expend-mu_neighbor, axis=2)

    loss = weights * distance**2
    heap_elements = [
        (loss[i, j], i, KNN_array[i, j]) 
        for i in range(n_samples) 
        for j in range(KNN_array.shape[1])
    ]
    heapq.heapify(heap_elements)
    inertia = heap_elements
    t1 = time.time()

    ############################################################################
    '''
    d = moments_2.shape[1]
    n_nodes = 2 * n_samples - c_true
    expanded_moments_1 = np.zeros(n_nodes)
    expanded_moments_1[:n_samples] = moments_1
    expanded_moments_2 = np.zeros((n_nodes, d))
    expanded_moments_2[:n_samples, :] = moments_2
    moments_1 = expanded_moments_1
    moments_2 = expanded_moments_2

    parent = np.arange(n_nodes, dtype=np.intp)
    c_now, k = n_samples, n_samples

    while c_now > c_true:
        loss, i, j = heapq.heappop(inertia)
        if parent[i] == i and parent[j] == j:   # active clusters
            # print(f'{i} + {j} -> {k}')
            parent[i], parent[j], parent[k] = k, k, k
            moments_1[k] = moments_1[i] + moments_1[j]  
            moments_2[k] = moments_2[i] + moments_2[j] 
            c_now -= 1
            k += 1
        else:       # inactive clusters
            while parent[i] != i:
                i = parent[i]
            while parent[j] != j:
                j = parent[j] 
            if i == j:
                continue 

            weight = (moments_1[i] * moments_1[j]) / (moments_1[i] + moments_1[j]) 
            distance = np.linalg.norm((moments_2[i]/moments_1[i] - moments_2[j]/moments_1[j])) 
            inertia_ij = weight * distance **2
            heapq.heappush(inertia, (inertia_ij, i, j))
    '''

    # ---------- Select implement according to backend ----------
    if backend == "cython":
        if not HAS_CYTHON_BACKEND:
            raise RuntimeError(
                "backend='cython' but no compiled Cython module stage2_core.merge_loop found,"
                "Please compile first or use  backend='python' or 'auto' instead."
            )
        parent, _, _, dt = _merge_loop_cython(inertia, moments_1, moments_2, c_true)

    elif backend == "python":
        parent, dt = _merge_loop_python(moments_1, moments_2, c_true, inertia)

    elif backend == "auto":
        if HAS_CYTHON_BACKEND:
            print('cython')
            parent, _, _, dt = _merge_loop_cython(inertia, moments_1, moments_2, c_true)
        else:
            print('python')
            parent, dt = _merge_loop_python(moments_1, moments_2, c_true, inertia)
    else:
        raise ValueError("backend must be 'auto'、'cython' or 'python', currently{}".format(backend))
    
    t2 = time.time()
    get_parents = get_label(parent)
    y_pred = get_parents[:n_samples]
    unique_values = np.unique(y_pred)

    y_pred_mapped = np.searchsorted(unique_values, y_pred)
    return y_pred_mapped, t2-t1    

## Algorithm 2 in paper
def req_numclust_late_update(c, data, req_clust, verbose=False, return_knn_time=False):
    c_, mat, NC, SSE = get_merge([], c, data)     
    n_samples = mat.shape[0]
    k = min(6, n_samples)  
    t1 = time.time()

    if n_samples <= 100:
        orig_dist = metrics.pairwise.pairwise_distances(mat, mat)
        np.fill_diagonal(orig_dist, 1e12)
        KNN = np.argsort(orig_dist, axis=1)[:, :k-1]    
        # KNND = np.take_along_axis(orig_dist, KNN, axis=1)    

    elif n_samples <= 100:
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(mat)
        KNN = knn.kneighbors(mat,return_distance=False)
        KNN = KNN[:, 1:] 

    else:
        nnd = NNDescent(mat, n_neighbors=k)
        KNN, KNND = nnd.neighbor_graph
        # KNND = KNND[:, 1:] 
        KNN = KNN[:, 1:]   

    t2 = time.time()


    KNN_list = KNN.tolist()
    moments_1 = NC
    moments_2 = mat * NC.reshape(-1, 1)

    mat_pred, t5 = stage2(KNN_list, moments_1, moments_2, c_true = req_clust)
    
    y_pred = mat_pred[c]
    if return_knn_time:
        return y_pred, t2-t1, t5
    else:
        return y_pred
    
def EHKM(data, req_clust = None, ensure_early_exit = True, verbose = False, return_knn_time=False):   

    data = data.astype(np.float32)
    n = np.shape(data)[0]   
    NC = np.ones(n, dtype=int) 
    SSE = np.zeros(n)
                                        
    NN, time1 = clust_rank(M = data, NC=NC, SSE=SSE, k = 1,return_knn_time=return_knn_time)     
    u, num_clusters = get_cluster(NN = NN)  
    c_ , mat, NC, SSE = get_merge([], u, data)    

    exit_clust = 2      
    c = c_
    num_clusters = [num_clusters]

    epoch = 0
    time2_list = []
    while exit_clust > 1:
        NN, time2 = clust_rank(mat, NC, SSE=SSE, k =5, return_knn_time=return_knn_time)   
        u, num_clusters_curr = get_cluster(NN)
        c_, mat, NC, SSE = get_merge(c_, u, data)

        time2_list.append(time2)
        num_clusters.append(num_clusters_curr)
        c = np.column_stack((c, c_))
        exit_clust = num_clusters[-2] - num_clusters_curr  

        if num_clusters_curr == 1 or exit_clust < 1:    
            num_clusters = num_clusters[:-1]   
            c = c[:, :-1]
            break

    time3 = 0
    time5 = 0
    nums_list2 = []    
    if req_clust is not None:
        if req_clust not in num_clusters:
            if req_clust > num_clusters[0]:
                print(f'requested number of clusters are larger than first partition with {num_clusters[0]} clusters . Returning {num_clusters[0]} clusters')
                req_c = c[:, 0]
            else: 
                ind = [i for i, v in enumerate(num_clusters) if v >= req_clust]

                req_c, time3, time5 = req_numclust_late_update(c[:, ind[-1]].copy(), data, req_clust, verbose=False, return_knn_time=True)
        else:
            req_c = c[:, num_clusters.index(req_clust)]
            
    else:
        req_c = None
    time_list = [time1, time2_list, time3, time5]

    if return_knn_time:
        return c, num_clusters, req_c, time_list
    else:
        return c, num_clusters, req_c
    


# stage2_core.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True
import heapq
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from time import time

# inertia: Python heap list [(loss, i, j),...]
def merge_loop(object inertia,
               np.ndarray[np.double_t, ndim=1] moments_1,
               np.ndarray[np.double_t, ndim=2] moments_2,
               int c_true):

    """
    inertia: Python heap [(loss, i, j), ...]
    moments_1: shape (n_samples,)
    moments_2: shape (n_samples, d)
    """

    cdef Py_ssize_t n_samples = moments_1.shape[0]
    cdef Py_ssize_t d = moments_2.shape[1]
    cdef Py_ssize_t n_nodes = 2 * n_samples - c_true

    # Extend the moments array to n_nodes
    cdef np.ndarray[np.double_t, ndim=1] m1 = np.zeros(n_nodes, dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=2] m2 = np.zeros((n_nodes, d), dtype=np.double)
    m1[:n_samples] = moments_1
    m2[:n_samples, :] = moments_2

    cdef np.ndarray[np.intp_t, ndim=1] parent = np.empty(n_nodes, dtype=np.intp)

    cdef Py_ssize_t i, j, root_i, root_j, k
    cdef int c_now
    cdef double w, dist, dx, inertia_ij
    cdef object item
    cdef double loss

    # initialize parent
    for i in range(n_nodes):
        parent[i] = i

    c_now = n_samples
    k = n_samples

    t1 = time()
    while c_now > c_true:
        # Get the current minimum (loss, i, j)
        item = heapq.heappop(inertia)
        loss = <double>item[0]
        i = <Py_ssize_t>item[1]
        j = <Py_ssize_t>item[2]

        # find the current root node (union find set)
        root_i = i
        while parent[root_i] != root_i:
            root_i = parent[root_i]

        root_j = j
        while parent[root_j] != root_j:
            root_j = parent[root_j]

        if root_i == root_j:
            continue

        # If i, j are themselves the respective roots and have not been merged yet, a true merge is performed
        if parent[i] == i and parent[j] == j and i == root_i and j == root_j:
            parent[i] = k
            parent[j] = k
            parent[k] = k

            m1[k] = m1[i] + m1[j]
            for d_idx in range(d):
                m2[k, d_idx] = m2[i, d_idx] + m2[j, d_idx]

            c_now -= 1
            k += 1
        else:
            # Otherwise: Calculate the inertia of these two "current root" clusters and repress them back to the heap
            w = (m1[root_i] * m1[root_j]) / (m1[root_i] + m1[root_j])

            dist = 0.0
            for d_idx in range(d):
                dx = m2[root_i, d_idx] / m1[root_i] - m2[root_j, d_idx] / m1[root_j]
                dist += dx * dx

            inertia_ij = w * dist
            heapq.heappush(inertia, (inertia_ij, root_i, root_j))

    t2 = time()
    return parent, m1, m2, t2 - t1

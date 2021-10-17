import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import sklearn

# _utils is from sklearn, _utils_xinrui is compiled using AOCC and with -O0 option
# from sklearn_xinrui.manifold import _utils
# import _utils_xinrui
# import _utils_3900x
# mypy error: Module 'DRE' has no attribute '_utils_tsne'
from . import _utils_tsne  # type: ignore
# from sklearn_xinrui.manifold import _utils_xinrui_ofast
from umap_xinrui.umap_ import fuzzy_simplicial_set
import numba

import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist
from scipy.special import gamma
from scipy.stats import chi


np.random.seed(42)
NPY_INFINITY = np.inf
SMOOTH_K_TOLERANCE = 1e-5  # 1e-5
MACHINE_EPSILON_NP = np.finfo(np.double).eps
# MACHINE_EPSILON_NP = 1e-14
MACHINE_EPSILON_TORCH = torch.finfo(torch.float32).eps  # used for Q function correction (prevent nan)
MACHINE_EPSILON_SPACE = 0.01  # used for Q function correction (prevent nan)
# MACHINE_EPSILON_TORCH = 1e-3

# <================ New calculation method for P matrix and KL divergence: ================>


def pairwise_distances(x):
    """
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = (x**2).sum(1).view(-1, 1)
    dist2 = x_norm + x_norm.view(1, -1) - 2.0 * torch.mm(x, torch.transpose(x, 0, 1))
    return dist2


'''
=========================================================================
                            calculation: (P)
=========================================================================
'''


@numba.njit(parallel=True)
def fast_knn_indices(X, n_neighbors, k):
    knn_indices = np.empty((X.shape[0], k), dtype=np.int32)
    for row in numba.prange(X.shape[0]):
        # v = np.argsort(X[row])  # Need to call argsort this way for numba
        v = X[row].argsort(kind="quicksort")
        v = v[n_neighbors:(n_neighbors+k)]
        knn_indices[row] = v
    return knn_indices


def nearest_neighbors(
    X,
    n_neighbors, k,
):
    knn_indices = fast_knn_indices(X, n_neighbors, k)
    knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()
    # Prune any nearest neighbours that are infinite distance apart.
    disconnected_index = knn_dists == np.inf
    knn_indices[disconnected_index] = -1
    return knn_indices, knn_dists


@numba.njit(
    locals={
        "psum": numba.types.float32,
        "lo": numba.types.float32,
        "mid": numba.types.float32,
        "hi": numba.types.float32,
    },
    fastmath=True,
)  # benchmarking `parallel=True` shows it to *decrease* performance
def binary_search_sigma(distances, k, chi_concern_rate, n_iter=64, bandwidth=1.0):  # n_iter=64
    target = (1-chi_concern_rate) * np.log2(k) * bandwidth
    sigma = np.zeros(distances.shape[0], dtype=np.float32)
    last_nn = np.zeros(distances.shape[0], dtype=np.float32)

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0
        last_nn[i] = np.min(distances[i])
        for n in range(n_iter):

            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = distances[i, j] - last_nn[i]
                if d >= 0:
                    psum += (1-chi_concern_rate) * np.exp(-(np.power(d, 2) / mid))  # exp2
                    # psum += (1 - chi_concern_rate) * np.exp(-(d / mid))  # exp1
                # else:
                #     psum += 1-chi_concern_rate

            if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == NPY_INFINITY:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0

        sigma[i] = mid
    return sigma, last_nn


@numba.njit(
    locals={
        "knn_dists": numba.types.float32[:, ::1],
        "sigmas": numba.types.float32[::1],
        "rhos": numba.types.float32[::1],
        "val": numba.types.float32,
    },
    parallel=True,
    fastmath=True,
)
def compute_membership_strengths(
    knn_indices, knn_dists, sigmas, rhos, chi_concern_rate, return_dists=False, bipartite=False,
):
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]

    rows = np.zeros(knn_indices.size, dtype=np.int32)
    cols = np.zeros(knn_indices.size, dtype=np.int32)
    vals = np.zeros(knn_indices.size, dtype=np.float32)
    if return_dists:
        dists = np.zeros(knn_indices.size, dtype=np.float32)
    else:
        dists = None

    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            # If applied to an adjacency matrix points shouldn't be similar to themselves.
            # If applied to an incidence matrix (or bipartite) then the row and column indices are different.
            if (bipartite == False) & (knn_indices[i, j] == i):
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0 or sigmas[i] == 0.0:
                val = 1-chi_concern_rate
            else:
                val = (1-chi_concern_rate) * np.exp(-(np.power((knn_dists[i, j] - rhos[i]), 2) / (sigmas[i])))  # exp2
                # val = (1-chi_concern_rate) * np.exp(-(knn_dists[i, j] - rhos[i]) / (sigmas[i]))  # exp1

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val
            if return_dists:
                dists[i * n_neighbors + j] = knn_dists[i, j]

    return rows, cols, vals, dists


def conditional_probability_p(tsne_perplexity, umap_n_neighbors, X, p_type='tsne'):
    D = sklearn.metrics.pairwise_distances(X)
    # builtin function from sklearn:
    if p_type == 'tsne':
        D = np.square(D)
        P = _utils_tsne._binary_search_perplexity(D, tsne_perplexity, 1)  # If use ofast, sometimes some elements are nan!
        P = P + P.T
        P = P / (2*D.shape[0])
        # sum_P = np.maximum(P.sum(), MACHINE_EPSILON)
        # P /= sum_P
    elif p_type == 'umap':
        P, _, _ = fuzzy_simplicial_set(D,
                                       umap_n_neighbors,
                                       None,
                                       "precomputed",
                                       {},
                                       None,
                                       None,
                                       angular=False,
                                       set_op_mix_ratio=1.0,
                                       local_connectivity=1.0,
                                       apply_set_operations=True,
                                       verbose=True,
                                       return_dists=None,
                                       )
        P = P.toarray()
        P = P / np.sum(P)  # normalization
    else:
        raise TypeError('[DRE] p type unavailable!')
    return P


'''
=========================================================================
                            calculation: (Q)
=========================================================================
'''


def calculate_probability_q(D2, a, b, q_type='tsne'):
    # a = 1.929
    # b = 0.7915

    # the entrance of q calculation:
    if q_type == 'tsne':
        qij = t_dist(D2, 1, 1)
    elif q_type == 'umap':
        # qij = wij_fixed(D2, a, b)
        qij = wij_fixed(D2, a, b)
    else:
        raise TypeError('unavailable q type!')
    return qij


def find_ab_params(spread, min_dist):

    def curve(x, a, b):
        return (1.0 + a * x ** 2) ** -b

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]


def t_dist(D2, a, b):
    eps = torch.tensor([MACHINE_EPSILON_NP]).to('cuda')
    non_zeros = torch.ones(D2.shape[0], D2.shape[0]).to('cuda') - torch.eye(D2.shape[0], D2.shape[0]).to('cuda')

    # qij = 1. / (1. + a * (D2 ** b))
    qij = torch.pow((1+a*(torch.pow(D2, b))), -1)
    qij *= non_zeros
    qij = torch.maximum(qij / (torch.sum(qij)), eps)
    return qij


def wij2(D2, a, b):
    eps = torch.tensor([MACHINE_EPSILON_NP]).to('cuda')
    eps2 = torch.tensor([MACHINE_EPSILON_TORCH]).to('cuda')
    non_zeros = torch.ones(D2.shape[0], D2.shape[0]).to('cuda') - torch.eye(D2.shape[0], D2.shape[0]).to('cuda')
    qij = 1. / (1. + a * ((D2+eps2) ** b))
    qij *= non_zeros
    qij = torch.maximum(qij, eps)  # without normalization
    # qij = torch.maximum(qij / (torch.sum(qij)), eps)
    return qij


def wij_fixed(D2, a, b):
    eps = torch.tensor([MACHINE_EPSILON_NP]).to('cuda')
    non_zeros = torch.ones(D2.shape[0], D2.shape[0]).to('cuda') - torch.eye(D2.shape[0], D2.shape[0]).to('cuda')
    qij = (1. + a * D2) ** -b
    qij *= non_zeros
    # qij = torch.maximum(qij, eps)  # without normalization
    qij = torch.maximum(qij / (torch.sum(qij)), eps)
    return qij


'''
=========================================================================
                            Loss functions:
=========================================================================
'''


def kl_divergence(X_embedded, P, a, b, q_type):  # skip_num_points=0, compute_error=True):
    # X_embedded = params.reshape(n_samples, n_components)

    # Q is a heavy-tailed distribution: Student's t-distribution
    eps = torch.tensor([MACHINE_EPSILON_NP]).to('cuda')

    dist = pairwise_distances(X_embedded)

    Q = calculate_probability_q(dist, a, b, q_type=q_type)

    # np.save('P.npy', np.array(P.to('cpu').detach()))
    # np.save('Q.npy', np.array(Q.to('cpu').detach()))
    kl_divergence = torch.sum(P * torch.log(torch.maximum(P, eps) / Q))

    return kl_divergence


def fuzzy_set_cross_entropy(P, Y, a, b, q_type='tsne'):
    eps = torch.tensor([MACHINE_EPSILON_NP]).to('cuda')
    # sum_Y = torch.sum(torch.square(Y), dim=1)
    D = pairwise_distances(Y)
    D += 2*torch.eye(D.shape[0], D.shape[0]).to('cuda')

    Q = calculate_probability_q(D, a, b, q_type=q_type)

    C1 = torch.sum(P * torch.log(torch.maximum(P, eps) / Q))
    C2 = torch.sum((1 - P) * torch.log(torch.maximum(1-P, eps) / torch.maximum(1-Q, eps)))
    C = C1 + C2
    return C


'''
=========================================================================
                            Loss functions:
=========================================================================
'''


def loss_function(X, Y, a, b, type='pre'):
    if type == 'pre' or type == 're1' or type == 're2' or type == 're3':
        return kl_divergence(Y, X, a, b, q_type='tsne')
    elif type == 're_umap':
        return fuzzy_set_cross_entropy(X, Y, a, b, q_type='umap')
    else:
        raise TypeError('[DRE] the input DRE type is wrong.')




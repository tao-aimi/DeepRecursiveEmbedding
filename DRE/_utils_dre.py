import numpy as np
import copy
import locale

import torch

import sklearn
import sklearn.metrics

from . import _utils_tsne  # type: ignore
# from sklearn_xinrui.manifold import _utils_xinrui_ofast
import numba

import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.optimize import curve_fit


np.random.seed(42)
MACHINE_EPSILON_NP = np.finfo(np.double).eps
# MACHINE_EPSILON_NP = 1e-14
MACHINE_EPSILON_TORCH = torch.finfo(torch.float32).eps  # used for Q function correction (prevent nan)
MACHINE_EPSILON_SPACE = 0.01  # used for Q function correction (prevent nan)
# MACHINE_EPSILON_TORCH = 1e-3
locale.setlocale(locale.LC_NUMERIC, "C")

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

SMOOTH_K_TOLERANCE = 1e-5  # 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf
MACHINE_EPSILON = np.finfo(np.double).eps

DISCONNECTION_DISTANCES = {
    "correlation": 1,
    "cosine": 1,
    "hellinger": 1,
    "jaccard": 1,
    "dice": 1,
}

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

#
# @numba.njit(
#     locals={
#         "psum": numba.types.float32,
#         "lo": numba.types.float32,
#         "mid": numba.types.float32,
#         "hi": numba.types.float32,
#     },
#     fastmath=True,
# )  # benchmarking `parallel=True` shows it to *decrease* performance
# def binary_search_sigma(distances, k, chi_concern_rate, n_iter=64, bandwidth=1.0):  # n_iter=64
#     target = (1-chi_concern_rate) * np.log2(k) * bandwidth
#     sigma = np.zeros(distances.shape[0], dtype=np.float32)
#     last_nn = np.zeros(distances.shape[0], dtype=np.float32)
#
#     for i in range(distances.shape[0]):
#         lo = 0.0
#         hi = NPY_INFINITY
#         mid = 1.0
#         last_nn[i] = np.min(distances[i])
#         for n in range(n_iter):
#
#             psum = 0.0
#             for j in range(1, distances.shape[1]):
#                 d = distances[i, j] - last_nn[i]
#                 if d >= 0:
#                     psum += (1-chi_concern_rate) * np.exp(-(np.power(d, 2) / mid))  # exp2
#                     # psum += (1 - chi_concern_rate) * np.exp(-(d / mid))  # exp1
#                 # else:
#                 #     psum += 1-chi_concern_rate
#
#             if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
#                 break
#
#             if psum > target:
#                 hi = mid
#                 mid = (lo + hi) / 2.0
#             else:
#                 lo = mid
#                 if hi == NPY_INFINITY:
#                     mid *= 2
#                 else:
#                     mid = (lo + hi) / 2.0
#
#         sigma[i] = mid
#     return sigma, last_nn
#

@numba.njit(
    locals={
        "psum": numba.types.float32,
        "lo": numba.types.float32,
        "mid": numba.types.float32,
        "hi": numba.types.float32,
    },
    fastmath=True,
)  # benchmarking `parallel=True` shows it to *decrease* performance
def smooth_knn_dist(distances, k, n_iter=64, local_connectivity=1.0, bandwidth=1.0):  # n_iter=64
    """Compute a continuous version of the distance to the kth nearest
    neighbor. That is, this is similar to knn-distance but allows continuous
    k values rather than requiring an integral k. In essence we are simply
    computing the distance such that the cardinality of fuzzy set we generate
    is k.

    Parameters
    ----------
    distances: array of shape (n_samples, n_neighbors)
        Distances to nearest neighbors for each samples. Each row should be a
        sorted list of distances to a given samples nearest neighbors.

    k: float
        The number of nearest neighbors to approximate for.

    n_iter: int (optional, default 64)
        We need to binary search for the correct distance value. This is the
        max number of iterations to use in such a search.

    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.

    bandwidth: float (optional, default 1)
        The target bandwidth of the kernel, larger values will produce
        larger return values.

    Returns
    -------
    knn_dist: array of shape (n_samples,)
        The distance to kth nearest neighbor, as suitably approximated.

    nn_dist: array of shape (n_samples,)
        The distance to the 1st nearest neighbor for each point.
    """
    target = np.log2(k) * bandwidth
    rho = np.zeros(distances.shape[0], dtype=np.float32)
    result = np.zeros(distances.shape[0], dtype=np.float32)

    mean_distances = np.mean(distances)

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0

        # TODO: This is very inefficient, but will do for now. FIXME
        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if non_zero_dists.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = non_zero_dists[index - 1]
                if interpolation > SMOOTH_K_TOLERANCE:
                    rho[i] += interpolation * (
                        non_zero_dists[index] - non_zero_dists[index - 1]
                    )
            else:
                rho[i] = interpolation * non_zero_dists[0]
                # rho[i] = interpolation * non_zero_dists[1]  # for bayes!
        elif non_zero_dists.shape[0] > 0:
            rho[i] = np.max(non_zero_dists)

        for n in range(n_iter):

            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = distances[i, j] - rho[i]
                if d > 0:
                    psum += np.exp(-(d / mid))
                else:
                    psum += 1.0

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

        result[i] = mid

        # TODO: This is very inefficient, but will do for now. FIXME
        if rho[i] > 0.0:
            mean_ith_distances = np.mean(ith_distances)
            if result[i] < MIN_K_DIST_SCALE * mean_ith_distances:
                result[i] = MIN_K_DIST_SCALE * mean_ith_distances
        else:
            if result[i] < MIN_K_DIST_SCALE * mean_distances:
                result[i] = MIN_K_DIST_SCALE * mean_distances

    return result, rho


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
    knn_indices, knn_dists, sigmas, rhos, return_dists=False, bipartite=False,
):
    """Construct the membership strength data for the 1-skeleton of each local
    fuzzy simplicial set -- this is formed as a sparse matrix where each row is
    a local fuzzy simplicial set, with a membership strength for the
    1-simplex to each other data point.

    Parameters
    ----------
    knn_indices: array of shape (n_samples, n_neighbors)
        The indices on the ``n_neighbors`` closest points in the dataset.

    knn_dists: array of shape (n_samples, n_neighbors)
        The distances to the ``n_neighbors`` closest points in the dataset.

    sigmas: array of shape(n_samples)
        The normalization factor derived from the metric tensor approximation.

    rhos: array of shape(n_samples)
        The local connectivity adjustment.

    return_dists: bool (optional, default False)
        Whether to return the pairwise distance associated with each edge

    bipartite: bool (optional, default False)
        Does the nearest neighbour set represent a bipartite graph?  That is are the
        nearest neighbour indices from the same point set as the row indices?

    Returns
    -------
    rows: array of shape (n_samples * n_neighbors)
        Row data for the resulting sparse matrix (coo format)

    cols: array of shape (n_samples * n_neighbors)
        Column data for the resulting sparse matrix (coo format)

    vals: array of shape (n_samples * n_neighbors)
        Entries for the resulting sparse matrix (coo format)

    dists: array of shape (n_samples * n_neighbors)
        Distance associated with each entry in the resulting sparse matrix
    """
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
                val = 1.0
            else:
                # *******************************************************************************
                # ******************************** main function ********************************
                # *******************************************************************************
                val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val
            if return_dists:
                dists[i * n_neighbors + j] = knn_dists[i, j]

    return rows, cols, vals, dists


def fuzzy_simplicial_set(
    X,
    n_neighbors,
    random_state,
    metric,
    metric_kwds={},
    knn_indices=None,
    knn_dists=None,
    angular=False,
    set_op_mix_ratio=1.0,
    local_connectivity=1.0,
    apply_set_operations=True,
    verbose=False,
    return_dists=None,
):
    if knn_indices is None or knn_dists is None:

        # numba:
        knn_indices, knn_dists = nearest_neighbors(
            X, n_neighbors, n_neighbors
        )

    knn_dists = knn_dists.astype(np.float32)

    # numba: This step already compute the P matrix inplicitly, TODO: extract this P matrix
    # only use knn_dists
    sigmas, rhos = smooth_knn_dist(
        knn_dists, float(n_neighbors), local_connectivity=float(local_connectivity),
    )

    # numba:
    # use knn_dists and knn_indices
    rows, cols, vals, dists = compute_membership_strengths(
        knn_indices, knn_dists, sigmas, rhos, return_dists
    )

    result = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(X.shape[0], X.shape[0])
    )
    result.eliminate_zeros()

    if apply_set_operations:
        transpose = result.transpose()

        prod_matrix = result.multiply(transpose)

        result = (
            set_op_mix_ratio * (result + transpose - prod_matrix)
            + (1.0 - set_op_mix_ratio) * prod_matrix
        )

    result.eliminate_zeros()

    if return_dists is None:
        return result, sigmas, rhos
    else:
        if return_dists:
            dmat = scipy.sparse.coo_matrix(
                (dists, (rows, cols)), shape=(X.shape[0], X.shape[0])
            )

            dists = dmat.maximum(dmat.transpose()).todok()
        else:
            dists = None

        return result, sigmas, rhos, dists


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




"""Capacity-aware assignment of coherent serving links."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from .arith import f


def round_graph(n, t, x, fixed=None):
    M, K = (n["M"], n["K"])
    a = np.asarray(x)
    a = np.maximum(0, a) * n["a"]
    a = a / np.maximum(a.sum(0), 1e-12)
    assert np.all(n["w"] == 1)
    cap = np.array([min(K, int(f(c) // f(t))) for c in n["C"]])
    owners = np.repeat(np.arange(M), cap)
    if len(owners) < K:
        return None
    if fixed and len(fixed) == M * K:
        A = np.array([fixed[j] for j in range(M * K)], bool).reshape(M, K)
    else:
        quality = a + 1e-05 * np.log1p(n["a"] ** 2)
        rr, cc = linear_sum_assignment(-quality[owners].T)
        A = np.zeros((M, K), bool)
        A[owners[cc], rr] = True
        for m in range(M):
            for _ in range(cap[m] - int(A[m].sum())):
                options = [k for k in range(K) if not A[m, k] and A[:, k].sum() < n["Dmax"]]
                if not options:
                    break
                k = max(options, key=lambda k: quality[m, k])
                A[m, k] = True
    if np.any(A.sum(0) < 1) or np.any(A.sum(0) > n["Dmax"]) or np.any(A.sum(1) > cap):
        return None
    return A

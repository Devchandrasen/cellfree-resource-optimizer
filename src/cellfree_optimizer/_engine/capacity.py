"""Capacity."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def check_parameters(n):
    M, K = (n["M"], n["K"])
    assert M >= 1 and K >= 1 and (1 <= n["Dmax"] <= M)
    assert np.shape(n["C"]) == (M,) and np.shape(n["w"]) == (K,)
    assert np.all(np.isfinite(n["C"])) and np.all(n["C"] > 0)
    assert np.all(np.isfinite(n["w"])) and np.all(n["w"] > 0)
    assert 0 < n["rho"] < 1


def coherent_upper(n):
    """Safe pure-CJT payload cap; stronger unit/equal-weight occupancy bound."""
    C, w = (n["C"], n["w"])
    upper = min(float(C.max() / w.max()), float(C.sum() / w.sum()))
    if np.allclose(w, w[0], rtol=0, atol=1e-12):
        if np.allclose(C, C[0], rtol=0, atol=1e-12):
            return min(upper, float(C[0] / w[0] / np.ceil(n["K"] / n["M"])))
        lo = 0.0
        hi = upper
        for _ in range(45):
            t = (lo + hi) / 2
            if np.minimum(n["K"], np.floor(C / (w[0] * t) + 1e-12)).sum() >= n["K"]:
                lo = t
            else:
                hi = t
        upper = min(upper, hi + 1e-10)
    return upper


def slots(n, target):
    """Necessary AP occupancy limit; exact for common weights, safe otherwise."""
    if target <= 0:
        return np.full(n["M"], n["K"], int)
    return np.clip(np.floor(n["C"] / (target * n["w"].min()) + 1e-09), 0, n["K"]).astype(int)


def round_slots(n, score, rng, power=1.0, noise=0.0, active=None, reserved=(), target=None):
    active = np.arange(n["K"]) if active is None else np.asarray(active, int)
    A = np.zeros((n["M"], n["K"]), bool)
    if len(active) == 0:
        return A
    q = slots(n, target if target is not None else coherent_upper(n))
    q = np.minimum(q, len(active))
    q[list(reserved)] = 0
    if q.sum() < len(active):
        return None
    aps = np.repeat(np.arange(n["M"]), q)
    degree = n["Dmax"]
    quality = np.maximum(np.asarray(score[:, active], float), 1e-15).copy()
    if noise:
        quality *= np.exp(rng.normal(0, noise, quality.shape))
    quality /= max(quality.max(), 1e-15)
    users = np.repeat(active, degree)
    local = np.repeat(np.arange(len(active)), degree)
    dd = np.tile(np.arange(degree), len(active))
    cost = quality[aps[:, None], local[None, :]] / (dd[None, :] + 1.0) ** power
    cost[:, dd == 0] += 1000
    rr, cc = linear_sum_assignment(-cost)
    A[aps[rr], users[cc]] = True
    if not np.all(A[:, active].any(axis=0)):
        return None
    for m in rng.permutation(n["M"]):
        while A[m].sum() < q[m]:
            eligible = active[~A[m, active] & (A[:, active].sum(axis=0) < n["Dmax"])]
            if not len(eligible):
                break
            col = (
                np.searchsorted(active, eligible)
                if np.all(np.diff(active) >= 0)
                else np.array([np.where(active == k)[0][0] for k in eligible])
            )
            value = quality[m, col] / (A[:, eligible].sum(axis=0) + 1.0) ** power
            A[m, eligible[np.argmax(value)]] = True
    return A

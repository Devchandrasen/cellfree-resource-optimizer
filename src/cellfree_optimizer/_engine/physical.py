"""Statistical radio rates, feasible recovery and constraint checks."""

import numpy as np


def rates(n, A, z, x):
    z = np.asarray(z, bool).reshape(n["K"])
    p = x * x
    signal = n["a"] ** 2 * p
    D = 1 + np.einsum("mki,mi->k", n["g"], p)
    rc = np.where(z, n["rho"] / np.log(2) * np.log1p(np.sum(n["a"] * x, axis=0) ** 2 / D), 0)
    rn = np.zeros_like(x)
    for k in range(n["K"]):
        if z[k]:
            continue
        tail = 0.0
        for m in n["order"][:, k][::-1]:
            if A[m, k]:
                rn[m, k] = n["rho"] / np.log(2) * np.log1p(signal[m, k] / (D[k] + tail))
            tail += signal[m, k]
    return (rc, rn)


def certify(n, A, z, x, t, u):
    A = np.asarray(A, bool).reshape(n["M"], n["K"])
    z = np.asarray(z, bool).reshape(n["K"])
    x = np.clip(np.asarray(x, float).reshape(A.shape), 0, A.astype(float))
    x /= np.maximum(1, np.linalg.norm(x, axis=1))[:, None]
    rc, rn = rates(n, A, z, x)
    u = np.clip(np.asarray(u, float).reshape(A.shape), 0, rn)
    u[~A] = 0
    u[:, z] = 0
    u *= 1 - 1e-10
    c = np.zeros(n["K"])
    c[z] = np.minimum(n["w"][z] * max(0, t), rc[z]) * (1 - 1e-10)
    t = max(0, min(float(t), float(np.min((c + u.sum(axis=0)) / n["w"]))))
    c[z] = n["w"][z] * t
    for k in np.where(~z)[0]:
        if u[:, k].sum() > 0:
            u[:, k] *= n["w"][k] * t / u[:, k].sum()
    load = A[:, z] @ c[z] + u.sum(axis=1)
    ratio = min(1, float(np.min(n["C"] / np.maximum(load, 1e-300))))
    if ratio < 1:
        ratio *= 1 - 1e-10
        t *= ratio
        c *= ratio
        u *= ratio
    return dict(A=A, z=z, x=x, t=t, u=u, rC=c)


def validate(n, s):
    A = s["A"]
    z = s["z"]
    x = s["x"]
    t = s["t"]
    u = s["u"]
    c = s["rC"]
    rc, rn = rates(n, A, z, x)
    load = A[:, z] @ c[z] + u.sum(axis=1)
    R = c + u.sum(axis=0)
    residual = max(
        0,
        float(np.max(x * x @ np.ones(n["K"]) - 1)),
        float(np.max((load - n["C"]) / np.maximum(1, n["C"]))),
        float(np.max((n["w"] * t - R) / np.maximum(1, n["w"] * t))),
        float(np.max((c[z] - rc[z]) / np.maximum(1, rc[z]))) if z.any() else 0,
        float(np.max((u - rn) / np.maximum(1, rn))),
        float(np.max(np.abs(x[~A]))) if (~A).any() else 0,
        float(np.max(np.abs(u[~A]))) if (~A).any() else 0,
        float(np.max(1 - A.sum(axis=0))),
        float(np.max(A.sum(axis=0) - n["Dmax"])),
    )
    assert np.all(np.isfinite(x)) and residual <= 2e-06, (residual, t)
    return residual


def isolated_bounds(n):
    M, K = (n["M"], n["K"])
    D = n["Dmax"]
    a2 = n["a"] ** 2
    own = n["g"][:, np.arange(K), np.arange(K)]
    csolo = np.zeros((K, D))
    nsolo = np.zeros(K)
    for k in range(K):
        for d in range(1, D + 1):
            csolo[k, d - 1] = (
                n["rho"] / np.log(2) * np.log1p(np.sort(a2[:, k] / (own[:, k] + 1 / d))[-d:].sum())
            )
        lo = 0.0
        hi = min(
            float(np.max(a2[:, k] / np.maximum(own[:, k], 1e-300))),
            float(np.sort(a2[:, k])[-D:].sum()),
        )
        for _ in range(60):
            eta = (lo + hi) / 2
            if np.sort(np.maximum(0, a2[:, k] - eta * own[:, k]))[-D:].sum() >= eta:
                lo = eta
            else:
                hi = eta
        nsolo[k] = n["rho"] / np.log(2) * np.log1p(hi) + 1e-08
    return (csolo + 1e-08, nsolo)

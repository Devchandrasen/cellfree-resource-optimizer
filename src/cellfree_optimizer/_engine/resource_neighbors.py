"""Resource neighbors."""

import numpy as np
from .budget_core import payload_upper


def resource_neighbors(n, s):
    A = s["A"]
    z = s["z"]
    M, K = A.shape
    t = s["t"] + 0.0004
    den = 1 + np.einsum("mki,mi->k", n["g"], s["x"] ** 2)
    quality = n["a"] / np.sqrt(den)[None, :]
    cov = np.maximum((quality * s["x"]).sum(axis=0), 1e-12)
    copies = A[:, z] @ n["w"][z]
    residual = np.maximum(0, n["C"] - t * copies)
    load = s["t"] * copies + np.asarray(s["u"]).sum(axis=1)
    pressure = load / np.maximum(n["C"], 1e-12)
    out = []
    seen = set()

    def push(B, score):
        h = B.tobytes()
        if h in seen or np.any(B.sum(axis=0) < 1) or np.any(B.sum(axis=0) > n["Dmax"]):
            return
        seen.add(h)
        cap = payload_upper(n, B, z)
        if cap < t - 1e-08:
            return
        out.append((score + min(0.1, cap / max(t, 1e-09) - 1), B, z.copy()))

    for k in np.flatnonzero(z):
        active = np.flatnonzero(A[:, k])
        inactive = np.flatnonzero(~A[:, k])
        for m in active:
            if pressure[m] < 0.8:
                continue
            loss = quality[m, k] * s["x"][m, k] / cov[k]
            if len(active) > 1:
                B = A.copy()
                B[m, k] = False
                push(B, pressure[m] - loss)
            for l in inactive:
                if residual[l] < t * n["w"][k] - 1e-08:
                    continue
                B = A.copy()
                B[m, k] = False
                B[l, k] = True
                push(B, pressure[m] - loss + 0.15 * quality[l, k] / max(quality[:, k].max(), 1e-12))
    for k in np.flatnonzero(~z):
        rank = np.argsort(-(quality[:, k] ** 2 * np.maximum(residual, 0.005)))
        for degree in sorted(set([2, 4, int(n["Dmax"])])):
            B = A.copy()
            B[:, k] = False
            B[rank[:degree], k] = True
            push(B, 0.8)
        for m in np.flatnonzero(A[:, k]):
            for l in rank[:4]:
                if A[l, k]:
                    continue
                B = A.copy()
                B[m, k] = False
                B[l, k] = True
                push(B, 0.6 + pressure[m] * float(s["u"][m, k]) / max(s["t"], 1e-12))
    out.sort(key=lambda v: v[0], reverse=True)
    return [(A, z) for _, A, z in out[:48]]

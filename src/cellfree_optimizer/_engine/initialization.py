"""Initialization."""

import time
import numpy as np
import scipy.sparse as sp
from .budget_core import Expired, payload_upper
from .fast_kernels import coherent, solve_reusable
from .capacity import coherent_upper, round_slots, slots
from .search import seed_graphs
from .assignment import round_graph


def envelope(n, b, end, target, weight, owner):
    b.check(end)
    M, K = (n["M"], n["K"])
    E = M * K
    q = slots(n, target)
    if q.sum() < K:
        return None
    base = sp.vstack([-sp.eye(E), sp.kron(sp.eye(M), np.ones((1, K)))], format="csc")
    blocks = [base]
    rhs = [np.r_[np.zeros(E), np.sqrt(q)]]
    from .budget_core import clarabel

    cones = [clarabel.NonnegativeConeT(E + M)]
    for m in range(M):
        B = sp.coo_matrix(
            (-np.ones(K), (np.arange(1, K + 1), m * K + np.arange(K))), shape=(K + 1, E)
        ).tocsc()
        blocks.append(B)
        rhs.append(np.r_[1.0, np.zeros(K)])
        cones.append(clarabel.SecondOrderConeT(K + 1))
    eta = np.expm1(np.log(2) * target * n["w"] / n["rho"])
    scale = np.sqrt(1 + n["g"].sum(axis=(0, 2)) / max(1, K * K))
    for k in range(K):
        row = np.zeros(E)
        row[np.arange(M) * K + k] = -n["a"][:, k] / np.sqrt(eta[k]) / scale[k]
        B = sp.vstack(
            [
                sp.csc_matrix(row[None, :]),
                sp.csc_matrix((1, E)),
                -sp.diags(np.sqrt(n["g"][:, k, :]).ravel() / scale[k]),
            ],
            format="csc",
        )
        blocks.append(B)
        rhs.append(np.r_[0.0, 1 / scale[k], np.zeros(E)])
        cones.append(clarabel.SecondOrderConeT(E + 2))
    if owner.solver is not None:
        owner.solver.update(q=np.asarray(weight).ravel())
    result = solve_reusable(
        owner,
        sp.vstack(blocks, format="csc"),
        np.concatenate(rhs),
        np.asarray(weight).ravel(),
        cones,
        b,
        end,
    )
    if result.x is None or not np.isfinite(result.x).all():
        return None
    x = np.maximum(0, np.asarray(result.x).reshape(M, K))
    x /= np.maximum(1, np.linalg.norm(x, axis=1))[:, None]
    D = 1 + np.einsum("mki,mi->k", n["g"], x * x)
    rate = np.min(n["rho"] / np.log(2) * np.log1p((n["a"] * x).sum(axis=0) ** 2 / D) / n["w"])
    return x if rate >= target * (1 - 3e-05) else None


def initialise(n, b, rng, stats, end):
    start = time.perf_counter()
    z = np.ones(n["K"], bool)
    seen = set()
    relax = None

    def polish(A, parent=None):
        if A is None:
            return
        b.check(end)
        key = A.tobytes()
        if key in seen:
            return
        seen.add(key)
        if payload_upper(n, A, z) < b.best["t"] + 0.0001:
            return
        target = None if b.best["t"] < 1e-06 else b.best["t"] + 0.0004
        s = coherent(n, A, b, parent, end, rounds=12, target=target)
        b.accept(s, "cold coherent seed")
        stats["seed_graphs"] += 1

    seeds = seed_graphs(n, rng)
    polish(seeds[0])
    hi = coherent_upper(n)
    owner = type("Holder", (), dict(solver=None))()
    previous = None
    for iteration in range(10):
        if b.left(end) < 0.08:
            break
        target = hi * (1 - 1e-08) if iteration % 3 != 2 else (hi + b.best["t"]) / 2
        target = max(target, b.best["t"] + 0.0004)
        weight = np.ones_like(n["a"]) if previous is None else 1 / np.maximum(previous, 0.025)
        weight /= max(float(weight.mean()), 1e-12)
        if iteration >= 6:
            weight *= np.exp(rng.normal(0, 0.18, weight.shape))
        x = envelope(n, b, end, target, weight, owner)
        stats["seed_envelopes"] += 1
        if x is None:
            hi = (hi + b.best["t"]) / 2
            previous = None
            continue
        previous = x
        relax = x
        score = n["a"] * x
        score /= np.maximum(score.sum(axis=0), 1e-15)[None, :]
        targets = [target, b.best["t"] + 0.0004]
        for trial in targets:
            polish(round_graph(n, trial, x), b.best)
            for power in [0.5, 1.5]:
                polish(
                    round_slots(
                        n,
                        score,
                        rng,
                        power=power,
                        noise=0.0 if iteration < 4 else 0.12,
                        target=trial,
                    ),
                    b.best,
                )
        if b.best["t"] >= coherent_upper(n) - 2e-05:
            break
    for A in seeds[1:]:
        if b.left(end) < 0.05:
            break
        polish(A, b.best)
    if relax is None:
        relax = n["a"] / np.maximum(1, np.linalg.norm(n["a"], axis=1))[:, None]
    from .root_guidance import guide

    root_x = guide(n, b, rng, stats, min(end, time.perf_counter() + 4.0), polish, relax=relax)
    if root_x is not None:
        relax = root_x
    stats["seed_seconds"] = time.perf_counter() - start
    stats["seed_rate"] = b.best["t"]
    return relax

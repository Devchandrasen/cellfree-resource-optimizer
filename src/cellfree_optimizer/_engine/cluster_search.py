"""Cluster search."""

import time
import itertools
import collections
import warnings
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, Bounds, LinearConstraint
from .physical import isolated_bounds
from .budget_core import Budget, Expired, payload_upper
from .fast_kernels import coherent
from .search import seed_graphs, neighbors, key
from .capacity import slots, round_slots, check_parameters, coherent_upper
from .joint_general import relaxed_coherent
from .mode_search import bundled_graphs, probe


def fair_cluster(n, parent, rng, budget, end=None, noise=0.0, limit=0.18):
    """Max-min amplitude proxy with integral AP occupancy, not a rate bound."""
    budget.check(end)
    M, K = (n["M"], n["K"])
    E = M * K
    target = parent["t"] + 0.0004
    q = slots(n, target)
    if q.sum() < K:
        return None
    den = 1 + np.einsum("mki,mi->k", n["g"], parent["x"] ** 2)
    eta = np.expm1(np.log(2) * max(target, 0.01) * n["w"] / n["rho"])
    quality = n["a"] / np.sqrt(np.maximum(q, 1))[:, None] / np.sqrt(den * eta)[None, :]
    if noise:
        quality *= np.exp(rng.normal(0, noise, quality.shape))
    quality /= max(1.0, quality.max())
    ap = sp.kron(sp.eye(M), np.ones((1, K)), format="csc")
    user = sp.kron(np.ones((1, M)), sp.eye(K), format="csc")
    cover = sp.coo_matrix(
        (quality.ravel(), (np.tile(np.arange(K), M), np.arange(E))), shape=(K, E)
    ).tocsc()
    matrix = sp.vstack(
        [
            sp.hstack([ap, sp.csc_matrix((M, 1))]),
            sp.hstack([user, sp.csc_matrix((K, 1))]),
            sp.hstack([cover, -np.ones((K, 1))]),
        ],
        format="csc",
    )
    objective = np.r_[np.full(E, 1e-07), -1.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = milp(
            objective,
            integrality=np.r_[np.ones(E), 0],
            bounds=Bounds(np.zeros(E + 1), np.r_[np.repeat(q > 0, K).astype(float), np.inf]),
            constraints=LinearConstraint(
                matrix,
                np.r_[np.zeros(M), np.ones(K), np.zeros(K)],
                np.r_[q, np.full(K, n["Dmax"]), np.full(K, np.inf)],
            ),
            options=dict(
                time_limit=max(0.005, min(limit, budget.left(end) - 0.006)),
                threads=1,
                mip_rel_gap=0.015,
            ),
        )
    budget.calls += 1
    if r.x is None or not np.isfinite(r.x).all():
        return None
    A = r.x[:E].reshape(M, K) > 0.5
    if np.any(A.sum(axis=1) > q) or np.any(A.sum(axis=0) < 1) or np.any(A.sum(axis=0) > n["Dmax"]):
        return None
    return A


def ranked_moves(n, parent, rng, budget):
    """All feasible one-link transfers and two-AP swaps, no random 160 cap."""
    A = parent["A"]
    M, K = A.shape
    q = slots(n, parent["t"] + 0.0004)
    degrees = A.sum(axis=0)
    loads = A.sum(axis=1)
    den = 1 + np.einsum("mki,mi->k", n["g"], parent["x"] ** 2)
    quality = n["a"] / np.sqrt(np.maximum(1, loads))[:, None] / np.sqrt(den)[None, :]
    coverage = np.maximum((quality * A).sum(axis=0), 1e-12)
    pool = []
    seen = set()
    z = np.ones(K, bool)

    def push(B):
        h = B.tobytes()
        if h in seen:
            return
        seen.add(h)
        if (
            np.any(B.sum(axis=1) > q)
            or np.any(B.sum(axis=0) < 1)
            or np.any(B.sum(axis=0) > n["Dmax"])
        ):
            return
        ratio = np.maximum((quality * B).sum(axis=0) / coverage, 1e-12)
        score = 10 * float(ratio.min()) + float(np.log(ratio).sum()) / K
        pool.append((score, float(rng.random()), B))

    for k in range(K):
        budget.check()
        active = np.flatnonzero(A[:, k])
        inactive = np.flatnonzero(~A[:, k])
        if degrees[k] < n["Dmax"]:
            for m in inactive:
                if loads[m] < q[m]:
                    B = A.copy()
                    B[m, k] = True
                    push(B)
        if degrees[k] > 1:
            for m in active:
                B = A.copy()
                B[m, k] = False
                push(B)
        for m in active:
            for l in inactive:
                if loads[l] < q[l]:
                    B = A.copy()
                    B[m, k] = False
                    B[l, k] = True
                    push(B)
    links = list(zip(*np.nonzero(A)))
    for m, k in links:
        if degrees[k] > 1:
            for i in range(K):
                if i != k and (not A[m, i]) and (degrees[i] < n["Dmax"]):
                    B = A.copy()
                    B[m, k] = False
                    B[m, i] = True
                    push(B)
    for (m, k), (l, i) in itertools.combinations(links, 2):
        if m != l and k != i and (not A[m, i]) and (not A[l, k]):
            B = A.copy()
            B[m, k] = False
            B[l, i] = False
            B[m, i] = True
            B[l, k] = True
            push(B)
    pool.sort(key=lambda t: (t[0], t[1]))
    return [B for _, _, B in pool]


def run(n, policy="joint_cluster", seconds=60, seed=20261027):
    pure = policy == "strongC"
    check_parameters(n)
    rng = np.random.default_rng(seed)
    mrng = np.random.default_rng(seed + 19727)
    bestC = None
    cache = {}
    examined = {}
    events = []
    stats = collections.Counter()

    class LedgerBudget(Budget):
        def accept(self, s, label):
            nonlocal bestC
            accepted = super().accept(s, label)
            if self.left() >= 0 and s["z"].all():
                if bestC is None or s["t"] > bestC["t"]:
                    bestC = s
                h = key(s["A"], s["z"])
                if h not in cache or s["t"] > cache[h]["t"]:
                    cache[h] = s
            return accepted

    b = LedgerBudget(n, "CJT" if pure else "joint", seconds)
    start = b.start
    zC = np.ones(n["K"], bool)
    relax = None
    nsolo = None
    pending = []
    parent_key = None
    last_mip = -100.0
    tested_since_mip = 0
    rest = 0

    def event(label):
        events.append(
            dict(
                stage=label,
                seconds=time.perf_counter() - start,
                rate=b.best["t"],
                coherent=bestC["t"],
            )
        )

    def polish(A, end=None):
        h = key(A, zC)
        target = None if bestC is None else bestC["t"] + 0.0004
        if h in examined and (examined[h] is None or target >= examined[h]):
            stats["cache_hits"] += 1
            return
        if bestC is not None and payload_upper(n, A, zC) < target - 1e-08:
            return
        s = coherent(n, A, b, bestC, end, target=target)
        b.accept(s, "coherent backbone")
        stats["coherent_graphs"] += 1
        examined[h] = None if target is None or s["t"] >= target * (1 - 2e-05) else target

    def backbone(end):
        nonlocal pending, parent_key, last_mip, tested_since_mip, rest
        while b.left(end) > 0.025:
            b.check(end)
            now = time.perf_counter()
            if now - last_mip > 1.25 and tested_since_mip >= 24 or last_mip < 0:
                last_mip = now
                tested_since_mip = 0
                rest += 1
                A = fair_cluster(
                    n, bestC, rng, b, end, noise=0.0 if rest == 1 else min(0.8, 0.12 + 0.025 * rest)
                )
                stats["fair_cluster_calls"] += 1
                if A is not None:
                    polish(A, end)
                continue
            h = key(bestC["A"], zC)
            if parent_key != h:
                pending = ranked_moves(n, bestC, rng, b)
                parent_key = h
                stats["expanded_parents"] += 1
            if pending:
                A = pending[-1]
                polish(A, end)
                pending.pop()
                tested_since_mip += 1
            else:
                tested_since_mip += 24
                A = round_slots(
                    n,
                    relax,
                    rng,
                    power=rng.uniform(0.4, 2.5),
                    noise=rng.uniform(0.15, 1.0),
                    target=bestC["t"] + 0.0004,
                )
                if A is not None:
                    polish(A, end)
                stats["rounded_restarts"] += 1

    try:
        seed_end = start + min(8.0, seconds * 0.3)
        relax, relax_upper = relaxed_coherent(n, b, seed_end)
        if relax is None:
            relax = n["a"] / np.maximum(1, np.linalg.norm(n["a"], axis=1))[:, None]
        graphbank = []
        for power in [0.5, 1.0, 2.0]:
            for noise in [0.0, 0.25, 0.6]:
                A = round_slots(n, relax, rng, power, noise, target=max(1e-06, relax_upper))
                if A is not None:
                    graphbank.append(A)
        graphbank += seed_graphs(n, rng)
        for A in graphbank:
            if b.left(seed_end) < 0.03:
                break
            if key(A, zC) in cache:
                continue
            try:
                polish(A, seed_end)
            except Expired:
                if b.left() < 0.01:
                    raise
                break
        if bestC is None:
            polish(seed_graphs(n, rng)[0])
        nsolo = isolated_bounds(n)[1]
        try:
            backbone(start + seconds * (1.0 if pure else 0.55))
        except Expired:
            if b.left() < 0.01:
                raise
        event("coherent prefix")
        if pure:
            raise Expired
        tried = set()
        probe_round = 0
        coh_reward = 0.0
        mix_reward = 0.0
        mixed_time = 0.0
        coh_time = 0.0
        while b.left() > 0.035:
            if not np.any(nsolo / n["w"] > b.best["t"] + 0.0002):
                stats["no_eligible_mode"] += 1
                backbone(b.deadline)
                break
            duration = 1.5 if mix_reward > coh_reward + 0.0001 else 3.0
            before = b.best["t"]
            t0 = time.perf_counter()
            try:
                backbone(min(b.deadline, t0 + duration))
            except Expired:
                if b.left() < 0.01:
                    raise
            used = time.perf_counter() - t0
            coh_time += used
            coh_reward = 0.6 * coh_reward + 0.4 * max(0, b.best["t"] - before) / max(used, 0.001)
            probe_round += 1
            pool = bundled_graphs(n, bestC, nsolo, relax, mrng)
            if not b.best["z"].all():
                pool += neighbors(n, b.best, "joint", mrng, probe_round)[:40]
            probe_start = time.perf_counter()
            probe_end = min(
                b.deadline, probe_start + (3.0 if mix_reward > coh_reward + 0.0001 else 1.25)
            )
            before = b.best["t"]
            count = 0
            for A, z in pool:
                if b.left(probe_end) < 0.04:
                    break
                h = key(A, z)
                if h in tried or payload_upper(n, A, z) < b.best["t"] + 0.0001:
                    continue
                if z.all():
                    try:
                        polish(A, probe_end)
                    except Expired:
                        break
                    tried.add(h)
                    continue
                if payload_upper(n, A, zC) >= b.best["t"] + 0.0004:
                    try:
                        polish(A, probe_end)
                    except Expired:
                        break
                    tried.add(h)
                    stats["coherent_dominance_redirects"] += 1
                    continue
                end = min(probe_end, time.perf_counter() + 0.65)
                parents = [bestC, b.best]
                warm = min(parents, key=lambda p: np.sum(p["z"] != z) + 0.02 * np.sum(p["A"] != A))
                try:
                    s = probe(n, A, z, b, warm, end, b.best["t"] + 0.0004, stats, iterations=22)
                    b.accept(s, "mixed probe")
                    stats["mixed_graphs"] += 1
                    tried.add(h)
                except Expired:
                    stats["mixed_subtimeouts"] += 1
                    tried.add(h)
                    if b.left() < 0.01:
                        raise
                count += 1
            used = time.perf_counter() - probe_start
            mixed_time += used
            mix_reward = 0.6 * mix_reward + 0.4 * max(0, b.best["t"] - before) / max(used, 0.001)
            stats["mixed_probe_batches"] += 1
            if count == 0:
                try:
                    backbone(min(b.deadline, time.perf_counter() + 2.0))
                except Expired:
                    if b.left() < 0.01:
                        raise
            event("interleaved search")
        stats["interleaved_coherent_milliseconds"] = round(coh_time * 1000)
        stats["mixed_milliseconds"] = round(mixed_time * 1000)
    except Expired:
        pass
    return (
        b.best,
        dict(
            policy=policy,
            budget_seconds=seconds,
            wall_seconds=time.perf_counter() - start,
            cpu_seconds=time.process_time() - b.cpu_start,
            accepted_point_seconds=b.trace[-1]["seconds"],
            trace=b.trace,
            events=events,
            solver_calls=b.calls,
            proposals=b.proposals,
            late_proposals_rejected=b.late,
            max_residual=b.max_residual,
            stop="shared monotonic deadline",
            seed=seed,
            solver_threads=1,
            independent_cold_start=True,
            stats=dict(stats),
            coherent_lower=None if bestC is None else bestC["t"],
            native_statuses=getattr(b, "native_statuses", {}),
        ),
    )

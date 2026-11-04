"""Joint general."""

import time
import itertools
import collections
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from .physical import isolated_bounds, validate
from .budget_core import Budget, Expired, fixed_power, warm_power, payload_upper, clarabel, nc_radio
from .fast_kernels import Aggregate, coherent, mixed, solve_reusable
from .search import seed_graphs, neighbors, key
from .capacity import coherent_upper, slots, check_parameters, round_slots


def relaxed_coherent(n, budget, end=None):
    """Convex amplitude envelope sum_k x_mk<=1; rounding is independently solved."""
    M, K = (n["M"], n["K"])
    E = M * K
    base = sp.vstack([-sp.eye(E), sp.kron(sp.eye(M), np.ones((1, K)))], format="csc")
    power_blocks = []
    for m in range(M):
        rr = np.arange(1, K + 1)
        cc = m * K + np.arange(K)
        power_blocks.append(sp.coo_matrix((-np.ones(K), (rr, cc)), shape=(K + 1, E)).tocsc())
    cones = (
        [clarabel.NonnegativeConeT(E + M)]
        + [clarabel.SecondOrderConeT(K + 1) for _ in range(M)]
        + [clarabel.SecondOrderConeT(E + 2) for _ in range(K)]
    )
    scale = np.sqrt(1 + np.einsum("mki,mi->k", n["g"], np.ones((M, K)) / K**2))
    owner = type("Holder", (), dict(solver=None, indices=None, indptr=None))()
    low = 0.0
    upper = coherent_upper(n)
    best = None
    for j in range(13):
        target = upper if j == 0 else (low + upper) / 2
        eta = np.expm1(np.log(2) * target * n["w"] / n["rho"])
        blocks = [base] + power_blocks
        b = [np.r_[np.zeros(E), np.sqrt(slots(n, target))]] + [
            np.r_[1.0, np.zeros(K)] for _ in range(M)
        ]
        for k in range(K):
            row = np.zeros(E)
            row[np.arange(M) * K + k] = -n["a"][:, k] / np.sqrt(eta[k]) / scale[k]
            blocks.append(
                sp.vstack(
                    [
                        sp.csc_matrix(row[None, :]),
                        sp.csc_matrix((1, E)),
                        -sp.diags(np.sqrt(n["g"][:, k, :]).ravel() / scale[k]),
                    ],
                    format="csc",
                )
            )
            b.append(np.r_[0.0, 1 / scale[k], np.zeros(E)])
        try:
            r = solve_reusable(
                owner,
                sp.vstack(blocks, format="csc"),
                np.concatenate(b),
                np.ones(E) * 0.001,
                cones,
                budget,
                end,
            )
        except Expired:
            break
        good = False
        if r.x is not None and np.isfinite(r.x).all():
            x = np.maximum(0, np.asarray(r.x).reshape(M, K))
            x /= np.maximum(1, np.linalg.norm(x, axis=1))[:, None]
            D = 1 + np.einsum("mki,mi->k", n["g"], x * x)
            rate = float(
                np.min(n["rho"] / np.log(2) * np.log1p((n["a"] * x).sum(axis=0) ** 2 / D) / n["w"])
            )
            if rate >= target * (1 - 2e-05):
                best = x
                low = target
                good = True
        if not good:
            upper = target
        if upper - low < 0.0002:
            break
    return (best, upper)


def coherent_moves(n, s, relax, rng):
    A = s["A"]
    M, K = A.shape
    if np.any(slots(n, s["t"] + 0.0004) > 1) or np.any(A.sum(axis=1) > 1):
        return neighbors(n, s, "CJT", rng, 0)
    owner = np.where(A.any(axis=1), np.argmax(A, axis=1), -1)
    counts = A.sum(axis=0)
    pool = []
    for m in range(M):
        old = owner[m]
        if old >= 0 and counts[old] <= 1:
            continue
        for k in range(K):
            if k == old or counts[k] >= n["Dmax"]:
                continue
            B = A.copy()
            B[m, :] = False
            B[m, k] = True
            score = relax[m, k] - (relax[m, old] if old >= 0 else 0)
            pool.append((score, B))
    for m, l in itertools.combinations(range(M), 2):
        i, j = (owner[m], owner[l])
        if i < 0 or j < 0 or i == j:
            continue
        B = A.copy()
        B[m, i] = False
        B[l, j] = False
        B[m, j] = True
        B[l, i] = True
        score = relax[m, j] + relax[l, i] - relax[m, i] - relax[l, j]
        pool.append((score, B))
    rng.shuffle(pool)
    pool.sort(key=lambda p: p[0], reverse=True)
    return [(B, np.ones(K, bool)) for _, B in pool]


def mixed_graphs(n, c, nsolo, relax, rng):
    M, K = (n["M"], n["K"])
    A = c["A"]
    rank = np.argsort(-n["beta"], axis=0)
    eligible = np.where(nsolo / n["w"] > c["t"] + 0.0002)[0]
    if not len(eligible):
        return []
    pressure = np.sum(c["x"] ** 2, axis=0)
    priority = (nsolo / (n["w"] * max(c["t"], 1e-09)) - 1) / (
        1 + 0.4 * np.maximum(0, A.sum(axis=0) - 1)
    )
    eligible = eligible[np.argsort(-priority[eligible])]
    pool = []
    subsets = [(int(k),) for k in eligible]
    subsets += list(itertools.combinations(eligible[:6], 2))
    for subset in subsets:
        z = np.ones(K, bool)
        z[list(subset)] = False
        Cusers = np.where(z)[0]
        B = A.copy()
        for k in subset:
            original = np.where(A[:, k])[0].tolist()
            extras = [m for m in rank[:, k] if m not in original]
            B[:, k] = False
            B[(original + extras)[: n["Dmax"]], k] = True
        pool.append((B.copy(), z.copy()))
        freed = np.where(A[:, list(subset)].any(axis=1))[0]
        candidates = []
        for m in freed:
            for k in Cusers:
                if B[m, k] or B[:, k].sum() >= n["Dmax"]:
                    continue
                if c["t"] * (B[m, z] @ n["w"][z] + n["w"][k]) > n["C"][m] + 1e-08:
                    continue
                value = n["a"][m, k] / max(1e-15, np.sum(n["a"][:, k] * c["x"][:, k]))
                candidates.append((value * max(pressure[k], 0.0001), m, k))
        for _, m, k in sorted(candidates, reverse=True)[:3]:
            BB = B.copy()
            BB[m, k] = True
            pool.append((BB, z.copy()))
        reservations = [()]
        if len(subset) == 1:
            reservations.append((int(rank[0, subset[0]]),))
        for reserved in reservations:
            BB = round_slots(
                n, relax, rng, power=1.0, active=Cusers, reserved=reserved, target=c["t"] + 0.0004
            )
            if BB is None:
                continue
            BB[:, ~z] = B[:, ~z]
            pool.append((BB, z.copy()))
    return pool


def run(n, policy="joint", seconds=60, seed=20261026):
    pure = policy == "fastC"
    budget = Budget(n, "CJT" if pure else "joint", seconds)
    rng = np.random.default_rng(seed)
    check_parameters(n)
    K = n["K"]
    zC = np.ones(K, bool)
    cache = {}
    examined = {}
    stages = []
    start = budget.start
    cap_upper = coherent_upper(n)
    nsolo = isolated_bounds(n)[1]
    relax = None
    bestC = None
    stage = "initialization"
    stats = collections.Counter()

    def store(s, label):
        nonlocal bestC
        budget.accept(s, label)
        if budget.left() < 0:
            return
        h = key(s["A"], s["z"])
        if h not in cache or s["t"] > cache[h]["t"]:
            cache[h] = s
        if s["z"].all() and (bestC is None or s["t"] > bestC["t"]):
            bestC = s

    def polish(A, parent=None, target=None, end=None):
        h = key(A, zC)
        if h in examined and (
            examined[h] is None or (target is not None and target >= examined[h])
        ):
            stats["coherent_cache_hits"] += 1
            return cache[h]
        s = coherent(n, A, budget, parent, end, target=target)
        store(s, "coherent incumbent")
        stats["coherent_graphs"] += 1
        if h in cache:
            examined[h] = None if target is None or s["t"] >= target * (1 - 2e-05) else target
        return s

    try:
        seed_end = start + min(8.0, seconds * 0.3)
        relax, relax_upper = relaxed_coherent(n, budget, seed_end)
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
            if budget.left(seed_end) < 0.03:
                break
            if key(A, zC) in cache:
                continue
            if bestC is not None and payload_upper(n, A, zC) < bestC["t"] - 1e-05:
                continue
            try:
                polish(A, end=seed_end)
            except Expired:
                if budget.left() < 0.01:
                    raise
                break
        if bestC is None:
            A = seed_graphs(n, rng)[0]
            polish(A)
        stage = "coherent search"
        coh_end = start + seconds * (1.0 if pure else 0.58)
        stagnant = 0
        round_id = 0
        tested = set()
        expanded = set()
        try:
            while budget.left(coh_end) > 0.03:
                round_id += 1
                before = bestC["t"]
                parent_key = key(bestC["A"], zC)
                moves = [] if parent_key in expanded else coherent_moves(n, bestC, relax, rng)
                improved = False
                for A, z in moves:
                    budget.check(coh_end)
                    h = key(A, z)
                    if h in tested:
                        continue
                    tested.add(h)
                    if payload_upper(n, A, z) < bestC["t"] + 0.0001:
                        continue
                    s = polish(A, bestC, target=bestC["t"] + 0.0004, end=coh_end)
                    if s["t"] > before + 0.0005:
                        improved = True
                        break
                if improved:
                    stagnant = 0
                    continue
                expanded.add(parent_key)
                stagnant += 1
                if not pure and stagnant >= 2:
                    break
                for _ in range(4):
                    A = round_slots(
                        n,
                        relax,
                        rng,
                        power=rng.uniform(0.4, 2.5),
                        noise=rng.uniform(0.25, 1.4),
                        target=bestC["t"] + 0.0004,
                    )
                    if A is not None:
                        polish(A, target=bestC["t"] + 0.0004, end=coh_end)
                if pure and bestC["t"] >= cap_upper - 0.0005:
                    break
        except Expired:
            if budget.left() < 0.01:
                raise
        stages.append(dict(stage=stage, seconds=time.perf_counter() - start, rate=bestC["t"]))
        if pure:
            raise Expired
        if bestC["t"] >= cap_upper - 0.0005 and np.max(nsolo / n["w"]) <= bestC["t"] + 0.0002:
            raise Expired
        if np.min(nsolo / n["w"]) > budget.best["t"] + 0.0002:
            try:
                end = min(budget.deadline, time.perf_counter() + 1.0)
                point, upper = nc_radio(n, np.ones_like(n["fixedA"], bool), budget, end)
                if point is not None:
                    store(point, "own all-NCJT seed")
                    if upper > budget.best["t"] + 0.0002:
                        s = mixed(
                            n,
                            point["A"],
                            point["z"],
                            budget,
                            point,
                            end,
                            iterations=15,
                            aggregate=False,
                        )
                        store(s, "own all-NCJT refinement")
            except Expired:
                if budget.left() < 0.01:
                    raise
        stage = "mixed search"
        tried = set()
        round_id = 0
        while budget.left() > 0.05:
            round_id += 1
            pool = mixed_graphs(n, bestC, nsolo, relax, rng)
            if not budget.best["z"].all():
                pool += neighbors(n, budget.best, "joint", rng, round_id)[:40]
            progress = False
            for A, z in pool:
                budget.check()
                h = key(A, z)
                if h in tried or payload_upper(n, A, z) < budget.best["t"] + 0.0001:
                    continue
                tried.add(h)
                before = budget.best["t"]
                if z.all():
                    s = polish(A, bestC, target=budget.best["t"] + 0.0004)
                else:
                    end = min(
                        budget.deadline,
                        time.perf_counter() + min(1.2, max(0.15, budget.left() * 0.08)),
                    )
                    parents = [bestC, budget.best]
                    warm = min(
                        parents, key=lambda p: np.sum(p["z"] != z) + 0.02 * np.sum(p["A"] != A)
                    )
                    try:
                        s = mixed(
                            n,
                            A,
                            z,
                            budget,
                            warm,
                            end,
                            iterations=22,
                            aggregate=True,
                            target=budget.best["t"] + 0.0004,
                        )
                        store(s, "mixed incumbent")
                        stats["mixed_graphs"] += 1
                    except Expired:
                        if budget.left() < 0.01:
                            raise
                        stats["mixed_subtimeouts"] += 1
                progress |= budget.best["t"] > before + 0.0004
            if not progress:
                for _ in range(10):
                    budget.check()
                    A = round_slots(
                        n,
                        relax,
                        rng,
                        power=rng.uniform(0.4, 2.5),
                        noise=rng.uniform(0.2, 1.5),
                        target=bestC["t"] + 0.0004,
                    )
                    if A is not None:
                        polish(A, target=bestC["t"] + 0.0004)
                if round_id > 2 and (not any(nsolo / n["w"] > budget.best["t"] + 0.001)):
                    stage = "coherent fallback"
    except Expired:
        pass
    return (
        budget.best,
        dict(
            policy=policy,
            budget_seconds=seconds,
            wall_seconds=time.perf_counter() - start,
            cpu_seconds=time.process_time() - budget.cpu_start,
            accepted_point_seconds=budget.trace[-1]["seconds"],
            trace=budget.trace,
            events=stages,
            solver_calls=budget.calls,
            proposals=budget.proposals,
            late_proposals_rejected=budget.late,
            max_residual=budget.max_residual,
            stop="budget or coherent payload bound",
            seed=seed,
            solver_threads=1,
            independent_cold_start=True,
            stage=stage,
            stats=dict(stats),
            coherent_lower=None if bestC is None else bestC["t"],
            native_statuses=getattr(budget, "native_statuses", {}),
        ),
    )

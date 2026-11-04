"""Search."""

import itertools
import time
import warnings
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment, milp, LinearConstraint, Bounds
from .budget_core import (
    Budget,
    Expired,
    optimize,
    coherent,
    nc_radio,
    fixed_power,
    warm_power,
    payload_upper,
)
from .capacity import coherent_upper, round_slots


def key(A, z):
    return A.tobytes() + z.tobytes()


def matching(n):
    if n["K"] > n["M"]:
        return round_slots(
            n, n["a"], np.random.default_rng(0), target=coherent_upper(n) * (1 - 1e-07)
        )
    own = n["g"][:, np.arange(n["K"]), np.arange(n["K"])]
    q = n["rho"] / np.log(2) * np.log1p(n["a"] ** 2 / (1 + own)) / n["w"][None, :]
    values = np.unique(q)
    lo = 0
    hi = len(values) - 1
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        cost = (q < values[mid]).astype(float) * 1000000.0 - q / np.maximum(1, q.max())
        rows, cols = linear_sum_assignment(cost)
        if np.all(q[rows, cols] >= values[mid]):
            best = (rows, cols)
            lo = mid + 1
        else:
            hi = mid - 1
    A = np.zeros_like(q, dtype=bool)
    A[best] = True
    return A


def seed_graphs(n, rng):
    M, K = (n["M"], n["K"])
    rank = np.argsort(-n["beta"], axis=0)
    match = matching(n)
    bank = [match]
    own = n["g"][:, np.arange(K), np.arange(K)]
    quality = n["a"] / np.sqrt(1 + own)
    for style in range(5):
        A = match.copy()
        for m in rng.permutation(np.where(~A.any(axis=1))[0]):
            current = np.sum(quality * A, axis=0)
            score = quality[m] / np.maximum(current, 1e-12) ** (1 + style * 0.5)
            score[A.sum(axis=0) >= n["Dmax"]] = -np.inf
            if not np.isfinite(score).any():
                break
            A[m, np.argmax(score)] = True
        bank.append(A)
    for degree in [d for d in [1, 2, 4, n["Dmax"]] if d <= n["Dmax"]]:
        A = np.zeros((M, K), bool)
        for k in range(K):
            A[rank[:degree, k], k] = True
        bank.append(A)
        B = match.copy()
        for k in range(K):
            needed = max(0, degree - int(B[:, k].sum()))
            extra = [m for m in rank[:, k] if not B[m, k]][:needed]
            B[extra, k] = True
        bank.append(B)
    result = []
    seen = set()
    for A in bank:
        if A.tobytes() not in seen:
            seen.add(A.tobytes())
            result.append(A)
    return result


def cjt_mip(n, target, budget, limit=4.0, joint=False):
    """General fixed-target pure-CJT outer-approximation MILP proposal."""
    if joint:
        raise ValueError("legacy MIP supports pure CJT only")
    budget.check()
    if target <= 1e-09:
        return None
    M, K = (n["M"], n["K"])
    D = M * K
    pairs = list(itertools.combinations(range(M), 2))
    F = len(pairs)
    N = 2 * D + K * F
    rows = []
    lb = []
    ub = []

    def add(row, low=-np.inf, high=np.inf):
        rows.append(row)
        lb.append(low)
        ub.append(high)

    for m in range(M):
        add({m * K + k: target * n["w"][k] for k in range(K)}, high=n["C"][m])
        add({D + m * K + k: 1 for k in range(K)}, high=1)
    eta = np.expm1(np.log(2) * target * n["w"] / n["rho"])
    for k in range(K):
        add({m * K + k: 1 for m in range(M)}, low=1, high=n["Dmax"])
        scale = float(1 + n["g"][:, k, :].max(axis=1).sum())
        row = {D + m * K + i: n["g"][m, k, i] / scale for m in range(M) for i in range(K)}
        for m in range(M):
            row[D + m * K + k] -= n["a"][m, k] ** 2 / (eta[k] * scale)
            add({D + m * K + k: 1, m * K + k: -1}, high=0)
        for j, (m, l) in enumerate(pairs):
            row[2 * D + k * F + j] = -2 * n["a"][m, k] * n["a"][l, k] / (eta[k] * scale)
        add(row, high=-1 / scale + 1e-11)
        for j, (m, l) in enumerate(pairs):
            y = 2 * D + k * F + j
            add({y: 1, m * K + k: -1}, high=0)
            add({y: 1, l * K + k: -1}, high=0)
            for lam in np.geomspace(1 / 16, 16, 5):
                add({y: 1, D + m * K + k: -0.5 * lam, D + l * K + k: -0.5 / lam}, high=0)
    rr = []
    cc = []
    vv = []
    for i, row in enumerate(rows):
        for j, value in row.items():
            rr.append(i)
            cc.append(j)
            vv.append(value)
    matrix = sp.coo_matrix((vv, (rr, cc)), shape=(len(rows), N)).tocsc()
    available = min(limit, max(0.001, budget.left() - 0.15))
    budget.check()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = milp(
            np.zeros(N),
            integrality=np.r_[np.ones(D), np.zeros(N - D)],
            bounds=Bounds(0, np.ones(N)),
            constraints=LinearConstraint(matrix, np.array(lb), np.array(ub)),
            options={"threads": 1, "time_limit": available, "node_limit": 10000, "mip_rel_gap": 0},
        )
    budget.calls += 1
    budget.events.append(
        dict(
            stage="CJT MILP",
            target=float(target),
            status=int(r.status),
            elapsed=time.perf_counter() - budget.start,
        )
    )
    if r.x is None:
        return None
    A = r.x[:D].reshape(M, K) > 0.5
    p = np.maximum(0, r.x[D : 2 * D].reshape(M, K))
    if not np.all(A.any(axis=0)):
        return None
    return (A, np.ones(K, bool), np.sqrt(p))


def neighbors(n, s, policy, rng, round_id):
    A = s["A"]
    z = s["z"]
    M, K = A.shape
    pool = []
    rank = np.argsort(-n["beta"], axis=0)

    def push(B, zz):
        if np.all(B.sum(axis=0) >= 1) and np.all(B.sum(axis=0) <= n["Dmax"]):
            pool.append((B, zz))

    if policy in ("CJT", "NCJT", "joint"):
        owner = A.sum(axis=1)
        for k in rng.permutation(K):
            active = np.where(A[:, k])[0]
            inactive = np.where(~A[:, k])[0]
            for m in inactive[:]:
                if len(active) < n["Dmax"]:
                    B = A.copy()
                    B[m, k] = True
                    push(B, z.copy())
            if len(active) > 1:
                for m in active:
                    B = A.copy()
                    B[m, k] = False
                    push(B, z.copy())
            for m in active:
                for mm in rank[:4, k]:
                    if not A[mm, k]:
                        B = A.copy()
                        B[m, k] = False
                        B[mm, k] = True
                        push(B, z.copy())
            for degree in [d for d in [1, 2, 4, n["Dmax"]] if d <= n["Dmax"]]:
                B = A.copy()
                B[:, k] = False
                B[rank[:degree, k], k] = True
                push(B, z.copy())
        if policy in ("CJT", "joint"):
            links = list(zip(*np.nonzero(A & z[None, :])))
            for m, k in links:
                if A[:, k].sum() > 1:
                    for i in range(K):
                        if i != k and (not A[m, i]):
                            B = A.copy()
                            B[m, k] = False
                            B[m, i] = True
                            push(B, z.copy())
            for (m, k), (l, i) in itertools.combinations(links, 2):
                if k != i and m != l and (not A[m, i]) and (not A[l, k]):
                    B = A.copy()
                    B[m, k] = False
                    B[l, i] = False
                    B[m, i] = True
                    B[l, k] = True
                    push(B, z.copy())
    if policy in ("fixed", "joint"):
        for size in [1, 2]:
            for ks in itertools.combinations(range(K), size):
                zz = z.copy()
                zz[list(ks)] = ~zz[list(ks)]
                push(A.copy(), zz)
                if policy == "joint":
                    B = A.copy()
                    for k in ks:
                        B[:, k] = False
                        if zz[k]:
                            allowed = np.where(~np.any(B[:, zz], axis=1))[0]
                            if not len(allowed):
                                allowed = rank[:, k]
                            chosen = allowed[np.argmax(n["a"][allowed, k])]
                            B[chosen, k] = True
                        else:
                            B[rank[: n["Dmax"], k], k] = True
                    push(B, zz.copy())
    rng.shuffle(pool)
    unique = []
    seen = set()
    for B, zz in pool:
        h = key(B, zz)
        if h in seen or payload_upper(n, B, zz) < s["t"] - 1e-05:
            continue
        seen.add(h)
        unique.append((B, zz))
    changed = [p for p in unique if not np.array_equal(p[1], z)]
    same = [p for p in unique if np.array_equal(p[1], z)]
    return changed[:64] + same[:96] if changed else same[:160]


def run(n, policy, seconds=60, seed=20261026):
    if policy != "CJT":
        raise ValueError("Generalized legacy control exposes pure CJT only")
    rng = np.random.default_rng(seed)
    budget = Budget(n, policy, seconds)
    K = n["K"]
    allC = np.ones(K, bool)
    allN = ~allC
    graphbank = seed_graphs(n, rng)
    cache = {}
    beam = []
    round_id = 0
    stop = "budget"
    nc_upper = None

    def evaluate(A, z, parent=None, it=22, end=None):
        s = optimize(n, A, z, budget, parent, end, iterations=it)
        h = key(A, z)
        if h not in cache or s["t"] > cache[h]["t"]:
            cache[h] = s
        if not any((key(p["A"], p["z"]) == key(A, z) for p in beam)):
            beam.append(s)
        return s

    try:
        if policy in ("CJT", "joint"):
            seed_end = (
                budget.start + min(seconds * 0.23, 12.0)
                if policy == "joint"
                else budget.start + min(seconds * 0.3, 18.0)
            )
            for A in graphbank:
                if budget.left(seed_end) < 0.08:
                    break
                try:
                    evaluate(A, allC, end=seed_end)
                except Expired:
                    break
        if policy in ("NCJT", "fixed", "joint"):
            init_end = (
                budget.start + min(seconds * 0.45, 25.0)
                if policy == "joint"
                else budget.start + min(seconds * 0.3, 18.0)
            )
            A = n["fixedA"] if policy == "fixed" else np.ones_like(n["fixedA"], bool)
            try:
                point, nc_upper = nc_radio(n, A, budget, init_end)
                if point is not None:
                    cache[key(point["A"], point["z"])] = point
                    beam.append(point)
                    if policy == "NCJT" and nc_upper - point["t"] <= 0.0005:
                        stop = "numerical radio bound reached"
                        raise Expired
                    evaluate(point["A"], allN, point, it=30, end=init_end)
            except Expired:
                if stop != "budget":
                    raise
            if policy in ("NCJT", "fixed"):
                for A in [n["fixedA"]] if policy == "fixed" else graphbank[-6:]:
                    if budget.left(init_end) < 0.1:
                        break
                    try:
                        evaluate(A, allN, end=init_end, it=14)
                    except Expired:
                        break
        if policy == "joint":
            hybrid_end = budget.start + min(8.0, seconds * 0.35)
            try:
                evaluate(n["fixedA"], allN, budget.best, it=20, end=hybrid_end)
                for k in range(K):
                    if budget.left(hybrid_end) < 0.08:
                        break
                    zz = allN.copy()
                    zz[k] = True
                    evaluate(n["fixedA"], zz, budget.best, it=14, end=hybrid_end)
            except Expired:
                pass
        if policy == "fixed":
            mode_end = budget.start + seconds * 0.65
            for size in range(1, K + 1):
                for ids in itertools.combinations(range(K), size):
                    if budget.left(mode_end) < 0.08:
                        break
                    z = np.zeros(K, bool)
                    z[list(ids)] = True
                    if payload_upper(n, n["fixedA"], z) < budget.best["t"] - 1e-05:
                        continue
                    try:
                        evaluate(n["fixedA"], z, budget.best, it=16, end=mode_end)
                    except Expired:
                        break
                if budget.left(mode_end) < 0.05:
                    break
        while budget.left() > 0.025:
            round_id += 1
            overall_upper = (
                coherent_upper(n)
                if policy == "CJT"
                else (nc_upper if policy == "NCJT" else max(2.0, nc_upper))
                if nc_upper is not None
                else np.inf
            )
            if budget.best["t"] >= overall_upper - 0.0005:
                stop = "numerical radio/payload upper reached"
                break
            if (
                policy == "NCJT"
                and nc_upper is not None
                and (budget.best["t"] >= nc_upper - 0.0005)
            ):
                stop = "numerical radio bound reached"
                break
            if policy in ("CJT", "joint") and round_id % 4 == 1 and (budget.left() > 1.2):
                target = min(
                    coherent_upper(n) * (1 - 1e-05), max(1e-05, budget.best["t"] * 1.006 + 0.001)
                )
                proposal = cjt_mip(
                    n,
                    target,
                    budget,
                    limit=min(4.0, budget.left() * 0.18),
                    joint=policy == "joint" and round_id % 8 == 5,
                )
                if proposal is not None:
                    A, z, x = proposal
                    parent = dict(x=x)
                    evaluate(A, z, parent, it=24)
            if not beam:
                beam = [budget.best]
            states = list(cache.values())
            category = round_id % 4
            subset = [
                p
                for p in states
                if (
                    p["z"].all()
                    if category == 1
                    else not p["z"].any()
                    if category == 2
                    else p["z"].any() and (not p["z"].all())
                )
            ]
            parent = (
                max(subset, key=lambda p: p["t"])
                if policy == "joint" and category and subset
                else budget.best
            )
            pool = neighbors(n, parent, policy, rng, round_id)
            ranked = []
            for A, z in pool:
                budget.check()
                h = key(A, z)
                if h in cache:
                    continue
                candidates = []
                for warm in [parent, None]:
                    p = fixed_power(n, A, z, warm_power(n, A, warm), budget)
                    budget.accept(p, "AP/mode screen")
                    candidates.append(p)
                p = max(candidates, key=lambda p: p["t"])
                ranked.append(p)
            ranked.sort(key=lambda p: p["t"], reverse=True)
            selected = ranked[:2]
            alternatives = [p for p in ranked if not np.array_equal(p["z"], parent["z"])]
            selected += alternatives[:2] if alternatives else ranked[2:4]
            used = set()
            for p in selected:
                h = key(p["A"], p["z"])
                if h not in used:
                    evaluate(p["A"], p["z"], p, it=26)
                    used.add(h)
            if not ranked or round_id % 3 == 0:
                p = budget.best
                xx = p["x"] * np.exp(rng.normal(0, 0.4, p["x"].shape))
                evaluate(p["A"], p["z"], dict(x=xx), it=35)
                if policy in ("CJT", "joint"):
                    score = n["a"] * np.exp(rng.normal(0, 0.35, n["a"].shape))
                    B = round_slots(n, score, rng, target=max(1e-06, budget.best["t"] + 0.0004))
                    if B is not None:
                        evaluate(B, allC, it=14)
    except Expired:
        pass
    elapsed = time.perf_counter() - budget.start
    cpu = time.process_time() - budget.cpu_start
    return (
        budget.best,
        dict(
            policy=policy,
            budget_seconds=seconds,
            wall_seconds=elapsed,
            cpu_seconds=cpu,
            accepted_point_seconds=budget.trace[-1]["seconds"],
            trace=budget.trace,
            events=budget.events,
            solver_calls=budget.calls,
            proposals=budget.proposals,
            late_proposals_rejected=budget.late,
            max_residual=budget.max_residual,
            stop=stop,
            rounds=round_id,
            seed=seed,
            nc_radio_upper=nc_upper,
            independent_cold_start=True,
            solver_threads=1,
        ),
    )

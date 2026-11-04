"""Mode search."""

import itertools
import numpy as np
from .budget_core import fixed_power, warm_power
from .fast_kernels import Aggregate, FastWM
from .joint_general import mixed_graphs


def checked_infeasibility(model, r):
    """Conservative floating-point Farkas check; no exact infeasibility claim."""
    if str(r.status) != "PrimalInfeasible" or r.z is None:
        return False
    z = np.asarray(r.z, float)
    b = model.last_b
    A = model.last_A
    if z.shape != b.shape or not np.isfinite(z).all():
        return False
    product = float(b @ z)
    if product >= -1e-08:
        return False
    z = z / -product
    if np.max(np.abs(A.T @ z)) > 2e-07:
        return False
    count = model.last_linear_rows
    if z[:count].min() < -2e-08:
        return False
    if count < len(z):
        pairs = z[count:].reshape(-1, 3)
        if np.min(pairs[:, 0] - np.linalg.norm(pairs[:, 1:], axis=1)) < -2e-08:
            return False
    return True


def probe(n, A, z, budget, parent, end, target, stats, iterations=22):
    best = fixed_power(n, A, z, warm_power(n, A, parent), budget, end)
    budget.accept(best, "mixed warm code LP")
    model = Aggregate(n, A, z)
    xx, r = model.solve(target, budget, end)
    if checked_infeasibility(model, r):
        stats["checked_aggregate_target_rejections"] += 1
        return best
    starts = [best]
    if xx is not None:
        p = fixed_power(n, A, z, xx, budget, end)
        budget.accept(p, "mixed aggregate target seed")
        starts.append(p)
        if p["t"] > best["t"]:
            best = p
    wm = FastWM(n, A, z)
    for start in sorted(starts, key=lambda s: s["t"], reverse=True)[:2]:
        x = warm_power(n, A, start)
        previous = -1.0
        stall = 0
        for i in range(iterations):
            budget.check(end)
            xx, r = wm.step(x, budget, end)
            if xx is None:
                break
            p = fixed_power(n, A, z, xx, budget, end)
            if p["t"] > best["t"]:
                best = p
                budget.accept(best, "mixed WMMSE refinement")
            if p["t"] < previous - 2e-06:
                break
            stall = stall + 1 if p["t"] - previous < 2e-05 else 0
            previous = p["t"]
            x = p["x"]
            if i >= 4 and stall >= 2:
                break
    return best


def bundled_graphs(n, c, nsolo, relax, rng):
    """One/two mode flips coupled to multiple useful coherent AP additions."""
    A = c["A"]
    M, K = A.shape
    target = c["t"] + 0.0004
    rank = np.argsort(-n["beta"], axis=0)
    pressure = np.sum(c["x"] ** 2, axis=0)
    amp = np.maximum((n["a"] * c["x"]).sum(axis=0), 1e-12)
    opportunity = n["a"] / amp[None, :] * pressure[None, :]
    eligible = np.where(nsolo / n["w"] > target)[0]
    scores = []
    for k in eligible:
        others = np.arange(K) != k
        value = np.where(~A & others[None, :], opportunity, 0).max(axis=1)
        scores.append(
            float(value[A[:, k]].sum())
            * min(1.0, max(0.0, nsolo[k] / n["w"][k] / max(c["t"], 0.01) - 1))
        )
    eligible = eligible[np.argsort(-np.asarray(scores))]
    pool = []
    subsets = [(int(k),) for k in eligible] + list(itertools.combinations(eligible[:6], 2))
    for subset in subsets:
        z = np.ones(K, bool)
        z[list(subset)] = False
        freed = np.flatnonzero(A[:, list(subset)].any(axis=1))
        B = A.copy()
        for k in subset:
            original = np.flatnonzero(A[:, k]).tolist()
            extra = [m for m in rank[:, k] if m not in original]
            B[:, k] = False
            B[(original + extra)[: n["Dmax"]], k] = True
        for variant in range(2):
            BB = B.copy()
            gain = np.ones(K)
            for m in freed:
                candidates = np.flatnonzero(z & ~BB[m] & (BB.sum(axis=0) < n["Dmax"]))
                candidates = [
                    k
                    for k in candidates
                    if target * (BB[m, z] @ n["w"][z] + n["w"][k]) <= n["C"][m] + 1e-08
                ]
                if not candidates:
                    continue
                values = opportunity[m, candidates] / gain[candidates]
                k = candidates[int(np.argmax(values))]
                BB[m, k] = True
                gain[k] += n["a"][m, k] / amp[k] * (1.0 if variant == 0 else 10.0)
            pool.append((BB, z.copy()))
    return pool + mixed_graphs(n, c, nsolo, relax, rng)

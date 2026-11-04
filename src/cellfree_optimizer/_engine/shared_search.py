"""Shared search."""

import time
import collections
import numpy as np
from .physical import isolated_bounds
from .budget_core import Budget, Expired, payload_upper, nc_radio
from .fast_kernels import coherent
from .search import seed_graphs, neighbors, key
from .capacity import slots, round_slots, check_parameters, coherent_upper
from .cluster_search import fair_cluster, ranked_moves
from .joint_general import relaxed_coherent
from .mode_search import bundled_graphs, probe
from .pure_nc import initialise_nc, refine_nc

EPS = 0.0004


def coherent_projection(n, A, parent, target):
    """Repair a proposed graph to target-dependent pure-CJT occupancy.

    Prefer proposed edges, then the parent's edges. Mandatory user coverage is
    obtained by the same capacity-slot assignment routine; no power is donated.
    Both enabled and disabled policies call this exact operator.
    """
    q = slots(n, target)
    if q.sum() < n["K"]:
        return None
    if np.all(A.sum(axis=1) <= q) and payload_upper(n, A, np.ones(n["K"], bool)) >= target:
        return A.copy()
    den = 1 + np.einsum("mki,mi->k", n["g"], parent["x"] ** 2)
    quality = n["a"] / np.sqrt(den)[None, :]
    quality = quality / np.maximum(quality.max(axis=0), 1e-15)[None, :]
    score = quality * (1.0 + 2.0 * A + 0.5 * parent["A"])
    return round_slots(n, score, np.random.default_rng(0), power=1.0, noise=0.0, target=target)


def run(n, policy, seconds=60, seed=20261028):
    assert policy in ["backboneC", "modeOff", "joint"]
    enabled = policy == "joint"
    pure_backbone = policy == "backboneC"
    bestC = None
    bestN = None
    cache = {}
    examined = {}
    events = []
    stats = collections.Counter()

    class LedgerBudget(Budget):
        def accept(self, s, label):
            nonlocal bestC, bestN
            accepted = super().accept(s, label)
            if self.left() >= 0 and s["z"].all():
                if bestC is None or s["t"] > bestC["t"]:
                    bestC = s
                h = key(s["A"], s["z"])
                if h not in cache or s["t"] > cache[h]["t"]:
                    cache[h] = s
            if self.left() >= 0 and (not s["z"].any()):
                if bestN is None or s["t"] > bestN["t"]:
                    bestN = s
            return accepted

    b = LedgerBudget(n, "joint" if enabled else "CJT", seconds)
    start = b.start
    check_parameters(n)
    rng = np.random.default_rng(seed)
    mrng = np.random.default_rng(seed + 19727)
    nrng = np.random.default_rng(seed + 40709)
    zC = np.ones(n["K"], bool)
    zN = ~zC
    capC = coherent_upper(n)
    totalFH = float(n["C"].sum() / n["w"].sum())
    relax = None
    nsolo = None
    pending = []
    parent_key = None
    last_mip = -100.0
    tested_since_mip = 0
    rest = 0
    stop = "shared deadline"
    tried = set()
    nc_seed = None
    coop_upper = min(
        float(n["C"].max() / n["w"].min()), float(n["C"].sum() / (n["w"].sum() + n["w"].min()))
    )

    def event(label):
        events.append(
            dict(
                stage=label,
                seconds=time.perf_counter() - start,
                rate=b.best["t"],
                coherent=bestC["t"],
            )
        )

    def saturated():
        return bestC is not None and bestC["t"] >= capC - 2e-06

    def polish(A, end=None):
        h = key(A, zC)
        target = None if bestC is None else bestC["t"] + EPS
        if h in examined and (examined[h] is None or target >= examined[h]):
            stats["cache_hits"] += 1
            return
        if bestC is not None and payload_upper(n, A, zC) < target - 1e-08:
            return
        s = coherent(n, A, b, bestC, end, target=target)
        b.accept(s, "shared coherent backbone")
        stats["coherent_graphs"] += 1
        examined[h] = None if target is None or s["t"] >= target * (1 - 2e-05) else target

    def backbone(end):
        nonlocal pending, parent_key, last_mip, tested_since_mip, rest
        while b.left(end) > 0.025:
            b.check(end)
            if saturated():
                stats["coherent_bound_stops"] += 1
                return
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
                polish(pending[-1], end)
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
                    target=bestC["t"] + EPS,
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
            if b.left(seed_end) < 0.03 or saturated():
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
        if enabled:
            if b.best["t"] >= totalFH - 2e-06:
                stop = "global total-fronthaul bound"
                raise Expired
            end = min(b.deadline, time.perf_counter() + min(8.0, seconds * 0.3))
            initialise_nc(n, b, lambda: bestN, nrng, stats, end)
            if bestN is not None and bestN["t"] > coop_upper + 2e-06:
                stats["NC_endpoint_dominance"] = 1
                stop = "NC-equivalent improvement domain"
                event("NC endpoint dominance")
                refine_nc(n, b, lambda: bestN, nrng, stats, b.deadline)
                raise Expired
        try:
            backbone(start + seconds * (1.0 if pure_backbone else 0.55))
        except Expired:
            if b.left() < 0.01:
                raise
        event("shared coherent prefix")
        if pure_backbone or (not enabled and saturated()):
            stop = "coherent payload bound" if saturated() else stop
            raise Expired
        probe_round = 0
        coh_reward = 0.0
        mix_reward = 0.0
        while b.left() > 0.035:
            if enabled and bestN is not None and (bestN["t"] > coop_upper + 2e-06):
                stats["NC_endpoint_dominance"] = 1
                stop = "NC-equivalent improvement domain"
                event("NC endpoint dominance")
                refine_nc(n, b, lambda: bestN, nrng, stats, b.deadline)
                break
            if enabled and b.best["t"] >= totalFH - 2e-06:
                stop = "global total-fronthaul bound"
                break
            if not enabled and saturated():
                stop = "coherent payload bound"
                break
            if not np.any(nsolo / n["w"] > b.best["t"] + 0.0002):
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
            coh_reward = 0.6 * coh_reward + 0.4 * max(0, b.best["t"] - before) / max(used, 0.001)
            probe_round += 1
            pool = bundled_graphs(n, bestC, nsolo, relax, mrng)
            if enabled and bestN is not None:
                pool += neighbors(n, bestN, "NCJT", mrng, probe_round)[:40]
            if enabled and (not b.best["z"].all()):
                pool += neighbors(n, b.best, "joint", mrng, probe_round)[:40]
                pool += neighbors(n, b.best, "NCJT", mrng, probe_round)[:40]
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
                if h in tried:
                    continue
                projected = coherent_projection(n, A, bestC, bestC["t"] + EPS)
                stats["shared_AP_hypotheses"] += 1
                if projected is not None:
                    try:
                        polish(projected, probe_end)
                    except Expired:
                        break
                    stats["coherent_projections"] += 1
                if not enabled or z.all():
                    tried.add(h)
                    count += 1
                    continue
                if payload_upper(n, A, z) < b.best["t"] + 0.0001:
                    tried.add(h)
                    continue
                end = min(probe_end, time.perf_counter() + 0.65)
                parents = [bestC, b.best]
                warm = min(parents, key=lambda p: np.sum(p["z"] != z) + 0.02 * np.sum(p["A"] != A))
                try:
                    s = probe(n, A, z, b, warm, end, b.best["t"] + EPS, stats)
                    b.accept(s, "NC-enabled shared AP proposal")
                    stats["mixed_graphs"] += 1
                    tried.add(h)
                except Expired:
                    stats["mixed_subtimeouts"] += 1
                    tried.add(h)
                    if b.left() < 0.01:
                        raise
                count += 1
            used = time.perf_counter() - probe_start
            mix_reward = 0.6 * mix_reward + 0.4 * max(0, b.best["t"] - before) / max(used, 0.001)
            if count == 0:
                try:
                    backbone(min(b.deadline, time.perf_counter() + 2.0))
                except Expired:
                    if b.left() < 0.01:
                        raise
            event("shared AP-proposal batches")
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
            stop=stop,
            seed=seed,
            solver_threads=1,
            independent_cold_start=True,
            stats=dict(stats),
            coherent_lower=None if bestC is None else bestC["t"],
            coherent_point=bestC,
            native_statuses=getattr(b, "native_statuses", {}),
            coherent_analytic_upper=capC,
            total_fronthaul_upper=totalFH,
            coherent_cooperation_upper=coop_upper,
            NC_lower=None if bestN is None else bestN["t"],
        ),
    )

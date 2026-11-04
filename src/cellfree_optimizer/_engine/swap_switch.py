"""Swap-and-switch mode search with fixed or adaptive AP association."""

import time
import collections
import numpy as np
from .budget_core import Budget, Expired, payload_upper
from .sca import optimize
from .search import key, neighbors
from . import shared_search
from . import pure_nc


def run(n, variant="ap", seconds=60, seed=20261030):
    assert variant in ["fixed", "twoStage", "ap"]
    budget = Budget(n, "fixed" if variant == "fixed" else "joint", seconds)
    rng = np.random.default_rng(seed)
    start = budget.start
    stats = collections.Counter()
    cache = {}
    ap_graphs = [n["fixedA"].copy()]
    stage_records = []
    current = None
    round_id = 0
    fixed = None

    def evaluate(A, z, parent=None, force=False):
        nonlocal current
        budget.check()
        A = np.asarray(A, bool)
        z = np.asarray(z, bool)
        assert np.all((A.sum(axis=0) >= 1) & (A.sum(axis=0) <= n["Dmax"]))
        h = key(A, z)
        if h in cache and (not force):
            stats["cached_utilities"] += 1
            return cache[h]
        if current is not None and payload_upper(n, A, z) <= current["t"] + 2e-05:
            stats["payload_pruned"] += 1
            return None
        p = optimize(n, A, z, budget, parent, iterations=30)
        if h not in cache or p["t"] > cache[h]["t"]:
            cache[h] = p
        stats["SCA_graphs"] += 1
        return p

    def improve(point, best):
        return (
            point
            if point is not None and (best is None or point["t"] > best["t"] + 2e-05)
            else best
        )

    try:
        if variant != "fixed":
            for endpoint in ["C", "N"]:
                budget.check()
                allocation = min(6.0, max(0.01, budget.left() * 0.2))
                if endpoint == "C":
                    p, meta = shared_search.run(n, "backboneC", allocation, seed)
                else:
                    p, meta = pure_nc.run(n, allocation, seed)
                stage_records.append(
                    dict(
                        endpoint=endpoint,
                        wall_seconds=meta["wall_seconds"],
                        cpu_seconds=meta["cpu_seconds"],
                        rate=p["t"],
                        solver_calls=meta["solver_calls"],
                    )
                )
                budget.calls += meta["solver_calls"]
                budget.accept(p, "own paid AP-preselection " + endpoint)
                ap_graphs.append(p["A"].copy())
                cache[key(p["A"], p["z"])] = p
                current = improve(p, current)
            current = budget.best
            if variant == "twoStage":
                fixed = current["A"].copy()
                ap_graphs = [fixed]
        else:
            fixed = n["fixedA"].copy()
        graph = ap_graphs[0] if current is None else current["A"]
        if current is None:
            for z in [np.zeros(n["K"], bool), rng.random(n["K"]) > 0.5, np.ones(n["K"], bool)]:
                point = evaluate(graph, z, current)
                current = improve(point, current)
        while budget.left() > 0.03:
            before = current["t"]
            round_id += 1
            for k in range(n["K"]):
                best = current
                for other in np.flatnonzero(current["z"] != current["z"][k]):
                    zz = current["z"].copy()
                    zz[[k, other]] = ~zz[[k, other]]
                    best = improve(evaluate(current["A"], zz, current), best)
                    stats["swap_evaluations"] += 1
                current = best
                zz = current["z"].copy()
                zz[k] = ~zz[k]
                point = evaluate(current["A"], zz, current)
                current = improve(point, current)
                stats["switch_evaluations"] += 1
                if variant == "ap":
                    proposals = []
                    rank = np.argsort(-n["beta"][:, k])
                    others = current["A"].copy()
                    others[:, k] = False
                    available = n["C"] - current["t"] * (
                        others[:, current["z"]] @ n["w"][current["z"]]
                    )
                    for degree in [d for d in [1, 2, 4, n["Dmax"]] if d <= n["Dmax"]]:
                        B = current["A"].copy()
                        Z = current["z"].copy()
                        Z[k] = ~Z[k]
                        allowed = (
                            rank[available[rank] >= n["w"][k] * current["t"] - 1e-09]
                            if Z[k]
                            else rank
                        )
                        if len(allowed) < degree:
                            continue
                        B[:, k] = False
                        B[allowed[:degree], k] = True
                        proposals.append((B, Z))
                    pool = neighbors(
                        n, current, "CJT" if current["z"].any() else "NCJT", rng, round_id
                    )
                    proposals += pool[:4]
                    for A, z in proposals:
                        current = improve(evaluate(A, z, current), current)
                        stats["AP_extension_evaluations"] += 1
            if current["t"] <= before + 2e-05:
                stats["matching_stable_passes"] += 1
                graph = ap_graphs[int(rng.integers(len(ap_graphs)))]
                zz = rng.random(n["K"]) > 0.5
                point = evaluate(graph, zz, None, force=True)
                if point is None:
                    zz = np.zeros(n["K"], bool)
                    warm = dict(
                        x=np.maximum(current["x"] * graph, 0.001 * graph)
                        * np.exp(rng.normal(0, 0.6, graph.shape))
                    )
                    point = evaluate(graph, zz, warm, force=True)
                    stats["power_restarts"] += 1
                if point is not None:
                    current = point
                    stats["multistart_restarts"] += 1
                elif len(cache) >= 2 ** n["K"] and variant != "ap":
                    break
            if budget.best["t"] >= n["C"].sum() / n["w"].sum() - 2e-06:
                break
    except Expired:
        pass
    if fixed is not None:
        assert np.array_equal(budget.best["A"], fixed)
    return (
        budget.best,
        dict(
            policy="swap_switch_" + variant,
            budget_seconds=seconds,
            wall_seconds=time.perf_counter() - start,
            cpu_seconds=time.process_time() - budget.cpu_start,
            accepted_point_seconds=budget.trace[-1]["seconds"],
            trace=budget.trace,
            events=budget.events,
            solver_calls=budget.calls,
            late_proposals_rejected=budget.late,
            max_residual=budget.max_residual,
            solver_threads=1,
            independent_cold_start=True,
            seed=seed,
            stats=dict(stats),
            AP_preselection_stages=stage_records,
            adaptation="Max-min utility with independent starts and AP refinements",
            native_statuses=getattr(budget, "native_statuses", {}),
        ),
    )

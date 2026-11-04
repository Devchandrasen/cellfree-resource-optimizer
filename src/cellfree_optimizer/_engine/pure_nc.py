"""Pure nc."""

import time
import collections
import numpy as np
from .budget_core import Budget, Expired, nc_radio, payload_upper
from .physical import isolated_bounds
from .search import neighbors, key, seed_graphs
from .mode_search import probe


def initialise_nc(n, b, get_nc, rng, stats, end):
    radio_upper = None
    try:
        point, radio_upper = nc_radio(n, np.ones_like(n["fixedA"], bool), b, end)
    except Expired:
        if b.left() < 0.01:
            raise
        point = None
    seeds = seed_graphs(n, rng)
    if b.left(end) > 0.04:
        try:
            nc_radio(n, seeds[0], b, end)
        except Expired:
            if b.left() < 0.01:
                raise
    current = get_nc()
    bank = [current["A"]] if current is not None else []
    bank += [n["fixedA"]] + seeds[-6:]
    seen = set()
    for A in bank:
        if b.left(end) < 0.04:
            break
        h = key(A, np.zeros(n["K"], bool))
        if h in seen:
            continue
        seen.add(h)
        parent = get_nc()
        target = 0.01 if parent is None else max(0.01, parent["t"] + 0.0004)
        try:
            s = probe(n, A, np.zeros(n["K"], bool), b, parent, end, target, stats, iterations=22)
            b.accept(s, "independent NC basin initializer")
            stats["NC_initializer_graphs"] += 1
        except Expired:
            if b.left() < 0.01:
                raise
            break
    return radio_upper


def refine_nc(n, b, get_nc, rng, stats, end):
    z = np.zeros(n["K"], bool)
    tried = set()
    turn = 0
    analytic = min(
        float(n["C"].sum() / n["w"].sum()), float(np.min(isolated_bounds(n)[1] / n["w"]))
    )
    while b.left(end) > 0.035:
        parent = get_nc()
        if parent is None:
            break
        if parent["t"] >= analytic - 2e-06:
            return "analytic NC upper bound"
        turn += 1
        pool = neighbors(n, parent, "NCJT", rng, turn)
        if turn % 2 == 0:
            pool = [(parent["A"].copy(), z.copy())] + pool
        progress = False
        for A, zz in pool:
            parent = get_nc()
            h = key(A, z)
            if h in tried and (not np.array_equal(A, parent["A"])):
                continue
            if payload_upper(n, A, z) < parent["t"] + 0.0001:
                continue
            subend = min(end, b.deadline, time.perf_counter() + 0.8)
            before = parent["t"]
            warm = parent
            if h in tried:
                warm = dict(x=parent["x"] * np.exp(rng.normal(0, 0.35, A.shape)))
            try:
                s = probe(n, A, z, b, warm, subend, parent["t"] + 0.0004, stats, iterations=25)
                b.accept(s, "AP-optimized own NC basin")
                stats["NC_graphs"] += 1
                tried.add(h)
            except Expired:
                stats["NC_subtimeouts"] += 1
                tried.add(h)
                if b.left() < 0.01:
                    raise
            progress |= get_nc()["t"] > before + 0.0004
            if progress:
                break
            b.check(end)
        if not pool:
            break
    return "shared NC basin deadline"


def run(n, seconds=60, seed=20261028):
    bestN = None

    class NCLedger(Budget):
        def accept(self, s, label):
            nonlocal bestN
            accepted = super().accept(s, label)
            if self.left() >= 0 and (not s["z"].any()):
                if bestN is None or s["t"] > bestN["t"]:
                    bestN = s
            return accepted

    b = NCLedger(n, "NCJT", seconds)
    bestN = b.best
    rng = np.random.default_rng(seed + 40709)
    stats = collections.Counter()
    radio_upper = None
    stop = "shared deadline"
    try:
        end = min(b.deadline, b.start + min(8.0, seconds * 0.3))
        radio_upper = initialise_nc(n, b, lambda: bestN, rng, stats, end)
        stop = refine_nc(n, b, lambda: bestN, rng, stats, b.deadline)
    except Expired:
        pass
    return (
        b.best,
        dict(
            policy="pureNC",
            budget_seconds=seconds,
            wall_seconds=time.perf_counter() - b.start,
            cpu_seconds=time.process_time() - b.cpu_start,
            accepted_point_seconds=b.trace[-1]["seconds"],
            trace=b.trace,
            events=[],
            solver_calls=b.calls,
            proposals=b.proposals,
            late_proposals_rejected=b.late,
            max_residual=b.max_residual,
            stop=stop,
            seed=seed,
            solver_threads=1,
            independent_cold_start=True,
            stats=dict(stats),
            native_statuses=getattr(b, "native_statuses", {}),
            numerical_NC_radio_upper=radio_upper,
            upper_note="LP bracket is guidance only; no exact global certificate or untimed point donation.",
        ),
    )

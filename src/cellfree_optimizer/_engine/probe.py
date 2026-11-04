"""Probe."""

import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from .budget_core import fixed_power, warm_power
from .probe_steps import Probe as Base
from .structured import admissible
from .search import key


def sparse_supports(n, s):
    A = s["A"]
    z = s["z"]
    nc = np.flatnonzero(~z)
    out = []
    pruned = A.copy()
    pruned[:, nc] &= s["u"][:, nc] > 1e-10
    if np.all(pruned.sum(axis=0) >= 1):
        out.append(pruned)
    if len(nc) > 1:
        score = s["u"][:, nc] + 1e-05 * n["a"][:, nc] / np.maximum(n["a"][:, nc].max(axis=0), 1e-15)
        rows, cols = linear_sum_assignment(np.where(A[:, nc], -score, 1000000.0).T)
        if len(rows) == len(nc) and all((A[m, nc[k]] for k, m in zip(rows, cols))):
            B = A.copy()
            B[:, nc] = False
            owners = {int(m): int(nc[k]) for k, m in zip(rows, cols)}
            for m, k in owners.items():
                B[m, k] = True
            for m in range(n["M"]):
                if m in owners:
                    continue
                eligible = [k for k in nc if A[m, k] and B[:, k].sum() < n["Dmax"]]
                if eligible:
                    k = max(
                        eligible,
                        key=lambda k: float(
                            s["u"][m, k] + 1e-05 * n["a"][m, k] / max(n["a"][:, k].max(), 1e-15)
                        ),
                    )
                    B[m, k] = True
            if np.all(B.sum(axis=0) >= 1):
                out.append(B)
    return out


class Probe:
    def __init__(self, n, A, z, parent):
        self.n = n
        self.A = A
        self.z = z
        self.parent = parent
        self.contexts = None
        self.best = None
        self.done = False

    def advance(self, b, end, target, stats):
        if self.contexts is None:
            s = fixed_power(self.n, self.A, self.z, warm_power(self.n, self.A, self.parent), b, end)
            self.best = s
            b.accept(s, "original mixed warm point")
            pool = sparse_supports(self.n, s) + [self.A]
            unique = {}
            for A in pool:
                unique[key(A, self.z)] = A
            pool = sorted(unique.values(), key=lambda A: not admissible(A, self.z))
            self.contexts = [Base(self.n, A, self.z, s) for A in pool]
            stats["sparse_NC_contexts"] += len(pool) - 1
        for context in self.contexts:
            if context.done:
                continue
            b.check(end)
            subend = min(end, time.perf_counter() + (0.5 if context.structured else 0.8))
            p = context.advance(b, subend, target, stats)
            if p["t"] > self.best["t"]:
                self.best = p
        self.done = all((c.done for c in self.contexts))
        return self.best

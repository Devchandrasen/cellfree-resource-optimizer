"""Capacity transitions."""

import numpy as np
from .capacity import slots
from .cluster_search import ranked_moves


class Cliff:
    def __init__(self, n, parent, target, rng):
        self.n = n
        self.parent = parent
        self.target = target
        self.rng = rng
        self.q = slots(n, target)
        self.done = False
        self.level = 0
        self.beam = []
        self.seen = set()
        self.inflight = None
        A = parent["A"]
        den = 1 + np.einsum("mki,mi->k", n["g"], parent["x"] ** 2)
        self.quality = n["a"] / np.sqrt(den)[None, :]
        eta = np.expm1(np.log(2) * target * n["w"] / n["rho"])
        self.required = np.sqrt(eta)
        candidates = [A.copy()]
        for m in range(n["M"]):
            while candidates and any((B[m].sum() > self.q[m] for B in candidates)):
                pool = []
                for B in candidates:
                    if B[m].sum() <= self.q[m]:
                        pool.append(B)
                        continue
                    degree = B.sum(axis=0)
                    for k in np.flatnonzero(B[m]):
                        base = B.copy()
                        base[m, k] = False
                        if degree[k] > 1:
                            pool.append(base)
                            continue
                        for l in range(n["M"]):
                            if l == m or base[l, k]:
                                continue
                            if base[l].sum() < self.q[l]:
                                C = base.copy()
                                C[l, k] = True
                                pool.append(C)
                            else:
                                for i in np.flatnonzero(base[l] & (degree > 1)):
                                    C = base.copy()
                                    C[l, i] = False
                                    C[l, k] = True
                                    pool.append(C)
                unique = {B.tobytes(): B for B in pool}
                candidates = sorted(unique.values(), key=self.score, reverse=True)[:64]
        candidates = [
            B
            for B in candidates
            if np.all(B.sum(axis=1) <= self.q)
            and np.all(B.sum(axis=0) >= 1)
            and np.all(B.sum(axis=0) <= n["Dmax"])
        ]
        self.pending = sorted(candidates, key=self.score, reverse=True)[:48]

    def score(self, A):
        signal = (self.quality * A / np.sqrt(np.maximum(1, A.sum(axis=1)))[:, None]).sum(axis=0)
        ratio = np.maximum(signal / self.required, 1e-14)
        return float(ratio.min() + 0.03 * np.log(ratio).mean())

    def next(self, budget):
        if self.inflight is not None:
            return self.inflight
        while not self.pending:
            if self.level >= 2 or not self.beam:
                self.done = True
                return None
            self.level += 1
            pool = []
            for point in sorted(self.beam, key=lambda s: s["t"], reverse=True)[:4]:
                proxy = dict(point)
                proxy["t"] = self.target - 0.0004
                pool += ranked_moves(self.n, proxy, self.rng, budget)[-16:]
            self.pending = sorted(pool, key=self.score, reverse=True)
            self.beam = []
        while self.pending:
            A = self.pending.pop(0)
            h = A.tobytes()
            if h in self.seen:
                continue
            self.seen.add(h)
            self.inflight = A
            return A
        return self.next(budget)

    def submit(self, point):
        self.beam.append(point)
        self.beam = sorted(self.beam, key=lambda s: s["t"], reverse=True)[:8]
        self.inflight = None

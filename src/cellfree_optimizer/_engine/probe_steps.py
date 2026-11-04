"""Probe steps."""

import numpy as np
from .probe_base import Probe as OldProbe
from .structured import Structured, admissible, checked_ray
from .budget_core import fixed_power, warm_power, payload_upper
from .mode_search import checked_infeasibility


class Probe(OldProbe):
    def __init__(self, n, A, z, parent):
        super().__init__(n, A, z, parent)
        self.structured = admissible(A, z)
        self.lo = 0.0
        self.hi = None
        self.trial = None
        self.tested = False

    def advance(self, b, end, target, stats):
        if not self.structured:
            return super().advance(b, end, target, stats)
        self.visits += 1
        stats["structured_probe_visits"] += 1
        b.check(end)
        if self.stage == 0:
            self.best = fixed_power(
                self.n, self.A, self.z, warm_power(self.n, self.A, self.parent), b, end
            )
            if (
                np.array_equal(self.A, self.parent["A"])
                and np.array_equal(self.z, self.parent["z"])
                and (self.parent["t"] > self.best["t"])
            ):
                self.best = self.parent
            b.accept(self.best, "structured mixed warm point")
            self.lo = self.best["t"]
            self.hi = payload_upper(self.n, self.A, self.z)
            self.stage = 1
        if self.aggregate is None:
            self.aggregate = Structured(self.n, self.A, self.z)
        while not self.done:
            b.check(end)
            if self.hi - self.lo < 0.0002 or self.iterations >= 16 or self.hi < target - 2e-06:
                self.done = True
                self.reason = "structured fixed-graph bracket"
                break
            if self.pending is None:
                self.trial = target if not self.tested else (self.lo + self.hi) / 2
                if self.trial > self.hi + 1e-10:
                    self.done = True
                    self.reason = "structured payload screen"
                    break
                xx, r = self.aggregate.solve(self.trial, b, end)
                stats["structured_conic_calls"] += 1
                if r is None or checked_ray(self.aggregate, r):
                    self.hi = min(self.hi, self.trial)
                    self.tested = True
                    self.iterations += 1
                    if self.lo < target - 2e-06:
                        self.done = True
                        self.reason = "structured target screen"
                    continue
                if xx is None:
                    self.done = True
                    self.reason = "structured numerical unknown"
                    break
                self.pending = xx
            p = fixed_power(self.n, self.A, self.z, self.pending, b, end)
            self.pending = None
            self.iterations += 1
            self.tested = True
            if p["t"] > self.best["t"]:
                self.best = p
                b.accept(p, "structured convex power/code point")
            if p["t"] >= self.trial * (1 - 2e-05):
                self.lo = max(self.lo, p["t"])
            else:
                self.hi = min(self.hi, self.trial)
                if self.lo < target - 2e-06:
                    self.done = True
                    self.reason = "structured recovered target failure"
        return self.best

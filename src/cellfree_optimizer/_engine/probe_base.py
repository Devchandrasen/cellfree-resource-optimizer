"""Probe base."""

import numpy as np
from .budget_core import fixed_power, warm_power
from .fast_kernels import Aggregate, FastWM
from .mode_search import checked_infeasibility


class Probe:
    def __init__(self, n, A, z, parent):
        self.n = n
        self.A = A.copy()
        self.z = z.copy()
        self.parent = parent
        self.best = None
        self.aggregate = None
        self.wm = None
        self.x = None
        self.pending = None
        self.stage = 0
        self.iterations = 0
        self.stall = 0
        self.previous = -1.0
        self.done = False
        self.reason = None
        self.visits = 0
        self.starts = []
        self.in_start = 0

    def advance(self, b, end, target, stats):
        self.visits += 1
        stats["persistent_probe_visits"] += 1
        if self.visits > 1:
            stats["persistent_probe_resumes"] += 1
        b.check(end)
        if self.stage == 0:
            self.best = fixed_power(
                self.n, self.A, self.z, warm_power(self.n, self.A, self.parent), b, end
            )
            b.accept(self.best, "mixed warm code LP")
            self.starts.append(self.best)
            self.stage = 1
        if self.stage == 1:
            if self.aggregate is None:
                self.aggregate = Aggregate(self.n, self.A, self.z)
            xx, r = self.aggregate.solve(target, b, end)
            if checked_infeasibility(self.aggregate, r):
                self.done = True
                self.reason = "floating aggregate target screen"
                stats["checked_aggregate_target_rejections"] += 1
                return self.best
            self.pending = xx
            self.stage = 2
        if self.stage == 2:
            if self.pending is not None:
                p = fixed_power(self.n, self.A, self.z, self.pending, b, end)
                b.accept(p, "aggregate seed")
                self.starts.append(p)
                if p["t"] > self.best["t"]:
                    self.best = p
            self.pending = None
            self.starts.sort(key=lambda p: p["t"], reverse=True)
            first = self.starts.pop(0)
            self.x = first["x"].copy()
            self.stage = 3
        if self.wm is None:
            self.wm = FastWM(self.n, self.A, self.z)
        while not self.done:
            b.check(end)
            if self.pending is None:
                self.pending, r = self.wm.step(self.x, b, end)
                if self.pending is None:
                    self.done = True
                    self.reason = "no finite WM step"
                    break
            p = fixed_power(self.n, self.A, self.z, self.pending, b, end)
            self.pending = None
            self.iterations += 1
            self.in_start += 1
            stats["persistent_WM_iterations"] += 1
            if p["t"] > self.best["t"]:
                self.best = p
                b.accept(p, "continued power/code refinement")
            self.stall = self.stall + 1 if p["t"] - self.previous < 2e-05 else 0
            decreased = p["t"] < self.previous - 2e-06
            self.previous = p["t"]
            self.x = p["x"].copy()
            if decreased or self.in_start >= 22 or (self.in_start >= 5 and self.stall >= 2):
                if self.starts:
                    self.x = self.starts.pop(0)["x"].copy()
                    self.previous = -1.0
                    self.stall = 0
                    self.in_start = 0
                    stats["second_power_basins"] += 1
                else:
                    self.done = True
                    self.reason = "two-basin WM stationarity or iteration cap"
        return self.best

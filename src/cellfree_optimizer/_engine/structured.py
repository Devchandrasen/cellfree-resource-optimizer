"""Structured."""

import numpy as np
import scipy.sparse as sp
from .fast_kernels import Aggregate, solve_reusable
from .budget_core import clarabel, fixed_power, warm_power, payload_upper, Expired
from .fast_kernels import settings


def solve_tight(owner, A, b, q, cones, budget, end):
    budget.check(end)
    st = settings(budget, end)
    st.tol_feas = 1e-12
    st.tol_gap_abs = 1e-10
    st.tol_gap_rel = 1e-10
    st.max_iter = 180
    if (
        owner.solver is not None
        and A.shape == owner.shape
        and np.array_equal(A.indices, owner.indices)
        and np.array_equal(A.indptr, owner.indptr)
    ):
        owner.solver.update(A=A.data, b=b, settings=st)
    else:
        owner.solver = clarabel.DefaultSolver(sp.csc_matrix((len(q), len(q))), q, A, b, cones, st)
        owner.shape = A.shape
        owner.indices = A.indices.copy()
        owner.indptr = A.indptr.copy()
    budget.calls += 1
    r = owner.solver.solve()
    info = owner.solver.get_info().linsolver
    assert info.threads == 1 and info.name == "qdldl"
    if not hasattr(budget, "native_statuses"):
        budget.native_statuses = {}
    budget.native_statuses[str(r.status)] = budget.native_statuses.get(str(r.status), 0) + 1
    if not hasattr(budget, "structured_statuses"):
        budget.structured_statuses = {}
    budget.structured_statuses[str(r.status)] = budget.structured_statuses.get(str(r.status), 0) + 1
    return r


def admissible(A, z):
    return bool(
        np.any(~np.asarray(z, bool))
        and np.all(np.asarray(A, bool)[:, ~np.asarray(z, bool)].sum(axis=1) <= 1)
    )


def checked_ray(model, r):
    """Conservative floating screen only; never an accepted global certificate."""
    if r is None or str(r.status) not in ["PrimalInfeasible", "AlmostPrimalInfeasible"]:
        return False
    z = np.asarray(r.z, float).copy()
    if not np.isfinite(z).all() or not np.max(abs(z), initial=0):
        return False
    z /= np.max(abs(z))
    L = model.last_linear_rows
    z[:L] = np.maximum(0, z[:L])
    pos = L + model.last_zero_rows
    for _ in model.pairs:
        z[pos] = max(0, z[pos], np.nextafter(np.linalg.norm(z[pos + 1 : pos + 3]), np.inf))
        pos += 3
    residual = model.last_A.T @ z
    box = np.r_[
        1 / model.power_scale,
        [1 / np.sqrt(model.power_scale[i] * model.power_scale[j]) for _, i, j in model.pairs],
    ]
    box = np.nextafter(box, np.inf)
    margin = float(np.minimum(0, residual) @ box - model.last_b @ z)
    return margin > 1e-07 * max(1.0, abs(float(model.last_b @ z)))


class Structured(Aggregate):
    def __init__(self, n, A, z):
        assert admissible(A, z)
        super().__init__(n, A, z)
        self.nc = np.flatnonzero(~self.z)
        self.power_scale = 1 / np.maximum(
            1.0, np.maximum(n["a"][self.mm, self.kk] ** 2, self.G.max(axis=0))
        )
        self.G *= self.power_scale[None, :]
        self.desired[:, : self.E] *= self.power_scale[None, :]
        for l, (_, i, j) in enumerate(self.pairs):
            self.desired[:, self.E + l] *= np.sqrt(self.power_scale[i] * self.power_scale[j])
        ap = np.zeros((n["M"], self.V))
        ap[self.mm, np.arange(self.E)] = self.power_scale
        self.linear = sp.vstack([-sp.eye(self.V), sp.csc_matrix(ap)], format="csc")
        self.q = np.r_[np.full(self.E, 0.001), np.zeros(len(self.pairs))]

    def solve(self, target, budget, end=None):
        n = self.n
        eta = np.expm1(np.log(2) * target * n["w"] / n["rho"])
        residual = n["C"] - target * self.coherent_load
        if np.any(residual < -1e-12):
            return (None, None)
        residual = np.maximum(0, residual)
        E, V = (self.E, self.V)
        radio = np.zeros_like(self.desired)
        radio[:, :E] = eta[:, None] * self.G
        radio -= self.desired
        scale = np.maximum(1, np.max(abs(radio), axis=1))
        cc = np.flatnonzero(self.z)
        rows = [self.linear, sp.csc_matrix(radio[cc] / scale[cc, None])]
        rhs = [np.r_[np.zeros(V), np.ones(n["M"])], -eta[cc] / scale[cc]]
        caps = []
        limits = []
        for edge, (m, k) in enumerate(zip(self.mm, self.kk)):
            if self.z[k]:
                continue
            q = np.expm1(np.log(2) * residual[m] / n["rho"])
            co = np.zeros(V)
            co[:E] = -q * self.G[k]
            co[edge] += n["a"][m, k] ** 2 * self.power_scale[edge]
            tail = (self.kk == k) & (n["rank"][self.mm, k] > n["rank"][m, k])
            co[np.flatnonzero(tail)] -= q * n["a"][self.mm[tail], k] ** 2 * self.power_scale[tail]
            sc = max(1.0, float(np.max(abs(co))))
            caps.append(co / sc)
            limits.append(q / sc)
        rows.append(sp.csc_matrix(np.asarray(caps).reshape(-1, V)))
        rhs.append(np.asarray(limits))
        linear = sp.vstack(rows, format="csc")
        equal = sp.csc_matrix(radio[self.nc] / scale[self.nc, None])
        A = sp.vstack([linear, equal, self.pairmatrix], format="csc")
        b = np.r_[
            np.concatenate(rhs), -eta[self.nc] / scale[self.nc], np.zeros(self.pairmatrix.shape[0])
        ]
        cones = [
            clarabel.NonnegativeConeT(linear.shape[0]),
            clarabel.ZeroConeT(len(self.nc)),
        ] + self.cones[1:]
        self.last_A = A
        self.last_b = b
        self.last_linear_rows = linear.shape[0]
        self.last_zero_rows = len(self.nc)
        r = solve_tight(self, A, b, self.q, cones, budget, end)
        if r.x is None or not np.isfinite(r.x).all():
            return (None, r)
        x = np.zeros_like(self.A, dtype=float)
        x[self.mm, self.kk] = np.sqrt(np.maximum(0, np.asarray(r.x[:E])) * self.power_scale)
        x /= np.maximum(1, np.linalg.norm(x, axis=1))[:, None]
        return (x, r)


def optimize(n, A, z, budget, parent=None, end=None, target=None, rounds=18):
    best = fixed_power(n, A, z, warm_power(n, A, parent), budget, end)
    budget.accept(best, "structured fixed-graph warm point")
    if (
        parent is not None
        and np.array_equal(A, parent["A"])
        and np.array_equal(z, parent["z"])
        and (parent["t"] > best["t"])
    ):
        best = parent
        budget.accept(best, "same-graph feasible parent retained")
    model = Structured(n, A, z)
    lo = best["t"]
    hi = payload_upper(n, A, z)
    signal = np.where(z, np.sum(n["a"] * A, axis=0) ** 2, np.sum(n["a"] ** 2 * A, axis=0))
    hi = min(hi, float(np.min(n["rho"] / np.log(2) * np.log1p(signal) / n["w"])))
    for j in range(rounds):
        budget.check(end)
        if hi - lo < 0.0002:
            break
        t = target if j == 0 and target is not None else hi if j == 0 else (lo + hi) / 2
        if t > hi + 1e-10:
            break
        xx, r = model.solve(t, budget, end)
        if xx is not None:
            p = fixed_power(n, A, z, xx, budget, end)
            budget.accept(p, "structured convex power/code recovery")
            if p["t"] > best["t"]:
                best = p
        if best["t"] >= t * (1 - 2e-05):
            lo = best["t"]
        elif r is None or str(r.status) in [
            "PrimalInfeasible",
            "AlmostPrimalInfeasible",
            "Solved",
            "AlmostSolved",
        ]:
            hi = t
        else:
            break
        if j == 0 and target is not None and (best["t"] < target * (1 - 2e-05)):
            break
    return (best, model)

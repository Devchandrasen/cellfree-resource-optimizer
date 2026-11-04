"""Fast kernels."""

import itertools
import time
import numpy as np
import scipy.sparse as sp
from .budget_core import clarabel, Expired, WMModel, fixed_power, warm_power, payload_upper


def settings(budget, end=None):
    s = clarabel.DefaultSettings()
    s.verbose = False
    s.max_threads = 1
    s.direct_solve_method = "qdldl"
    s.max_iter = 120
    s.tol_gap_abs = 1e-08
    s.tol_gap_rel = 1e-08
    s.tol_feas = 1e-09
    s.presolve_enable = False
    s.input_sparse_dropzeros = False
    s.time_limit = max(0.001, budget.left(end) - 0.003)
    return s


def solve_reusable(owner, A, b, q, cones, budget, end=None):
    budget.check(end)
    st = settings(budget, end)
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
    result = owner.solver.solve()
    info = owner.solver.get_info().linsolver
    assert info.name == "qdldl" and info.threads == 1
    if not hasattr(budget, "native_statuses"):
        budget.native_statuses = {}
    status = str(result.status)
    budget.native_statuses[status] = budget.native_statuses.get(status, 0) + 1
    return result


class Aggregate:
    """Exact fixed-mode aggregate radio in p=x^2; individual NCJT FH relaxed."""

    def __init__(self, n, A, z):
        self.n = n
        self.A = np.asarray(A, bool)
        self.z = np.asarray(z, bool)
        self.mm, self.kk = np.nonzero(A)
        E = len(self.mm)
        self.E = E
        self.pairs = []
        for k in np.where(z)[0]:
            ids = np.where(self.kk == k)[0]
            self.pairs.extend(((k, i, j) for i, j in itertools.combinations(ids, 2)))
        F = len(self.pairs)
        V = E + F
        self.V = V
        G = np.array([n["g"][self.mm, k, self.kk] for k in range(n["K"])])
        self.G = G
        desired = np.zeros((n["K"], V))
        for k in range(n["K"]):
            ids = np.where(self.kk == k)[0]
            desired[k, ids] = n["a"][self.mm[ids], k] ** 2
        for l, (k, i, j) in enumerate(self.pairs):
            desired[k, E + l] = 2 * n["a"][self.mm[i], k] * n["a"][self.mm[j], k]
        self.desired = desired
        ap = np.zeros((n["M"], V))
        ap[self.mm, np.arange(E)] = 1
        self.linear = sp.vstack([-sp.eye(V), sp.csc_matrix(ap)], format="csc")
        self.cones = [clarabel.NonnegativeConeT(V + n["M"] + n["K"])]
        rr = []
        cc = []
        dd = []
        for l, (k, i, j) in enumerate(self.pairs):
            r = 3 * l
            rr.extend([r, r, r + 1, r + 2, r + 2])
            cc.extend([i, j, E + l, i, j])
            dd.extend([-1, -1, -2, -1, 1])
            self.cones.append(clarabel.SecondOrderConeT(3))
        self.pairmatrix = sp.coo_matrix((dd, (rr, cc)), shape=(3 * F, V)).tocsc()
        self.q = np.r_[np.ones(E), np.zeros(F)] * 0.001
        coherent_load = self.A[:, self.z] @ n["w"][self.z]
        self.q[:E] *= 1 + 3 * (~self.z[self.kk] & (coherent_load[self.mm] > 0))
        self.forwarding_cuts = []
        self.coherent_load = coherent_load
        for k in np.where(~self.z)[0]:
            busy = np.where(self.A[:, k] & (coherent_load > 0))[0]
            busy = busy[np.argsort(-n["a"][busy, k])[:8]]
            for size in [1, 2]:
                for blocked in itertools.combinations(busy, size):
                    outside = (self.kk == k) & ~np.isin(self.mm, blocked)
                    self.forwarding_cuts.append((k, np.array(blocked, int), outside))
        self.solver = None
        self.indices = None
        self.indptr = None

    def solve(self, target, budget, end=None):
        n = self.n
        eta = np.expm1(np.log(2) * target * n["w"] / n["rho"])
        radio = np.zeros_like(self.desired)
        radio[:, : self.E] = eta[:, None] * self.G
        radio -= self.desired
        scale = np.maximum(1, np.max(np.abs(radio), axis=1))
        cutrows = []
        cutrhs = []
        for k, blocked, outside in self.forwarding_cuts:
            required = target * n["w"][k] - np.sum(
                n["C"][blocked] - target * self.coherent_load[blocked]
            )
            if required <= 1e-07:
                continue
            e = np.expm1(np.log(2) * required / n["rho"])
            row = np.zeros(self.V)
            row[: self.E] = e * self.G[k]
            ids = np.flatnonzero(outside)
            row[ids] -= n["a"][self.mm[ids], k] ** 2
            sc = max(1, np.max(np.abs(row)))
            cutrows.append(row / sc)
            cutrhs.append(-e / sc)
        linear = sp.vstack(
            [self.linear, sp.csc_matrix(radio / scale[:, None])]
            + ([sp.csc_matrix(np.array(cutrows))] if cutrows else []),
            format="csc",
        )
        A = sp.vstack([linear, self.pairmatrix], format="csc")
        b = np.r_[
            np.zeros(self.V),
            np.ones(n["M"]),
            -eta / scale,
            cutrhs,
            np.zeros(self.pairmatrix.shape[0]),
        ]
        cones = [clarabel.NonnegativeConeT(linear.shape[0])] + self.cones[1:]
        self.last_A = A
        self.last_b = b
        self.last_linear_rows = linear.shape[0]
        r = solve_reusable(self, A, b, self.q, cones, budget, end)
        if r.x is None or not np.isfinite(r.x).all():
            return (None, r)
        x = np.zeros_like(self.A, dtype=float)
        x[self.mm, self.kk] = np.sqrt(np.maximum(0, np.asarray(r.x[: self.E])))
        x /= np.maximum(1, np.linalg.norm(x, axis=1))[:, None]
        return (x, r)


def coherent(n, A, budget, parent=None, end=None, rounds=14, target=None):
    z = np.ones(n["K"], bool)
    best = fixed_power(n, A, z, warm_power(n, A, parent), budget, end)
    budget.accept(best, "fast coherent initialization")
    model = Aggregate(n, A, z)
    low = best["t"]
    upper = payload_upper(n, A, z)
    isolated = n["rho"] / np.log(2) * np.log1p(np.sum(n["a"] * A, axis=0) ** 2) / n["w"]
    upper = min(upper, float(isolated.min()))
    if target is not None:
        x, r = model.solve(target, budget, end)
        if x is not None:
            p = fixed_power(n, A, z, x, budget, end)
            if p["t"] > best["t"]:
                best = p
                budget.accept(best, "coherent target test")
        if best["t"] < target * (1 - 2e-05):
            return best
        low = best["t"]
    for j in range(rounds):
        if upper - low < 0.0002:
            break
        budget.check(end)
        t = upper if j == 0 else (low + upper) / 2
        x, r = model.solve(t, budget, end)
        if x is not None:
            p = fixed_power(n, A, z, x, budget, end)
            if p["t"] > best["t"]:
                best = p
                budget.accept(best, "exact power-lift coherent polish")
        if best["t"] >= t * (1 - 2e-05):
            low = best["t"]
        else:
            upper = t
    return best


class FastWM:
    """Same surrogate, with numeric updates of one fixed CSC sparsity pattern."""

    def __init__(self, n, A, z):
        self.base = WMModel(n, A, z)
        m = self.base
        self.n = n
        const = sp.vstack([b[0] for b in m.blocks], format="coo")
        bb = np.concatenate([b[1] for b in m.blocks])
        offset = len(bb)
        rows = list(const.row)
        cols = list(const.col)
        values = list(const.data)
        tags = [0] * len(values)
        streams = [0] * len(values)
        self.first = []
        self.last = []
        bparts = [bb]
        cones = m.cones.copy()

        def append(r, c, v, tag, j):
            rows.append(r)
            cols.append(c)
            values.append(v)
            tags.append(tag)
            streams.append(j)

        for j, L in enumerate(m.own):
            first = offset
            last = offset + L.shape[0] + 1
            self.first.append(first)
            self.last.append(last)
            nz = np.flatnonzero(m.s[j])
            for r in [first, last]:
                for c in nz:
                    append(r, c, -m.s[j, c], 1, j)
                append(r, m.E + m.users[j], 1.0, 2, j)
                append(r, m.ir + j, 1.0, 0, j)
            coo = L.tocoo()
            for r, c, v in zip(coo.row, coo.col, coo.data):
                append(first + 1 + r, c, -2 * v, 3, j)
            bparts.append(np.r_[1.0, np.zeros(L.shape[0]), -1.0])
            cones.append(clarabel.SecondOrderConeT(L.shape[0] + 2))
            offset = last + 1
        order = np.lexsort((np.asarray(rows), np.asarray(cols)))
        self.values = np.asarray(values)[order]
        self.tags = np.asarray(tags)[order]
        self.streams = np.asarray(streams)[order]
        self.A = sp.coo_matrix(
            (self.values, (np.asarray(rows)[order], np.asarray(cols)[order])), shape=(offset, m.nv)
        ).tocsc()
        assert len(self.A.data) == len(self.values)
        self.bb = np.concatenate(bparts)
        self.cones = cones
        self.q = np.zeros(m.nv)
        self.q[-1] = -1
        self.own = sp.vstack(m.own, format="csc")
        self.own_offsets = np.r_[0, np.cumsum([L.shape[0] for L in m.own])]
        self.solver = None
        self.indices = None
        self.indptr = None

    def step(self, x, budget, end=None):
        m = self.base
        q = x[m.mm, m.kk]
        d = m.s @ q
        own = np.add.reduceat((self.own @ q) ** 2, self.own_offsets[:-1])
        total = 1 + m.base[m.users] @ (q * q) + own
        den = np.maximum(total - d * d, 1e-100)
        rx = d / total
        w = total / den
        factor = self.n["rho"] / np.log(2)
        alpha = factor * w * rx**2
        beta = factor * (np.log(w) + 1 - w * (1 + rx**2))
        linear = 2 * factor * w * rx
        factors = np.ones(len(self.values))
        for tag, vector in [(1, linear), (2, alpha), (3, np.sqrt(alpha))]:
            mask = self.tags == tag
            factors[mask] = vector[self.streams[mask]]
        self.A.data[:] = self.values * factors
        b = self.bb.copy()
        b[self.first] += beta
        b[self.last] += beta
        self.last_b = b
        r = solve_reusable(self, self.A, b, self.q, self.cones, budget, end)
        if r.x is None or not np.isfinite(r.x).all():
            return (None, r)
        xx = np.zeros_like(x)
        xx[m.mm, m.kk] = np.maximum(0, np.asarray(r.x[: m.E]))
        xx /= np.maximum(1, np.linalg.norm(xx, axis=1))[:, None]
        return (xx, r)


def mixed(n, A, z, budget, parent=None, end=None, iterations=30, aggregate=True, target=None):
    best = fixed_power(n, A, z, warm_power(n, A, parent), budget, end)
    budget.accept(best, "mixed warm code LP")
    starts = [best]
    if aggregate and budget.left(end) > 0.025:
        model = Aggregate(n, A, z)
        desired = target or min(
            payload_upper(n, A, z) * 0.999, max(best["t"] * 1.005, budget.best["t"] * 1.002)
        )
        for t in [desired, min(desired, max(0.1, best["t"]) * 0.98)]:
            if t <= 0:
                continue
            xx, r = model.solve(t, budget, end)
            if xx is not None:
                p = fixed_power(n, A, z, xx, budget, end)
                starts.append(p)
                budget.accept(p, "mixed aggregate-radio seed")
                if p["t"] > best["t"]:
                    best = p
    model = FastWM(n, A, z)
    for start in sorted(starts, key=lambda s: s["t"], reverse=True)[:2]:
        x = warm_power(n, A, start)
        previous = -1.0
        stall = 0
        for i in range(iterations):
            budget.check(end)
            xx, r = model.step(x, budget, end)
            if xx is None:
                break
            p = fixed_power(n, A, z, xx, budget, end)
            if p["t"] > best["t"]:
                best = p
                budget.accept(best, "reused-CSC power/code WMMSE")
            if p["t"] < previous - 2e-06:
                break
            stall = stall + 1 if p["t"] - previous < 2e-05 else 0
            previous = p["t"]
            x = p["x"]
            if i >= 4 and stall >= 2:
                break
    return best

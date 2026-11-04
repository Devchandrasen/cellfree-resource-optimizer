"""Successive convex power and payload allocation."""

import time
import numpy as np
import scipy.sparse as sp
from .budget_core import fixed_power, warm_power, Expired, payload_upper
from .fast_kernels import solve_reusable


class PublishedSCA:
    def __init__(self, n, A, z):
        self.n = n
        self.A = np.asarray(A, bool)
        self.z = np.asarray(z, bool)
        self.mm, self.kk = np.nonzero(self.A)
        self.E = len(self.mm)
        E = self.E
        M, K = self.A.shape
        users = []
        desired = []
        variance = []
        traffic = []
        base = np.array([n["g"][self.mm, k, self.kk] for k in range(K)])
        for k in range(K):
            ids = np.flatnonzero(self.kk == k)
            streams = [None] if self.z[k] else list(ids[np.argsort(n["rank"][self.mm[ids], k])])
            for edge in streams:
                signal = np.zeros(E)
                noise = base[k].copy()
                flow = np.zeros(M)
                if edge is None:
                    signal[ids] = n["a"][self.mm[ids], k]
                    flow[self.A[:, k]] = 1
                else:
                    m = self.mm[edge]
                    signal[edge] = n["a"][m, k]
                    tail = ids[n["rank"][self.mm[ids], k] > n["rank"][m, k]]
                    noise[tail] += n["a"][self.mm[tail], k] ** 2
                    flow[m] = 1
                users.append(k)
                desired.append(signal)
                variance.append(noise)
                traffic.append(flow)
        self.users = np.array(users, int)
        self.S = np.array(desired)
        self.G = np.array(variance)
        self.F = np.array(traffic).T
        self.J = len(users)
        J = self.J
        V = E + 2 * J + 1
        self.V = V
        self.factor = n["rho"] / np.log(2)
        flow = np.zeros((M, V))
        flow[:, E + J : E + 2 * J] = self.F
        service = np.zeros((K, V))
        service[np.arange(K), -1] = n["w"]
        for j, k in enumerate(self.users):
            service[k, E + J + j] = -1
        self.staticA = sp.vstack(
            [-sp.eye(V, format="csc"), sp.csc_matrix(flow), sp.csc_matrix(service)], format="csc"
        )
        self.staticb = np.r_[np.zeros(V), n["C"], np.zeros(K)]
        self.staticcones = None
        self.power = []
        for m in range(M):
            ids = np.flatnonzero(self.mm == m)
            rows = np.arange(1, len(ids) + 1)
            self.power.append(
                sp.coo_matrix((-np.ones(len(ids)), (rows, ids)), shape=(len(ids) + 1, V)).tocsc()
            )
        self.solver = None
        self.indices = None
        self.indptr = None

    def anchors(self, x):
        q = np.asarray(x)[self.mm, self.kk]
        desired = self.S @ q
        den = 1 + self.G @ (q * q)
        return (desired / den, desired * desired / den)

    def matrices(self, x):
        from .budget_core import clarabel

        E, J, V = (self.E, self.J, self.V)
        h, xi0 = self.anchors(x)
        blocks = [self.staticA]
        rhs = [self.staticb]
        cones = [clarabel.NonnegativeConeT(len(self.staticb))]
        for matrix in self.power:
            blocks.append(matrix)
            rhs.append(np.r_[1.0, np.zeros(matrix.shape[0] - 1)])
            cones.append(clarabel.SecondOrderConeT(matrix.shape[0]))
        for j in range(J):
            d = -2 * h[j] * self.S[j]
            nz = np.flatnonzero(d)
            rr = list(np.r_[np.zeros(len(nz), int), np.full(len(nz), E + 1, int)])
            cc = list(np.r_[nz, nz])
            vv = list(np.r_[d[nz], d[nz]])
            rr.extend([0, E + 1])
            cc.extend([E + j, E + j])
            vv.extend([1.0, 1.0])
            rr.extend(range(1, E + 1))
            cc.extend(range(E))
            vv.extend(-2 * h[j] * np.sqrt(self.G[j]))
            matrix = sp.coo_matrix((vv, (rr, cc)), shape=(E + 2, V)).tocsc()
            blocks.append(matrix)
            rhs.append(np.r_[1 - h[j] ** 2, np.zeros(E), -1 - h[j] ** 2])
            cones.append(clarabel.SecondOrderConeT(E + 2))
            dlog = 1 + np.log1p(xi0[j])
            matrix = sp.coo_matrix(
                (
                    [-1.0, 1 / self.factor, 1.0, 1 / self.factor],
                    ([0, 0, 2, 2], [E + j, E + J + j, E + j, E + J + j]),
                ),
                shape=(3, V),
            ).tocsc()
            blocks.append(matrix)
            rhs.append(np.array([dlog + 1, 2 * np.sqrt(1 + xi0[j]), dlog - 1]))
            cones.append(clarabel.SecondOrderConeT(3))
        objective = np.zeros(V)
        objective[-1] = -1
        return (sp.vstack(blocks, format="csc"), np.concatenate(rhs), objective, cones)

    def step(self, x, budget, end=None):
        A, b, q, cones = self.matrices(x)
        result = solve_reusable(self, A, b, q, cones, budget, end)
        if result.x is None or not np.isfinite(result.x).all():
            return (None, result)
        xx = np.zeros(self.A.shape)
        xx[self.mm, self.kk] = np.maximum(0, np.asarray(result.x[: self.E]))
        xx /= np.maximum(1, np.linalg.norm(xx, axis=1))[:, None]
        return (xx, result)


def optimize(n, A, z, budget, parent=None, end=None, iterations=30):
    best = fixed_power(n, A, z, warm_power(n, A, parent), budget, end)
    budget.accept(best, "TWC26-MMF original-code initialization")
    if best["t"] >= payload_upper(n, A, z) - 2e-06:
        return best
    model = PublishedSCA(n, A, z)
    x = best["x"]
    stall = 0
    for iteration in range(iterations):
        budget.check(end)
        previous = best["t"]
        xx, result = model.step(x, budget, end)
        if xx is None:
            break
        point = fixed_power(n, A, z, xx, budget, end)
        budget.accept(point, "TWC26 Eq34-41 MMF SCA")
        if point["t"] > best["t"]:
            best = point
        if point["t"] < previous - 2e-05:
            break
        stall = stall + 1 if best["t"] - previous < 2e-05 else 0
        x = point["x"]
        if iteration >= 4 and stall >= 2:
            break
    return best

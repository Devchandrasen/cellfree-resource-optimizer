"""Budget core."""

from contextlib import contextmanager
from contextvars import ContextVar

import os
from pathlib import Path
import sys
import time
import warnings
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
import clarabel
from .physical import certify, validate, rates

_SESSION = ContextVar("acceptance_session", default=None)


@contextmanager
def acceptance_session(seconds):
    """Cap every nested controller by the enclosing public acceptance clock."""
    start = time.perf_counter()
    state = dict(start=start, deadline=start + seconds, engine_start=None)
    token = _SESSION.set(state)
    try:
        yield state
    finally:
        _SESSION.reset(token)


class Expired(Exception):
    pass


class Budget:
    def __init__(self, n, policy, seconds):
        self.n = n
        self.policy = policy
        self.seconds = float(seconds)
        self.start = time.perf_counter()
        self.cpu_start = time.process_time()
        self.deadline = self.start + self.seconds
        session = _SESSION.get()
        if session is not None:
            self.deadline = min(self.deadline, session["deadline"])
            self.seconds = max(0.0, self.deadline - self.start)
            if session["engine_start"] is None:
                session["engine_start"] = self.start
        A = np.asarray(n["fixedA"], bool)
        z = np.full(n["K"], policy in ("CJT", "joint"), bool)
        self.best = certify(
            n, A, z, np.zeros_like(A, dtype=float), 0, np.zeros_like(A, dtype=float)
        )
        self.trace = [dict(seconds=0.0, rate=0.0, stage="zero")]
        self.events = []
        self.calls = 0
        self.proposals = 0
        self.late = 0
        self.max_residual = 0.0

    def left(self, end=None):
        return min(self.deadline, end or self.deadline) - time.perf_counter()

    def check(self, end=None):
        if self.left(end) < 0.004:
            raise Expired

    def accept(self, s, stage):
        self.proposals += 1
        res = validate(self.n, s)
        self.max_residual = max(self.max_residual, res)
        z = np.asarray(s["z"], bool)
        assert self.policy != "CJT" or z.all()
        assert self.policy != "NCJT" or not z.any()
        assert self.policy != "fixed" or np.array_equal(s["A"], self.n["fixedA"])
        elapsed = time.perf_counter() - self.start
        if elapsed > self.seconds:
            self.late += 1
            return False
        if s["t"] > self.best["t"] + 1e-10:
            self.best = s
            self.trace.append(dict(seconds=elapsed, rate=float(s["t"]), stage=stage))
            return True
        return False


def solve_cone(q, blocks, cones, budget, end=None):
    budget.check(end)
    A = sp.vstack([x[0] for x in blocks], format="csc")
    b = np.concatenate([x[1] for x in blocks])
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 120
    settings.max_threads = 1
    settings.direct_solve_method = "qdldl"
    settings.tol_gap_abs = 1e-08
    settings.tol_gap_rel = 1e-08
    settings.tol_feas = 1e-08
    settings.time_limit = max(0.001, budget.left(end) - 0.003)
    solver = clarabel.DefaultSolver(
        sp.csc_matrix((len(q), len(q))), np.asarray(q, float), A, b, cones, settings
    )
    budget.calls += 1
    result = solver.solve()
    info = solver.get_info().linsolver
    assert info.name == "qdldl" and info.threads == 1
    return result


def fixed_power(n, A, z, x, budget, end=None):
    budget.check(end)
    A = np.asarray(A, bool)
    z = np.asarray(z, bool)
    x = np.maximum(0, np.asarray(x, float)) * A
    x /= np.maximum(1, np.linalg.norm(x, axis=1))[:, None]
    rc, rn = rates(n, A, z, x)
    u = np.zeros_like(x)
    if z.all():
        copies = A @ n["w"]
        t = min(float(np.min(rc / n["w"])), float(np.min(n["C"] / np.maximum(copies, 1e-300))))
    else:
        mm, kk = np.nonzero(A & ~z[None, :])
        E = len(mm)
        matrix = np.zeros((n["M"] + n["K"], E + 1))
        matrix[mm, np.arange(E)] = 1
        matrix[: n["M"], -1] = A[:, z] @ n["w"][z]
        matrix[n["M"] + kk, np.arange(E)] = -1
        matrix[n["M"] :, -1] = n["w"]
        rhs = np.r_[n["C"], np.where(z, rc, 0)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = linprog(
                np.r_[np.zeros(E), -1.0],
                A_ub=matrix,
                b_ub=rhs,
                bounds=list(zip(np.zeros(E), rn[mm, kk])) + [(0, None)],
                method="highs",
                options={"time_limit": max(0.001, budget.left(end) - 0.002), "threads": 1},
            )
        budget.calls += 1
        if r.x is None:
            raise Expired if budget.left(end) < 0.005 else RuntimeError(r.message)
        t = max(0, float(r.x[-1]))
        u[mm, kk] = np.maximum(0, r.x[:E])
    s = certify(n, A, z, x, t, u)
    validate(n, s)
    return s


def warm_power(n, A, parent=None):
    if parent is None:
        quality = np.sum(n["a"] ** 2 * A, axis=0)
        x = A * (n["w"] / np.sqrt(np.maximum(quality, 1e-100)))[None, :]
        x /= np.maximum(np.linalg.norm(x, axis=1), 1e-12)[:, None]
    else:
        x = np.maximum(parent["x"] * A, 0.001 * A)
        x /= np.maximum(1, np.linalg.norm(x, axis=1))[:, None]
    return x


def payload_upper(n, A, z):
    copies = A[:, z] @ n["w"][z]
    return min(
        float(np.min(n["C"] / np.maximum(copies, 1e-300))),
        float(n["C"].sum() / (copies.sum() + n["w"][~z].sum())),
    )


def coherent(n, A, budget, parent=None, end=None, rounds=14):
    A = np.asarray(A, bool)
    M, K = (n["M"], n["K"])
    mm, kk = np.nonzero(A)
    E = len(mm)
    z = np.ones(K, bool)
    best = fixed_power(n, A, z, warm_power(n, A, parent), budget, end)
    budget.accept(best, "CJT initial")
    upper = payload_upper(n, A, z)
    low = best["t"]
    isolated = n["rho"] / np.log(2) * np.log1p(np.sum(n["a"] * A, axis=0) ** 2) / n["w"]
    upper = min(upper, float(isolated.min()))
    for iteration in range(rounds):
        budget.check(end)
        if upper - low <= 0.0002:
            break
        target = upper if iteration == 0 else (low + upper) / 2
        eta = np.expm1(np.log(2) * n["w"] * target / n["rho"])
        blocks = [(-sp.eye(E, format="csc"), np.zeros(E))]
        cones = [clarabel.NonnegativeConeT(E)]
        for m in range(M):
            ids = np.where(mm == m)[0]
            if len(ids):
                block = sp.lil_matrix((len(ids) + 1, E))
                block[np.arange(1, len(ids) + 1), ids] = -1
                blocks.append((block.tocsc(), np.r_[1.0, np.zeros(len(ids))]))
                cones.append(clarabel.SecondOrderConeT(len(ids) + 1))
        scale = np.sqrt(1 + np.einsum("mki,mi->k", n["g"], best["x"] ** 2))
        for k in range(K):
            ids = np.where(kk == k)[0]
            row = np.zeros(E)
            row[ids] = -n["a"][mm[ids], k] / np.sqrt(eta[k]) / scale[k]
            block = sp.vstack(
                [
                    sp.csc_matrix(row[None, :]),
                    sp.csc_matrix((1, E)),
                    -sp.diags(np.sqrt(n["g"][mm, k, kk]) / scale[k]),
                ],
                format="csc",
            )
            blocks.append((block, np.r_[0.0, 1 / scale[k], np.zeros(E)]))
            cones.append(clarabel.SecondOrderConeT(E + 2))
        result = solve_cone(np.zeros(E), blocks, cones, budget, end)
        if result.x is not None and np.isfinite(result.x).all():
            x = np.zeros((M, K))
            x[mm, kk] = np.maximum(0, result.x)
            point = fixed_power(n, A, z, x, budget, end)
            if point["t"] > best["t"]:
                best = point
                budget.accept(best, "full-range CJT SOC")
        if best["t"] >= target * (1 - 2e-06):
            low = max(low, best["t"])
        else:
            upper = target
    return best


class WMModel:
    def __init__(self, n, A, z):
        self.n = n
        self.A = np.asarray(A, bool)
        self.z = np.asarray(z, bool)
        M, K = (n["M"], n["K"])
        self.mm, self.kk = np.nonzero(A)
        E = len(self.mm)
        self.E = E
        self.base = np.array([n["g"][self.mm, k, self.kk] for k in range(K)])
        desired = []
        own = []
        users = []
        aps = []
        for k in range(K):
            ids = np.where(self.kk == k)[0]
            if z[k]:
                row = np.zeros(E)
                row[ids] = n["a"][self.mm[ids], k]
                desired.append(row)
                own.append(sp.csc_matrix(row[None, :]))
                users.append(k)
                aps.append(-1)
            else:
                for j in ids:
                    m = self.mm[j]
                    row = np.zeros(E)
                    row[j] = n["a"][m, k]
                    later = ids[n["rank"][self.mm[ids], k] >= n["rank"][m, k]]
                    L = sp.coo_matrix(
                        (n["a"][self.mm[later], k], (np.arange(len(later)), later)),
                        shape=(len(later), E),
                    ).tocsc()
                    desired.append(row)
                    own.append(L)
                    users.append(k)
                    aps.append(m)
        self.s = np.array(desired)
        self.own = own
        self.users = np.array(users)
        self.S = len(users)
        S = self.S
        self.nv = E + K + S + 1
        nv = self.nv
        self.it = nv - 1
        self.ir = E + K
        H = np.zeros((K, S))
        G = np.zeros((M, S))
        for j, (k, m) in enumerate(zip(users, aps)):
            H[k, j] = 1
            if m < 0:
                G[:, j] = A[:, k]
            else:
                G[m, j] = 1
        linear = sp.lil_matrix((nv + M + K, nv))
        linear[:nv, :] = -sp.eye(nv)
        linear[nv : nv + M, self.ir : self.ir + S] = G
        linear[nv + M :, self.ir : self.ir + S] = -H
        linear[nv + M :, self.it] = n["w"][:, None]
        self.blocks = [(linear.tocsc(), np.r_[np.zeros(nv), n["C"], np.zeros(K)])]
        self.cones = [clarabel.NonnegativeConeT(nv + M + K)]
        for m in range(M):
            ids = np.where(self.mm == m)[0]
            if len(ids):
                B = sp.lil_matrix((len(ids) + 1, nv))
                B[np.arange(1, len(ids) + 1), ids] = -1
                self.blocks.append((B.tocsc(), np.r_[1.0, np.zeros(len(ids))]))
                self.cones.append(clarabel.SecondOrderConeT(len(ids) + 1))
        for k in range(K):
            B = sp.lil_matrix((E + 2, nv))
            B[0, E + k] = -1
            B[-1, E + k] = -1
            B[1:-1, :E] = -2 * sp.diags(np.sqrt(self.base[k]))
            self.blocks.append((B.tocsc(), np.r_[1.0, np.zeros(E), -1.0]))
            self.cones.append(clarabel.SecondOrderConeT(E + 2))

    def coefficients(self, x):
        q = x[self.mm, self.kk]
        d = self.s @ q
        total = (
            1 + self.base[self.users] @ (q * q) + np.array([np.sum((L @ q) ** 2) for L in self.own])
        )
        den = np.maximum(total - d * d, 1e-100)
        receiver = d / total
        weight = total / den
        factor = self.n["rho"] / np.log(2)
        alpha = factor * weight * receiver**2
        constant = factor * (np.log(weight) + 1 - weight * (1 + receiver**2))
        linear = (2 * factor * weight * receiver)[:, None] * self.s
        return (alpha, constant, linear)

    def step(self, x, budget, end=None):
        alpha, constant, linear = self.coefficients(x)
        blocks = self.blocks.copy()
        cones = self.cones.copy()
        for j in range(self.S):
            row = np.zeros(self.nv)
            row[: self.E] = -linear[j]
            row[self.E + self.users[j]] = alpha[j]
            row[self.ir + j] = 1
            L = sp.hstack(
                [
                    -2 * np.sqrt(alpha[j]) * self.own[j],
                    sp.csc_matrix((self.own[j].shape[0], self.nv - self.E)),
                ],
                format="csc",
            )
            B = sp.vstack(
                [sp.csc_matrix(row[None, :]), L, sp.csc_matrix(row[None, :])], format="csc"
            )
            blocks.append((B, np.r_[constant[j] + 1, np.zeros(L.shape[0]), constant[j] - 1]))
            cones.append(clarabel.SecondOrderConeT(L.shape[0] + 2))
        objective = np.zeros(self.nv)
        objective[-1] = -1
        result = solve_cone(objective, blocks, cones, budget, end)
        xx = np.zeros_like(x)
        if result.x is None or not np.isfinite(result.x).all():
            return (None, result)
        xx[self.mm, self.kk] = np.maximum(0, np.asarray(result.x[: self.E]))
        xx /= np.maximum(1, np.linalg.norm(xx, axis=1))[:, None]
        return (xx, result)


def optimize(n, A, z, budget, parent=None, end=None, iterations=22):
    z = np.asarray(z, bool)
    if z.all():
        return coherent(n, A, budget, parent, end)
    x = warm_power(n, A, parent)
    best = fixed_power(n, A, z, x, budget, end)
    budget.accept(best, "power initialization")
    model = WMModel(n, A, z)
    stall = 0
    for j in range(iterations):
        budget.check(end)
        xx, result = model.step(x, budget, end)
        if xx is None:
            break
        point = fixed_power(n, A, z, xx, budget, end)
        previous = best["t"]
        if point["t"] > best["t"]:
            best = point
            budget.accept(best, "joint power/code WMMSE")
        if point["t"] < previous - 2e-06:
            break
        x = point["x"]
        stall = stall + 1 if best["t"] - previous < 1e-05 else 0
        if j >= 4 and stall >= 2:
            break
    return best


def nc_radio(n, A, budget, end=None, rounds=17):
    """Aggregate LP relaxation; only independently coded/checked powers are accepted."""
    M, K = (n["M"], n["K"])
    mm, kk = np.nonzero(A)
    E = len(mm)
    low = 0.0
    upper = float(np.min(n["rho"] / np.log(2) * np.log1p(np.sum(n["a"] ** 2 * A, axis=0)) / n["w"]))
    powers = np.zeros((M, K))
    last = None
    for j in range(rounds):
        budget.check(end)
        target = (low + upper) / 2
        eta = np.expm1(np.log(2) * target * n["w"] / n["rho"])
        radio = np.array([eta[k] * n["g"][mm, k, kk] for k in range(K)])
        for k in range(K):
            ids = kk == k
            radio[k, ids] -= n["a"][mm[ids], k] ** 2
        ap = np.zeros((M, E))
        ap[mm, np.arange(E)] = 1
        scales = np.maximum(1, np.max(np.abs(radio), axis=1))
        matrix = np.vstack([radio / scales[:, None], ap])
        rhs = np.r_[-eta / scales, np.ones(M)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = linprog(
                np.zeros(E),
                A_ub=matrix,
                b_ub=rhs,
                bounds=(0, 1),
                method="highs",
                options={"threads": 1, "time_limit": max(0.001, budget.left(end) - 0.002)},
            )
        budget.calls += 1
        if r.success:
            low = target
            powers[mm, kk] = np.maximum(0, r.x)
            last = powers.copy()
        elif r.status == 2:
            upper = target
        else:
            break
        if upper - low < 0.0002:
            break
    if last is None:
        return (None, float(upper))
    support = A.copy()
    for k in range(K):
        ids = np.where(A[:, k])[0]
        keep = ids[np.argsort(last[ids, k])[-n["Dmax"] :]]
        support[:, k] = False
        support[keep, k] = True
    if budget.policy == "fixed":
        support = np.asarray(n["fixedA"], bool)
    point = fixed_power(n, support, np.zeros(K, bool), np.sqrt(last) * support, budget, end)
    budget.accept(point, "aggregate NCJT LP + exact SIC coding")
    return (point, float(upper))

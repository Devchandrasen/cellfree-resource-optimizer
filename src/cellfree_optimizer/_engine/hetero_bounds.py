"""Heterogeneous radio and forwarding bounds."""

import os
from fractions import Fraction as F
import math
import json
import hashlib
import numpy as np
from .bounds import outward_rate, rate_lower_fraction

GRID = 2**40
SQRT_GRID = 2**50


def rf(x):
    return F.from_float(float(x))


def pair(q):
    return [str(q.numerator), str(q.denominator)]


def sqrt_bound(q, up=False, scale=SQRT_GRID):
    assert q >= 0
    r = math.isqrt(q.numerator * scale * scale // q.denominator)
    if up and r * r * q.denominator < q.numerator * scale * scale:
        r += 1
    return F(r, scale)


def eta_down(rate, rho):
    rate = F(rate)
    assert rate > 0
    e = rf(np.expm1(float(rate) * np.log(2) / rho))
    assert e > 0
    for _ in range(8):
        if rf(outward_rate(e, rho)) <= rate:
            return e
        e = rf(np.nextafter(float(e), -np.inf))
    lo = F(0)
    hi = e
    for _ in range(60):
        mid = (lo + hi) / 2
        if rf(outward_rate(mid, rho)) <= rate:
            lo = mid
        else:
            hi = mid
    assert rf(outward_rate(lo, rho)) <= rate
    return lo


def eta_up(rate, rho):
    rate = F(rate)
    e = float(np.expm1(float(rate) * np.log(2) / rho))
    assert 0 <= float(rate) < 8
    for _ in range(64):
        if rate_lower_fraction(rf(e), rho) >= rate:
            return rf(e)
        e = float(np.nextafter(e, np.inf))
    raise AssertionError("Directed exponent check did not bracket the requested code")


def model_hash(n):
    h = hashlib.sha256()
    for key in ["b", "zeta", "P", "C", "w"]:
        h.update(np.asarray(n[key], dtype="<f8").tobytes())
    h.update(np.array([n["rho"], n["Dmax"]], dtype="<f8").tobytes())
    return h.hexdigest()


def floors(n, t, mode, steps=16):
    M, K = (n["M"], n["K"])
    assert 1 <= n["Dmax"] <= M and int(n["Dmax"]) == n["Dmax"]
    assert np.shape(n["b"]) == (M, K) and np.shape(n["zeta"]) == (M, K, K)
    assert all(
        (
            np.shape(n[key]) == (size,) and np.all(np.isfinite(n[key]))
            for key, size in [("P", M), ("C", M), ("w", K)]
        )
    )
    assert np.all(np.isfinite(n["b"])) and np.all(np.isfinite(n["zeta"]))
    if "nu" in n:
        cross = np.array(n["nu"], copy=True).reshape(M, K, K)
        assert np.all(np.isfinite(cross))
        cross[:, np.arange(K), np.arange(K)] = 0
        assert not np.any(cross), "Requires zero cross-user coherent means"
    t = rf(t)
    P = list(map(rf, n["P"]))
    C = list(map(rf, n["C"]))
    w = list(map(rf, n["w"]))
    eta = [eta_down(t * v, n["rho"]) for v in w]
    assert (
        mode in ["C", "N", "J"]
        and 0 < n["rho"] < 1
        and all((v > 0 for v in P + C + w))
        and np.all(n["b"] >= 0)
        and np.all(n["zeta"] >= 0)
    )
    if mode == "J" and t * sum(w) > sum(C):
        return dict(excluded=True, reason="joint_total_forwarding", steps=0, eta=eta)
    eligible = [list(range(M)) for _ in range(K)]
    degree = [1] * K
    reason = None
    if mode in ["C", "J"]:
        sorted_w = sorted(w)
        q = []
        for cap in C:
            mass = F(0)
            count = 0
            for val in sorted_w:
                if t * (mass + val) > cap:
                    break
                mass += val
                count += 1
            q.append(count)
        for i in range(K):
            eligible[i] = [m for m in range(M) if C[m] >= w[i] * t]
            degree[i] = min(n["Dmax"], len(eligible[i]), 1 + int((sum(C) / t - sum(w)) // w[i]))
            if mode == "C":
                degree[i] = min(degree[i], sum(q) - K + 1)
        if mode == "C" and min(degree) < 1:
            reason = "coherent_capacity_coverage"
    bb = [[rf(v) ** 2 for v in row] for row in n["b"]]
    T = []
    if reason:
        return dict(
            excluded=True, reason=reason, steps=0, eta=eta, eligible=eligible, degree=degree
        )

    def ratio(k, i, ids, limit, coherent):
        if limit < 1:
            return F(0)
        ratios = []
        for m in ids:
            zeta = rf(n["zeta"][m, k, i])
            assert zeta >= 0
            if not zeta:
                if bb[m][i]:
                    return None
                continue
            ratios.append(bb[m][i] / zeta)
        return sum(sorted(ratios, reverse=True)[:limit]) if coherent else max(ratios, default=F(0))

    for k in range(K):
        row = []
        for i in range(K):
            H = ratio(k, i, eligible[i], degree[i], mode != "N")
            if mode == "J":
                N = ratio(k, i, range(M), 1, False)
                H = None if H is None or N is None else max(H, N)
            if H == 0:
                return dict(
                    excluded=True,
                    reason="unservable_user",
                    steps=0,
                    eta=eta,
                    eligible=eligible,
                    degree=degree,
                )
            row.append(0 if H is None else int(eta[i] / H * GRID))
        T.append(row)
    ell = [GRID] * K
    upper = [
        1 + sum((P[m] * max((rf(n["zeta"][m, k, i]) for i in range(K))) for m in range(M)))
        for k in range(K)
    ]
    for r in range(steps):
        ell = [GRID + sum((T[k][i] * ell[i] for i in range(K))) // GRID for k in range(K)]
        if any((F(ell[k], GRID) > upper[k] for k in range(K))):
            return dict(
                excluded=True,
                reason="interference_floor_above_power_upper",
                steps=r + 1,
                eta=eta,
                ell=ell,
                T=T,
                eligible=eligible,
                degree=degree,
            )
    return dict(
        excluded=False, steps=steps, eta=eta, ell=ell, T=T, eligible=eligible, degree=degree
    )


def support_maximum(n, t, weights, m):
    t = rf(t)
    C = rf(n["C"][m])
    w = list(map(rf, n["w"]))
    values = [weights[k] ** 2 * rf(n["P"][m]) * rf(n["b"][m, k]) ** 2 for k in range(n["K"])]
    if len(set(w)) == 1:
        q = min(n["K"], int(C // (t * w[0])))
        return sum(sorted(values, reverse=True)[:q])
    best = F(0)
    for mask in range(1 << n["K"]):
        ids = [k for k in range(n["K"]) if mask >> k & 1]
        if t * sum((w[k] for k in ids)) <= C:
            best = max(best, sum((values[k] for k in ids)))
    return best


def coherent_certificate(n, t, weights=None, steps=16):
    lower = floors(n, t, "C", steps)
    record = dict(
        kind="heterogeneous_C_support",
        target_hex=float(t).hex(),
        model_sha256=model_hash(n),
        steps=lower["steps"],
        grid=GRID,
        eta_lower=[pair(v) for v in lower["eta"]],
    )
    if lower["excluded"]:
        record.update(excluded=True, reason=lower["reason"])
        return record
    if weights is None:
        weights = np.ones(n["K"])
    weights = list(map(rf, weights))
    assert all((v >= 0 for v in weights)) and sum(weights) > 0
    left = sum(
        (weights[k] * sqrt_bound(lower["eta"][k] * F(lower["ell"][k], GRID)) for k in range(n["K"]))
    )
    right = sum((sqrt_bound(support_maximum(n, t, weights, m), True) for m in range(n["M"])))
    record.update(
        excluded=left > right,
        reason="weighted_support_inequality" if left > right else "UNKNOWN",
        lambda_hex=[float(v).hex() for v in weights],
        left_lower=pair(left),
        right_upper=pair(right),
        margin_lower=float(left - right),
    )
    return record


def guide_weights(n, t, steps=16):
    """Convex support-function search; floating outcome alone never proves exclusion."""
    from scipy.optimize import minimize

    lower = floors(n, t, "C", steps)
    if lower["excluded"]:
        return None
    req = np.sqrt(
        np.array([float(lower["eta"][k] * F(lower["ell"][k], GRID)) for k in range(n["K"])])
    )
    a = n["a"] / req[None, :]
    q = np.minimum(n["K"], np.floor(n["C"] / t + 1e-12).astype(int))
    assert np.all(n["w"] == 1), "Guide currently unit-weight;certificate supports general weights"

    def fun(y):
        val = 0.0
        grad = np.zeros_like(y)
        for m in range(n["M"]):
            if q[m] <= 0:
                continue
            active = np.argsort(-((a[m] * y) ** 2))[: q[m]]
            v = np.linalg.norm(a[m, active] * y[active])
            val += v
            if v > 1e-30:
                grad[active] += a[m, active] ** 2 * y[active] / v
        return (val, grad)

    candidates = [(fun(np.ones(n["K"]) / n["K"])[0], 1 / (n["K"] * req))]
    for start in [np.ones(n["K"]) / n["K"], req / req.sum()]:
        r = minimize(
            fun,
            start,
            jac=True,
            bounds=[(0, 1)] * n["K"],
            constraints=[
                {"type": "eq", "fun": lambda y: y.sum() - 1, "jac": lambda y: np.ones_like(y)}
            ],
            method="SLSQP",
            options={"maxiter": 180, "ftol": 1e-10},
        )
        if not np.all(np.isfinite(r.x)):
            continue
        y = np.maximum(0, r.x)
        if y.sum() <= 0:
            continue
        y /= y.sum()
        candidates.append((fun(y)[0], y / req))
    return min(candidates, key=lambda x: x[0])[1]


def noncoherent_cut_certificate(n, t, steps=16):
    lower = floors(n, t, "N", steps)
    record = dict(
        kind="heterogeneous_N_cut",
        target_hex=float(t).hex(),
        model_sha256=model_hash(n),
        steps=lower["steps"],
        grid=GRID,
        eta_lower=[pair(v) for v in lower["eta"]],
    )
    if lower["excluded"]:
        record.update(excluded=True, reason=lower["reason"])
        return record
    K, M = (n["K"], n["M"])
    Q = 2**44
    caps = [math.ceil(rf(v) * Q) for v in n["C"]]
    demand = [int(rf(t) * rf(v) * Q) for v in n["w"]]
    radii = np.array(
        [
            [
                math.ceil(
                    rf(
                        outward_rate(
                            rf(n["P"][m]) * rf(n["b"][m, k]) ** 2 / F(lower["ell"][k], GRID),
                            n["rho"],
                        )
                    )
                    * Q
                )
                for k in range(K)
            ]
            for m in range(M)
        ],
        dtype=np.int64,
    )
    sums = np.zeros((M, 1 << K), dtype=np.int64)
    needs = np.zeros(1 << K, dtype=np.int64)
    for mask in range(1, 1 << K):
        bit = mask & -mask
        k = bit.bit_length() - 1
        previous = mask ^ bit
        sums[:, mask] = sums[:, previous] + radii[:, k]
        needs[mask] = needs[previous] + demand[k]
    available = np.minimum(np.array(caps)[:, None], sums).sum(0)
    mask = int(np.argmax(needs - available))
    gap = int(needs[mask] - available[mask])
    record.update(
        excluded=gap > 0,
        reason="optimistic_flow_cut" if gap > 0 else "UNKNOWN",
        cut_mask=mask,
        flow_grid=Q,
        left_lower_int=int(needs[mask]),
        right_upper_int=int(available[mask]),
    )
    return record


def minimum_square_mass(caps, total):
    """Exact clipped equal-fill minimum;caps,total are nonnegative rationals."""
    caps = sorted(caps)
    assert 0 < total <= sum(caps)
    mass = F(0)
    squared = F(0)
    for j, cap in enumerate(caps):
        remaining = len(caps) - j
        level = (total - mass) / remaining
        if level <= cap:
            return squared + (total - mass) ** 2 / remaining
        mass += cap
        squared += cap * cap
    assert mass == total
    return squared


def joint_conversion_certificate(n, t, nc_certificate=None, steps=16, variant="full"):
    assert variant in ["full", "constant", "degree", "noiseCap"]
    lower = floors(n, t, "J", steps if variant == "full" else 0)
    record = dict(
        kind="joint_NC_conversion",
        variant=variant,
        target_hex=float(t).hex(),
        model_sha256=model_hash(n),
        steps=lower["steps"],
        grid=GRID,
        eta_lower=[pair(v) for v in lower["eta"]],
    )
    if lower["excluded"]:
        record.update(excluded=True, reason=lower["reason"])
        return record
    h = rf(t)
    kappas = []
    forced = []
    for k in range(n["K"]):
        caps = sorted(
            [sqrt_bound(rf(n["P"][m]), True) * rf(n["b"][m, k]) for m in lower["eligible"][k]],
            reverse=True,
        )[: max(0, lower["degree"][k])]
        Q = sqrt_bound(lower["eta"][k] * F(lower["ell"][k], GRID))
        impossible = not caps or Q > sum(caps) or (not sum(caps))
        if impossible or len(caps) == 1:
            kappa = F(1)
        elif Q == 0:
            kappa = F(len(caps))
        else:
            kappa = Q * Q / minimum_square_mass(caps, Q)
        if variant == "constant":
            kappa = F(n["Dmax"])
        elif variant == "degree":
            kappa = F(max(1, lower["degree"][k]))
        assert kappa >= 1
        kappas.append(pair(kappa))
        forced.append(impossible)
        service = (
            rf(t)
            if kappa == 1
            else rate_lower_fraction(lower["eta"][k] / kappa, n["rho"]) / rf(n["w"][k])
        )
        h = min(h, service)
    record.update(converted_service_lower=pair(h), kappa_upper=kappas, coherent_impossible=forced)
    if nc_certificate is not None:
        from .bounds import check_certificate

        assert np.all(n["w"] == 1) and check_certificate(n, nc_certificate)
        upper = rf(nc_certificate["radio_rate_upper"])
        record.update(
            excluded=h > upper,
            reason="NC_conversion_above_global_N_upper" if h > upper else "UNKNOWN",
            NC_global_certificate=nc_certificate,
        )
    if not record.get("excluded", False):
        candidate = float(np.nextafter(float(h), -np.inf))
        cut = noncoherent_cut_certificate(n, candidate, steps)
        record.update(
            excluded=cut["excluded"],
            reason="NC_conversion_to_excluded_flow_target" if cut["excluded"] else "UNKNOWN",
            NC_target_certificate=cut,
        )
    return record

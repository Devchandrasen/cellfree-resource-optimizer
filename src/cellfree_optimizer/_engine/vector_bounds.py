"""Vector resource bounds with rational validation helpers."""

from fractions import Fraction as F
import numpy as np
from .bounds import rate_lower_fraction, check_certificate
from .hetero_bounds import (
    floors,
    pair,
    rf,
    model_hash,
    GRID,
    sqrt_bound,
    minimum_square_mass,
    noncoherent_cut_certificate,
)


def linear_separator(n, parent):
    from scipy.optimize import linprog

    K, M = (n["K"], n["M"])
    gamma = [
        F(*map(int, e)) / F(*map(int, q))
        for e, q in zip(parent["eta_lower"], parent["kappa_upper"])
    ]
    y = np.array(list(map(float, gamma)))
    matrix = np.zeros((M * K, K + M))
    for m in range(M):
        for i in range(K):
            matrix[m * K + i, :K] = -n["P"][m] * y * n["zeta"][m, :, i]
            matrix[m * K + i, i] += n["P"][m] * n["b"][m, i] ** 2
            matrix[m * K + i, K + m] = -1
    objective = np.r_[-y, np.ones(M)]
    equality = np.r_[np.ones(K), np.zeros(M)][None, :]
    solved = linprog(
        objective,
        A_ub=matrix,
        b_ub=np.zeros(M * K),
        A_eq=equality,
        b_eq=np.ones(1),
        bounds=(0, None),
        method="highs",
        options={"threads": 1},
    )
    lam = np.maximum(0.0, solved.x[:K]) if solved.success else np.ones(K)
    if not np.all(np.isfinite(lam)) or lam.sum() <= 0:
        lam = np.ones(K)
    lam /= lam.sum()
    weights = list(map(rf, lam))
    left = sum((weights[k] * gamma[k] for k in range(K)))
    right = sum(
        (
            rf(n["P"][m])
            * max(
                F(0),
                max(
                    (
                        weights[i] * rf(n["b"][m, i]) ** 2
                        - sum((weights[k] * gamma[k] * rf(n["zeta"][m, k, i]) for k in range(K)))
                        for i in range(K)
                    )
                ),
            )
            for m in range(M)
        )
    )
    return dict(
        lambda_hex=[float(v).hex() for v in weights],
        left_lower=pair(left),
        right_upper=pair(right),
        excluded=left > right,
        guidance_status=int(solved.status),
    )


def geometry_record(n, t, variant="full"):
    assert variant in ["full", "degree"]
    q = floors(n, t, "J", 16 if variant == "full" else 0)
    r = dict(
        kind="joint_NC_conversion",
        variant=variant,
        target_hex=float(t).hex(),
        model_sha256=model_hash(n),
        steps=q["steps"],
        grid=GRID,
        eta_lower=[pair(v) for v in q["eta"]],
    )
    if q["excluded"]:
        return dict(r, excluded=True, reason=q["reason"])
    kappas, forced, service = ([], [], rf(t))
    for k in range(n["K"]):
        caps = sorted(
            [sqrt_bound(rf(n["P"][m]), True) * rf(n["b"][m, k]) for m in q["eligible"][k]],
            reverse=True,
        )[: max(0, q["degree"][k])]
        demand = sqrt_bound(q["eta"][k] * F(q["ell"][k], GRID))
        impossible = not caps or demand > sum(caps) or (not sum(caps))
        if impossible or len(caps) == 1:
            kappa = F(1)
        elif demand == 0:
            kappa = F(len(caps))
        else:
            kappa = demand * demand / minimum_square_mass(caps, demand)
        if variant == "degree":
            kappa = F(max(1, q["degree"][k]))
        assert kappa >= 1
        kappas.append(pair(kappa))
        forced.append(bool(impossible))
        guaranteed = (
            rf(t)
            if kappa == 1
            else rate_lower_fraction(q["eta"][k] / kappa, n["rho"]) / rf(n["w"][k])
        )
        service = min(service, guaranteed)
    return dict(
        r,
        excluded=False,
        reason="UNKNOWN",
        kappa_upper=kappas,
        coherent_impossible=forced,
        converted_service_lower=pair(service),
    )


def certificate(n, t, nc=None, variant="full"):
    parent = geometry_record(n, t, variant)
    r = dict(
        kind="vector_NC_conversion",
        variant=variant,
        target_hex=float(t).hex(),
        model_sha256=model_hash(n),
        geometry=parent,
        excluded=parent["excluded"],
        reason=parent["reason"],
    )
    if r["excluded"]:
        return r
    if nc is not None:
        assert np.all(n["w"] == 1) and check_certificate(n, nc)
        r["NC_global_certificate"] = nc
        if F(*map(int, parent["converted_service_lower"])) > rf(nc["radio_rate_upper"]):
            return dict(r, excluded=True, reason="scalar_service_above_N_radio_upper")
    separator = linear_separator(n, parent)
    r["linear_separator"] = separator
    if separator["excluded"]:
        return dict(r, excluded=True, reason="vector_linear_power_separator")
    weights = []
    for eta, kappa in zip(parent["eta_lower"], parent["kappa_upper"]):
        lower = rate_lower_fraction(F(*map(int, eta)) / F(*map(int, kappa)), n["rho"])
        value = float(np.nextafter(float(lower), -np.inf))
        assert 0 < rf(value) <= lower
        weights.append(value)
    converted = dict(n, w=np.asarray(weights))
    child = noncoherent_cut_certificate(converted, 1.0)
    return dict(
        r,
        weights_hex=[v.hex() for v in weights],
        vector_NC_certificate=child,
        excluded=child["excluded"],
        reason="vector_" + child["reason"],
    )

"""Directed-rounding helpers for conservative service bounds."""

import os
from fractions import Fraction as F
from decimal import Decimal, localcontext, ROUND_CEILING, ROUND_FLOOR
import warnings
import hashlib
import functools
import numpy as np
from scipy.optimize import linprog


def outward_rate(eta, rho):
    """Upper rho*log2(1+eta), via positive atanh series and a bounded tail."""
    z = eta / (eta + 2)
    with localcontext() as ctx:
        ctx.prec = 90
        ctx.rounding = ROUND_CEILING
        zz = Decimal(z.numerator) / Decimal(z.denominator)
        zz2 = zz * zz
        term = zz
        total = Decimal(0)
        for j in range(256):
            total += term / Decimal(2 * j + 1)
            term *= zz2
        with localcontext() as down:
            down.prec = 90
            down.rounding = ROUND_FLOOR
            denominator = Decimal(513) * (Decimal(1) - zz2)
        assert denominator > 0
        ln_upper = 2 * (total + term / denominator)
    with localcontext() as ctx:
        ctx.prec = 90
        ctx.rounding = ROUND_FLOOR
        zz = Decimal(1) / Decimal(3)
        zz2 = zz * zz
        term = zz
        total = Decimal(0)
        for j in range(256):
            total += term / Decimal(2 * j + 1)
            term *= zz2
        ln2_lower = 2 * total
    with localcontext() as ctx:
        ctx.prec = 90
        ctx.rounding = ROUND_CEILING
        rr = F.from_float(float(rho))
        rho_upper = Decimal(rr.numerator) / Decimal(rr.denominator)
        upper = rho_upper * ln_upper / ln2_lower
        return float(np.nextafter(float(upper), np.inf))


@functools.lru_cache(maxsize=1)
def ln2_upper():
    with localcontext() as ctx:
        ctx.prec = 90
        ctx.rounding = ROUND_CEILING
        z = Decimal(1) / Decimal(3)
        z2 = z * z
        term = z
        total = Decimal(0)
        for j in range(256):
            total += term / Decimal(2 * j + 1)
            term *= z2
        with localcontext() as down:
            down.prec = 90
            down.rounding = ROUND_FLOOR
            denominator = Decimal(513) * (Decimal(1) - z2)
        return 2 * (total + term / denominator)


def rate_lower_fraction(eta, rho):
    z = eta / (eta + 2)
    denominator = ln2_upper()
    with localcontext() as ctx:
        ctx.prec = 90
        ctx.rounding = ROUND_FLOOR
        zz = Decimal(z.numerator) / Decimal(z.denominator)
        zz2 = zz * zz
        term = zz
        total = Decimal(0)
        for j in range(256):
            total += term / Decimal(2 * j + 1)
            term *= zz2
        rr = F.from_float(float(rho))
        rho_lower = Decimal(rr.numerator) / Decimal(rr.denominator)
        return F(rho_lower * 2 * total / denominator)


def moment_hash(n):
    values = [np.asarray(n[k], dtype="<f8").tobytes() for k in ["b", "P", "zeta"]]
    values.append(np.asarray([n["rho"]], dtype="<f8").tobytes())
    return hashlib.sha256(b"".join(values)).hexdigest()


def check_achievable_lower(n, s, margin=2e-06):
    """Exact power/forwarding checks plus directed-rounded radio lower bounds.

    Uniformly shrink amplitudes by 1-2^-40 and recode each user at t-margin.
    This certificate treats supplied b/P/zeta/rho moments as real constants.
    Common AP power, as in this study, makes coherent signal power rational.
    """
    M, K = (n["M"], n["K"])
    assert np.all(n["w"] == 1)
    A = np.asarray(s["A"], bool)
    z = np.asarray(s["z"], bool).reshape(-1)
    assert np.all(A.sum(axis=0) >= 1) and np.all(A.sum(axis=0) <= n["Dmax"])
    alpha = 1 - F(1, 2**40)
    target = F.from_float(float(s["t"])) - F.from_float(float(margin))
    assert target > 0
    x = [[F.from_float(float(s["x"][m, k])) * alpha for k in range(K)] for m in range(M)]
    p = [[v * v for v in row] for row in x]
    for m in range(M):
        assert sum(p[m]) <= 1 and all((v >= 0 for v in x[m]))
        assert all((A[m, k] or x[m][k] == 0 for k in range(K)))
    power = [F.from_float(float(v)) for v in n["P"]]
    assert all((v == power[0] for v in power))
    bb = [[F.from_float(float(n["b"][m, k])) for k in range(K)] for m in range(M)]
    den = [
        1
        + sum(
            (
                power[m] * F.from_float(float(n["zeta"][m, k, i])) * p[m][i]
                for m in range(M)
                for i in range(K)
            )
        )
        for k in range(K)
    ]
    u = [[F(0) for k in range(K)] for m in range(M)]
    for k in range(K):
        if z[k]:
            signal = power[0] * sum((bb[m][k] * x[m][k] for m in range(M))) ** 2
            assert rate_lower_fraction(signal / den[k], n["rho"]) >= target
        else:
            old = [F.from_float(float(s["u"][m, k])) for m in range(M)]
            mass = sum(old)
            assert mass > 0 and all((v >= 0 for v in old))
            for m in range(M):
                u[m][k] = old[m] * target / mass
            tail = F(0)
            for m in n["order"][:, k][::-1]:
                m = int(m)
                signal = power[m] * bb[m][k] ** 2 * p[m][k]
                if A[m, k]:
                    assert u[m][k] <= rate_lower_fraction(signal / (den[k] + tail), n["rho"])
                else:
                    assert u[m][k] == 0
                tail += signal
            assert sum((u[m][k] for m in range(M))) == target
    for m in range(M):
        load = target * int((A[m] & z).sum()) + sum((u[m][k] for k in range(K)))
        assert load <= F.from_float(float(n["C"][m]))
    return dict(
        all_passed=True,
        lower_rate=float(np.nextafter(float(target), -np.inf)),
        target_numerator=str(target.numerator),
        target_denominator=str(target.denominator),
        amplitude_multiplier_numerator=str(alpha.numerator),
        amplitude_multiplier_denominator=str(alpha.denominator),
        radio_arithmetic="90-digit directed Decimal atanh series; exact Fraction power, interference, SIC, code flow and fronthaul",
    )


def numerical_weights(n):
    """LPs choose useful nonnegative weights; their statuses are NOT proofs."""
    M, K = (n["M"], n["K"])
    E = M * K
    a2 = n["a"] ** 2
    G = n["g"].transpose(1, 0, 2).reshape(K, E)
    S = np.zeros((K, E))
    for k in range(K):
        S[k, np.arange(M) * K + k] = a2[:, k]
    AP = np.zeros((M, E))
    for m in range(M):
        AP[m, m * K : (m + 1) * K] = 1
    own = n["g"][:, np.arange(K), np.arange(K)]
    ratio = np.divide(a2, own, out=np.full_like(a2, np.inf), where=own > 0)
    low = 0.0
    high = min(float(a2.sum(axis=0).min()), float(ratio.max(axis=0).min()))
    for _ in range(22):
        eta = (low + high) / 2
        R = eta * G - S
        scale = np.maximum(1, np.max(abs(R), axis=1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = linprog(
                np.zeros(E),
                A_ub=np.vstack([R / scale[:, None], AP]),
                b_ub=np.r_[-eta / scale, np.ones(M)],
                bounds=(0, None),
                method="highs",
                options={"threads": 1, "time_limit": 1.0},
            )
        if r.success:
            low = eta
        elif r.status == 2:
            high = eta
        else:
            break
        if high - low < 2e-06:
            break
    eta = max(1e-07, low * 0.999)
    R = eta * G - S
    scale = np.maximum(1, np.max(abs(R), axis=1))
    A = np.vstack([np.c_[R / scale[:, None], np.zeros(K)], np.c_[AP, -np.ones(M)]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = linprog(
            np.r_[np.zeros(E), 1.0],
            A_ub=A,
            b_ub=np.r_[-eta / scale, np.zeros(M)],
            bounds=(0, None),
            method="highs",
            options={"threads": 1, "time_limit": 2.0},
        )
    if r.success:
        lam = np.maximum(0, -np.asarray(r.ineqlin.marginals[:K]) / scale)
        if np.isfinite(lam).all() and lam.sum() > 0:
            return (lam, dict(guide_eta=low, dual_status=int(r.status)))
    return (
        np.ones(K),
        dict(guide_eta=low, dual_status=int(r.status), fallback="uniform nonnegative weights"),
    )


def rational_root(n, weights):
    M, K = (n["M"], n["K"])
    lam = [F.from_float(float(v)) for v in weights]
    assert all((v >= 0 for v in lam)) and sum(lam) > 0
    power = [F.from_float(float(v)) for v in n["P"]]
    aa = [[power[m] * F.from_float(float(n["b"][m, k])) ** 2 for k in range(K)] for m in range(M)]
    dd = [
        [
            power[m] * sum((lam[k] * F.from_float(float(n["zeta"][m, k, i])) for k in range(K)))
            for i in range(K)
        ]
        for m in range(M)
    ]
    intercept = [[lam[i] * aa[m][i] for i in range(K)] for m in range(M)]
    mass = sum(lam)

    def active(eta):
        chosen = []
        for m in range(M):
            values = [intercept[m][i] - eta * dd[m][i] for i in range(K)]
            i = max(range(K), key=values.__getitem__)
            chosen.append(i if values[i] > 0 else -1)
        return chosen

    def value(eta):
        return (
            sum(
                (
                    max([F(0)] + [intercept[m][i] - eta * dd[m][i] for i in range(K)])
                    for m in range(M)
                )
            )
            - eta * mass
        )

    hi = sum((max(v) for v in intercept)) / mass + 1
    lo = 0.0
    up = float(hi)
    a = np.asarray([[float(v) for v in row] for row in intercept])
    d = np.asarray([[float(v) for v in row] for row in dd])
    for _ in range(100):
        mid = (lo + up) / 2
        if np.maximum(0, (a - mid * d).max(axis=1)).sum() - mid * float(mass) > 0:
            lo = mid
        else:
            up = mid
    chosen = active(F.from_float(up))
    for iteration in range(30):
        top = sum((intercept[m][i] for m, i in enumerate(chosen) if i >= 0))
        bottom = mass + sum((dd[m][i] for m, i in enumerate(chosen) if i >= 0))
        eta = top / bottom
        check = value(eta)
        if check == 0:
            return (
                eta,
                dict(
                    exact_piecewise_root=True,
                    active_iterations=iteration + 1,
                    residual_num="0",
                    residual_den="1",
                ),
            )
        chosen = active(eta)
    low = F(0)
    upper = hi
    for _ in range(100):
        mid = (low + upper) / 2
        if value(mid) > 0:
            low = mid
        else:
            upper = mid
    residual = value(upper)
    assert residual <= 0
    return (
        upper,
        dict(
            exact_piecewise_root=False,
            active_iterations=30,
            residual_num=str(residual.numerator),
            residual_den=str(residual.denominator),
        ),
    )


def build_certificate(n):
    assert np.all(n["w"] == 1) and np.all(n["a"] >= 0) and np.all(n["g"] >= 0)
    lam, guide = numerical_weights(n)
    eta, proof = rational_root(n, lam)
    rate = outward_rate(eta, n["rho"])
    return dict(
        kind="weighted NC radio relaxation; exact rational max-affine inequality",
        weights_hex=[float(v).hex() for v in lam],
        eta_numerator=str(eta.numerator),
        eta_denominator=str(eta.denominator),
        radio_rate_upper=rate,
        rho_hex=float(n["rho"]).hex(),
        coefficient_sha256=moment_hash(n),
        guide=guide,
        proof=proof,
        scope="Pure NC radio, all links allowed; FH/Dmax removed. Supplied IEEE b/P/zeta/rho moments treated as real constants; a^2=P*b^2 and g=P*zeta evaluated exactly.",
    )


def check_certificate(n, certificate):
    assert moment_hash(n) == certificate["coefficient_sha256"]
    lam = [F.from_float(float.fromhex(v)) for v in certificate["weights_hex"]]
    eta = F(int(certificate["eta_numerator"]), int(certificate["eta_denominator"]))
    mass = sum(lam)
    assert all((v >= 0 for v in lam)) and mass > 0 and (eta >= 0)
    rhs = F(0)
    for m in range(n["M"]):
        values = [F(0)]
        for i in range(n["K"]):
            power = F.from_float(float(n["P"][m]))
            b = F.from_float(float(n["b"][m, i]))
            d = power * sum(
                (lam[k] * F.from_float(float(n["zeta"][m, k, i])) for k in range(n["K"]))
            )
            values.append(lam[i] * power * b * b - eta * d)
        rhs += max(values)
    assert rhs - eta * mass <= 0
    assert certificate["radio_rate_upper"] >= outward_rate(eta, n["rho"])
    return True


def mode_bounds(n, C, certificate):
    radio = certificate["radio_rate_upper"]
    total = float(np.nextafter(float(F(n["M"]) * F.from_float(float(C)) / n["K"]), np.inf))
    nc = min(radio, total)
    coherent = C / np.ceil(n["K"] / n["M"])
    if n["M"] == n["K"]:
        coherent = min(coherent, max(C / 2, nc))
    cooperation = min(
        C, float(np.nextafter(float(F(n["M"]) * F.from_float(float(C)) / (n["K"] + 1)), np.inf))
    )
    return dict(
        pure_C_upper=float(coherent),
        pure_N_upper=float(nc),
        NC_radio_upper=radio,
        coherent_cooperation_upper=float(cooperation),
        joint_upper=float(min(total, max(cooperation, radio))),
        equal_load_structural_bound=n["M"] == n["K"],
    )

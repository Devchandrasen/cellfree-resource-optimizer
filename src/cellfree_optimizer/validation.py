"""Independent non-repairing checks for exported solutions."""

import numpy as np


def validate_solution(network, solution, tolerance=2e-6):
    n = network.moments
    m, k = n["M"], n["K"]
    a = np.asarray(solution["A"])
    z = np.asarray(solution["z"]).reshape(-1)
    x = np.asarray(solution["x"], dtype=float)
    u = np.asarray(solution["u"], dtype=float)
    c = np.asarray(solution["rC"], dtype=float).reshape(-1)
    t = float(solution["t"])
    if (
        a.shape != (m, k)
        or z.shape != (k,)
        or x.shape != (m, k)
        or u.shape != (m, k)
        or c.shape != (k,)
    ):
        raise ValueError("Solution dimensions do not match the network")
    if not all(np.isfinite(v).all() for v in [a, z, x, u, c, t]):
        raise ValueError("Solution contains a nonfinite value")
    if not np.isin(a, [0, 1]).all() or not np.isin(z, [0, 1]).all():
        raise ValueError("Association and mode arrays must be binary")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    a, z = a.astype(bool), z.astype(bool)
    power = x**2
    denominator = 1 + np.einsum("mki,mi->k", n["g"], power)
    coherent_radio = np.where(
        z, n["rho"] * np.log2(1 + np.sum(n["a"] * x, axis=0) ** 2 / denominator), 0
    )
    noncoherent_radio = np.zeros((m, k))
    for user in range(k):
        if z[user]:
            continue
        tail = 0.0
        for ap in n["order"][:, user][::-1]:
            signal = n["a"][ap, user] ** 2 * power[ap, user]
            if a[ap, user]:
                noncoherent_radio[ap, user] = n["rho"] * np.log2(
                    1 + signal / (denominator[user] + tail)
                )
            tail += signal
    load = a[:, z] @ c[z] + u.sum(axis=1)
    delivered = c + u.sum(axis=0)
    residuals = dict(
        nonnegative=max(0.0, -t, float(-x.min()), float(-u.min()), float(-c.min())),
        ap_power=float(np.maximum(0, power.sum(axis=1) - 1).max()),
        support=float(np.abs(x[~a]).max(initial=0)),
        payload_support=float(np.abs(u[~a]).max(initial=0)),
        coherent_mode=float(np.abs(c[~z]).max(initial=0)),
        noncoherent_mode=float(np.abs(u[:, z]).max(initial=0)),
        coverage=float(np.maximum(0, 1 - a.sum(axis=0)).max()),
        association_limit=float(np.maximum(0, a.sum(axis=0) - n["Dmax"]).max()),
        fronthaul=float(np.maximum(0, (load - n["C"]) / np.maximum(1, n["C"])).max()),
        service=float(np.maximum(0, (n["w"] * t - delivered) / np.maximum(1, n["w"] * t)).max()),
        coherent_radio=float(
            np.maximum(0, (c - coherent_radio) / np.maximum(1, coherent_radio)).max()
        ),
        noncoherent_radio=float(
            np.maximum(0, (u - noncoherent_radio) / np.maximum(1, noncoherent_radio)).max()
        ),
    )
    maximum = max(residuals.values())
    return dict(
        feasible=maximum <= tolerance,
        max_residual=maximum,
        residuals=residuals,
        delivered_service=delivered.tolist(),
        fronthaul_load=load.tolist(),
        normalized_ap_power=power.sum(axis=1).tolist(),
    )

"""Correlated Rayleigh channels and finite-MMSE statistical MRT moments."""

from dataclasses import dataclass
import numpy as np
from numpy.polynomial.hermite import hermgauss

from .config import NetworkConfig


@dataclass(slots=True)
class Network:
    config: NetworkConfig
    moments: dict
    ap_positions: np.ndarray
    user_positions: np.ndarray


def generate_network(config=None):
    """Generate a periodic square deployment without running any optimizer.

    All gains are divided by receiver noise power. Power values remain in watts.
    Independent random streams keep fronthaul changes from changing the channels.
    """
    c = config or NetworkConfig()
    geometry_seed, shadow_seed, capacity_seed = np.random.SeedSequence(c.seed).spawn(3)
    rng = np.random.default_rng(geometry_seed)
    ap = rng.uniform(0, c.side_length, (c.aps, 2))
    ue = rng.uniform(0, c.side_length, (c.users, 2))
    delta = (
        ue[None, :, :] - ap[:, None, :] + c.side_length / 2
    ) % c.side_length - c.side_length / 2
    distance = np.sqrt(np.sum(delta**2, axis=2) + c.height**2)
    between = np.abs(ue[:, None, :] - ue[None, :, :])
    between = np.minimum(between, c.side_length - between)
    covariance = 2 ** (-np.linalg.norm(between, axis=2) / c.shadow_distance)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    shadow_root = (eigenvectors * np.sqrt(np.maximum(0, eigenvalues))) @ eigenvectors.T
    shadow = (
        c.shadow_std_db
        * (shadow_root @ np.random.default_rng(shadow_seed).standard_normal((c.users, c.aps))).T
    )
    gain_db = c.pathloss_intercept_db - c.pathloss_slope * np.log10(distance) + shadow
    noise = 1.380649e-23 * c.temperature_kelvin * c.bandwidth_hz * 10 ** (c.noise_figure_db / 10)
    beta = 10 ** (gain_db / 10) / noise
    nodes, weights = hermgauss(c.quadrature_nodes)
    weights = weights / np.sqrt(np.pi)
    b = np.empty((c.aps, c.users))
    zeta = np.empty((c.aps, c.users, c.users))
    q = c.pilot_length * c.pilot_power
    for m in range(c.aps):
        covariances = []
        for k in range(c.users):
            angle = np.arctan2(delta[m, k, 1], delta[m, k, 0])
            angles = angle + np.sqrt(2) * np.deg2rad(c.angular_spread_degrees) * nodes
            steering = np.exp(1j * np.pi * np.arange(c.antennas)[:, None] * np.sin(angles))
            r = beta[m, k] * (steering * weights) @ steering.conj().T
            covariances.append((r + r.conj().T) / 2)
        covariances = np.asarray(covariances)
        for i, r in enumerate(covariances):
            phi = q * r @ np.linalg.solve(np.eye(c.antennas) + q * r, r)
            trace = max(float(np.trace(phi).real), np.finfo(float).tiny)
            b[m, i] = np.sqrt(trace)
            zeta[m, :, i] = np.maximum(0, np.einsum("kab,ba->k", covariances, phi).real / trace)
    capacities = np.geomspace(1.0, c.fronthaul_ratio, c.aps)
    capacities *= c.fronthaul / capacities.mean()
    capacities = np.random.default_rng(capacity_seed).permutation(capacities)
    power = np.full(c.aps, c.ap_power)
    order = np.argsort(-(b**2), axis=0, kind="stable")
    rank = np.empty_like(order)
    for k in range(c.users):
        rank[order[:, k], k] = np.arange(c.aps)
    moments = dict(
        M=c.aps,
        K=c.users,
        N=c.antennas,
        Dmax=c.max_aps,
        P=power,
        C=capacities,
        w=np.ones(c.users),
        rho=1 - c.pilot_length / c.coherence_symbols,
        b=b,
        beta=beta,
        zeta=zeta,
        nu=np.zeros_like(zeta),
        a=b * np.sqrt(power)[:, None],
        g=zeta * power[:, None, None],
        order=order,
        rank=rank,
    )
    return Network(c, moments, ap, ue)

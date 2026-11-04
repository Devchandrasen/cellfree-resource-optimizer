from dataclasses import replace
import numpy as np
import pytest
from cellfree_optimizer import NetworkConfig, generate_network
from cellfree_optimizer.io import save_network, load_network


@pytest.mark.parametrize(
    "change",
    [
        {"aps": 0},
        {"users": -1},
        {"antennas": True},
        {"max_aps": 15},
        {"pilot_length": 9},
        {"coherence_symbols": 14},
        {"fronthaul": 0},
        {"fronthaul": float("nan")},
        {"ap_power": -1},
        {"seed": -1},
        {"fronthaul_ratio": 0.5},
        {"quadrature_nodes": 0},
        {"angular_spread_degrees": -1},
    ],
)
def test_invalid_configuration(change):
    with pytest.raises(ValueError):
        NetworkConfig(**change)


def test_reproducible_channels_and_separate_capacity_stream():
    cfg = NetworkConfig(aps=4, antennas=3, users=3, max_aps=2, pilot_length=3)
    first, second = generate_network(cfg), generate_network(cfg)
    changed = generate_network(replace(cfg, fronthaul=2))
    for key in ["beta", "b", "zeta", "C", "order", "rank"]:
        np.testing.assert_array_equal(first.moments[key], second.moments[key])
    np.testing.assert_array_equal(first.moments["beta"], changed.moments["beta"])
    np.testing.assert_array_equal(first.moments["b"], changed.moments["b"])
    np.testing.assert_allclose(first.moments["C"], 2 * changed.moments["C"])
    assert first.moments["zeta"].shape == (4, 3, 3)
    assert np.isfinite(first.moments["zeta"]).all()
    assert np.all(first.moments["b"] > 0)
    assert np.mean(first.moments["C"]) == pytest.approx(cfg.fronthaul)


def test_scalar_mmse_moments_match_closed_form():
    cfg = NetworkConfig(aps=3, antennas=1, users=2, max_aps=2, pilot_length=2, seed=81)
    n = generate_network(cfg).moments
    q = cfg.pilot_length * cfg.pilot_power
    np.testing.assert_allclose(n["b"] ** 2, q * n["beta"] ** 2 / (1 + q * n["beta"]), rtol=1e-12)
    np.testing.assert_allclose(
        n["zeta"], np.repeat(n["beta"][:, :, None], cfg.users, axis=2), rtol=1e-12
    )


def test_network_roundtrip_without_pickle(tmp_path):
    net = generate_network(NetworkConfig(aps=3, antennas=2, users=2, max_aps=2))
    save_network(net, tmp_path / "network.npz")
    loaded = load_network(tmp_path / "network.npz")
    assert loaded.config == net.config
    for key in net.moments:
        np.testing.assert_array_equal(loaded.moments[key], net.moments[key])

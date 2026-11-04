import numpy as np
import pytest
from cellfree_optimizer import NetworkConfig, generate_network, optimize
from cellfree_optimizer.optimizer import METHODS


@pytest.mark.parametrize("method", METHODS)
def test_methods_obey_physics_and_acceptance_deadline(method):
    n = generate_network(
        NetworkConfig(aps=4, users=3, antennas=2, max_aps=2, pilot_length=3, seed=19)
    )
    result = optimize(n, method=method, seconds=2, seed=3)
    assert result.validation["feasible"]
    assert result.service > 0
    assert result.metadata["accepted_point_seconds"] <= 2
    assert result.metadata["independent_cold_start"]
    rates = [p["rate"] for p in result.metadata["trace"]]
    assert all(b >= a for a, b in zip(rates, rates[1:]))
    z = result.solution["z"]
    if method in ["strong-cjt", "mode-informed-cjt"]:
        assert z.all()
    if method == "pure-ncjt":
        assert not z.any()
    if method == "adaptive-pure":
        assert z.all() or not z.any()
    if method == "fixed-hybrid":
        expected = np.zeros_like(result.solution["A"])
        order = np.argsort(-n.moments["beta"], axis=0, kind="stable")
        for k in range(n.config.users):
            expected[order[: n.config.max_aps, k], k] = True
        np.testing.assert_array_equal(expected, result.solution["A"])


@pytest.mark.parametrize(
    "kwargs", [{"method": "unknown"}, {"seconds": 0}, {"seconds": float("inf")}, {"seed": -1}]
)
def test_invalid_runtime_options(kwargs):
    n = generate_network(NetworkConfig(aps=2, users=1, antennas=1, max_aps=1))
    with pytest.raises(ValueError):
        optimize(n, **kwargs)


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_seed_graphs_respect_small_association_limits(limit):
    from cellfree_optimizer._engine.search import seed_graphs

    n = generate_network(NetworkConfig(aps=4, users=3, antennas=2, max_aps=limit))
    for graph in seed_graphs(n.moments, np.random.default_rng(4)):
        assert np.all(graph.sum(axis=0) >= 1)
        assert np.all(graph.sum(axis=0) <= limit)


@pytest.mark.parametrize("method", ["joint", "pure-ncjt"])
def test_single_ap_single_user_endpoint(method):
    n = generate_network(NetworkConfig(aps=1, users=1, antennas=1, max_aps=1, pilot_length=1))
    result = optimize(n, method, seconds=1)
    assert result.validation["feasible"]
    assert result.service > 0


def test_more_users_than_access_points():
    n = generate_network(NetworkConfig(aps=3, users=5, antennas=2, max_aps=2, pilot_length=5))
    result = optimize(n, "strong-cjt", seconds=2)
    assert result.validation["feasible"]
    assert result.service > 0

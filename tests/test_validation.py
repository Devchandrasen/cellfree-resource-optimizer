from copy import deepcopy
import numpy as np
import pytest
from cellfree_optimizer import Network, NetworkConfig, validate_solution


def fixture_point():
    config = NetworkConfig(aps=2, users=1, antennas=1, max_aps=2, pilot_length=1)
    n = dict(
        M=2,
        K=1,
        N=1,
        Dmax=2,
        a=np.array([[2.0], [1.0]]),
        g=np.array([[[0.4]], [[0.2]]]),
        C=np.array([4.0, 4.0]),
        w=np.ones(1),
        rho=0.9,
        order=np.array([[0], [1]]),
    )
    net = Network(config, n, np.zeros((2, 2)), np.zeros((1, 2)))
    s = dict(
        A=np.ones((2, 1), bool),
        z=np.ones(1, bool),
        x=np.array([[0.3], [0.4]]),
        t=0.2,
        rC=np.array([0.2]),
        u=np.zeros((2, 1)),
    )
    return net, s


def test_coherent_rate_and_load():
    net, s = fixture_point()
    den = 1 + 0.4 * 0.3**2 + 0.2 * 0.4**2
    rate = 0.9 * np.log2(1 + (2 * 0.3 + 0.4) ** 2 / den)
    s["t"] = rate
    s["rC"][:] = rate
    v = validate_solution(net, s)
    assert v["feasible"]
    np.testing.assert_allclose(v["fronthaul_load"], [rate, rate])
    assert v["delivered_service"] == pytest.approx([rate])


def test_noncoherent_sic_telescopes_and_forwards_only_own_stream():
    net, s = fixture_point()
    s["z"][:] = False
    den = 1 + 0.4 * 0.3**2 + 0.2 * 0.4**2
    rn = 0.9 * np.log2([1 + (2 * 0.3) ** 2 / (den + 0.4**2), 1 + 0.4**2 / den])
    s["u"][:, 0] = rn
    s["rC"][:] = 0
    s["t"] = float(rn.sum())
    assert rn.sum() == pytest.approx(0.9 * np.log2(1 + ((2 * 0.3) ** 2 + 0.4**2) / den))
    v = validate_solution(net, s)
    assert v["feasible"]
    np.testing.assert_allclose(v["fronthaul_load"], rn)


@pytest.mark.parametrize(
    "corruption",
    [
        "negative_power",
        "wrong_mode",
        "excess_power",
        "unserved_user",
        "missing_payload",
        "excess_fronthaul",
        "negative_rate",
    ],
)
def test_validator_rejects_corrupt_allocations_without_repair(corruption):
    net, s = fixture_point()
    if corruption == "negative_power":
        s["x"][0, 0] = -0.3
    if corruption == "wrong_mode":
        s["u"][0, 0] = 0.1
    if corruption == "excess_power":
        s["x"][0, 0] = 2
    if corruption == "unserved_user":
        s["A"][:] = False
    if corruption == "missing_payload":
        s["rC"][:] = 0
    if corruption == "excess_fronthaul":
        net.moments["C"][:] = 0.01
    if corruption == "negative_rate":
        s["t"] = -0.1
    before = deepcopy(s)
    assert not validate_solution(net, s)["feasible"]
    for key in s:
        np.testing.assert_array_equal(s[key], before[key])


@pytest.mark.parametrize("field,value", [("A", 0.5), ("x", float("nan"))])
def test_malformed_inputs_raise(field, value):
    net, s = fixture_point()
    s[field] = np.full((2, 1), value)
    with pytest.raises(ValueError):
        validate_solution(net, s)

import numpy as np
from cellfree_optimizer import NetworkConfig, generate_network
from cellfree_optimizer._engine import budget_core


def test_nested_budget_is_capped_by_outer_clock_and_late_points_rejected(monkeypatch):
    n = generate_network(NetworkConfig(aps=2, users=1, antennas=1, max_aps=1)).moments
    n["fixedA"] = np.array([[True], [False]])
    with budget_core.acceptance_session(1) as clock:
        outer = budget_core.Budget(n, "joint", 10)
        nested = budget_core.Budget(n, "joint", 10)
        assert outer.deadline <= clock["deadline"]
        assert nested.deadline <= clock["deadline"]
        assert clock["engine_start"] == outer.start
        monkeypatch.setattr(budget_core.time, "perf_counter", lambda: clock["deadline"] + 0.1)
        assert not outer.accept(outer.best, "late candidate")
        assert outer.late == 1
    assert budget_core._SESSION.get() is None

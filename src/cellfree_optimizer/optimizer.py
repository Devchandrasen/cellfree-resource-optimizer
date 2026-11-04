"""Public dispatch and a shared runtime contract for all methods."""

from dataclasses import dataclass
import time
import numpy as np
from threadpoolctl import threadpool_limits

from .validation import validate_solution

METHODS = (
    "joint",
    "strong-cjt",
    "mode-informed-cjt",
    "pure-ncjt",
    "adaptive-pure",
    "fixed-hybrid",
    "swap-switch",
)


@dataclass(slots=True)
class OptimizationResult:
    method: str
    solution: dict
    metadata: dict
    validation: dict

    @property
    def service(self):
        return float(self.solution["t"])


def optimize(network, method="joint", seconds=60.0, seed=0):
    """Run one cold start. Imports and channel generation precede the budget.

    Association templates, random starts, optimization and candidate checks are
    charged to the deadline. Late candidates are rejected. Native calls can return
    after the deadline, so elapsed wall time can exceed the acceptance budget.
    """
    from ._engine import adaptive, joint, pure_nc, swap_switch
    from ._engine.budget_core import acceptance_session
    from ._engine.capacity import check_parameters

    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}. Choose from {', '.join(METHODS)}")
    seconds = float(seconds)
    if not np.isfinite(seconds) or seconds < 1:
        raise ValueError("seconds must be finite and at least 1")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    n = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in network.moments.items()
    }
    check_parameters(n)
    if not np.array_equal(n["w"], np.ones(n["K"])):
        raise ValueError("The packaged association controller supports unit service weights")
    with threadpool_limits(limits=1), acceptance_session(seconds) as clock:
        start = clock["start"]
        cpu_start = time.process_time()
        rank = np.argsort(-n["beta"], axis=0, kind="stable")
        n["fixedA"] = np.zeros((n["M"], n["K"]), dtype=bool)
        for user in range(n["K"]):
            n["fixedA"][rank[: n["Dmax"], user], user] = True
        prep = time.perf_counter() - start
        remaining = max(0.0, seconds - prep)
        if method in ("joint", "strong-cjt", "mode-informed-cjt"):
            policy = {"joint": "joint", "strong-cjt": "backboneC", "mode-informed-cjt": "modeOff"}[
                method
            ]
            solution, meta = joint.run(n, policy, remaining, int(seed))
        elif method == "pure-ncjt":
            solution, meta = pure_nc.run(n, remaining, int(seed))
        elif method == "adaptive-pure":
            solution, meta = adaptive.run(n, "backboneC", remaining, int(seed))
        else:
            solution, meta = swap_switch.run(
                n, "fixed" if method == "fixed-hybrid" else "ap", remaining, int(seed)
            )
        elapsed = time.perf_counter() - start
        cpu_elapsed = time.process_time() - cpu_start
        engine_offset = (clock["engine_start"] or start) - start
    meta.pop("coherent_point", None)
    meta["policy"] = method
    meta["engine_budget_seconds"] = meta["budget_seconds"]
    meta["budget_seconds"] = seconds
    meta["association_template_seconds"] = prep
    meta["wall_seconds"] = elapsed
    meta["cpu_seconds"] = cpu_elapsed
    meta["accepted_point_seconds"] += engine_offset
    for field in ("trace", "events"):
        for entry in meta.get(field, []):
            entry["seconds"] += engine_offset
    check = validate_solution(network, solution)
    if not check["feasible"]:
        raise RuntimeError(f"Invalid optimizer output: {check['residuals']}")
    if meta["accepted_point_seconds"] > seconds + 1e-9:
        raise RuntimeError("The returned incumbent was accepted after its budget")
    if method in ("strong-cjt", "mode-informed-cjt") and not np.all(solution["z"]):
        raise RuntimeError("A pure coherent method returned a noncoherent mode")
    if method == "pure-ncjt" and np.any(solution["z"]):
        raise RuntimeError("The pure noncoherent method returned a coherent mode")
    if method == "adaptive-pure" and not (np.all(solution["z"]) or not np.any(solution["z"])):
        raise RuntimeError("The adaptive pure-mode method returned mixed modes")
    if method == "fixed-hybrid" and not np.array_equal(solution["A"], n["fixedA"]):
        raise RuntimeError("The fixed-association method changed its serving graph")
    return OptimizationResult(method, solution, meta, check)

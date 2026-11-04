"""Root guidance."""

import time
import itertools
import numpy as np
from .lifted_model import Model
from .budget_core import clarabel
from .capacity import coherent_upper, round_slots
from .assignment import round_graph


def guide(n, b, rng, stats, end, polish, relax=None):

    def settings():
        st = clarabel.DefaultSettings()
        st.verbose = False
        st.max_threads = 1
        st.direct_solve_method = "qdldl"
        st.max_iter = 90
        st.tol_feas = 1e-08
        st.time_limit = max(0.001, b.left(end) - 0.005)
        return st

    hi = coherent_upper(n)
    last = None
    if relax is not None:
        D = 1 + np.einsum("mki,mi->k", n["g"], relax * relax)
        hint = float(
            np.min(n["rho"] / np.log(2) * np.log1p((n["a"] * relax).sum(axis=0) ** 2 / D) / n["w"])
        )
        hi = min(hi, max(hint, b.best["t"] + 0.02))
    for trial in range(2):
        if b.left(end) < 0.3:
            break
        target = (
            b.best["t"] + 0.0004
            if trial == 0
            else max(b.best["t"] + 0.0004, (b.best["t"] + hi) / 2)
        )
        m = Model(n, target, power=True, sparse=False, label=True)
        b.check(end)
        st = settings()
        m.solver.update(settings=st)
        base = m.b.copy()
        domains = [{}]
        for turn in range(9):
            if not domains or b.left(end) < 0.08:
                break
            fixed = domains.pop(0)
            rhs = base.copy()
            for j, v in fixed.items():
                lo, up = m.box[j]
                rhs[lo] = -v
                rhs[up] = v
            st = settings()
            m.solver.update(b=rhs, settings=st)
            result = m.solver.solve()
            b.calls += 1
            stats["cold_root_solves"] += 1
            if str(result.status) not in ["Solved", "AlmostSolved"] or result.x is None:
                if not fixed:
                    hi = target
                continue
            values = np.asarray(result.x)
            x = np.maximum(0, values[m.vmap["x"]])
            y = np.clip(values[m.vmap["y"]], 0, 1)
            if not np.isfinite(x).all():
                continue
            last = x
            score = n["a"] * x
            score /= np.maximum(score.sum(axis=0), 1e-15)[None, :]
            for t in [target, b.best["t"] + 0.0004]:
                polish(round_graph(n, t, x), dict(x=x))
                polish(
                    round_slots(n, score * (0.3 + y), rng, power=1.0, noise=0.0, target=t),
                    dict(x=x),
                )
            if turn == 0:
                uncertainty = (y * (1 - y)).sum(axis=1)
                candidates = [mm for mm in range(n["M"]) if 0 < m.q[mm] < n["K"]]
                for mm in sorted(candidates, key=lambda mm: uncertainty[mm], reverse=True)[:2]:
                    ranks = np.argsort(-(y[mm] + 0.2 * score[mm]))
                    q = m.q[mm]
                    base_set = set(map(int, ranks[:q]))
                    sets = [base_set]
                    for old in ranks[max(0, q - 2) : q]:
                        for new in ranks[q : min(n["K"], q + 2)]:
                            sets.append(base_set - {int(old)} | {int(new)})
                    for members in sets[:4]:
                        domains.append(
                            {int(m.vmap["y"][mm, k]): int(k in members) for k in range(n["K"])}
                        )
    return last

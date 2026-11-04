"""Model."""

import clarabel
import numpy as np
from scipy import sparse
from .arith import F, f, root, scale, GRID, lower_int, upper_int, pair, unpack, tangent
from .hetero_bounds import eta_down


class Model:
    def __init__(self, n, t, kind="C"):
        assert kind == "C"
        self.n = n
        self.t = float(t)
        self.kind = kind
        M, K = (n["M"], n["K"])
        J = M * K
        self.vmap = {
            key: np.arange(j * J, (j + 1) * J).reshape(M, K)
            for j, key in enumerate(["x", "y", "p"])
        }
        self.size = 3 * J
        self.rows = []
        self.blocks = []
        self.spec = []
        self.box = []
        for j in range(self.size):
            self.box.append((len(self.rows), len(self.rows) + 1))
            self.lin({j: -1}, 0, ("boxlo", j))
            self.lin({j: 1}, 1, ("boxhi", j))
        x, y, p = [self.vmap[k] for k in ["x", "y", "p"]]
        eta = [eta_down(f(t) * f(w), n["rho"]) for w in n["w"]]
        self.etas = eta
        q = []
        for m in range(M):
            self.lin({int(j): 1 for j in p[m]}, 1, ("APpower", m))
            if np.all(n["w"] == n["w"][0]):
                cap = min(K, int(f(n["C"][m]) // (f(t) * f(n["w"][0]))))
                q.append(cap)
                self.lin({int(j): 1 for j in y[m]}, cap, ("Cslots", m))
            else:
                q.append(K)
                self.lin(
                    {int(y[m, k]): f(t) * f(n["w"][k]) for k in range(K)},
                    f(n["C"][m]),
                    ("Ctraffic", m),
                )
            for k in range(K):
                self.lin({int(p[m, k]): 1, int(y[m, k]): -1}, 0, ("power_support", m, k))
                if f(n["C"][m]) < f(t) * f(n["w"][k]):
                    self.lin({int(y[m, k]): 1}, 0, ("eligibility", m, k))
        for k in range(K):
            self.lin({int(j): -1 for j in y[:, k]}, -1, ("coverage", k))
            self.lin({int(j): 1 for j in y[:, k]}, n["Dmax"], ("degree", k))
        self.linear_count = len(self.rows)
        self.blocks.append(("nonnegative", self.linear_count))
        for m in range(M):
            for k in range(K):
                self.soc(
                    [
                        {int(p[m, k]): 1, int(y[m, k]): 1},
                        {int(x[m, k]): 2},
                        {int(p[m, k]): 1, int(y[m, k]): -1},
                    ],
                    [0, 0, 0],
                    ("perspective", m, k),
                )
        for k in range(K):
            total = F(1) + sum(
                (f(n["P"][m]) * f(n["zeta"][m, k, i]) for m in range(M) for i in range(K))
            )
            s = 1
            while s * s < total:
                s *= 2
            c = [root(f(n["P"][m]) * f(n["b"][m, k]) ** 2 / eta[k] / s**2, True) for m in range(M)]
            maps = [{int(x[m, k]): c[m] for m in range(M)}, {}]
            constants = [F(0), F(1, s)]
            for m in range(M):
                if q[m] == 1:
                    maps.append(
                        {
                            int(x[m, i]): root(f(n["P"][m]) * f(n["zeta"][m, k, i]) / s**2)
                            for i in range(K)
                        }
                    )
                    constants.append(F(0))
                else:
                    for i in range(K):
                        maps.append(
                            {int(x[m, i]): root(f(n["P"][m]) * f(n["zeta"][m, k, i]) / s**2)}
                        )
                        constants.append(F(0))
            self.soc(maps, constants, ("Cradio", k))
        self.compile()

    def lin(self, coef, b, tag):
        coef = {int(k): F(v) for k, v in coef.items() if v}
        b = F(b)
        s = scale(list(coef.values()) + [b])
        rr = {k: lower_int(v / s) for k, v in coef.items()}
        bi = upper_int(b / s)
        self.spec.append(dict(type="linear", tag=list(tag), start=len(self.rows), length=1))
        self.rows.append((rr, bi))

    def soc(self, maps, constants, tag):
        start = len(self.rows)
        for coefficients, b in zip(maps, constants):
            row = {int(j): -int(F(v) * GRID) for j, v in coefficients.items() if v}
            assert all((F(v) * GRID == int(F(v) * GRID) for v in coefficients.values()))
            bb = F(b) * GRID
            assert bb == int(bb)
            self.rows.append((row, int(bb)))
        self.blocks.append(("soc", len(maps)))
        self.spec.append(dict(type="soc", tag=list(tag), start=start, length=len(maps)))

    def compile(self):
        ii = []
        jj = []
        vv = []
        for i, (r, b) in enumerate(self.rows):
            for j, v in r.items():
                ii.append(i)
                jj.append(j)
                vv.append(v / GRID)
        self.A = sparse.csc_matrix((vv, (ii, jj)), shape=(len(self.rows), self.size))
        self.b = np.array([b / GRID for _, b in self.rows])
        self.cones = [
            clarabel.NonnegativeConeT(length)
            if kind == "nonnegative"
            else clarabel.SecondOrderConeT(length)
            for kind, length in self.blocks
        ]
        settings = clarabel.DefaultSettings()
        settings.max_threads = 1
        settings.direct_solve_method = "qdldl"
        settings.verbose = False
        settings.max_iter = 150
        settings.tol_feas = 1e-09
        settings.tol_gap_abs = 1e-09
        settings.tol_gap_rel = 1e-09
        self.solver = clarabel.DefaultSolver(
            sparse.csc_matrix((self.size, self.size)),
            np.zeros(self.size),
            self.A,
            self.b,
            self.cones,
            settings,
        )
        assert (
            self.solver.get_info().linsolver.threads == 1
            and self.solver.get_info().linsolver.name == "qdldl"
        )

    def solve(self, fixed=None):
        fixed = fixed or {}
        bb = self.b.copy()
        for j, val in fixed.items():
            lo, hi = self.box[int(j)]
            bb[lo] = -val
            bb[hi] = val
        self.solver.update(b=bb)
        sol = self.solver.solve()
        return (sol, bb)

    def metadata(self):
        from .hetero_bounds import model_hash

        out = dict(
            kind=self.kind,
            model_sha256=model_hash(self.n),
            target_hex=self.t.hex(),
            grid=GRID,
            eta_lower=[pair(v) for v in self.etas],
            primitives=self.spec,
            size=self.size,
        )
        if self.kind == "J":
            out["geometry"] = self.geometry
        out["linear_solver"] = dict(
            name=self.solver.get_info().linsolver.name,
            threads=self.solver.get_info().linsolver.threads,
        )
        return out

    def witness(self, sol, fixed=None):
        import math

        raw = np.array(sol.z, float)
        H = 2**40
        if not np.all(np.isfinite(raw)) or max(abs(raw), default=0) == 0:
            return None
        raw /= max(abs(raw))
        z = [int(round(v * H)) for v in raw]
        pos = 0
        for kind, length in self.blocks:
            if kind == "nonnegative":
                z[pos : pos + length] = [max(0, a) for a in z[pos : pos + length]]
            else:
                z[pos] = max(
                    0, z[pos], math.isqrt(sum((a * a for a in z[pos + 1 : pos + length]))) + 1
                )
            pos += length
        result = self.replay(z, fixed or {})
        if result <= 0:
            return None
        import base64
        import zlib

        return dict(
            z_b85=base64.b85encode(zlib.compress(np.array(z, dtype="<i8").tobytes(), 6)).decode(),
            length=len(z),
            margin=str(result),
        )

    def replay(self, z, fixed):
        r = [0] * self.size
        bdot = 0
        overrides = {}
        for j, val in fixed.items():
            lo, hi = self.box[int(j)]
            overrides[lo] = -int(val) * GRID
            overrides[hi] = int(val) * GRID
        for i, ((coef, b), zz) in enumerate(zip(self.rows, z)):
            if not zz:
                continue
            bdot += overrides.get(i, b) * zz
            for j, v in coef.items():
                r[j] += v * zz
        return sum((min(0, v) for v in r)) - bdot


class Mixed(Model):
    def __init__(self, n, t, radio=False):
        from .vector_bounds import geometry_record
        from .hetero_bounds import floors

        self.n = n
        self.t = float(t)
        self.kind = "J"
        M, K = (n["M"], n["K"])
        J = M * K
        self.vmap = {
            key: np.arange(j * J, (j + 1) * J).reshape(M, K)
            for j, key in enumerate(["x", "p", "yc", "yn", "u"])
        }
        self.vmap["z"] = np.arange(5 * J, 5 * J + K)
        self.size = 5 * J + K
        self.rows = []
        self.blocks = []
        self.spec = []
        self.box = []
        self.geometry = geometry_record(n, t)
        self.floor = floors(n, t, "J")
        assert not self.geometry["excluded"], (
            "Handle geometric exclusion before constructing mixed model"
        )
        for j in range(self.size):
            self.box.append((len(self.rows), len(self.rows) + 1))
            self.lin({j: -1}, 0, ("boxlo", j))
            self.lin({j: 1}, 1, ("boxhi", j))
        x, p, yc, yn, u = [self.vmap[k] for k in ["x", "p", "yc", "yn", "u"]]
        z = self.vmap["z"]
        eta = list(map(unpack, self.geometry["eta_lower"]))
        self.etas = eta
        gamma = [eta[k] / unpack(self.geometry["kappa_upper"][k]) for k in range(K)]
        weights = [f(t) * f(v) for v in n["w"]]
        a2 = [[f(n["P"][m]) * f(n["b"][m, k]) ** 2 for k in range(K)] for m in range(M)]
        g = [
            [[f(n["P"][m]) * f(n["zeta"][m, k, i]) for i in range(K)] for k in range(K)]
            for m in range(M)
        ]
        bar = [1 + sum((max(g[m][k]) for m in range(M))) for k in range(K)]
        for m in range(M):
            self.lin({int(j): 1 for j in p[m]}, 1, ("APpower", m))
            self.lin(
                {
                    **{int(yc[m, k]): weights[k] for k in range(K)},
                    **{int(u[m, k]): weights[k] for k in range(K)},
                },
                f(n["C"][m]),
                ("Jtraffic", m),
            )
            if np.all(n["w"] == n["w"][0]):
                self.lin(
                    {int(j): 1 for j in yc[m]},
                    min(K, int(f(n["C"][m]) // weights[0])),
                    ("Jslots", m),
                )
            for k in range(K):
                for co, tag in [
                    ({int(yc[m, k]): 1, int(z[k]): -1}, "Cgate"),
                    ({int(yn[m, k]): 1, int(z[k]): 1}, "Ngate"),
                    ({int(yc[m, k]): 1, int(yn[m, k]): 1}, "Asum"),
                    ({int(p[m, k]): 1, int(yc[m, k]): -1, int(yn[m, k]): -1}, "Psupport"),
                    ({int(p[m, k]): 1, int(x[m, k]): -1}, "Psecant"),
                    ({int(u[m, k]): 1, int(yn[m, k]): -1}, "Ugate"),
                ]:
                    self.lin(co, 1 if tag in ["Ngate", "Asum"] else 0, (tag, m, k))
                if f(n["C"][m]) < weights[k]:
                    self.lin({int(yc[m, k]): 1}, 0, ("Jeligibility", m, k))
                B = a2[m][k] / F(self.floor["ell"][k], 2**40)
                for p0 in [F(0), F(1, 32), F(1, 8), F(1, 2), F(1)]:
                    aa, bb = tangent(B, p0, n["rho"])
                    self.lin(
                        {int(u[m, k]): weights[k], int(p[m, k]): -aa},
                        bb,
                        ("Ntangent", m, k, str(p0)),
                    )
        for k in range(K):
            self.lin({**{int(j): 1 for j in u[:, k]}, int(z[k]): 1}, 1, ("Ntotalhi", k))
            self.lin({**{int(j): -1 for j in u[:, k]}, int(z[k]): -1}, -1, ("Ntotallo", k))
            self.lin({**{int(j): -1 for j in yc[:, k]}, int(z[k]): 1}, 0, ("Ccoverage", k))
            self.lin({**{int(j): -1 for j in yn[:, k]}, int(z[k]): -1}, -1, ("Ncoverage", k))
            self.lin(
                {**{int(j): 1 for j in yc[:, k]}, **{int(j): 1 for j in yn[:, k]}},
                n["Dmax"],
                ("Jdegree", k),
            )
            d = self.floor["degree"][k]
            self.lin({**{int(j): 1 for j in yc[:, k]}, int(z[k]): -max(0, d)}, 0, ("Cdegree", k))
            if self.geometry["coherent_impossible"][k]:
                self.lin({int(z[k]): 1}, 0, ("forcedN", k))
            Q = root(eta[k] * F(self.floor["ell"][k], 2**40))
            co = {int(x[m, k]): -root(a2[m][k], True) for m in range(M)}
            co[int(z[k])] = Q
            self.lin(co, 0, ("Camplitude", k))
            co = {
                int(p[m, i]): gamma[k] * g[m][k][i] - (a2[m][i] if i == k else 0)
                for m in range(M)
                for i in range(K)
            }
            self.lin(co, -gamma[k], ("vectorSINR", k))
            bonus = (eta[k] - gamma[k]) * F(self.floor["ell"][k], 2**40)
            co[int(z[k])] = -bonus
            self.lin(co, -gamma[k] - bonus, ("modeSINRfloor", k))
            co = {
                int(p[m, i]): eta[k] * g[m][k][i] - (a2[m][i] if i == k else 0)
                for m in range(M)
                for i in range(K)
            }
            co[int(z[k])] = -eta[k] * bar[k]
            self.lin(co, -eta[k], ("NCaggregate", k))
            co[int(z[k])] = -(eta[k] - gamma[k]) * bar[k]
            self.lin(co, -eta[k], ("modeSINRupper", k))
        self.linear_count = len(self.rows)
        self.blocks.append(("nonnegative", self.linear_count))
        for m in range(M):
            for k in range(K):
                self.soc(
                    [
                        {int(p[m, k]): 1, int(yc[m, k]): 1, int(yn[m, k]): 1},
                        {int(x[m, k]): 2},
                        {int(p[m, k]): 1, int(yc[m, k]): -1, int(yn[m, k]): -1},
                    ],
                    [0, 0, 0],
                    ("Jperspective", m, k),
                )
        for k in range(K) if radio else []:
            total = 1 + sum((g[m][k][i] for m in range(M) for i in range(K)))
            s = 1
            while s * s < total:
                s *= 2
            c = [root(a2[m][k] / eta[k] / s**2, True) for m in range(M)]
            H = root(bar[k] / s**2, True)
            maps = [{**{int(x[m, k]): c[m] for m in range(M)}, int(z[k]): -H}, {}]
            constants = [H, F(1, s)]
            for m in range(M):
                for i in range(K):
                    maps.append({int(x[m, i]): root(g[m][k][i] / s**2)})
                    constants.append(0)
            self.soc(maps, constants, ("JCradio", k))
        self.compile()

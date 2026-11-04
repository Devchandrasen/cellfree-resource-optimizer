"""Lifted model."""

from .model import Model as OriginalC, Mixed as OriginalJ
from .arith import F, f, root, tangent
import numpy as np
from .hetero_bounds import eta_down
from .reduced_native import Reduced


class Lift:
    def compile(self):
        n = self.n
        M, K = (n["M"], n["K"])
        J = M * K
        base = self.size
        oldrows = self.rows
        oldspec = self.spec
        linear = self.linear_count
        oldcones = [
            (p, oldrows[p["start"] : p["start"] + p["length"]])
            for p in oldspec
            if p["type"] == "soc"
        ]
        self.rows = oldrows[:linear]
        self.spec = [dict(p) for p in oldspec if p["type"] == "linear"]
        self.blocks = []
        cursor = base
        self.cross = {}
        self.alloc = {}
        self.vmap = dict(self.vmap)
        self.a2 = [[f(n["P"][m]) * f(n["b"][m, k]) ** 2 for k in range(K)] for m in range(M)]
        self.g = [
            [[f(n["P"][m]) * f(n["zeta"][m, k, i]) for i in range(K)] for k in range(K)]
            for m in range(M)
        ]
        self.bar = [1 + sum((max(self.g[m][k]) for m in range(M))) for k in range(K)]
        self.scales = []
        for k in range(K):
            s = 1
            while s * s < 1 + sum((self.g[m][k][i] for m in range(M) for i in range(K))):
                s *= 2
            self.scales.append(s)
        self.q = [min(K, int(f(n["C"][m]) // (f(self.t) * min(map(f, n["w"]))))) for m in range(M)]
        if self.kind == "J":
            for key in ["pc", "pn", "xc", "xn"]:
                self.vmap[key] = np.arange(cursor, cursor + J).reshape(M, K)
                cursor += J
            self.vmap["h"] = np.arange(cursor, cursor + K)
            cursor += K
            if self.features.get("demand", False):
                self.vmap["d"] = np.arange(cursor, cursor + K)
                cursor += K
                self.vmap["dl"] = np.arange(cursor, cursor + J).reshape(M, K)
                cursor += J
        if self.features["power"]:
            for k in range(K):
                eligible = [
                    m
                    for m in range(M)
                    if f(n["C"][m]) >= f(self.t) * f(n["w"][k]) and self.a2[m][k] > 0
                ]
                for j, m in enumerate(eligible):
                    for l in eligible[j + 1 :]:
                        self.cross[k, m, l] = cursor
                        cursor += 1
        if self.features["sparse"]:
            self.vmap["e"] = np.arange(cursor, cursor + J).reshape(M, K)
            cursor += J
            for m in range(M):
                if 1 < self.q[m] < K:
                    for k in range(K):
                        v = list(range(cursor, cursor + K))
                        cursor += K
                        t = list(range(cursor, cursor + K))
                        cursor += K
                        self.alloc[m, k] = (v, t)
        self.size = cursor
        for j in range(base, self.size):
            self.box.append((len(self.rows), len(self.rows) + 1))
            self.lin({j: -1}, 0, ("boxlo", j))
            self.lin({j: 1}, 1, ("boxhi", j))
        if self.kind == "J":
            self.split_linear()
        if self.features["power"]:
            for k in range(K):
                self.power_linear(k)
            if self.features.get("label", False):
                xx = self.vmap["x" if self.kind == "C" else "xc"]
                for (k, m, l), j in self.cross.items():
                    self.lin({j: 1, int(xx[m, k]): -1}, 0, ("coupledgmLabel", k, m, l, 0))
                    self.lin({j: 1, int(xx[l, k]): -1}, 0, ("coupledgmLabel", k, m, l, 1))
                    self.lin(
                        {int(xx[m, k]): 1, int(xx[l, k]): 1, j: -1},
                        1,
                        ("coupledgmLabel", k, m, l, 2),
                    )
        if self.features["sparse"]:
            e = self.vmap["e"]
            for m in range(M):
                for k in range(K):
                    B = root(max(self.g[m][k]) / self.scales[k] ** 2, True)
                    if self.features.get("label", False):
                        self.lin({int(e[m, k]): 1}, B, ("coupledEcap", m, k))
                    if self.q[m] == 0:
                        self.lin({int(e[m, k]): 1}, 0, ("coupledzeroE", m, k))
                    elif self.q[m] == 1:
                        xx = self.vmap["x" if self.kind == "C" else "xc"]
                        co = {
                            int(xx[m, i]): root(self.g[m][k][i] / self.scales[k] ** 2)
                            for i in range(K)
                        }
                        co[int(e[m, k])] = -1
                        self.lin(co, 0, ("coupledoneE", m, k))
                    elif self.q[m] < K:
                        v, t = self.alloc[m, k]
                        for i in range(K):
                            self.lin({v[i]: 1, int(e[m, k]): -1}, 0, ("coupledallocGate", m, k, i))
                        if self.features.get("label", False):
                            yy = self.vmap["y" if self.kind == "C" else "yc"]
                            xx = self.vmap["x" if self.kind == "C" else "xc"]
                            for i in range(K):
                                self.lin(
                                    {v[i]: 1, int(yy[m, i]): -B},
                                    0,
                                    ("coupledallocLabel", m, k, i, 0),
                                )
                                self.lin(
                                    {int(e[m, k]): 1, v[i]: -1, int(yy[m, i]): B},
                                    B,
                                    ("coupledallocLabel", m, k, i, 1),
                                )
                                self.lin(
                                    {t[i]: 1, int(yy[m, i]): -B},
                                    0,
                                    ("coupledallocLabel", m, k, i, 2),
                                )
                                self.lin(
                                    {
                                        t[i]: 1,
                                        int(xx[m, i]): -root(
                                            self.g[m][k][i] / self.scales[k] ** 2, True
                                        ),
                                    },
                                    0,
                                    ("coupledallocLabel", m, k, i, 3),
                                )
                        self.lin(
                            {**{j: 1 for j in v}, int(e[m, k]): -self.q[m]},
                            0,
                            ("coupledallocCount", m, k),
                        )
                        self.lin(
                            {**{j: 1 for j in t}, int(e[m, k]): -1}, 0, ("coupledallocSum", m, k)
                        )
        self.linear_count = len(self.rows)
        self.blocks = [("nonnegative", self.linear_count)]
        for p, rows in oldcones:
            p = dict(p)
            p["start"] = len(self.rows)
            self.spec.append(p)
            self.rows.extend(rows)
            self.blocks.append(("soc", len(rows)))
        if self.kind == "J":
            for m in range(M):
                for k in range(K):
                    for key, yy in [("c", "yc"), ("n", "yn")]:
                        pp = int(self.vmap["p" + key][m, k])
                        xx = int(self.vmap["x" + key][m, k])
                        y = int(self.vmap[yy][m, k])
                        self.soc(
                            [{pp: 1, y: 1}, {xx: 2}, {pp: 1, y: -1}],
                            [0, 0, 0],
                            ("coupledsplitCone", m, k, key),
                        )
        for (k, m, l), j in self.cross.items():
            pp = self.vmap["p" if self.kind == "C" else "pc"]
            a, b = (int(pp[m, k]), int(pp[l, k]))
            self.soc([{a: 1, b: 1}, {j: 2}, {a: 1, b: -1}], [0, 0, 0], ("coupledgm", k, m, l))
        if self.features["sparse"]:
            for m in range(M):
                for k in range(K):
                    xx = self.vmap["x" if self.kind == "C" else "xc"]
                    e = int(self.vmap["e"][m, k])
                    s = self.scales[k]
                    if self.q[m] >= K:
                        self.soc(
                            [{e: 1}]
                            + [{int(xx[m, i]): root(self.g[m][k][i] / s**2)} for i in range(K)],
                            [0] * (K + 1),
                            ("coupleddenseE", m, k),
                        )
                    elif 1 < self.q[m] < K:
                        v, t = self.alloc[m, k]
                        for i in range(K):
                            self.soc(
                                [
                                    {t[i]: 1, v[i]: 1},
                                    {int(xx[m, i]): 2 * root(self.g[m][k][i] / s**2)},
                                    {t[i]: 1, v[i]: -1},
                                ],
                                [0, 0, 0],
                                ("coupledallocCone", m, k, i),
                            )
            for k in range(K):
                self.support_radio(k)
        super().compile()

    def split_linear(self):
        n = self.n
        M, K = (n["M"], n["K"])
        v = self.vmap
        for m in range(M):
            for k in range(K):
                for sign in [-1, 1]:
                    self.lin(
                        {
                            int(v["p"][m, k]): sign,
                            int(v["pc"][m, k]): -sign,
                            int(v["pn"][m, k]): -sign,
                        },
                        0,
                        ("coupledsplitP", m, k, sign),
                    )
                    self.lin(
                        {
                            int(v["x"][m, k]): sign,
                            int(v["xc"][m, k]): -sign,
                            int(v["xn"][m, k]): -sign,
                        },
                        0,
                        ("coupledsplitX", m, k, sign),
                    )
                for key, yy in [("pc", "yc"), ("pn", "yn")]:
                    self.lin(
                        {int(v[key][m, k]): 1, int(v[yy][m, k]): -1},
                        0,
                        ("coupledsplitGate", m, k, key),
                    )
                B = self.a2[m][k] / F(self.floor["ell"][k], 2**40)
                for p0 in [F(0), F(1, 32), F(1, 8), F(1, 2), F(1)]:
                    a, b = tangent(B, p0, n["rho"])
                    self.lin(
                        {
                            int(v["u"][m, k]): f(self.t) * f(n["w"][k]),
                            int(v["pn"][m, k]): -a,
                            int(v["yn"][m, k]): -b,
                        },
                        0,
                        ("coupledsplitRate", m, k, str(p0)),
                    )
        for k in range(K):
            h, z = (int(v["h"][k]), int(v["z"][k]))
            bar = self.bar[k]
            ell = F(self.floor["ell"][k], 2**40)
            if self.features.get("demand", False):
                self.demand_linear(k)
            self.lin({h: -bar, z: ell}, 0, ("coupledDproduct", k, 0))
            self.lin(
                {
                    **{int(v["p"][m, i]): self.g[m][k][i] for m in range(M) for i in range(K)},
                    h: -bar,
                    z: bar,
                },
                bar - 1,
                ("coupledDproduct", k, 1),
            )
            self.lin({h: 1, z: -1}, 0, ("coupledDproduct", k, 2))
            self.lin(
                {
                    **{int(v["p"][m, i]): -self.g[m][k][i] for m in range(M) for i in range(K)},
                    h: bar,
                    z: -ell,
                },
                1 - ell,
                ("coupledDproduct", k, 3),
            )
            co = {
                int(v["p"][m, i]): self.etas[k] * self.g[m][k][i]
                for m in range(M)
                for i in range(K)
            }
            for m in range(M):
                co[int(v["pn"][m, k])] = -self.a2[m][k]
            co[h] = -self.etas[k] * bar
            self.lin(co, -self.etas[k], ("coupledNpower", k))
            if self.features["subset"]:
                strong = sorted(range(M), key=lambda m: (-float(self.a2[m][k]), m))
                small = sorted(range(M), key=lambda m: (float(n["C"][m]), m))
                sets = {tuple(range(M))}
                for order in [strong, small]:
                    for size in [2, 4, 8, M]:
                        if size <= M:
                            sets.add(tuple(sorted(order[:size])))
                for j, m in enumerate(strong[:4]):
                    for l in strong[:4][j + 1 :]:
                        sets.add(tuple(sorted([m, l])))
                for members in sorted(sets):
                    mask = sum((1 << m for m in members))
                    total = sum((self.a2[m][k] for m in members))
                    if not total:
                        continue
                    for p0 in [F(0), F(1, 32), F(1, 8), F(1, 2), F(1)]:
                        a, b = tangent(total / ell, p0, n["rho"])
                        co = {int(v["u"][m, k]): f(self.t) * f(n["w"][k]) for m in members}
                        co.update({int(v["pn"][m, k]): -a * self.a2[m][k] / total for m in members})
                        co[z] = b
                        self.lin(co, b, ("coupledsubsetRate", k, mask, str(p0)))
            if self.features.get("flow", False):
                strong = sorted(range(M), key=lambda m: (-float(self.a2[m][k]), m))
                small = sorted(range(M), key=lambda m: (float(n["C"][m]), m))
                sets = {tuple([m]) for m in range(M)}
                for order in [strong, small]:
                    for length in [2, 4]:
                        if length <= M:
                            sets.add(tuple(sorted(order[:length])))
                for j, m in enumerate(strong[:4]):
                    for l in strong[:4][j + 1 :]:
                        sets.add(tuple(sorted([m, l])))
                for members in sorted(sets):
                    Q = sum((self.q[m] for m in members))
                    delta = (
                        f(self.t) * f(n["w"][k])
                        - sum((f(n["C"][m]) for m in members))
                        + Q * f(self.t) * min(map(f, n["w"]))
                    )
                    if delta <= 0:
                        continue
                    gamma = eta_down(delta, n["rho"])
                    if gamma <= 0:
                        continue
                    co = {
                        int(v["p"][m, i]): gamma * self.g[m][k][i]
                        for m in range(M)
                        for i in range(K)
                    }
                    for m in range(M):
                        if m not in members:
                            co[int(v["pn"][m, k])] = -self.a2[m][k]
                    co[int(v["h"][k])] = -gamma * bar
                    for m in members:
                        for i in range(K):
                            co[int(v["yc"][m, i])] = gamma * bar
                    self.lin(
                        co,
                        gamma * bar * Q - gamma,
                        (
                            "coupledflow",
                            k,
                            sum((1 << m for m in members)),
                            str(gamma.numerator),
                            str(gamma.denominator),
                        ),
                    )

    def demand_linear(self, k):
        n = self.n
        M, K = (n["M"], n["K"])
        v = self.vmap
        bar = self.bar[k]
        ell = F(self.floor["ell"][k], 2**40)
        d = int(v["d"][k])
        for sign in [-1, 1]:
            co = {int(v["p"][m, i]): -sign * self.g[m][k][i] for m in range(M) for i in range(K)}
            co[d] = sign * bar
            self.lin(co, sign, ("coupleddenom", k, sign))
        self.lin({d: -1}, -ell / bar, ("coupleddenomFloor", k))
        for m in range(M):
            load = {int(v["yc"][m, i]): f(self.t) * f(n["w"][i]) / f(n["C"][m]) for i in range(K)}
            w = int(v["dl"][m, k])
            lo = ell / bar
            self.lin(
                {**{j: lo * a for j, a in load.items()}, w: -1}, 0, ("coupledloadProduct", k, m, 0)
            )
            self.lin({**load, d: 1, w: -1}, 1, ("coupledloadProduct", k, m, 1))
            self.lin({**{j: -a for j, a in load.items()}, w: 1}, 0, ("coupledloadProduct", k, m, 2))
            self.lin(
                {**{j: -lo * a for j, a in load.items()}, w: 1, d: -1},
                -lo,
                ("coupledloadProduct", k, m, 3),
            )
        sets = {tuple([m]) for m in range(M)}
        for order in [
            sorted(range(M), key=lambda m: (-float(self.a2[m][k]), m)),
            sorted(range(M), key=lambda m: (float(n["C"][m]), m)),
        ]:
            for length in [2, 4]:
                if length <= M:
                    sets.add(tuple(sorted(order[:length])))
        for members in sorted(sets):
            Csum = sum((f(n["C"][m]) for m in members))
            for p0 in [F(0), F(1), F(4), F(16)]:
                a, b = tangent(F(1), p0, n["rho"])
                co = {
                    d: (f(self.t) * f(n["w"][k]) - Csum - b) * bar,
                    int(v["h"][k]): -f(self.t) * f(n["w"][k]) * bar,
                }
                for m in members:
                    co[int(v["dl"][m, k])] = f(n["C"][m]) * bar
                for m in range(M):
                    if m not in members:
                        co[int(v["pn"][m, k])] = -a * self.a2[m][k]
                self.lin(co, 0, ("coupleddemand", k, sum((1 << m for m in members)), str(p0)))

    def power_linear(self, k):
        n = self.n
        M, K = (n["M"], n["K"])
        pp = self.vmap["p" if self.kind == "C" else "pc"]
        if self.kind == "C":
            co = {
                int(self.vmap["p"][m, i]): self.etas[k] * self.g[m][k][i]
                for m in range(M)
                for i in range(K)
            }
            rhs = -self.etas[k]
        else:
            co = {int(self.vmap["h"][k]): self.etas[k] * self.bar[k]}
            rhs = 0
        for m in range(M):
            co[int(pp[m, k])] = co.get(int(pp[m, k]), F(0)) - self.a2[m][k]
        for (i, m, l), j in self.cross.items():
            if i == k:
                co[j] = -2 * root(self.a2[m][k] * self.a2[l][k], True)
        self.lin(co, rhs, ("coupledCpower", k))

    def support_radio(self, k):
        n = self.n
        M, K = (n["M"], n["K"])
        s = self.scales[k]
        xx = self.vmap["x" if self.kind == "C" else "xc"]
        first = {int(xx[m, k]): root(self.a2[m][k] / self.etas[k] / s**2, True) for m in range(M)}
        constant = F(0)
        if self.kind == "J":
            H = root(self.bar[k] / s**2, True)
            first[int(self.vmap["z"][k])] = -H
            constant = H
        maps = [first, {}] + [{int(self.vmap["e"][m, k]): 1} for m in range(M)]
        bs = [constant, F(1, s)] + [0] * M
        if self.kind == "J":
            for m in range(M):
                for i in range(K):
                    maps.append({int(self.vmap["xn"][m, i]): root(self.g[m][k][i] / s**2)})
                    bs.append(0)
        self.soc(maps, bs, ("coupledsupportRadio", k))

    def metadata(self):
        r = super().metadata()
        r["coupled"] = dict(self.features)
        return r


class Model(Reduced, Lift, OriginalC):
    def __init__(self, n, t, power=True, sparse=True, label=True):
        self.features = dict(
            power=power,
            sparse=sparse,
            split=False,
            subset=False,
            label=label,
            flow=False,
            demand=False,
        )
        super().__init__(n, t)


class Mixed(Reduced, Lift, OriginalJ):
    def __init__(
        self,
        n,
        t,
        radio=True,
        power=True,
        sparse=True,
        subset=True,
        label=True,
        flow=True,
        demand=True,
    ):
        self.features = dict(
            power=power,
            sparse=sparse,
            split=True,
            subset=subset,
            label=label,
            flow=flow,
            demand=demand,
        )
        super().__init__(n, t, radio=radio)

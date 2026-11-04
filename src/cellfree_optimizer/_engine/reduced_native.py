"""Reduced native."""

import numpy as np
import collections
import types
import math
import base64
import zlib
from scipy import sparse
import clarabel
from .arith import f, GRID


def zeros(m, fixed):
    n = m.n
    M, K = (n["M"], n["K"])
    v = m.vmap
    Z = {int(j) for j, val in fixed.items() if val == 0}
    if m.kind == "C":
        for a in range(M):
            for i in range(K):
                if int(v["y"][a, i]) in Z or f(n["C"][a]) < f(m.t) * f(n["w"][i]):
                    Z.update(map(int, [v["y"][a, i], v["p"][a, i], v["x"][a, i]]))
    else:
        for i in range(K):
            zz = int(v["z"][i])
            isN = zz in Z or m.geometry["coherent_impossible"][i]
            isC = fixed.get(zz) == 1 or any((fixed.get(int(v["yc"][a, i])) == 1 for a in range(M)))
            if isN:
                Z.add(zz)
                Z.add(int(v["h"][i]))
                for a in range(M):
                    Z.update((int(v[key][a, i]) for key in ["yc", "pc", "xc"]))
            if isC:
                for a in range(M):
                    Z.update((int(v[key][a, i]) for key in ["yn", "u", "pn", "xn"]))
            for a in range(M):
                if int(v["yc"][a, i]) in Z or f(n["C"][a]) < f(m.t) * f(n["w"][i]):
                    Z.update((int(v[key][a, i]) for key in ["yc", "pc", "xc"]))
                if int(v["pc"][a, i]) in Z and int(v["pn"][a, i]) in Z:
                    Z.add(int(v["p"][a, i]))
                if int(v["xc"][a, i]) in Z and int(v["xn"][a, i]) in Z:
                    Z.add(int(v["x"][a, i]))
    pp = v["p" if m.kind == "C" else "pc"]
    xx = v["x" if m.kind == "C" else "xc"]
    yy = v["y" if m.kind == "C" else "yc"]
    if m.features.get("demand", False):
        for a in range(M):
            if m.q[a] == 0 or all((int(v["yc"][a, i]) in Z for i in range(K))):
                Z.update(map(int, v["dl"][a]))
    for (i, a, b), j in m.cross.items():
        if int(pp[a, i]) in Z or int(pp[b, i]) in Z:
            Z.add(j)
    if m.features["sparse"]:
        for a in range(M):
            if m.q[a] == 0 or all((int(xx[a, i]) in Z for i in range(K))):
                for i in range(K):
                    Z.add(int(v["e"][a, i]))
            for i in range(K):
                if (a, i) not in m.alloc:
                    continue
                vv, tt = m.alloc[a, i]
                for j in range(K):
                    if int(v["e"][a, i]) in Z or int(yy[a, j]) in Z:
                        Z.update([vv[j], tt[j]])
                    elif m.g[a][i][j] == 0:
                        Z.add(tt[j])
    return Z


class Reduced:
    def solve(self, fixed=None):
        fixed = fixed or {}
        Z = zeros(self, fixed)
        key = tuple(sorted(Z))
        cache = getattr(self, "_reduced_cache", None)
        conflict = next((int(j) for j, value in fixed.items() if value == 1 and int(j) in Z), None)
        if conflict is not None:
            zz = np.zeros(len(self.rows))
            zz[self.box[conflict][0]] = 1
            return (
                types.SimpleNamespace(x=np.zeros(self.size), z=zz, status="DerivedInfeasible"),
                self.b.copy(),
            )
        if cache is None:
            self._reduced_cache = cache = collections.OrderedDict()
        if key in cache:
            entry = cache.pop(key)
            cache[key] = entry
        else:
            keep = np.array([j for j in range(self.size) if j not in Z], int)
            rowmap = []
            cones = []
            pos = 0
            AA = self.A[:, keep].tocsr()
            for kind, length in self.blocks:
                ids = list(range(pos, pos + length))
                pos += length
                if kind == "nonnegative":
                    ids = [i for i in ids if AA.indptr[i + 1] > AA.indptr[i] or self.b[i] < 0]
                    if ids:
                        rowmap.extend(ids)
                        cones.append(clarabel.NonnegativeConeT(len(ids)))
                else:
                    if all((AA.indptr[i + 1] == AA.indptr[i] and self.b[i] == 0 for i in ids)):
                        continue
                    ids = [ids[0]] + [
                        i for i in ids[1:] if AA.indptr[i + 1] > AA.indptr[i] or self.b[i] != 0
                    ]
                    if len(ids) == 2:
                        a, b = (AA.getrow(ids[0]), AA.getrow(ids[1]))
                        same = (a - b).nnz == 0 and self.b[ids[0]] == self.b[ids[1]]
                        opposite = (a + b).nnz == 0 and self.b[ids[0]] == -self.b[ids[1]]
                        if same or opposite:
                            ids = ids[:1]
                    rowmap.extend(ids)
                    cones.append(
                        clarabel.NonnegativeConeT(1)
                        if len(ids) == 1
                        else clarabel.SecondOrderConeT(len(ids))
                    )
            settings = clarabel.DefaultSettings()
            settings.verbose = False
            settings.max_threads = 1
            settings.direct_solve_method = "qdldl"
            settings.max_iter = 150
            settings.tol_feas = 1e-09
            settings.tol_gap_abs = 1e-09
            settings.tol_gap_rel = 1e-09
            rowmap = np.asarray(rowmap, int)
            A = self.A[rowmap][:, keep].tocsc()
            solver = clarabel.DefaultSolver(
                sparse.csc_matrix((len(keep), len(keep))),
                np.zeros(len(keep)),
                A,
                self.b[rowmap],
                cones,
                settings,
            )
            assert (
                solver.get_info().linsolver.threads == 1
                and solver.get_info().linsolver.name == "qdldl"
            )
            entry = (solver, keep, rowmap)
            cache[key] = entry
            while len(cache) > 2:
                cache.popitem(last=False)
        solver, keep, rowmap = entry
        bb = self.b.copy()
        for j, val in fixed.items():
            lo, hi = self.box[int(j)]
            bb[lo] = -val
            bb[hi] = val
        solver.update(b=bb[rowmap])
        raw = solver.solve()
        xx = np.zeros(self.size)
        zz = np.zeros(len(self.rows))
        xx[keep] = raw.x
        zz[rowmap] = raw.z
        self.last_reduction = dict(columns=len(keep), rows=len(rowmap), zero_columns=len(Z))
        return (types.SimpleNamespace(x=xx, z=zz, status=raw.status), bb)

    def witness(self, sol, fixed=None):
        fixed = fixed or {}
        raw = np.asarray(sol.z, float)
        if not np.all(np.isfinite(raw)) or not np.max(abs(raw), initial=0):
            return None
        H = 2**48
        raw = raw / np.max(abs(raw))
        z = [int(round(a * H)) for a in raw]
        pos = 0
        for kind, length in self.blocks:
            if kind == "nonnegative":
                z[pos : pos + length] = [max(0, a) for a in z[pos : pos + length]]
            else:
                nn = sum((a * a for a in z[pos + 1 : pos + length]))
                ceil = math.isqrt(nn)
                if ceil * ceil < nn:
                    ceil += 1
                z[pos] = max(0, z[pos], ceil)
            pos += length
        Z = zeros(self, fixed)
        r = [0] * self.size
        rhs = 0
        override = {}
        for j, val in fixed.items():
            lo, hi = self.box[int(j)]
            override[lo] = -val * GRID
            override[hi] = val * GRID
        for i, ((co, b), a) in enumerate(zip(self.rows, z)):
            if a:
                rhs += override.get(i, b) * a
                for j, c in co.items():
                    r[j] += a * c
        margin = sum((min(0, a) for j, a in enumerate(r) if j not in Z)) - rhs
        if margin <= 0:
            return None
        return dict(
            z_b85=base64.b85encode(zlib.compress(np.asarray(z, dtype="<i8").tobytes(), 6)).decode(),
            length=len(z),
            margin=str(margin),
            zero_columns=sorted(Z),
            domain_schema="canonical_coupled",
        )

    def metadata(self):
        r = super().metadata()
        r["canonical_zero_domains"] = True
        return r

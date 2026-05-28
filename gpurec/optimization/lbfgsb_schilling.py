"""Reference-compatible L-BFGS-B kernels used by conformance tests.

The routines in this module are local Python ports of the Schilling L-BFGS-B
3.0 portability-spec kernels.  They intentionally preserve the spec's
Fortran-style loop order and one-based index conventions at the boundaries so
JSON vectors from ``docs/spec/data`` can be used directly.

Derived from Jonathan Schilling's BSD-3-Clause L-BFGS-B specification pack.
The upstream license is retained with the copied conformance fixtures under
``tests/data/lbfgsb_schilling/LICENSE``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import LinAlgError, cholesky, solve_triangular


@dataclass(frozen=True)
class SchillingActiveResult:
    prjctd: bool
    cnstnd: bool
    boxed: bool


def schilling_projgr(
    n: int,
    x: np.ndarray,
    lower: np.ndarray,
    u: np.ndarray,
    nbd: np.ndarray,
    g: np.ndarray,
) -> float:
    sbgnrm = 0.0
    for i in range(n):
        gi = g[i]
        if nbd[i] != 0:
            if gi < 0.0:
                if nbd[i] >= 2:
                    gi = max(x[i] - u[i], gi)
            else:
                if nbd[i] <= 2:
                    gi = min(x[i] - lower[i], gi)
        sbgnrm = max(sbgnrm, abs(gi))
    return float(sbgnrm)


def schilling_active(
    n: int,
    lower: np.ndarray,
    u: np.ndarray,
    nbd: np.ndarray,
    x: np.ndarray,
    iwhere: np.ndarray,
    iprint: int = -1,
) -> SchillingActiveResult:
    nbdd = 0
    prjctd = False
    for i in range(n):
        if nbd[i] > 0:
            if nbd[i] <= 2 and x[i] <= lower[i]:
                if x[i] < lower[i]:
                    prjctd = True
                    x[i] = lower[i]
                nbdd += 1
            elif nbd[i] >= 2 and x[i] >= u[i]:
                if x[i] > u[i]:
                    prjctd = True
                    x[i] = u[i]
                nbdd += 1

    cnstnd = False
    boxed = True
    for i in range(n):
        if nbd[i] != 2:
            boxed = False
        if nbd[i] == 0:
            iwhere[i] = -1
        else:
            cnstnd = True
            if nbd[i] == 2 and u[i] - lower[i] <= 0.0:
                iwhere[i] = 3
            else:
                iwhere[i] = 0

    _ = iprint
    _ = nbdd
    return SchillingActiveResult(prjctd=prjctd, cnstnd=cnstnd, boxed=boxed)


def schilling_bmv(
    m: int,
    sy: np.ndarray,
    wt: np.ndarray,
    col: int,
    v: np.ndarray,
    p: np.ndarray,
) -> None:
    if col == 0:
        return

    p[col] = v[col]
    for i in range(2, col + 1):
        s = 0.0
        for k in range(1, i):
            s += sy[i - 1, k - 1] * v[k - 1] / sy[k - 1, k - 1]
        p[col + i - 1] = v[col + i - 1] + s

    p[col : col + col] = solve_triangular(
        wt[:col, :col],
        p[col : col + col],
        lower=False,
        trans="T",
    )

    for i in range(1, col + 1):
        p[i - 1] = v[i - 1] / np.sqrt(sy[i - 1, i - 1])

    p[col : col + col] = solve_triangular(
        wt[:col, :col],
        p[col : col + col],
        lower=False,
        trans="N",
    )

    for i in range(1, col + 1):
        p[i - 1] = -p[i - 1] / np.sqrt(sy[i - 1, i - 1])

    for i in range(1, col + 1):
        s = 0.0
        for k in range(i + 1, col + 1):
            s += sy[k - 1, i - 1] * p[col + k - 1] / sy[i - 1, i - 1]
        p[i - 1] += s


def schilling_formt(
    m: int,
    wt: np.ndarray,
    sy: np.ndarray,
    ss: np.ndarray,
    col: int,
    theta: float,
) -> int:
    if col == 0:
        return 0

    for j in range(1, col + 1):
        wt[0, j - 1] = theta * ss[0, j - 1]

    for i in range(2, col + 1):
        for j in range(i, col + 1):
            s = 0.0
            for k in range(1, i):
                s += sy[i - 1, k - 1] * sy[j - 1, k - 1] / sy[k - 1, k - 1]
            wt[i - 1, j - 1] = s + theta * ss[i - 1, j - 1]

    try:
        chol = cholesky(wt[:col, :col].copy(), lower=False, check_finite=True)
    except LinAlgError:
        return -3
    for i in range(col):
        for j in range(i, col):
            wt[i, j] = chol[i, j]
    return 0


def schilling_cauchy(
    n: int,
    x: np.ndarray,
    lower: np.ndarray,
    u: np.ndarray,
    nbd: np.ndarray,
    g: np.ndarray,
    iorder: np.ndarray,
    iwhere: np.ndarray,
    t: np.ndarray,
    d: np.ndarray,
    xcp: np.ndarray,
    m: int,
    wy: np.ndarray,
    ws: np.ndarray,
    sy: np.ndarray,
    wt: np.ndarray,
    theta: float,
    col: int,
    head: int,
    p: np.ndarray,
    c: np.ndarray,
    wbp: np.ndarray,
    v: np.ndarray,
    sbgnrm: float,
    epsmch: float,
    iprint: int = -1,
) -> int:
    if sbgnrm <= 0.0:
        xcp[:] = x
        return 0

    bnded = True
    nfree = n + 1
    nbreak = 0
    ibkmin = 0
    bkmin = 0.0
    col2 = 2 * col
    f1 = 0.0
    for i in range(col2):
        p[i] = 0.0

    for i in range(1, n + 1):
        neggi = -g[i - 1]
        if iwhere[i - 1] != 3 and iwhere[i - 1] != -1:
            tl = 0.0
            tu = 0.0
            if nbd[i - 1] <= 2:
                tl = x[i - 1] - lower[i - 1]
            if nbd[i - 1] >= 2:
                tu = u[i - 1] - x[i - 1]
            xlower = (nbd[i - 1] <= 2) and (tl <= 0.0)
            xupper = (nbd[i - 1] >= 2) and (tu <= 0.0)

            iwhere[i - 1] = 0
            if xlower:
                if neggi <= 0.0:
                    iwhere[i - 1] = 1
            elif xupper:
                if neggi >= 0.0:
                    iwhere[i - 1] = 2
            else:
                if abs(neggi) <= 0.0:
                    iwhere[i - 1] = -3

        pointr = head
        if iwhere[i - 1] != 0 and iwhere[i - 1] != -1:
            d[i - 1] = 0.0
        else:
            d[i - 1] = neggi
            f1 = f1 - neggi * neggi
            for j in range(1, col + 1):
                p[j - 1] += wy[i - 1, pointr - 1] * neggi
                p[col + j - 1] += ws[i - 1, pointr - 1] * neggi
                pointr = (pointr % m) + 1
            if (nbd[i - 1] <= 2) and (nbd[i - 1] != 0) and (neggi < 0.0):
                nbreak += 1
                tl = x[i - 1] - lower[i - 1]
                iorder[nbreak - 1] = i
                t[nbreak - 1] = tl / (-neggi)
                if nbreak == 1 or t[nbreak - 1] < bkmin:
                    bkmin = t[nbreak - 1]
                    ibkmin = nbreak
            elif (nbd[i - 1] >= 2) and (neggi > 0.0):
                nbreak += 1
                tu = u[i - 1] - x[i - 1]
                iorder[nbreak - 1] = i
                t[nbreak - 1] = tu / neggi
                if nbreak == 1 or t[nbreak - 1] < bkmin:
                    bkmin = t[nbreak - 1]
                    ibkmin = nbreak
            else:
                nfree -= 1
                iorder[nfree - 1] = i
                if abs(neggi) > 0.0:
                    bnded = False

    if theta != 1.0:
        for j in range(col):
            p[col + j] *= theta

    xcp[:] = x
    if nbreak == 0 and nfree == n + 1:
        return 0

    for j in range(col2):
        c[j] = 0.0

    f2 = -theta * f1
    f2_org = f2
    if col > 0:
        schilling_bmv(m, sy, wt, col, p[:col2].copy(), v[:col2])
        f2 = f2 - float(np.dot(v[:col2], p[:col2]))
    dtm = -f1 / f2 if f2 != 0.0 else 0.0
    tsum = 0.0
    nseg = 1
    skip_final_motion = False

    if nbreak != 0:
        nleft = nbreak
        iter_ = 1
        tj = 0.0
        while True:
            tj0 = tj
            if iter_ == 1:
                tj = bkmin
                ibp = iorder[ibkmin - 1]
            else:
                if iter_ == 2 and ibkmin != nbreak:
                    t[ibkmin - 1] = t[nbreak - 1]
                    iorder[ibkmin - 1] = iorder[nbreak - 1]
                _schilling_hpsolb(nleft, t, iorder, iter_ - 2)
                tj = t[nleft - 1]
                ibp = iorder[nleft - 1]

            dt = tj - tj0
            if dtm < dt:
                break

            tsum += dt
            nleft -= 1
            iter_ += 1
            dibp = d[ibp - 1]
            d[ibp - 1] = 0.0
            if dibp > 0.0:
                zibp = u[ibp - 1] - x[ibp - 1]
                xcp[ibp - 1] = u[ibp - 1]
                iwhere[ibp - 1] = 2
            else:
                zibp = lower[ibp - 1] - x[ibp - 1]
                xcp[ibp - 1] = lower[ibp - 1]
                iwhere[ibp - 1] = 1

            if nleft == 0 and nbreak == n:
                dtm = dt
                skip_final_motion = True
                break

            nseg += 1
            dibp2 = dibp * dibp
            f1 = f1 + dt * f2 + dibp2 - theta * dibp * zibp
            f2 = f2 - theta * dibp2

            if col > 0:
                for j in range(col2):
                    c[j] += dt * p[j]
                pointr = head
                for j in range(1, col + 1):
                    wbp[j - 1] = wy[ibp - 1, pointr - 1]
                    wbp[col + j - 1] = theta * ws[ibp - 1, pointr - 1]
                    pointr = (pointr % m) + 1
                schilling_bmv(m, sy, wt, col, wbp[:col2].copy(), v[:col2])
                wmc = float(np.dot(c[:col2], v[:col2]))
                wmp = float(np.dot(p[:col2], v[:col2]))
                wmw = float(np.dot(wbp[:col2], v[:col2]))
                for j in range(col2):
                    p[j] -= dibp * wbp[j]
                f1 += dibp * wmc
                f2 += 2.0 * dibp * wmp - dibp2 * wmw

            f2 = max(epsmch * f2_org, f2)
            if nleft > 0:
                dtm = -f1 / f2
                continue
            if bnded:
                f1 = 0.0
                f2 = 0.0
                dtm = 0.0
            else:
                dtm = -f1 / f2
            break

    if not skip_final_motion:
        if dtm <= 0.0:
            dtm = 0.0
        tsum += dtm
        for i in range(n):
            xcp[i] = xcp[i] + tsum * d[i]
        if col > 0:
            for j in range(col2):
                c[j] += dtm * p[j]
    elif col > 0:
        for j in range(col2):
            c[j] += dtm * p[j]

    _ = iprint
    return int(nseg)


def _schilling_hpsolb(n: int, t: np.ndarray, iorder: np.ndarray, iheap: int) -> None:
    if n <= 1:
        return
    if iheap == 0:
        for k in range(2, n + 1):
            t_k = t[k - 1]
            i_k = iorder[k - 1]
            j = k
            while j > 1:
                parent = j // 2
                if t[parent - 1] <= t_k:
                    break
                t[j - 1] = t[parent - 1]
                iorder[j - 1] = iorder[parent - 1]
                j = parent
            t[j - 1] = t_k
            iorder[j - 1] = i_k
    if n > 1:
        t0 = t[0]
        i0 = iorder[0]
        t[0] = t[n - 1]
        iorder[0] = iorder[n - 1]
        t[n - 1] = t0
        iorder[n - 1] = i0
        n -= 1
        k = 1
        while True:
            j = 2 * k
            if j > n:
                break
            if j < n and t[j] < t[j - 1]:
                j += 1
            if t[k - 1] <= t[j - 1]:
                break
            t0 = t[k - 1]
            i0 = iorder[k - 1]
            t[k - 1] = t[j - 1]
            iorder[k - 1] = iorder[j - 1]
            t[j - 1] = t0
            iorder[j - 1] = i0
            k = j


def schilling_subsm(
    n: int,
    m: int,
    nsub: int,
    ind: np.ndarray,
    lower: np.ndarray,
    u: np.ndarray,
    nbd: np.ndarray,
    x: np.ndarray,
    d: np.ndarray,
    xp: np.ndarray,
    ws: np.ndarray,
    wy: np.ndarray,
    theta: float,
    xx: np.ndarray,
    gg: np.ndarray,
    col: int,
    head: int,
    wv: np.ndarray,
    wn: np.ndarray,
    iprint: int = -1,
) -> int:
    if nsub <= 0:
        return 0

    col2 = 2 * col
    pointr = head
    for i in range(1, col + 1):
        temp1 = 0.0
        temp2 = 0.0
        for j in range(1, nsub + 1):
            k = ind[j - 1] - 1
            temp1 += wy[k, pointr - 1] * d[j - 1]
            temp2 += ws[k, pointr - 1] * d[j - 1]
        wv[i - 1] = temp1
        wv[col + i - 1] = theta * temp2
        pointr = (pointr % m) + 1

    if col2 > 0:
        wv[:col2] = solve_triangular(
            wn[:col2, :col2],
            wv[:col2],
            lower=False,
            trans="T",
        )
        for i in range(col):
            wv[i] = -wv[i]
        wv[:col2] = solve_triangular(
            wn[:col2, :col2],
            wv[:col2],
            lower=False,
            trans="N",
        )

    pointr = head
    for jy in range(1, col + 1):
        js = col + jy
        for i in range(1, nsub + 1):
            k = ind[i - 1] - 1
            d[i - 1] += wy[k, pointr - 1] * wv[jy - 1] / theta + ws[
                k,
                pointr - 1,
            ] * wv[js - 1]
        pointr = (pointr % m) + 1
    for i in range(nsub):
        d[i] = d[i] / theta

    iword = 0
    xp[:] = x
    for i in range(1, nsub + 1):
        k = ind[i - 1] - 1
        dk = d[i - 1]
        xk = x[k]
        if nbd[k] != 0:
            if nbd[k] == 1:
                x[k] = max(lower[k], xk + dk)
                if x[k] == lower[k]:
                    iword = 1
            elif nbd[k] == 2:
                xk2 = max(lower[k], xk + dk)
                x[k] = min(u[k], xk2)
                if x[k] == lower[k] or x[k] == u[k]:
                    iword = 1
            elif nbd[k] == 3:
                x[k] = min(u[k], xk + dk)
                if x[k] == u[k]:
                    iword = 1
        else:
            x[k] = xk + dk

    if iword == 0:
        return iword

    dd_p = 0.0
    for i in range(n):
        dd_p += (x[i] - xx[i]) * gg[i]
    if dd_p > 0.0:
        x[:] = xp
    else:
        return iword

    alpha = 1.0
    temp1 = alpha
    ibd = 0
    for i in range(1, nsub + 1):
        k = ind[i - 1] - 1
        dk = d[i - 1]
        if nbd[k] != 0:
            if dk < 0.0 and nbd[k] <= 2:
                temp2 = lower[k] - x[k]
                if temp2 >= 0.0:
                    temp1 = 0.0
                elif dk * alpha < temp2:
                    temp1 = temp2 / dk
            elif dk > 0.0 and nbd[k] >= 2:
                temp2 = u[k] - x[k]
                if temp2 <= 0.0:
                    temp1 = 0.0
                elif dk * alpha > temp2:
                    temp1 = temp2 / dk
            if temp1 < alpha:
                alpha = temp1
                ibd = i

    if alpha < 1.0:
        dk = d[ibd - 1]
        k = ind[ibd - 1] - 1
        if dk > 0.0:
            x[k] = u[k]
            d[ibd - 1] = 0.0
        elif dk < 0.0:
            x[k] = lower[k]
            d[ibd - 1] = 0.0

    for i in range(1, nsub + 1):
        k = ind[i - 1] - 1
        x[k] = x[k] + alpha * d[i - 1]

    _ = iprint
    return iword


def run_schilling_case(subroutine: str, inputs: dict[str, Any]) -> dict[str, Any]:
    if subroutine == "projgr":
        return {
            "sbgnrm": schilling_projgr(
                inputs["n"],
                np.array(inputs["x"], dtype=np.float64),
                np.array(inputs["l"], dtype=np.float64),
                np.array(inputs["u"], dtype=np.float64),
                np.array(inputs["nbd"], dtype=np.int32),
                np.array(inputs["g"], dtype=np.float64),
            )
        }
    if subroutine == "active":
        n = inputs["n"]
        x = np.array(inputs["x_in"], dtype=np.float64).copy()
        iwhere = np.zeros(n, dtype=np.int32)
        result = schilling_active(
            n,
            np.array(inputs["l"], dtype=np.float64),
            np.array(inputs["u"], dtype=np.float64),
            np.array(inputs["nbd"], dtype=np.int32),
            x,
            iwhere,
            inputs["iprint"],
        )
        return {
            "x": x.tolist(),
            "iwhere": iwhere.tolist(),
            "prjctd": result.prjctd,
            "cnstnd": result.cnstnd,
            "boxed": result.boxed,
        }
    if subroutine == "bmv":
        p = np.array(inputs["p_in"], dtype=np.float64).copy()
        schilling_bmv(
            inputs["m"],
            np.array(inputs["sy"], dtype=np.float64),
            np.array(inputs["wt"], dtype=np.float64),
            inputs["col"],
            np.array(inputs["v"], dtype=np.float64),
            p,
        )
        return {"p": p.tolist()}
    if subroutine == "cauchy":
        n = inputs["n"]
        m = inputs["m"]
        col = inputs["col"]
        iwhere = np.array(inputs["iwhere_in"], dtype=np.int32).copy()
        iorder = np.zeros(n, dtype=np.int32)
        t = np.zeros(n, dtype=np.float64)
        d = np.zeros(n, dtype=np.float64)
        xcp = np.full(n, -42.0, dtype=np.float64)
        p = np.zeros(max(2, 2 * col), dtype=np.float64)
        c = np.zeros(max(2, 2 * col), dtype=np.float64)
        wbp = np.zeros(max(2, 2 * col), dtype=np.float64)
        v = np.zeros(max(2, 2 * col), dtype=np.float64)
        nseg = schilling_cauchy(
            n,
            np.array(inputs["x"], dtype=np.float64),
            np.array(inputs["l"], dtype=np.float64),
            np.array(inputs["u"], dtype=np.float64),
            np.array(inputs["nbd"], dtype=np.int32),
            np.array(inputs["g"], dtype=np.float64),
            iorder,
            iwhere,
            t,
            d,
            xcp,
            m,
            np.array(inputs["wy"], dtype=np.float64).reshape(n, max(1, col)),
            np.array(inputs["ws"], dtype=np.float64).reshape(n, max(1, col)),
            np.array(inputs["sy"], dtype=np.float64).reshape(m, m),
            np.array(inputs["wt"], dtype=np.float64).reshape(m, m),
            inputs["theta"],
            col,
            inputs["head"],
            p,
            c,
            wbp,
            v,
            inputs["sbgnrm"],
            inputs["epsmch"],
            inputs["iprint"],
        )
        return {"xcp": xcp.tolist(), "iwhere": iwhere.tolist(), "nseg": nseg}
    if subroutine == "subsm":
        n = inputs["n"]
        m = inputs["m"]
        nsub = inputs["nsub"]
        ind = (
            np.array(inputs["ind"], dtype=np.int32)
            if nsub > 0
            else np.zeros(0, dtype=np.int32)
        )
        x = np.array(inputs["x_in"], dtype=np.float64).copy()
        d = (
            np.array(inputs["d_in"], dtype=np.float64).copy()
            if inputs["d_in"]
            else np.zeros(max(1, nsub), dtype=np.float64)
        )
        xp = np.full(n, -42.0, dtype=np.float64)
        wv = np.zeros(2 * m, dtype=np.float64)
        iword = schilling_subsm(
            n,
            m,
            nsub,
            ind,
            np.array(inputs["l"], dtype=np.float64),
            np.array(inputs["u"], dtype=np.float64),
            np.array(inputs["nbd"], dtype=np.int32),
            x,
            d,
            xp,
            np.array(inputs["ws"], dtype=np.float64).reshape(n, m),
            np.array(inputs["wy"], dtype=np.float64).reshape(n, m),
            inputs["theta"],
            np.array(inputs["xx"], dtype=np.float64),
            np.array(inputs["gg"], dtype=np.float64),
            inputs["col"],
            inputs["head"],
            wv,
            np.array(inputs["wn_in"], dtype=np.float64).reshape(2 * m, 2 * m).copy(),
        )
        return {"x": x.tolist(), "iword": iword}
    raise ValueError(f"unsupported Schilling L-BFGS-B subroutine: {subroutine}")

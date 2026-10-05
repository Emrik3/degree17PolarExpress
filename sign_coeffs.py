#!/usr/bin/env python3
"""
Coefficients for a composition of odd polynomials approximating sign(x) on [l, 1].

Pipeline (one pass per iteration t = 0 .. iters-1):
  1. Remez: odd polynomial p_t of degree 2q+1 (q = 2**m) approximating 1 on [l_t, u_t].
     The lower bound is then updated, l_{t+1} = p_t(l_t), u_{t+1} = 2 - l_{t+1}.
  2. Gauss-Newton (damped, i.e. Levenberg-Marquardt): fit the efficient evaluation scheme (parameter vector theta,
     see `eval_poly`) to p_t. The solution for iteration t is the starting guess
     for iteration t+1.

Usage:
    python sign_coeffs.py --l 0.001 --iters 4
    python sign_coeffs.py --l 1e-4 --iters 5 --out coeffs --plot

Output (in --out, default "coeffs/"):
    coeffs_remez.npy   (iters, q+1)   p_t(x) = sum_i c[t,i] * x**(2i+1)
    coeffs_eval.npy    (iters, P)     flat theta for the evaluation scheme (see `unpack`)
    bounds.npy         (iters, 2)     [l_t, u_t] interval each polynomial was built for
"""

import argparse
import os
from math import comb

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial
from numpy.polynomial.chebyshev import chebvander


# ----------------------------------------------------------------------------
# Odd polynomials  p(x) = sum_i c[i] * x**(2i+1)
# ----------------------------------------------------------------------------
def odd_poly(c, x):
    """Evaluate sum_i c[i] x^(2i+1) (Horner in x^2)."""
    x = np.asarray(x, dtype=float)
    y = x * x
    out = np.zeros_like(x)
    for ci in c[::-1]:
        out = out * y + ci
    return x * out


def taylor_coeffs(q):
    """
    Degree-(2q+1) Taylor polynomial of sign at 1: truncation of
    x * (1 - x^2)^(-1/2) = sum_k binom(2k,k)/4^k (1 - x^2)^k.
    For q=2 this gives [1.875, -1.25, 0.375]; for q=8 the 109395/32768... set.
    """
    c = np.zeros(q + 1)
    for k in range(q + 1):
        w = comb(2 * k, k) / 4**k
        for j in range(k + 1):
            c[j] += w * comb(k, j) * (-1) ** j
    return c


# ----------------------------------------------------------------------------
# Remez
# ----------------------------------------------------------------------------
def interior_extrema(c, l, u):
    """
    Interior local extrema of an odd polynomial on (l, u).
    p'(x) = Q(x^2) with Q(y) = sum_i c[i](2i+1) y^i, so we find the roots of Q
    (degree q) instead of a degree-2q polynomial in x.
    """
    d = np.array([ci * (2 * i + 1) for i, ci in enumerate(c)])
    if len(d) < 2:
        return np.array([])
    Qd = d[::-1]  # descending powers for numpy
    dQ = np.polyder(Qd)
    y = np.roots(Qd)
    y = y[(np.abs(y.imag) <= 1e-8 * np.maximum(1.0, np.abs(y))) & (y.real > 0)].real
    for _ in range(3):  # a few Newton steps polish the roots
        slope = np.polyval(dQ, y)
        y = np.where(slope != 0, y - np.polyval(Qd, y) / np.where(slope == 0, 1, slope), y)
    x = np.sqrt(y[y > 0])
    return np.sort(x[(x > l) & (x < u)])


def odd_remez(q, l, u, tol=1e-15, max_iter=100):
    """
    Remez algorithm: best odd polynomial p(x) = x*r(x^2) of degree 2q+1 to the constant 1
    on [l, u]. Returns (c, E): q+1 monomial coefficients (p = sum c[i] x^(2i+1)) and |E|.

    The solve is done with r in a Chebyshev basis on [l^2, u^2]. A monomial Vandermonde system
    becomes hopelessly ill-conditioned once the interval is narrow (e.g. [0.98, 1.02]).
    The conversion to monomial coefficients is only done at the very end.
    """
    n = q + 2
    dom = [l * l, u * u]
    signs = (-1.0) ** np.arange(n)
    Y = Chebyshev.identity(domain=dom)  # the function y -> y
    x = np.sort(0.5 * (l + u) + 0.5 * (u - l) * np.cos(np.pi * np.arange(n) / (n - 1)))

    old_E = np.inf
    for _ in range(max_iter):
        s = (2 * x * x - (dom[0] + dom[1])) / (dom[1] - dom[0])
        A = np.empty((n, n))
        A[:, : q + 1] = chebvander(s, q)
        A[:, -1] = signs / x
        try:
            sol = np.linalg.solve(A, 1.0 / x)  # x r(x^2) + (-1)^j E = 1
        except np.linalg.LinAlgError as e:
            raise RuntimeError("Remez system is singular") from e
        r, E = Chebyshev(sol[:-1], domain=dom), sol[-1]

        # p'(x) = r(y) + 2 y r'(y) with y = x^2: a degree-q polynomial in y
        w = r + 2 * Y * r.deriv()
        yr = w.roots()
        yr = yr[np.abs(yr.imag) <= 1e-9 * np.maximum(1.0, np.abs(yr))].real
        ext = np.sort(np.sqrt(yr[(yr > dom[0]) & (yr < dom[1])]))
        if len(ext) != q:
            raise RuntimeError(f"Remez: found {len(ext)} interior extrema, expected {q}.")
        x = np.concatenate(([l], ext, [u]))
        if abs(abs(E) - abs(old_E)) < tol:
            break
        old_E = E

    c = r.convert(kind=Polynomial, domain=[-1, 1], window=[-1, 1]).coef
    c = np.concatenate([c, np.zeros(q + 1 - len(c))])
    return c, abs(E)


def remez_chain(l, iters, q, cushion=0.02407327424182761):
    """
    Composition of `iters` Remez polynomials.

    cushion: Remez is run on [max(l, cushion*u), u] and the polynomial is rescaled
             so that p(l) + p(u) = 2. Set to 0 to disable.
    Returns coefficient array (iters, q+1) and bounds array (iters, 2) = [l_t, u_t].
    """
    u = 1.0
    coeffs, bounds = [], []
    for _ in range(iters):
        bounds.append((l, u))
        if 1.0 - l <= 1e-9:
            # Already converged: use the Taylor polynomial at 1 (quadratic convergence).
            c = taylor_coeffs(q)
            u = 2.0 - l
        else:
            lo = max(l, cushion * u)
            try:
                c, _E = odd_remez(q, lo, u)
            except RuntimeError as e:
                # Interval so narrow that double precision cannot resolve the equioscillation:
                # the Taylor polynomial at 1 is as good as anything we can represent.
                print(f"  [note] Remez failed on [{lo:.4g}, {u:.4g}] ({e}); using Taylor polynomial.")
                c = taylor_coeffs(q)
                coeffs.append(c)
                l = float(odd_poly(c, l))  # ~ 1 - 1e-16: the next step is the fully converged one
                u = 2.0 - l
                continue
            if cushion * u > l:
                c = c * (2.0 / (odd_poly(c, l) + odd_poly(c, u)))
            l = float(odd_poly(c, l))
            u = 2.0 - l
        coeffs.append(c)
    return np.array(coeffs), np.array(bounds)


# ----------------------------------------------------------------------------
# Efficient evaluation scheme
#
#   out_0 = 1,  out_1 = x^2
#   out_{i+2} = c_i * (A_i . out[:i+2]) * (B_i . out[:i+2]),   i = 0..m-1
#   result = x * sum_k out_k          (a polynomial of degree 2^(m+1)+1 in x)
#
# theta = [A_0 .. A_{m-1}, B_0 .. B_{m-1}, c]  with |A_i| = |B_i| = i+2, |c| = m+2
# (c[m], c[m+1] are never used) TODO: Remove them
# ----------------------------------------------------------------------------
def n_params(m):
    return m * (m + 3) + (m + 2)


def unpack(theta, m):
    A, B, idx = [], [], 0
    for i in range(m):
        A.append(theta[idx : idx + i + 2])
        idx += i + 2
    for i in range(m):
        B.append(theta[idx : idx + i + 2])
        idx += i + 2
    c = theta[idx : idx + m + 2]
    return A, B, c


def pack(A, B, c):
    return np.concatenate([np.concatenate(A), np.concatenate(B), np.asarray(c, float)])


def _forward(theta, m, x):
    A, B, c = unpack(theta, m)
    out = np.empty((m + 2, x.size))
    out[0] = 1.0
    out[1] = x * x
    U, V = [], []
    for i in range(m):
        k = i + 2
        u = A[i] @ out[:k]
        v = B[i] @ out[:k]
        out[k] = c[i] * u * v
        U.append(u)
        V.append(v)
    return out, U, V, A, B, c


def eval_poly(theta, m, x):
    """Vectorised evaluation of the scheme at the points x."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = _forward(theta, m, x)[0]
    return x * out.sum(axis=0)


def eval_jac(theta, m, x):
    """Values and analytic Jacobian d value / d theta, shape (len(x), n_params)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out, U, V, A, B, c = _forward(theta, m, x)
    val = x * out.sum(axis=0)

    # Reverse sweep: g[k] = d(sum_j out_j) / d out_k
    g = np.ones((m + 2, x.size))
    for k in range(m + 1, -1, -1):
        for i in range(max(k - 1, 0), m):
            g[k] += g[i + 2] * c[i] * (A[i][k] * V[i] + B[i][k] * U[i])

    P = n_params(m)
    J = np.zeros((x.size, P))
    a_off = np.concatenate(([0], np.cumsum([i + 2 for i in range(m)])))
    nA = a_off[-1]
    for i in range(m):
        k = i + 2
        w = x * g[i + 2] * c[i]
        J[:, a_off[i] : a_off[i] + k] = (w * V[i])[:, None] * out[:k].T
        J[:, nA + a_off[i] : nA + a_off[i] + k] = (w * U[i])[:, None] * out[:k].T
        J[:, 2 * nA + i] = x * g[i + 2] * U[i] * V[i]
    return val, J


# ----------------------------------------------------------------------------
# Gauss-Newton fit of the scheme to a given odd polynomial
# ----------------------------------------------------------------------------
def fit_nodes(c, l, u, degenerate_width=1e-2):
    """Extrema of p on [l, u] together with the crossings of p with 1 between them."""
    q = len(c) - 1

    ext = np.concatenate(([l], interior_extrema(c, l, u), [u]))

    xs = np.linspace(l, u, 20001)
    d = odd_poly(c, xs) - 1.0
    idx = np.nonzero(d[:-1] * d[1:] < 0)[0]
    cross = xs[idx] - d[idx] * (xs[idx + 1] - xs[idx]) / (d[idx + 1] - d[idx])
    if len(cross) != len(ext) - 1:  # fall back to midpoints between extrema
        cross = 0.5 * (ext[:-1] + ext[1:])
    return np.sort(np.concatenate([ext, cross]))


def gauss_newton(theta0, m, nodes, target, tol=1e-15, max_iter=2000, lam0=1e-3, weights=None):
    """
    Damped Gauss-Newton (Levenberg-Marquardt) fit of the scheme to `target` at `nodes`.

    Step: argmin ||J p + res||^2 + lam ||p||^2. With lam -> 0 this is plain Gauss-Newton; the damping
    makes it robust when the warm start (previous iteration's theta) is far from the new target.
    lam shrinks after every accepted step and grows after every rejected one.

    weights: optional per-node weights; the minimised residual is weights * (g - target).
    The returned error is always the unweighted max residual.
    """
    theta = theta0.copy()
    P = theta.size
    w = np.ones(len(nodes)) if weights is None else np.asarray(weights, float)

    def residual(th):
        with np.errstate(all="ignore"):
            r = eval_poly(th, m, nodes) - target
        return r if np.all(np.isfinite(r)) else None

    res = residual(theta)
    if res is None:
        raise RuntimeError("Initial guess gives non-finite values at the nodes.")
    F, lam = 0.5 * (w * res) @ (w * res), lam0
    for _ in range(max_iter):
        if np.max(np.abs(res)) < tol:
            break
        _, J = eval_jac(theta, m, nodes)
        accepted = False
        while lam < 1e14:
            aug = np.vstack([w[:, None] * J, np.sqrt(lam) * np.eye(P)])
            step = np.linalg.lstsq(aug, np.concatenate([-w * res, np.zeros(P)]), rcond=None)[0]
            res_t = residual(theta + step)
            if res_t is not None and 0.5 * (w * res_t) @ (w * res_t) < F:
                accepted = True
                break
            lam *= 10
        if not accepted:
            break
        F_t = 0.5 * (w * res_t) @ (w * res_t)
        stalled = F - F_t <= 1e-16 * F
        theta, res, F = theta + step, res_t, F_t
        lam = max(lam * 0.3, 1e-15)
        if stalled:
            break
    return theta, float(np.max(np.abs(res)))


def fit_with_fallback(theta_prev, m, nodes, c, c_prev, accept=1e-15, gn_iters=2000, gn_tol=1e-15, weights=None):
    """
    Direct warm-started fit; if that stalls in a bad local minimum and a previous polynomial exists,
    continuation: fit the blend (1-s)*c_prev + s*c for s = 1/K, ..., 1, each step warm-started from the last.
    """
    target = odd_poly(c, nodes)
    theta, err = gauss_newton(theta_prev, m, nodes, target, gn_tol, gn_iters, weights=weights)
    if err <= accept or c_prev is None:
        return theta, err
    best = (theta, err)
    for K in (10, 40, 160):
        th = theta_prev
        for sc in np.linspace(0, 1, K + 1)[1:]:
            th, e = gauss_newton(th, m, nodes, odd_poly((1 - sc) * c_prev + sc * c, nodes), gn_tol, 300, weights=weights)
        if e < best[1]:
            best = (th, e)
        if e <= accept:
            break
    return best


def pin_fixed_point(theta, m, nodes, c, weights=(1,1,1,1), gn_iters=3000, gn_tol=1e-15):
    """
    For a converged-type step (p ~ 1 on its whole interval) make x = 1 an (almost) exact fixed point of the
    fitted scheme: add x = 1 as an extra node and refit with its weight increased gradually, each stage
    warm-started from the last. The error floor of the whole composition is |g(1) - 1|, and a plain
    least-squares fit leaves that at 1e-7..1e-9. Increasing the weight gradually keeps the fit on the
    interval essentially unchanged; jumping straight to a large weight does not (it stalls much worse).
    """
    nd = np.append(nodes, 1.0)
    target = odd_poly(c, nd)
    for wt in weights:
        w = np.ones(len(nd))
        w[-1] = wt
        theta, _ = gauss_newton(theta, m, nd, target, gn_tol, gn_iters, weights=w)
    err = float(np.max(np.abs(eval_poly(theta, m, nodes) - target[:-1])))
    return theta, err


def initial_guess(m, seed=0):
    """Starting guesses for all values of l for m=3 (degree 17); random otherwise."""
    if m == 3:
        A = [
            [8.1900628402201772e00, -1.1341497882285287e01],
            [5.2695286609514156e00, -1.1355755098785176e01, -8.5575587792319769e00],
            [2.9941920143595552e-01, 3.9368736424870771e-01, -1.3629994852556693e-01, 8.7543792821015454e-01],
        ]
        B = [
            [5.7224200607981572e00, -1.3349570510955273e01],
            [2.4027229678283074e01, -6.9033542160046899e00, -8.1581776809768751e00],
            [3.4280673478743980e-02, -5.9554503046777407e-01, -1.4997426806130245e00, 2.8797501228862719e00],
        ]
        c = [1.2650651343045305e-01, 3.6010993010002004e-03, 1.8975934474204015e00, -1.0, 1.0]
        return pack([np.array(a) for a in A], [np.array(b) for b in B], c)
    print(f"[warning] no built-in initial guess for m={m}; using a random one (may not converge).")
    return np.random.default_rng(seed).standard_normal(n_params(m))

# These are to coefficients that were used from the earlier code. Differs slightly. Probably because it became more accurate
"""co17 = [
    [
        8.1900628402201772e00, -1.1341497882285287e01,
        5.2695286609514156e00, -1.1355755098785176e01, -8.5575587792319769e00,
        2.9941920143595552e-01, 3.9368736424870771e-01, -1.3629994852556693e-01, 8.7543792821015454e-01,
        5.7224200607981572e00, -1.3349570510955273e01,
        2.4027229678283074e01, -6.9033542160046899e00, -8.1581776809768751e00,
        3.4280673478743980e-02, -5.9554503046777407e-01, -1.4997426806130245e00, 2.8797501228862719e00,
        1.2650651343045305e-01, 3.6010993010002004e-03, 1.8975934474204015e00,
        -1.0000000000000000e00, 1.0000000000000000e00,
    ],

    [
        1.1144074461067206e01, -6.9396828758687201e00,
        7.9990669472586662e00, -1.5203810732953235e01, -1.1864829667404932e01,
        4.8726421145938135e00, -5.8118918475287673e00, -4.1495523653680859e00, 3.7670050455967421e00,
        2.7920643049709448e00, -1.2173180152033812e01,
        1.8330227610086052e01, -1.2595988939658326e01, -8.5818359752227487e00,
        5.4573434354147443e-01, -6.5204669665263626e-01, -4.8864015526669952e-01, -3.6134936920750149e-01,
        -6.7015134954514274e-03, 1.3207317978063923e-02, -1.3271454360899630e00,
        -1.0000000000000000e00, 1.0000000000000000e00,
    ],

    [
        1.6299250683234664e01, -1.0916580743419150e01,
        3.7913032455099862e00, -1.2091853643971568e01, -8.4497129859472579e00,
        2.0443860705498897e00, -2.3035547895774053e00, -3.0226551737973932e00, -1.7898310627611815e00,
        5.2460060407784832e00, -1.6298612492955382e01,
        2.4110518765100437e01, -1.0443034129627778e01, -7.0686358901823443e00,
        -7.5667962333679650e00, 4.7505407528854944e00, 2.0656187790382869e00, 1.8643556133192400e00,
        -5.3633761697932086e-03, 2.0793412852751460e-02, -5.2926244835196729e-02,
        -1.0000000000000000e00, 1.0000000000000000e00,
    ],

    [
        1.6031574106559976e01, -1.1147498488902956e01,
        4.2026958242656027e00, -1.1950450600727466e01, -8.1313541853831222e00,
        2.5803801639056574e00, -1.6476836162133901e00, -2.8026154399513548e00, -1.9348904425589695e00,
        5.1105010106341799e00, -1.6447337553819647e01,
        2.4323150509918314e01, -9.7183194558286310e00, -7.3093303735788435e00,
        -7.5994104159191522e00, 4.7819666281751774e00, 1.6589010640978155e00, 1.7980217613035476e00,
        -3.6470399492116688e-03, 1.9560705832703747e-02, -3.5247930092353254e-02,
        -1.0000000000000000e00, 1.0000000000000000e00,
    ],

    [
        1.6031574105251433e01, -1.1147498491607324e01,
        4.2026958242813279e00, -1.1950450600711740e01, -8.1313541839234826e00,
        2.5803801639737705e00, -1.6476836161452770e00, -2.8026154399376000e00, -1.9348904438201786e00,
        5.1105010125580437e00, -1.6447337555126030e01,
        2.4323150509696994e01, -9.7183194560499491e00, -7.3093303735811146e00,
        -7.5994104159694267e00, 4.7819666281249029e00, 1.6589010640876634e00, 1.7980217614247784e00,
        -3.6470377160762795e-03, 1.9560699740462543e-02, -3.5247939607092903e-02,
        -1.0000000000000000e00, 1.0000000000000000e00,
    ],
]"""
# ----------------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------------
def compute(l, iters, m=3, cushion=0.02407327424182761, out_dir="coeffs/", gn_tol=1e-15, gn_iters=4000,
            pin_weights=(1e1, 1e2, 1e3, 1e4), degenerate_width=1e-3):
    q = 2**m
    print(f"l = {l}, iterations = {iters}, q = {q} (degree {2 * q + 1} per iteration)")

    remez_c, bounds = remez_chain(l, iters, q, cushion)

    theta = initial_guess(m)
    thetas = []
    for t in range(iters):
        
        lt, ut = bounds[t]
        c = remez_c[t]
        nodes = fit_nodes(c, lt, ut, degenerate_width)
        
        c_prev = remez_c[t - 1] if t > 0 else None
        theta, err_nodes = fit_with_fallback(theta, m, nodes, c, c_prev, gn_iters=gn_iters, gn_tol=gn_tol)

        # Converged-type step: p ~ 1 on its whole interval (to machine precision), so repeating it must
        # converge to 1. Pin the fixed point x = 1.
        """chk = np.linspace(lt, ut, 2001) if ut - lt > 1e-6 else nodes
        fixed_point_step = lt - 1e-12 <= 1.0 <= ut + 1e-12 and np.max(np.abs(odd_poly(c, chk) - 1.0)) < 1e-8
        if fixed_point_step:
            theta, err_nodes = pin_fixed_point(theta, m, nodes, c, pin_weights, gn_iters)"""
        fixed_point_step = False
        xs = np.linspace(lt, ut, 20001) if ut - lt > 1e-6 else np.linspace(1 - degenerate_width, 1 + degenerate_width, 2001)
        err_dense = np.max(np.abs(eval_poly(theta, m, xs) - odd_poly(c, xs)))
        print(f"  iter {t}: interval [{lt:.3e}, {ut:.6f}]  1-l_next = {1 - odd_poly(c, lt):.3e}  "
              f"fit err (nodes) = {err_nodes:.2e}, (dense) = {err_dense:.2e}"
              + (f", g(1)-1 = {eval_poly(theta, m, [1.0])[0] - 1:+.1e}" if fixed_point_step else ""))
        if err_dense > 1e-8:
            print("    [warning] evaluation-scheme fit is inaccurate for this iteration. Typical for the last, "
                  "nearly converged iterations (target ~ 1 on a tiny interval); use fewer iterations or try more gn_iters.")
        thetas.append(theta.copy())  # theta is also the warm start for iteration t+1

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "coeffs_remez.npy"), remez_c)
    np.save(os.path.join(out_dir, "coeffs_eval.npy"), np.array(thetas))
    np.save(os.path.join(out_dir, "bounds.npy"), bounds)
    print(f"Saved coeffs_remez.npy, coeffs_eval.npy, bounds.npy in {out_dir}")
    return remez_c, np.array(thetas), bounds


def compose_remez(remez_c, x):
    for c in remez_c:
        x = odd_poly(c, x)
    return x


def compose_eval(thetas, m, x):
    for th in thetas:
        x = eval_poly(th, m, x)
    return x

def plot_fit_debug(c, theta, m, l, u, nodes, title=""):
    import matplotlib.pyplot as plt

    # Dense points over the interval
    x = np.linspace(l, u, 5000)

    target = odd_poly(c, x)
    fitted = eval_poly(theta, m, x)


    target_nodes = odd_poly(c, nodes)
    fitted_nodes = eval_poly(theta, m, nodes)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # Polynomial values
    ax1.plot(x, target, label="target Remez polynomial $p(x)$")
    ax1.plot(x, fitted, "--", label="evaluation scheme $g_\\theta(x)$")

    # Nodes used by Gauss-Newton
    ax1.scatter(
        nodes,
        target_nodes,
        marker="o",
        s=45,
        zorder=5,
        label="GN nodes",
    )

    ax1.set_ylabel("value")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Residual
    residual = fitted - target
    residual_nodes = fitted_nodes - target_nodes

    ax2.plot(x, residual, label="$g_\\theta(x)-p(x)$")
    ax2.scatter(
        nodes,
        residual_nodes,
        marker="o",
        s=45,
        zorder=5,
        label="GN-node residual",
    )
    ax2.axhline(0.0, linestyle="--")

    ax2.set_xlabel("x")
    ax2.set_ylabel("residual")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--l", type=float, default=1e-3, help="lower end of the interval [l, 1]")
    ap.add_argument("--iters", type=int, default=4, help="number of composed polynomials")
    ap.add_argument("--m", type=int, default=3, help="evaluation depth; degree per iteration is 2*2**m+1")
    ap.add_argument("--cushion", type=float, default=0.02407327424182761, help="0 disables")
    ap.add_argument("--out", default="coeffs/", help="output directory")
    ap.add_argument("--plot", action="store_true", help="plot the composed approximation")
    a = ap.parse_args()

    remez_c, thetas, bounds = compute(a.l, a.iters, a.m, a.cushion, a.out)

    x = np.logspace(np.log10(a.l), 0, 5000)
    e_remez = np.max(np.abs(1 - compose_remez(remez_c, x)))
    e_eval = np.max(np.abs(1 - compose_eval(thetas, a.m, x)))
    print(f"max |1 - composition| on [{a.l}, 1], each polynomial applied once:  "
          f"Remez {e_remez:.3e},  evaluation scheme {e_eval:.3e}")
    # Repeating the last polynomial (what an iteration does once the saved ones run out) shows convergence.
    rep = 3
    e_remez_r = np.max(np.abs(1 - compose_remez(list(remez_c) + [remez_c[-1]] * rep, x)))
    e_eval_r = np.max(np.abs(1 - compose_eval(list(thetas) + [thetas[-1]] * rep, a.m, x)))
    print(f"  ... with the last polynomial repeated {rep} more times:  Remez {e_remez_r:.3e},  evaluation scheme {e_eval_r:.3e}")

    if a.plot:
        import matplotlib.pyplot as plt

        plt.loglog(x, np.abs(1 - compose_remez(remez_c, x)) + 1e-18, label="Remez")
        plt.loglog(x, np.abs(1 - compose_eval(thetas, a.m, x)) + 1e-18, "--", label="evaluation scheme")
        plt.xlabel("x")
        plt.ylabel("|1 - p(x)|")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
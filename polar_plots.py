#!/usr/bin/env python3
"""
Error-vs-matrix-multiplications plots for the polar factor, comparing
Polar Express, Jordan, Newton-Schulz and the proposed method.

The proposed method now reads the coefficients written by sign_coeffs.py
(coeffs_eval.npy and bounds.npy in COEFFS_DIR), so run that first:

    python sign_coeffs.py --l 1e-3 --iters 5 --out coeffs/
    python polar_plots.py --coeffs coeffs/                 # synthetic spectrum
    python polar_plots.py --coeffs coeffs/ --real          # h3_c_attn_grads.pt
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from itertools import repeat

import sign_coeffs as sc

COEFFS_DIR = "coeffs/"

# ----------------------------------------------------------------------------
# Polar Express coefficients (a, b, c): p(x) = a x + b x^3 + c x^5
# ----------------------------------------------------------------------------
coeffs_listPE = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),  # subsequent coeffs equal this numerically
]
# safety factor for numerical stability (but exclude last polynomial)
coeffs_listPE = [(a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in coeffs_listPE[:-1]] + [coeffs_listPE[-1]]


# ----------------------------------------------------------------------------
# One step of each method (3 / 2 / 3 / m+2 matrix-matrix multiplications)
# ----------------------------------------------------------------------------
DTYPE = torch.float64
def _orient(G):
    """double precision, and transpose so the Gram matrix X X^T is the small one."""
    X = G.to(DTYPE)
    tall = G.size(-2) > G.size(-1)
    return (X.mT if tall else X), tall


def polar_express_step(G, coeffs):
    """X <- a X + b (X X^T) X + c (X X^T)^2 X   (3 multiplications)."""
    a, b, c = coeffs
    X, tall = _orient(G)
    A = X @ X.mT
    X = a * X + (b * A + c * (A @ A)) @ X
    return X.mT if tall else X


def newton_schulz_step(G):
    X, tall = _orient(G)
    X = 1.5 * X - 0.5 * (X @ X.mT) @ X
    return X.mT if tall else X


def jordan_step(G):
    X, tall = _orient(G)
    A = X @ X.mT
    X = 3.4445 * X - 4.7750 * A @ X + 2.0315 * (A @ A) @ X
    return X.mT if tall else X


def proposed_step(G, theta, m):
    """
    One step of the proposed method with the evaluation scheme from sign_coeffs.py
    (matrix version of `sc.eval_poly`), using m + 2 multiplications:

        S = X X^T                                    (1)
        out_0 = I,  out_1 = S
        out_{i+2} = (sum_j A_i[j] out_j)(sum_j B_i[j] out_j)    (m, one product each)
        X <- (sum_k c_k out_k) X                     (1)
    """
    """off = sc.n_params(m) - (m + 2)  # start of c in theta
    theta[off:] /= 1.01"""
    A, B, c = sc.unpack(theta, m)
    X, tall = _orient(G)
    S = X @ X.mT
    eye = torch.eye(S.size(-1), dtype=S.dtype, device=S.device).expand_as(S)
    outs = [eye, S]
    for i in range(m):
        k = i + 2
        U = sum(A[i][j] * outs[j] for j in range(k))
        V = sum(B[i][j] * outs[j] for j in range(k))
        outs.append(U @ V)
    X = sum(c[k] * outs[k] for k in range(m + 2)) @ X
    return X.mT if tall else X


def SP8_coeffs(b):
    f4 = b[8]
    c1 = b[7] / (2 * f4)
    t2 = b[6] / f4 - c1**2
    t1 = b[5] / f4 - c1 * t2
    d0 = 0.25 * (1 - t2**2 + 4 * b[4] / f4 - 4 * c1 * t1)
    e2 = 0.5 * (t2 + 1)
    d2 = 0.5 * (t2 - 1)
    e1 = c1 * d0 + t1 * e2 - b[3] / f4
    d1 = t1 - e1
    f2 = b[2] - f4 * (d0 * e2 + d1 * e1)
    f1 = b[1] - f4 * d0 * e1
    f0 = b[0]

    return c1, d0, d1, d2, e1, e2, f0, f1, f2, f4


def SP8_step(G, coeffs):
    c1, d0, d1, d2, e1, e2, f0, f1, f2, f4 = coeffs
    safe = 1.01

    f0 = f0 / safe
    f1 = f1 / safe
    f2 = f2 / safe
    f4 = f4 / safe

    X, tall = _orient(G)

    S1 = X @ X.mT
    S0 = torch.eye(S1.size(-1), dtype=S1.dtype, device=S1.device).expand_as(S1)
    S2 = S1 @ S1
    S3 = S2 @ (c1 * S1 + S2)
    S4 = (d0 * S0 + d1 * S1 + d2 * S2 + S3) @ (e1 * S1 + e2 * S2 + S3)
    X = (f0 * S0 + f1 * S1 + f2 * S2 + f4 * S4) @ X
    return X.mT if tall else X



def load_proposed(directory=COEFFS_DIR, method="eval"):
    """Load the coefficients; returns (thetas or None, sp8 coeffs or None, m, l0)."""
    remez_c = np.load(os.path.join(directory, "coeffs_remez.npy"), allow_pickle=True)
    bounds = np.load(os.path.join(directory, "bounds.npy"), allow_pickle=True)
    m = int(round(np.log2(remez_c.shape[1] - 1)))  # remez_c has q + 1 = 2**m + 1 columns

    thetas = None
    if method in ("eval", "both"):
        thetas = np.load(os.path.join(directory, "coeffs_eval.npy"), allow_pickle=True)
        if thetas.ndim != 2 or thetas.shape[0] != remez_c.shape[0]:
            raise SystemExit("coeffs_eval.npy is empty or does not match coeffs_remez.npy "
                             "(was sign_coeffs.py run with --method sp8?).")
        assert sc.n_params(m) == thetas.shape[1], "coeffs_eval.npy has an unexpected shape"
        thetas = list(thetas)

    sp8_c = None
    if method in ("sp8", "both"):
        if m != 3:
            raise SystemExit("SP8 is only defined for m = 3 (degree 17).")
        sp8_c = [SP8_coeffs(c) for c in remez_c]

    l_last, u_last = bounds[-1]
    if u_last - l_last > 0.2:
        print(
            "[warning] the last saved polynomial was built for a wide interval "
            f"[{l_last:.3g}, {u_last:.3g}] and is repeated for all later steps, so the error will stall. "
            "Re-run sign_coeffs.py with more --iters."
        )
    return thetas, sp8_c, m, float(bounds[0][0])

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def save_curve(name, xs, ys):
    with open(f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y"])
        for x, y in zip(xs, ys):
            w.writerow([float(x), float(y)])  # float() also converts 0-dim torch tensors


def _curve(plot, mults, vals, label):
    plot(
        mults,
        vals,
        linewidth=3,
        marker="o",
        markersize=7,
        markeredgecolor="white",
        markeredgewidth=1.5,
        label=label,
    )


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False, fontsize=11)
    plt.tight_layout()

def _finite(X):
    return bool(torch.isfinite(X).all())

HYBRID_EVAL = 0   # number of leading steps that use the eval scheme; the rest use SP8

def hybrid_step(G, i, thetas, sp8_c, m, n_eval=None):
    """Steps 0..n_eval-1: eval scheme (thetas). Later steps: SP8 (sp8_c). Last polynomial repeats."""
    n_eval = HYBRID_EVAL if n_eval is None else n_eval
    j = min(i, len(thetas) - 1)
    if i < n_eval:
        return proposed_step(G, thetas[j], m)
    return SP8_step(G, sp8_c[j])


def run_methods(A, polar, thetas, sp8_c, m, k, err_fn):
    """
    Run all methods and return {name: (mults, errors)}.
    k scales the number of steps (30*k multiplications per method).
    """
    res = {}

    # Safety factor might be needed. Do not apply to the last polynomial.
    # The step result is linear in the c's (eval scheme) and in f0, f1, f2, f4 (SP8), so dividing
    # those by 1.01 scales the output of the polynomial by 1/1.01. Work on copies.
    """if thetas is not None:
        thetas = [th.copy() for th in thetas]
        off = sc.n_params(m) - (m + 2)  # start of c in theta
        for th in thetas[:-1]:
            th[off:] /= 1.01"""
    """if sp8_c is not None:
        sp8_c = [tuple(c[:6]) + tuple(c[6+i] / 1.01 for i in range(4)) if i < len(sp8_c) - 1 else tuple(c)
                for i, c in enumerate(sp8_c)]"""

    


    def run(name, X, step, n_steps, mults_per_step):
        errs, mults = [1.0], [0]
        for i in range(n_steps):
            X = step(X, i)
            if not _finite(X):
                print(f"[warning] {name}: non-finite values at step {i + 1}, stopping.")
                break
            mp = mults_per_step(i) if callable(mults_per_step) else mults_per_step
            mults.append(mults[-1] + mp)
            errs.append(err_fn(X - polar))
        res[name] = (mults, errs)

    def run_mach(name, X, step, n_steps, mults_per_step):
        errs, mults = [1.0], [0]
        for i in range(n_steps):
            G = step(X, i)
            if not _finite(G):
                print(f"[warning] {name}: non-finite values at step {i + 1}, stopping.")
                break
            mults.append(mults[-1] + mults_per_step)
            errs.append(err_fn(G - polar))
        res[name] = (mults, errs)

    if HYBRID_EVAL > 0 and thetas is not None and sp8_c is not None:
        run(f"Hybrid ({HYBRID_EVAL} eval + SP8)", A.clone(),
            lambda X, i: hybrid_step(X, i, thetas, sp8_c, m), 6 * k, m + 2)

    run("Polar Express", A.clone(),
        lambda X, i: polar_express_step(X, coeffs_listPE[min(i, len(coeffs_listPE) - 1)]), 10 * k, 3)
    run("Jordan", A.clone(), lambda X, i: jordan_step(X), 10 * k, 3)
    if thetas is not None:
        run("Proposed New Method", A.clone(),
            lambda X, i: proposed_step(X, thetas[min(i, len(thetas) - 1)], m), 6 * k, m + 2)
    if sp8_c is not None:  # SP8 also needs m + 2 = 5 multiplications per step
        run("SP8", A.clone(),
            lambda X, i: SP8_step(X, sp8_c[min(i, len(sp8_c) - 1)]), 6 * k, m + 2)
    if sp8_c is not None:  # SP8 also needs m + 2 = 5 multiplications per step
        run_mach("SP8-mach", A.clone(),
            lambda X, i: MachPolar17(X, i+1), 6 * k, m + 2)
    if N_PE > 0 and sp8_c is not None:
        run(f"{N_PE} PE + SP8", A.clone(),
            lambda X, i: pe_sp8_step(X, i, sp8_c),
            N_PE + 6 * k - SP8_START,
            lambda i: 3 if i < N_PE else m + 2)
    run("Newton-Schulz", A.clone(), lambda X, i: newton_schulz_step(X), 15 * k, 2)
    return res

N_PE = 0          # leading PE steps, then SP8 starting at sp8_c[SP8_START]
SP8_START = 1

def pe_sp8_step(G, i, sp8_c, n_pe=None, sp8_start=None):
    n_pe = N_PE if n_pe is None else n_pe
    sp8_start = SP8_START if sp8_start is None else sp8_start
    if i < n_pe:
        return polar_express_step(G, coeffs_listPE[min(i, len(coeffs_listPE) - 1)])
    j = min(i - n_pe + sp8_start, len(sp8_c) - 1)
    return SP8_step(G, sp8_c[j])


# ----------------------------------------------------------------------------
# Real gradient matrix
# ----------------------------------------------------------------------------
def real_plots(thetas, sp8_c, m, path="h3_c_attn_grads.pt"):
    """Relative Frobenius error of the polar factor of a gradient matrix from a real training run."""
    A = torch.load(path, map_location="cpu")
    if not torch.is_tensor(A):
        raise ValueError("Loaded object is not a tensor - inspect its structure first")
    A = A.double()
    if A.ndim > 2:
        A = A.reshape(-1, A.shape[-1])  # flatten leading batch/head dims

    

    s = torch.linalg.svdvals(A)
    print("shape:", A.shape)
    print("sigma max:", s.max().item(), " sigma min:", s.min().item(), " condition:", (s.max() / s.min()).item())

    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    polar = U @ Vh

    A = A / (A.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    s = torch.linalg.svdvals(A)
    print("Normalized: sigma max:", s.max().item(), " sigma min:", s.min().item())

    bounds = np.load("coeffs/bounds.npy")
    trace(lambda X, i: hybrid_step(X, i, thetas, sp8_c, m, HYBRID_EVAL), A, 4, bounds)


    pn = torch.linalg.matrix_norm(polar, ord="fro")
    res = run_methods(A, polar, thetas, sp8_c, m, 1,
                      lambda D: (torch.linalg.matrix_norm(D, ord="fro") / pn).item())

    plt.figure()
    for name, (mults, errs) in res.items():
        _curve(plt.plot, mults, errs, name)
    plt.xlabel("Matrix-Matrix Multiplications", fontsize=14)
    plt.ylabel("Relative Frobenius Error", fontsize=14)
    plt.ylim(0, 1.05)
    _style(plt.gca())

    print(f"Final error PE:  {res['Polar Express'][1][-1]}")
    if "Proposed New Method" in res:
        print(f"Final error new: {res['Proposed New Method'][1][-1]}")
    if "SP8" in res:
        print(f"Final error SP8: {res['SP8'][1][-1]}")
    for tag, name in (("NS", "Newton-Schulz"), ("PE", "Polar Express"),
                      ("PS", "Proposed New Method"), ("SP8", "SP8")):
        if name in res:
            save_curve(f"gradient_frobenius_error{tag}", *res[name])

# ----------------------------------------------------------------------------
# Synthetic matrix
# ----------------------------------------------------------------------------
def generate_log_spectrum_matrix(m, n, sigma_min=1e-6, sigma_max=1.0):
    """A = U diag(S) V^T with log-spaced singular values in [sigma_min, sigma_max], random orthogonal U, V."""
    k = min(m, n)
    U, _ = torch.linalg.qr(torch.randn(m, k, dtype=torch.double))
    V, _ = torch.linalg.qr(torch.randn(n, k, dtype=torch.double))
    S = torch.logspace(float(np.log10(sigma_min)), float(np.log10(sigma_max)), k, dtype=torch.double)
    return (U * S) @ V.T


def synthetic_plots(thetas, sp8_c, m, l, rows=6912, cols=768, k=2, show_bounds=False):
    """Spectral error for a matrix with log-spaced singular values in [l, 1]."""
    A = generate_log_spectrum_matrix(rows, cols, sigma_min=l, sigma_max=1.0)
    s = torch.linalg.svdvals(A)
    print("shape:", A.shape)
    print("sigma max:", s.max().item(), " sigma min:", s.min().item(), " condition:", (s.max() / s.min()).item())

    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    polar = U @ Vh

    res = run_methods(A, polar, thetas, sp8_c, m, k, lambda D: torch.linalg.matrix_norm(D, ord=2).item())
    print(f"Final error PE:  {res['Polar Express'][1][-1]:.6e}")
    if "Proposed New Method" in res:
        print(f"Final error new: {res['Proposed New Method'][1][-1]:.6e}")
    if "SP8" in res:
        print(f"Final error SP8: {res['SP8'][1][-1]:.6e}")

    plt.figure(figsize=(7, 5))
    for name, (mults, errs) in res.items():
        _curve(plt.semilogy, mults, errs, name)

    if show_bounds:  # theoretical (1-l)^(d^t) bounds, d = degree of one step
        for name, d, n_steps, mps in (
            ("Polar Express", 3, 10, 3),
            ("Newton-Schulz", 2, 15, 2),
            ("Proposed New Method", 2**m + 1, 6, m + 2),
            ("SP8", 2**m + 1, 6, m + 2),
        ):
            if name not in res:
                continue
            mults = [mps * i for i in range(n_steps + 1)]
            bound = [(1 - l) ** (d**i) for i in range(n_steps + 1)]
            plt.semilogy(mults, bound, ":", linewidth=2, label=f"Bound {name}")

    plt.xlabel("Matrix-Matrix Multiplications", fontsize=14)
    plt.ylabel("Spectral Error", fontsize=14)
    plt.ylim(1e-3, 1.05)
    _style(plt.gca())

    plt.savefig("synthetic_log_spectrum_all.pdf", format="pdf", bbox_inches="tight")
    plt.show()

import torch

co = [
    [
        [[9.9207257146298222e-01, -1.4235516131557799e+00],
         [2.1560096288878530e-02, -4.2394388872995729e-02, -3.4762121962591784e-02],
         [5.7824458888809327e-01, 7.6324504283468275e-01, -2.5144805370902978e-01, 1.6390118839860151e+00]],

        [[5.7359037938337600e+00, -1.3338110773060512e+01],
         [2.4029276702105410e+01, -6.9072391698233719e+00, -8.1494614566781305e+00],
         [1.6007261285642154e-02, -5.8894286688481978e-01, -1.5101956722858789e+00, 2.8974433977565046e+00]],

        [1.0257468033619634e+00, 9.7295695731870657e-01, 9.8987345313521957e-01,
         9.8775529118046457e-01, 9.7289904879592692e-01]
    ],

    [
        [[9.4845495878942543e-01, -6.7134996953364723e-01],
         [5.1893070875113842e-03, 1.6543154699996891e-02, -2.6783729222515894e-04],
         [7.1465894324723678e-01, -5.8740764671376872e-02, 6.7815227530155742e-01, 1.6890878813327208e+00]],

        [[4.5139574530344015e+00, -1.3718852058764924e+01],
         [2.4267006621831097e+01, -5.8464426566147178e+00, -8.3437639104953014e+00],
         [1.4524453419726200e-02, -4.5431278721528283e+00, 1.3247454633831042e+00, 2.9290006035401026e+00]],

        [4.8321966487003887e-01, 2.1805317162493094e+00, 2.9156016590579537e-01,
         9.2366431460497211e-01, 4.4757800811598036e-01]
    ],

    [
        [[5.9679407625575898e-01, -7.1406821178372559e-01],
         [1.7706777220322791e-02, 1.1210975381582290e-02, -1.9808472355266053e-04],
         [4.8716513558369573e-01, 1.2372684174028487e-01, 6.1061193648134415e-01, 1.8077965577033801e+00]],

        [[4.4438139469878619e+00, -1.3723989607227232e+01],
         [2.4266192632948240e+01, -5.8427078965542725e+00, -8.3487539677592739e+00],
         [3.9401188727617019e-02, -4.6485444858032423e+00, 1.1430040085032636e+00, 2.8500669589596805e+00]],

        [5.8973369609639603e-01, 2.2865686677622881e+00, 2.2757945783820258e-01,
         9.4630549385570162e-01, 5.1442384824079335e-01]
    ],

    [
        [[2.6492454347899608e-01, -7.3272214044817086e-01],
         [2.6951911234422685e-02, 2.6739125022399342e-03, 2.2446987559776961e-05],
         [7.2906207514580801e-01, 2.0665554796041963e-01, 5.0265752894077564e-01, 1.7318052615231099e+00]],

        [[4.4172730362614576e+00, -1.3723121867526277e+01],
         [2.4260881967348691e+01, -5.8419474133901863e+00, -8.3647225880816247e+00],
         [-1.0737654883562950e-01, -4.6420882158838390e+00, 9.3654164206949531e-01, 2.9214157294423653e+00]],

        [3.0895233883564349e-01, 2.1398828674953645e+00, 2.7467362132676126e-01,
         8.6774486543794682e-01, 4.4391888892478726e-01]
    ],

    [
        [[2.6492454348204864e-01, -7.3272214044511830e-01],
         [2.6951911296004459e-02, 2.6739125638217090e-03, 2.2447255640867362e-05],
         [7.2906207514750609e-01, 2.0665554796211774e-01, 5.0265752894816795e-01, 1.7318052615222017e+00]],

        [[4.4172730362616113e+00, -1.3723121867526125e+01],
         [2.4260881967348588e+01, -5.8419474133902884e+00, -8.3647225880820670e+00],
         [-1.0737654883729919e-01, -4.6420882158855088e+00, 9.3654164206222679e-01, 2.9214157294432583e+00]],

        [3.0895233883393197e-01, 2.1398828674936530e+00, 2.7467362131931067e-01,
         8.6774486543886220e-01, 4.4391888893319376e-01]
    ]
]

@torch.compile
def MachPolar17(G: torch.Tensor, steps: int) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()  # for speed
    if G.size(-2) > G.size(-1):
        X = X.mT  # this reduces FLOPs
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    m = 3

    for i in range(steps):
        coeffs = co[min(i, len(co) - 1)]
        A = coeffs[0]
        B = coeffs[1]
        c = coeffs[2]

        S = X @ X.mT
        eye = torch.eye(S.size(-1), dtype=S.dtype, device=S.device).expand_as(S)
        outs = [eye, S]
        for i in range(m):
            k = i + 2
            U = sum(A[i][j] * outs[j] for j in range(k))
            V = sum(B[i][j] * outs[j] for j in range(k))
            outs.append(U @ V)
        X = sum(c[k] * outs[k] for k in range(m + 2)) @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X



def trace(step, A, n_steps, bounds=None):
    global DTYPE
    Xb = A.bfloat16(); Xr = A.double()
    for i in range(n_steps):
        DTYPE = torch.float64
        Xr_next = step(Xr, i)                    # pure double trajectory
        one_ref = step(Xb.double(), i)           # double step from the bf16 iterate
        DTYPE = torch.bfloat16
        Xb_next = step(Xb, i)
        gen = (Xb_next.double() - one_ref).norm() / one_ref.norm()
        acc = (Xb_next.double() - Xr_next).norm() / Xr_next.norm()
        if _finite(Xb_next):
            sv = torch.linalg.svdvals(Xb_next.double())
            smin, smax = sv.min().item(), sv.max().item()
        else:
            smin = smax = float("nan")
        u_next = bounds[min(i + 1, len(bounds) - 1)][1] if bounds is not None else float("nan")
        print(f"step {i+1}: generated {gen:.2e} accumulated {acc:.2e} "
            f"sigma in [{smin:.3e}, {smax:.3e}]  sigma_max/u_next = {smax/u_next:.4f}")
        if not (smax < 1e3):    # also stops on huge finite values
            break
        Xb, Xr = Xb_next, Xr_next
    DTYPE = torch.float64

# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coeffs", default=COEFFS_DIR, help="directory written by sign_coeffs.py")
    ap.add_argument("--method", choices=["eval", "sp8", "both"], default="eval",
                    help="proposed method: Gauss-Newton fitted scheme (eval), SP8 (sp8, needs m=3), or both")
    ap.add_argument("--real", action="store_true", help="use h3_c_attn_grads.pt instead of the synthetic matrix")
    ap.add_argument("--l", type=float, default=None, help="smallest singular value (default: the l used for the coeffs)")
    ap.add_argument("--bounds", action="store_true", help="also plot the theoretical bounds")
    ap.add_argument("--hybrid", type=int, default=0,
                    help="use the eval scheme for the first N steps, then SP8 (forces --method both)")
    ap.add_argument("--npe", type=int, default=0, help="PE steps before SP8")
    ap.add_argument("--sp8-start", type=int, default=1, help="first SP8 polynomial index after the PE steps")
    a = ap.parse_args()
    global N_PE, SP8_START
    N_PE, SP8_START = a.npe, a.sp8_start
    if a.npe > 0 and a.method == "eval":
        a.method = "both"

    global HYBRID_EVAL
    HYBRID_EVAL = a.hybrid
    if a.hybrid > 0:
        a.method = "both"

    thetas, sp8_c, m, l0 = load_proposed(a.coeffs, a.method)

    n = len(thetas) if thetas is not None else len(sp8_c)
    print(f"Loaded {n} polynomials, m = {m} ({m + 2} multiplications per step), l = {l0}, method = {a.method}")
    

    if a.real:
        real_plots(thetas, sp8_c, m)
    else:
        synthetic_plots(thetas, sp8_c, m, a.l if a.l is not None else l0, show_bounds=a.bounds)
    plt.show()

if __name__ == "__main__":
    main()
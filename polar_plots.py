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
def _orient(G):
    """double precision, and transpose so the Gram matrix X X^T is the small one."""
    X = G.double()
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
        out_{i+2} = c_i (sum_j A_i[j] out_j)(sum_j B_i[j] out_j)    (m, one product each)
        X <- (sum_k out_k) X                         (1)
    """
    A, B, c = sc.unpack(theta, m)
    X, tall = _orient(G)
    S = X @ X.mT
    eye = torch.eye(S.size(-1), dtype=S.dtype, device=S.device).expand_as(S)
    outs = [eye, S]
    for i in range(m):
        k = i + 2
        U = sum(A[i][j] * outs[j] for j in range(k))
        V = sum(B[i][j] * outs[j] for j in range(k))
        outs.append(c[i] * (U @ V))
    X = sum(outs) @ X
    return X.mT if tall else X


def load_proposed(directory=COEFFS_DIR):
    """Load the evaluation coefficients; returns (list of theta, m, l0)."""
    thetas = np.load(os.path.join(directory, "coeffs_eval.npy"), allow_pickle=True)
    bounds = np.load(os.path.join(directory, "bounds.npy"), allow_pickle=True)
    m = int(round(np.sqrt(thetas.shape[1] + 2) - 2))  # P = m^2 + 4m + 2
    assert sc.n_params(m) == thetas.shape[1], "coeffs_eval.npy has an unexpected shape"
    l_last, u_last = bounds[-1]
    if u_last - l_last > 0.2:
        print(
            "[warning] the last saved polynomial was built for a wide interval "
            f"[{l_last:.3g}, {u_last:.3g}] and is repeated for all later steps, so the error will stall. "
            "Re-run sign_coeffs.py with more --iters."
        )
    return list(thetas), m, float(bounds[0][0])


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


def run_methods(A, polar, thetas, m, k, err_fn):
    """
    Run all four methods and return {name: (mults, errors)}.
    k scales the number of steps (30*k multiplications per method).
    """
    res = {}

    # Saftey factor might be needed. Do not apply to the last polynomial.
    """for i in range(3):
        for j in range(3):
            thetas[i][18+j] /= 1.01 **(j)"""

    def run(name, X, step, n_steps, mults_per_step):
        errs, mults = [1.0], [0]
        for i in range(n_steps):
            X = step(X, i)
            mults.append(mults[-1] + mults_per_step)
            errs.append(err_fn(X - polar))
        res[name] = (mults, errs)

    run("Polar Express", A.clone(),
        lambda X, i: polar_express_step(X, coeffs_listPE[min(i, len(coeffs_listPE) - 1)]), 10 * k, 3)
    run("Jordan", A.clone(), lambda X, i: jordan_step(X), 10 * k, 3)
    run("Proposed New Method", A.clone(),
        lambda X, i: proposed_step(X, thetas[min(i, len(thetas) - 1)], m), 6 * k, m + 2)
    run("Newton-Schulz", A.clone(), lambda X, i: newton_schulz_step(X), 15 * k, 2)
    return res


# ----------------------------------------------------------------------------
# Real gradient matrix
# ----------------------------------------------------------------------------
def real_plots(thetas, m, path="h3_c_attn_grads.pt"):
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

    pn = torch.linalg.matrix_norm(polar, ord="fro")
    res = run_methods(A, polar, thetas, m, 1, lambda D: (torch.linalg.matrix_norm(D, ord=2)).item())

    plt.figure()
    for name, (mults, errs) in res.items():
        _curve(plt.plot, mults, errs, name)
    plt.xlabel("Matrix-Matrix Multiplications", fontsize=14)
    plt.ylabel("Relative Frobenius Error", fontsize=14)
    plt.ylim(0, 1.05)
    _style(plt.gca())

    print(f"Final error PE:  {res['Polar Express'][1][-1]}")
    print(f"Final error new: {res['Proposed New Method'][1][-1]}")
    for tag, name in (("NS", "Newton-Schulz"), ("PE", "Polar Express"), ("PS", "Proposed New Method")):
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


def synthetic_plots(thetas, m, l, rows=6912, cols=768, k=2, show_bounds=False):
    """Spectral error for a matrix with log-spaced singular values in [l, 1]."""
    A = generate_log_spectrum_matrix(rows, cols, sigma_min=l, sigma_max=1.0)
    s = torch.linalg.svdvals(A)
    print("shape:", A.shape)
    print("sigma max:", s.max().item(), " sigma min:", s.min().item(), " condition:", (s.max() / s.min()).item())

    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    polar = U @ Vh

    res = run_methods(A, polar, thetas, m, k, lambda D: torch.linalg.matrix_norm(D, ord=2).item())
    print(f"Final error PE:  {res['Polar Express'][1][-1]:.6e}")
    print(f"Final error new: {res['Proposed New Method'][1][-1]:.6e}")

    plt.figure(figsize=(7, 5))
    for name, (mults, errs) in res.items():
        _curve(plt.semilogy, mults, errs, name)

    if show_bounds:  # theoretical (1-l)^(d^t) bounds, d = degree of one step
        for name, d, n_steps, mps in (
            ("Polar Express", 3, 10, 3),
            ("Newton-Schulz", 2, 15, 2),
            ("Proposed New Method", 2**m + 1, 6, m + 2),
        ):
            mults = [mps * i for i in range(n_steps + 1)]
            bound = [(1 - l) ** (d**i) for i in range(n_steps + 1)]
            plt.semilogy(mults, bound, ":", linewidth=2, label=f"Bound {name}")

    plt.xlabel("Matrix-Matrix Multiplications", fontsize=14)
    plt.ylabel("Spectral Error", fontsize=14)
    plt.ylim(1e-3, 1.05)
    _style(plt.gca())

    plt.savefig("synthetic_log_spectrum_all.pdf", format="pdf", bbox_inches="tight")
    plt.show()


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coeffs", default=COEFFS_DIR, help="directory written by sign_coeffs.py")
    ap.add_argument("--real", action="store_true", help="use h3_c_attn_grads.pt instead of the synthetic matrix")
    ap.add_argument("--l", type=float, default=None, help="smallest singular value (default: the l used for the coeffs)")
    ap.add_argument("--bounds", action="store_true", help="also plot the theoretical bounds")
    a = ap.parse_args()

    thetas, m, l0 = load_proposed(a.coeffs)
    print(f"Loaded {len(thetas)} polynomials, m = {m} ({m + 2} multiplications per step), l = {l0}")

    if a.real:
        real_plots(thetas, m)
    else:
        synthetic_plots(thetas, m, a.l if a.l is not None else l0, show_bounds=a.bounds)
    plt.show()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Load the .npy files written by sign_coeffs.py and plot the polynomial at every step.

    python plot_coeffs.py --dir coeffs/                 # show windows, write csv/
    python plot_coeffs.py --dir coeffs/ --save plots/   # write PNGs instead (no window)
    python plot_coeffs.py --csv data/ --every 4         # CSVs into data/, keep every 4th point

Figures:
  steps.png        one column per step: Remez polynomial vs evaluation scheme on [0, u_t]
                   (top), and |scheme - Remez| on [l_t, u_t] (bottom)
  composition.png  |1 - composition| on [l_0, 1] after every step, for both versions

CSV files (all with header "x,y"):
  step{t}_remez.csv, step{t}_scheme.csv, step{t}_diff.csv
  comp_remez_{k}.csv, comp_scheme_{k}.csv     (k = number of steps composed)
  bounds.csv                                  (columns: step,l,u)
"""

import argparse
import os

import numpy as np

import sign_coeffs as sc


def load(directory):
    remez_c = np.load(os.path.join(directory, "coeffs_remez.npy"), allow_pickle=True)
    thetas = np.load(os.path.join(directory, "coeffs_eval.npy"), allow_pickle=True)
    bounds = np.load(os.path.join(directory, "bounds.npy"), allow_pickle=True)
    # P = m^2 + 4m + 2  ->  m = sqrt(P + 2) - 2
    m = int(round(np.sqrt(thetas.shape[1] + 2) - 2))
    assert sc.n_params(m) == thetas.shape[1], "coeffs_eval.npy has an unexpected number of columns"
    return remez_c, thetas, bounds, m


def save_xy(directory, name, x, y, every=1):
    """Write one curve as an x,y CSV at full precision (optionally thinned)."""
    os.makedirs(directory, exist_ok=True)
    data = np.column_stack([np.asarray(x, dtype=float)[::every], np.asarray(y, dtype=float)[::every]])
    np.savetxt(os.path.join(directory, f"{name}.csv"), data,
               delimiter=",", header="x,y", comments="", fmt="%.17g")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="coeffs/", help="directory with the .npy files")
    ap.add_argument("--save", metavar="DIR", help="save PNGs here instead of opening windows")
    ap.add_argument("--csv", metavar="DIR", default="csv/", help="directory for the CSV data (default: csv/)")
    ap.add_argument("--every", type=int, default=1, help="keep every N-th point in the CSVs (smaller TikZ files)")
    a = ap.parse_args()

    if a.save:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    remez_c, thetas, bounds, m = load(a.dir)
    T = len(remez_c)
    print(f"Loaded {T} steps, q = {2**m}, m = {m}")

    # interval endpoints, useful for shading the [l, u] region in TikZ
    os.makedirs(a.csv, exist_ok=True)
    np.savetxt(os.path.join(a.csv, "bounds.csv"),
               np.column_stack([np.arange(T), [b[0] for b in bounds], [b[1] for b in bounds]]),
               delimiter=",", header="step,l,u", comments="", fmt=["%d", "%.17g", "%.17g"])

    # ---- per-step figure ---------------------------------------------------
    fig, axes = plt.subplots(2, T, figsize=(3.6 * T, 6), squeeze=False)
    for t in range(T):
        l, u = bounds[t]
        x = np.linspace(0, u, 4000)
        y_remez = sc.odd_poly(remez_c[t], x)
        y_scheme = sc.eval_poly(thetas[t], m, x)
        save_xy(a.csv, f"step{t}_remez", x, y_remez, a.every)
        save_xy(a.csv, f"step{t}_scheme", x, y_scheme, a.every)

        ax = axes[0, t]
        ax.plot(x, y_remez, label="Remez")
        ax.plot(x, y_scheme, "--", label="evaluation scheme")
        ax.axhline(1, color="gray", lw=0.6)
        ax.axvspan(l, u, color="tab:green", alpha=0.08)
        ax.set_ylim(-0.5, 2.5)
        ax.set_title(f"step {t}:  [{l:.3g}, {u:.4g}]")
        if t == 0:
            ax.legend(fontsize=8)

        if u - l > 1e-6:
            xi = np.linspace(l, u, 4000)
        else:  # degenerate interval: look just around 1
            xi = np.linspace(1 - 1e-2, 1 + 1e-2, 2000)
        diff = np.abs(sc.eval_poly(thetas[t], m, xi) - sc.odd_poly(remez_c[t], xi))
        save_xy(a.csv, f"step{t}_diff", xi, diff + 1e-18, a.every)

        axes[1, t].semilogy(xi, diff + 1e-18)
        axes[1, t].set_xlabel("x")
        axes[1, t].set_title(f"max diff {diff.max():.1e}", fontsize=9)
        print(f"  step {t}: max |scheme - Remez| on its interval = {diff.max():.2e}")
    axes[1, 0].set_ylabel("|scheme - Remez|")
    fig.tight_layout()

    # ---- composition figure ------------------------------------------------
    l0 = bounds[0][0]
    xs = np.logspace(np.log10(l0), 0, 4000)
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))
    xr, xe = xs.copy(), xs.copy()
    for t in range(T):
        xr = sc.odd_poly(remez_c[t], xr)
        xe = sc.eval_poly(thetas[t], m, xe)
        err_r = np.abs(1 - xr) + 1e-18
        err_e = np.abs(1 - xe) + 1e-18
        save_xy(a.csv, f"comp_remez_{t + 1}", xs, err_r, a.every)
        save_xy(a.csv, f"comp_scheme_{t + 1}", xs, err_e, a.every)

        (line,) = ax2.loglog(xs, err_r, label=f"Remez, {t + 1} steps")
        ax2.loglog(xs, err_e, "--", color=line.get_color())
    ax2.set_xlabel("x")
    ax2.set_ylabel("|1 - composition(x)|   (solid: Remez, dashed: scheme)")
    ax2.legend(fontsize=8)
    fig2.tight_layout()

    print(f"Saved CSV data in {a.csv}")

    if a.save:
        os.makedirs(a.save, exist_ok=True)
        fig.savefig(os.path.join(a.save, "steps.png"), dpi=130)
        fig2.savefig(os.path.join(a.save, "composition.png"), dpi=130)
        print(f"Saved steps.png and composition.png in {a.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
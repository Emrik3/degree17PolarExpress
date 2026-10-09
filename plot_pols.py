#!/usr/bin/env python3
"""
Load the .npy files written by sign_coeffs.py and plot the polynomial at every step.

    python plot_coeffs.py --dir coeffs/                 # show windows, write csv/
    python plot_coeffs.py --dir coeffs/ --method sp8    # SP8 instead of the fitted scheme
    python plot_coeffs.py --dir coeffs/ --method both   # Remez, fitted scheme and SP8
    python plot_coeffs.py --dir coeffs/ --save plots/   # write PNGs instead (no window)
    python plot_coeffs.py --csv data/ --every 4         # CSVs into data/, keep every 4th point

Figures:
  steps.png        one column per step: Remez polynomial vs the chosen scheme(s) on [0, u_t]
                   (top), and |scheme - Remez| on [l_t, u_t] (bottom)
  composition.png  |1 - composition| on [l_0, 1] after every step, for all versions

CSV files (all with header "x,y"):
  step{t}_remez.csv, step{t}_scheme.csv, step{t}_diff.csv           (method eval)
  step{t}_sp8.csv,   step{t}_sp8_diff.csv                           (method sp8)
  comp_remez_{k}.csv, comp_scheme_{k}.csv, comp_sp8_{k}.csv         (k = number of steps composed)
  bounds.csv                                                        (columns: step,l,u)

SP8 is built directly from the Remez coefficients (SP8_coeffs) and only exists for m = 3 (degree 17).
"""

import argparse
import os

import numpy as np

import sign_coeffs as sc


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


def SP8_step(x, coeffs):
    c1, d0, d1, d2, e1, e2, f0, f1, f2, f4 = coeffs

    S0 = 1
    S1 = x*x
    S2 = S1 * S1
    S3 = S2 * (c1 * S1 + S2)
    S4 = (d0 * S0 + d1 * S1 + d2 * S2 + S3) * (e1 * S1 + e2 * S2 + S3)
    X = (f0 * S0 + f1 * S1 + f2 * S2 + f4 * S4) * x
    return X

def compose_sp8(sp8_c, x):
    x = np.asarray(x, dtype=float)
    for c in sp8_c:
        x = SP8_step(x, c)
    return x



def load(directory, need_eval):
    remez_c = np.load(os.path.join(directory, "coeffs_remez.npy"), allow_pickle=True)
    bounds = np.load(os.path.join(directory, "bounds.npy"), allow_pickle=True)
    # q = 2**m, and remez_c has q+1 columns
    m = int(round(np.log2(remez_c.shape[1] - 1)))

    thetas = None
    if need_eval:
        path = os.path.join(directory, "coeffs_eval.npy")
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found; run sign_coeffs.py with --method eval/both, "
                             f"or use --method sp8 here.")
        thetas = np.load(path, allow_pickle=True)
        if thetas.ndim != 2 or thetas.shape[0] != remez_c.shape[0]:
            raise SystemExit("coeffs_eval.npy is empty or does not match coeffs_remez.npy "
                             "(was sign_coeffs.py run with --method sp8?).")
        assert sc.n_params(m) == thetas.shape[1], "coeffs_eval.npy has an unexpected number of columns"
    return remez_c, thetas, bounds, m


def save_xy(directory, name, x, y, every=1):
    """Write one curve as an x,y CSV at full precision (optionally thinned)."""
    os.makedirs(directory, exist_ok=True)
    data = np.column_stack([np.asarray(x, dtype=float)[::every], np.asarray(y, dtype=float)[::every]])
    np.savetxt(os.path.join(directory, f"{name}.csv"), data,
               delimiter=",", header="x,y", comments="", fmt="%.17g")


def build_schemes(method, remez_c, thetas, m):
    """
    List of schemes to compare against Remez. Each is a dict with
      label, tag (CSV name), diff (CSV name of the difference, with {t}), style, color,
      step(t, x) -> the t-th polynomial of that scheme evaluated at x.
    """
    schemes = []
    if method in ("eval", "both"):
        schemes.append(dict(
            label="evaluation scheme", tag="scheme", diff="step{t}_diff", style="--", color="tab:orange",
            step=lambda t, x: sc.eval_poly(thetas[t], m, x),
        ))
    if method in ("sp8", "both"):
        if m != 3:
            raise SystemExit("SP8 is only defined for m = 3 (degree 17).")
        sp8_c = [SP8_coeffs(c) for c in remez_c]
        #print(sp8_c)
        schemes.append(dict(
            label="SP8", tag="sp8", diff="step{t}_sp8_diff", style=":", color="tab:red",
            step=lambda t, x: SP8_step(np.asarray(x, dtype=float), sp8_c[t]),
        ))
    return schemes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="coeffs/", help="directory with the .npy files")
    ap.add_argument("--method", choices=["eval", "sp8", "both"], default="eval",
                    help="scheme to compare against Remez: Gauss-Newton fitted scheme (eval), "
                         "SP8 (sp8, needs m=3), or both")
    ap.add_argument("--save", metavar="DIR", help="save PNGs here instead of opening windows")
    ap.add_argument("--csv", metavar="DIR", default="csv/", help="directory for the CSV data (default: csv/)")
    ap.add_argument("--every", type=int, default=1, help="keep every N-th point in the CSVs (smaller TikZ files)")
    a = ap.parse_args()

    if a.save:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    remez_c, thetas, bounds, m = load(a.dir, need_eval=a.method in ("eval", "both"))
    schemes = build_schemes(a.method, remez_c, thetas, m)
    T = len(remez_c)
    print(f"Loaded {T} steps, q = {2**m}, m = {m}, method = {a.method}")

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
        save_xy(a.csv, f"step{t}_remez", x, y_remez, a.every)

        ax = axes[0, t]
        ax.plot(x, y_remez, color="tab:blue", label="Remez")

        if u - l > 1e-6:
            xi = np.linspace(l, u, 4000)
        else:  # degenerate interval: look just around 1
            xi = np.linspace(1 - 1e-2, 1 + 1e-2, 2000)
        ref_i = sc.odd_poly(remez_c[t], xi)

        title_diffs = []
        for s in schemes:
            with np.errstate(all="ignore"):
                y = s["step"](t, x)
                diff = np.abs(s["step"](t, xi) - ref_i)
            save_xy(a.csv, f"step{t}_{s['tag']}", x, y, a.every)
            save_xy(a.csv, s["diff"].format(t=t), xi, diff + 1e-18, a.every)

            ax.plot(x, y, s["style"], color=s["color"], label=s["label"])
            axes[1, t].semilogy(xi, diff + 1e-18, s["style"], color=s["color"], label=s["label"])
            dmax = np.nanmax(diff) if np.any(np.isfinite(diff)) else np.nan
            title_diffs.append(f"{s['label']} {dmax:.1e}")
            print(f"  step {t}: max |{s['label']} - Remez| on its interval = {dmax:.2e}")

        ax.axhline(1, color="gray", lw=0.6)
        ax.axvspan(l, u, color="tab:green", alpha=0.08)
        ax.set_ylim(-0.5, 2.5)
        ax.set_title(f"step {t}:  [{l:.3g}, {u:.4g}]")
        if t == 0:
            ax.legend(fontsize=8)

        axes[1, t].set_xlabel("x")
        axes[1, t].set_title("max diff " + ", ".join(title_diffs), fontsize=8)
    axes[1, 0].set_ylabel("|scheme - Remez|")
    if len(schemes) > 1:
        axes[1, 0].legend(fontsize=8)
    fig.tight_layout()

    # ---- composition figure ------------------------------------------------
    l0 = bounds[0][0]
    xs = np.logspace(np.log10(l0), 0, 4000)
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))
    xr = xs.copy()
    xcur = [xs.copy() for _ in schemes]
    for t in range(T):
        xr = sc.odd_poly(remez_c[t], xr)
        err_r = np.abs(1 - xr) + 1e-18
        save_xy(a.csv, f"comp_remez_{t + 1}", xs, err_r, a.every)
        (line,) = ax2.loglog(xs, err_r, label=f"Remez, {t + 1} steps")

        for i, s in enumerate(schemes):
            with np.errstate(all="ignore"):
                xcur[i] = s["step"](t, xcur[i])
            err = np.abs(1 - xcur[i]) + 1e-18
            save_xy(a.csv, f"comp_{s['tag']}_{t + 1}", xs, err, a.every)
            ax2.loglog(xs, err, s["style"], color=line.get_color())
    ax2.set_xlabel("x")
    styles = ", ".join(f"{st}: {lab}" for st, lab in
                       [("solid", "Remez")] + [({"--": "dashed", ":": "dotted"}[s["style"]], s["label"]) for s in schemes])
    ax2.set_ylabel(f"|1 - composition(x)|\n({styles})", fontsize=9)
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
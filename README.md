# higherOrderPolarExpress

Research code for building and evaluating high-order polynomial iterations for the polar decomposition / matrix sign function, with comparisons against Polar Express, Jordan and Newton–Schulz.

The method approximates `sign(x)` on `[l, 1]` by a **composition of odd polynomials** `p_0, p_1, ..., p_{T-1}`. Each `p_t` has degree `2q+1` (default `q = 8`, degree 17) and is applied to the matrix as `X ← p_t(X)`, with the polynomials chosen so that the singular values are driven to 1.

Each polynomial is computed in two stages:

1. **Remez** gives the best odd polynomial of degree `2q+1` approximating 1 on the current interval `[l_t, u_t]`. The next interval is `l_{t+1} = p_t(l_t)`, `u_{t+1} = 2 - l_{t+1}`.
2. **Fitting an evaluation scheme.** A polynomial of degree 17 can be evaluated with far fewer matrix products than Horner or the monomial form. The scheme's parameters are fitted to the Remez polynomial by (damped) Gauss–Newton. Each step is warm-started from the previous step's parameters.

## Repository contents

| File | Purpose |
|---|---|
| `sign_coeffs.py` | Computes the Remez polynomials, fits the evaluation scheme and writes the coefficient arrays to `coeffs/`. |
| `polar_plots.py` | Compares Polar Express, Jordan, Newton–Schulz and the proposed method on a synthetic or a real matrix; writes curves and figures. |
| `plot_pols.py` | Plots every saved polynomial (Remez vs. evaluation scheme) and the composition error. |
| `muon.py` | Compiled PyTorch implementation (`MachPolar17`) using fixed coefficients. |
| `coeffs/` | Saved coefficient and interval files (`coeffs_remez.npy`, `coeffs_eval.npy`, `bounds.npy`). |
| `h3_c_attn_grads.pt` | Example real gradient tensor used by `polar_plots.py --real`. |

## Quick start

```bash
# 1. generate coefficients for singular values in [1e-3, 1]
python sign_coeffs.py --l 1e-3 --iters 5 --out coeffs/

# 2. look at the fitted polynomials
python plot_pols.py --dir coeffs/

# 3. compare against the other methods (synthetic matrix)
python polar_plots.py --coeffs coeffs/

# 4. compare on a real gradient matrix
python polar_plots.py --coeffs coeffs/ --real
```

## Commands and flags

### `sign_coeffs.py` — generate coefficients

```bash
python sign_coeffs.py [--l L] [--iters N] [--m M] [--cushion C] [--out DIR] [--plot]
```

| Flag | Default | Description |
|---|---|---|
| `--l` | `1e-3` | Lower end of the interval `[l, 1]` that the singular values are assumed to lie in (smallest singular value after normalising the largest to 1). |
| `--iters` | `4` | Number of polynomials in the composition (number of iterations of the method). |
| `--m` | `3` | Depth of the evaluation scheme. The polynomial degree per iteration is `2·2^m + 1` (so `m = 3` gives degree 17) and the scheme uses `m + 2` matrix multiplications per iteration. The built-in initial guess only exists for `m = 3`; for other values a random start is used, which may not converge. |
| `--cushion` | `0.02407327424182761` | Remez is run on `[max(l, cushion·u), u]` and the polynomial is then rescaled so that `p(l) + p(u) = 2`. This trades a slightly worse worst case for better behaviour near the lower end. Use `0` to disable. |
| `--out` | `coeffs/` | Output directory. It is created if it does not exist. |
| `--plot` | off | After computing, plot `|1 − composition(x)|` on `[l, 1]` for the Remez polynomials and the fitted evaluation scheme. |

### `plot_pols.py` — inspect saved coefficients

```bash
python plot_pols.py [--dir DIR] [--save OUTDIR]
```

| Flag | Default | Description |
|---|---|---|
| `--dir` | `coeffs/` | Directory containing `coeffs_remez.npy`, `coeffs_eval.npy` and `bounds.npy`. |
| `--save` | none | Write `steps.png` and `composition.png` into this directory instead of opening windows. |

### `polar_plots.py` — compare against other methods

```bash
python polar_plots.py [--coeffs DIR] [--real] [--l L] [--bounds]
```

| Flag | Default | Description |
|---|---|---|
| `--coeffs` | `coeffs/` | Directory written by `sign_coeffs.py`. |
| `--real` | off | Use `h3_c_attn_grads.pt` instead of a synthetic matrix, and plot the relative Frobenius error. |
| `--l` | the `l` the coefficients were built for | Smallest singular value of the synthetic matrix. |
| `--bounds` | off | Synthetic run only: also plot the theoretical bounds `(1 − l)^(d^t)` (`d` is the degree of one step). |

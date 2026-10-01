"""
Avalanche statistics of the isolated mean-field population (Fig. 1C-D; Methods,
"Avalanche durations and sizes").

Isolated regions (G = 0) at a = 0.967 / 0.973 / 0.979 (super / critical / subcritical),
sigma = 0.677, 60 s at 20 kHz, first 5000 samples discarded. Avalanches are excursions of
|R(t)| below its median; duration T = number of time steps, size S = area between the
threshold and the signal over the event.

Exponents (critical regime): tau (sizes) and alpha (durations) by maximum likelihood
(powerlaw package, lower cutoff at the median), gamma from a log-log fit of S vs T;
log-likelihood-ratio test power law vs exponential for each regime.
Output: data/derived/avalanche_statistics.pkl
"""
from __future__ import annotations

import pickle
from pathlib import Path
import sys
import warnings

import numpy as np
from scipy.stats import linregress

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # src/ (paths.py, utilities)
import paths  # noqa: E402
import simulators_numba as sim  # noqa: E402
from supplementary_utils import find_events  # noqa: E402  (same event detection as all avalanche panels)

A_REG = (0.967, 0.973, 0.979)
SIGMA = 0.677


def main():
    import powerlaw
    W = np.zeros((3, 3))
    R = sim.simulate_mf(W, 0.0, a=np.array(A_REG), sigma=SIGMA, T_ms=60000.0, stride=1, seed=0)[5000:]
    out = {"a": A_REG, "trace": R[185000:205000].astype(np.float32), "S": [], "T": [], "fit": []}
    for j, a in enumerate(A_REG):
        x = np.abs(R[:, j].astype(float))
        th = float(np.median(x))
        st, en = find_events(x, th, dir=-1)
        T = np.array([e - s + 1 for s, e in zip(st, en)], float)
        S = np.array([np.trapezoid(th - x[s:e + 1]) for s, e in zip(st, en)])
        out["S"].append(S); out["T"].append(T)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Sp = S[S > 0]
            fs = powerlaw.Fit(Sp, xmin=np.percentile(Sp, 50), verbose=False)
            ft = powerlaw.Fit(T, xmin=np.percentile(T, 50), discrete=True, verbose=False)
            llr, p = fs.distribution_compare("power_law", "exponential")
        k = T > 5
        gamma = linregress(np.log10(T[k]), np.log10(np.clip(S[k], 1e-12, None))).slope
        tau, alpha = fs.power_law.alpha, ft.power_law.alpha
        out["fit"].append({"tau": tau, "alpha": alpha, "gamma": gamma, "llr_pl_vs_exp": llr, "p": p})
        print(f"a={a}: tau={tau:.2f} alpha={alpha:.2f} gamma={gamma:.2f} "
              f"(alpha-1)/(tau-1)={(alpha - 1) / (tau - 1):.2f}  power law vs exponential: R={llr:.1f}, p={p:.2g}")
    pickle.dump(out, open(paths.DERIVED / "avalanche_statistics.pkl", "wb"))


if __name__ == "__main__":
    main()

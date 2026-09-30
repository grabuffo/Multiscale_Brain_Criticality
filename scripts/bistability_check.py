"""
Bistable window of the isolated mean-field population (Methods, "Identifying the Bistable Regime")? Deterministic (eta = 0) Eq. (4), many random initial conditions per (a, sigma),
as in Suppl. Fig. S1C (250 initial conditions, R0 ~ U(0.01, 1), Psi0 ~ U(0, 2 pi)).

Reports, per (a, sigma): spread across initial conditions of the time-averaged R
over the LAST 10 s (true multistability), and the published statistic
(variance-to-mean ratio of the whole-run average) for short and long runs, to test
whether slow convergence near the bifurcation mimics bistability.
Output: data/derived/bistability.pkl
"""
from pathlib import Path
import pickle
import sys

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # src/ (paths.py, utilities)
import paths  # noqa: E402

OUT = paths.DERIVED / "bistability.pkl"
J, OMEGA, DT = 1.25, 1.0, 0.05
N_IC = 250


@njit(cache=True)
def run(a, sigma, R0, P0, n_steps, tail):
    """Heun, eta = 0. Returns whole-run mean R and last-`tail`-steps mean R, per node."""
    R = R0.copy(); P = P0.copy()
    n = R.size
    s_all = np.zeros(n); s_tail = np.zeros(n)
    s2 = sigma * sigma
    for t in range(n_steps):
        for i in range(n):
            r = R[i]; p = P[i]
            f1 = 0.5 * r * (J * (1 - r * r) - s2) - 0.5 * a[i] * (1 - r * r) * np.cos(p)
            g1 = OMEGA + a[i] * (1 + r * r) * np.sin(p) / (2 * r)
            rp = max(r + DT * f1, 1e-6); pp = p + DT * g1
            f2 = 0.5 * rp * (J * (1 - rp * rp) - s2) - 0.5 * a[i] * (1 - rp * rp) * np.cos(pp)
            g2 = OMEGA + a[i] * (1 + rp * rp) * np.sin(pp) / (2 * rp)
            R[i] = max(r + 0.5 * DT * (f1 + f2), 1e-6); P[i] = p + 0.5 * DT * (g1 + g2)
            s_all[i] += R[i]
            if t >= n_steps - tail:
                s_tail[i] += R[i]
    return s_all / n_steps, s_tail / tail


def main():
    rng = np.random.default_rng(0)
    out = {}
    for sigma, A in ((0.677, np.round(np.arange(0.955, 0.9901, 0.0005), 4)),
                     *[(s, np.round(np.arange(0.80, 1.1001, 0.004), 4)) for s in (0.45, 0.55, 0.65, 0.75, 0.85, 0.95)]):
        a = np.repeat(A, N_IC)
        R0 = rng.uniform(0.01, 1.0, a.size); P0 = rng.uniform(0, 2 * np.pi, a.size)
        res = {}
        for T_ms in (2000.0, 60000.0):
            m_all, m_tail = run(a, sigma, R0, P0, int(T_ms / DT), int(10000.0 / DT) if T_ms > 10000 else int(T_ms / DT))
            m_all = m_all.reshape(len(A), N_IC); m_tail = m_tail.reshape(len(A), N_IC)
            res[T_ms] = {"vmr_whole": m_all.var(1) / m_all.mean(1), "spread_tail": m_tail.std(1)}
        out[sigma] = {"a": A, **res}
        s60 = res[60000.0]["spread_tail"]; bi = A[s60 > 1e-3]
        k2 = np.argmax(res[2000.0]["vmr_whole"])
        print(f"sigma={sigma}: true bistable a (60 s, last 10 s spread>1e-3): "
              f"{(bi.min(), bi.max()) if bi.size else 'none'} | 2 s-run VMR peak at a={A[k2]} "
              f"({res[2000.0]['vmr_whole'][k2]:.1e})", flush=True)
    pickle.dump(out, open(OUT, "wb"))


if __name__ == "__main__":
    main()

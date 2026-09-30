"""
Augment paper/revision/derived_data/C1_indicators.pkl with the same three
long-range-correlation indicators (AC1, alpha_PSD, H_DFA) computed on the
*global signal* GS(t) = (1/N) sum_n z(R_n(t)) of the critical-regime
Allen-connectome simulations, one number per G.

This complements the per-region computation already cached in C1_indicators.pkl
so Suppl. Fig. S7 can compare the global-signal story with the regional one.
"""

from __future__ import annotations

import pickle
import time
import warnings
from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # revision/ (paths.py)
import paths  # noqa: E402
_sys.path.insert(0, str(paths.SRC))  # original src/ (functions.py, Utils.py, ...)
import sys

import nolds
import numpy as np
from scipy.signal import welch
from scipy.stats import linregress

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import revision_utils as ru  # noqa: E402

PROJECT_ROOT = paths.REPO
DERIVED = paths.DERIVED

SIM_DIR = paths.SIM_ROOT / "Gscan_connectome_crit"

TLEN = 5000
FS = 2000
NPERSEG = 4096
PSD_FMIN = 1.0
PSD_FMAX = 60.0
DFA_NVALS = np.unique(np.logspace(np.log10(32), np.log10(16000), 20).astype(int))


def _alpha_PSD(sig: np.ndarray) -> float:
    freqs, psd = welch(sig, fs=FS, nperseg=NPERSEG)
    mask = (freqs >= PSD_FMIN) & (freqs <= PSD_FMAX) & (psd > 0)
    if mask.sum() < 5:
        return np.nan
    s = linregress(np.log10(freqs[mask]), np.log10(psd[mask]))
    return -float(s.slope)


def _dfa_exponent(sig: np.ndarray) -> float:
    if sig.size < int(DFA_NVALS.max() * 4):
        return np.nan
    try:
        return float(
            nolds.dfa(
                sig.astype(np.float64),
                nvals=DFA_NVALS,
                overlap=True,
                order=1,
                fit_exp="poly",
            )
        )
    except Exception:
        return np.nan


def main() -> None:
    n_G = ru.N_G

    cache_path = DERIVED / "C1_indicators.pkl"
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    AC1_GS = np.full(n_G, np.nan)
    alpha_PSD_GS = np.full(n_G, np.nan)
    H_DFA_GS = np.full(n_G, np.nan)

    t_start = time.time()
    for iG in range(n_G):
        with open(SIM_DIR / f"data_G{iG}.pkl", "rb") as f:
            data = pickle.load(f)
        R = data["R"]
        GS = ru.global_signal(R, tlen=TLEN).astype(np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            AC1_GS[iG] = ru.AC1(GS, n=20)
            alpha_PSD_GS[iG] = _alpha_PSD(GS)
            H_DFA_GS[iG] = _dfa_exponent(GS)
        print(
            f"  G[{iG:2d}]={ru.G_GRID[iG]:.4f}  "
            f"AC1(GS)={AC1_GS[iG]:+.3f}  "
            f"alpha(GS)={alpha_PSD_GS[iG]:.3f}  "
            f"H_DFA(GS)={H_DFA_GS[iG]:.3f}"
        )

    print(f"elapsed: {time.time() - t_start:.1f}s")

    cache["AC1_GS"] = AC1_GS
    cache["alpha_PSD_GS"] = alpha_PSD_GS
    cache["H_DFA_GS"] = H_DFA_GS

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"Updated {cache_path}")


if __name__ == "__main__":
    main()

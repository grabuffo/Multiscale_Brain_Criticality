"""
Compute independent corroborating indicators of long-range temporal correlations
across all 60 cortical ROIs and the full G sweep for the critical regime
(Reviewer C, point 1).

Primary criticality argument lives in the multiscale link itself: AC1 of R_n(t)
peaks along the (a, sigma) ridge that also exhibits bistability (Suppl. Fig. S1C),
the hallmark of a hybrid-type (HT) transition in the underlying spiking network
[Buendia et al. 2021], where critical neuronal avalanches are independently
documented. This rules out Hopf and SNIC alternatives that would also produce
AC1 peaks without criticality.

To corroborate the AC1 reading at the whole-brain level we add two independent
indicators of long-range temporal correlations that are reliable in the
mean-field reduction (unlike avalanche scaling exponents, which require the
microscopic spiking structure and are reported in the spiking-network
companion paper):

  - alpha_PSD : 1/f spectral exponent, negative slope of log10(PSD) vs log10(f)
                in the 1-60 Hz band of R_n(t). Canonical critical value alpha ~ 1.
  - H_DFA    : detrended fluctuation analysis exponent of R_n(t) (Peng et al.
               1994). Reports long-range temporal correlations directly in the
               time domain, complementary to the spectral estimate. Computed
               with nolds (polynomial detrending, fit_exp='poly').
               Theoretical reference values:
                 white noise              H = 0.5
                 1/f noise (alpha = 1)    H ~ 1.0
                 brown / random walk      H = 1.5

Cache: paper/revision/derived_data/C1_indicators.pkl
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
DERIVED.mkdir(parents=True, exist_ok=True)

SIM_DIR = paths.SIM_ROOT / "Gscan_connectome_crit"

TLEN = 5000          # burn-in samples
FS = 2000            # Hz (R was downsampled by 10 from dt=0.05 ms)
NPERSEG = 4096       # Welch segment length (~2 s)
PSD_FMIN = 1.0
PSD_FMAX = 60.0

# DFA window sizes (in samples): log-spaced from 16 ms to 8 s, covering 2.7 decades.
# Largest window stays under 5% of usable record length (~355k samples), safely
# within the conventional 10% upper bound.
DFA_NVALS = np.unique(np.logspace(np.log10(32), np.log10(16000), 20).astype(int))


def _alpha_PSD(sig: np.ndarray) -> float:
    """Negative slope of log10(PSD) vs log10(f) in [PSD_FMIN, PSD_FMAX] (1/f exponent)."""
    freqs, psd = welch(sig, fs=FS, nperseg=NPERSEG)
    mask = (freqs >= PSD_FMIN) & (freqs <= PSD_FMAX) & (psd > 0)
    if mask.sum() < 5:
        return np.nan
    x = np.log10(freqs[mask])
    y = np.log10(psd[mask])
    s = linregress(x, y)
    return -float(s.slope)


def _dfa_exponent(sig: np.ndarray) -> float:
    """Hurst-style scaling exponent from DFA (nolds, polynomial detrending order 1).

    Robust to slow trends in R_n(t); reports long-range temporal correlations of
    the fluctuations around the local trend.
    """
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
    n_R = ru.N_REGIONS

    alpha_PSD = np.full((n_R, n_G), np.nan)
    H_DFA = np.full((n_R, n_G), np.nan)

    t_start = time.time()
    for iG in range(n_G):
        with open(SIM_DIR / f"data_G{iG}.pkl", "rb") as f:
            data = pickle.load(f)
        R = data["R"][TLEN:, :ru.N_REGIONS]
        t_G = time.time()
        for j in range(n_R):
            sig = R[:, j].astype(np.float64, copy=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                alpha_PSD[j, iG] = _alpha_PSD(sig)
                H_DFA[j, iG] = _dfa_exponent(sig)
        print(
            f"  G[{iG:2d}]={ru.G_GRID[iG]:.4f}  "
            f"alpha median={np.nanmedian(alpha_PSD[:, iG]):.3f}  "
            f"H_DFA median={np.nanmedian(H_DFA[:, iG]):.3f}  "
            f"({time.time() - t_G:.1f}s)"
        )

    print(f"elapsed: {time.time() - t_start:.1f}s")
    out = {
        "G_grid": ru.G_GRID,
        "alpha_PSD": alpha_PSD,
        "H_DFA": H_DFA,
        "DFA_nvals": DFA_NVALS,
        "PSD_band": (PSD_FMIN, PSD_FMAX),
        "fs": FS,
    }
    with open(DERIVED / "C1_indicators.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"Saved {DERIVED / 'C1_indicators.pkl'}")


if __name__ == "__main__":
    main()

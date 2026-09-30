"""
Compute global-signal AC1 and H_DFA per G for the structural-null surrogates
(S1 weight-shuffled full; S2 sparse Allen backbone; S3 sparse random) used in
Suppl. Fig. S6.

Purpose: distinguish locking from criticality at high G. Where AC1(GS) rebounds
to high values at high G in the random/sparse surrogates (S6 panel A), the DFA
exponent of GS(t) should:
  - sit near H = 1 if the dynamics are critical (1/f long-range correlations),
  - move toward H = 1.5 (random-walk-like) if the dynamics are locked / saturated.

Saves: paper/revision/derived_data/C3_surrogate_GS_DFA.pkl
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

# Surrogate names -> simulation directories
SIM_DIRS = {
    "allen": paths.SIM_ROOT / "Gscan_connectome_crit",
    "S1":    paths.SIM_ROOT / "Gscan_surrogate_S1",
    "S2":    paths.SIM_ROOT / "Gscan_surrogate_S2",
    "S3":    paths.SIM_ROOT / "Gscan_surrogate_S3",
}

TLEN = 5000
FS = 2000
DFA_NVALS = np.unique(np.logspace(np.log10(32), np.log10(16000), 20).astype(int))


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
    out = {"G_grid": ru.G_GRID}

    for graph, sim_dir in SIM_DIRS.items():
        AC1_GS = np.full(n_G, np.nan)
        H_DFA_GS = np.full(n_G, np.nan)
        t_g = time.time()
        for iG in range(n_G):
            pkl = sim_dir / f"data_G{iG}.pkl"
            if not pkl.exists():
                print(f"  [{graph}] G[{iG}] missing: {pkl}")
                continue
            with open(pkl, "rb") as f:
                data = pickle.load(f)
            R = data["R"]
            GS = ru.global_signal(R, tlen=TLEN).astype(np.float64)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                AC1_GS[iG] = ru.AC1(GS, n=20)
                H_DFA_GS[iG] = _dfa_exponent(GS)
            print(
                f"  [{graph}] G[{iG:2d}]={ru.G_GRID[iG]:.4f}  "
                f"AC1(GS)={AC1_GS[iG]:+.3f}  H_DFA(GS)={H_DFA_GS[iG]:.3f}"
            )
        print(f"  [{graph}] elapsed {time.time() - t_g:.0f}s")
        out[graph] = {"AC1_GS": AC1_GS, "H_DFA_GS": H_DFA_GS}

    out_path = DERIVED / "C3_surrogate_GS_DFA.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()

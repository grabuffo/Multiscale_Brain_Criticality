"""
Compute the comparison metrics for the Allen vs. S1 (weight-shuffled) experiment
addressing reviewer point C.3.

For one folder of `data_G%d.pkl` files (critical regime, 16 G values), compute:
  - AC1_local : per-region AC1, shape (60, 16)
  - AC1_GS    : AC1 of the global signal, shape (16,)
  - aval_GS   : avalanche durations of GS at each G; dict {iG: list[int]}
  - W         : the weights matrix used for that simulation (loaded separately)
  - instr     : in-strength derived from W, shape (60,)

Saves a single .pkl per graph at:
    paper/revision/derived_data/C3_metrics_{graph}.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # revision/ (paths.py)
import paths  # noqa: E402
_sys.path.insert(0, str(paths.SRC))  # original src/ (functions.py, Utils.py, ...)

import numpy as np
from scipy import stats

import revision_utils as ru

# Pull go_edge from src/ to keep edge-timeseries definition identical
HERE = Path(__file__).resolve().parent
FIRST_SUB_SRC = paths.SRC
sys.path.insert(0, str(FIRST_SUB_SRC))
from functions import go_edge  # noqa: E402


PROJECT_ROOT = paths.REPO
DERIVED = paths.DERIVED
EMP_DIR = paths.EMP_DIR
DERIVED.mkdir(parents=True, exist_ok=True)

# BOLD ROI selection (matches notebook 7, cell-21): drop 5 ROIs per hemisphere
# (10 of 60), leaving 50 ROIs aligned with empirical FC/dFC.
NREG_HALF = 30
BOLD_REMOVE_ROI = [19, 17, 14, 21, 22,
                   19 + NREG_HALF, 17 + NREG_HALF, 14 + NREG_HALF,
                   21 + NREG_HALF, 22 + NREG_HALF]
BOLD_REROI = np.delete(np.arange(60), BOLD_REMOVE_ROI)  # 50 ROIs
BOLD_TLEN = 20   # discard first 20 BOLD samples (matches notebook 7)
DFC_T = 160      # = BOLD length (180) − BOLD_TLEN (20)

GRAPHS = {
    # graph_name -> (sim_folder, weights_file_in_DERIVED)
    "allen": (
        paths.SIM_ROOT / "Gscan_connectome_crit",
        "W_allen_60.npy",
    ),
    "S1": (
        paths.SIM_ROOT / "Gscan_surrogate_S1",
        "W_surrogate_S1_shuffle.npy",
    ),
    "S2": (
        paths.SIM_ROOT / "Gscan_surrogate_S2",
        "W_surrogate_S2_sparse_allen.npy",
    ),
    "S3": (
        paths.SIM_ROOT / "Gscan_surrogate_S3",
        "W_surrogate_S3_sparse_random.npy",
    ),
    # B.4: structured heterogeneity — Allen weights, gradient a in [0.974, 0.978] mapped to in-strength rank.
    "B4grad": (
        paths.SIM_ROOT / "Gscan_B4_subcrit_gradient",
        "W_allen_60.npy",
    ),
    # Homogeneous subcritical baseline (a=0.979) for B.4 comparison.
    "allen_sub": (
        paths.SIM_ROOT / "Gscan_connectome_sub",
        "W_allen_60.npy",
    ),
    # B.4 v2: hubs at critical boundary, periphery deeper subcritical, a in [0.973, 0.978].
    "B4grad_v2": (
        paths.SIM_ROOT / "Gscan_B4_subcrit_gradient_v2",
        "W_allen_60.npy",
    ),
    # B.4 random control: same a values as B4grad_v2, but assigned to regions in
    # random order (seed=2026); breaks the in-strength -> distance-from-criticality
    # mapping while preserving the per-region a distribution.
    "B4grad_random": (
        paths.SIM_ROOT / "Gscan_B4_subcrit_gradient_random",
        "W_allen_60.npy",
    ),
}

CORTEX_60 = np.arange(ru.N_REGIONS)
TLEN = 5000


def _load_empirical():
    """Load empirical FC/dFC matrices keyed by subject id."""
    with open(EMP_DIR / "FCs_emp.pkl", "rb") as f:
        FCs_emp = pickle.load(f)
    with open(EMP_DIR / "dFCs_emp.pkl", "rb") as f:
        dFCs_emp = pickle.load(f)
    return FCs_emp, dFCs_emp


def compute_metrics_for_graph(graph: str) -> dict:
    sim_dir, weights_file = GRAPHS[graph]
    if not sim_dir.exists():
        raise FileNotFoundError(f"Simulation folder missing: {sim_dir}")
    W = np.load(DERIVED / weights_file)
    instr = ru.in_strength_from_weights(W)

    AC1_local = np.zeros((ru.N_REGIONS, ru.N_G))
    AC1_GS = np.zeros(ru.N_G)
    aval_GS: dict[int, list] = {}

    # BOLD-level summaries
    FC_sim = np.zeros((ru.N_G, len(BOLD_REROI), len(BOLD_REROI)))
    dFC_sim = np.zeros((ru.N_G, DFC_T, DFC_T))

    t0 = time.time()
    for iG in range(ru.N_G):
        pkl = sim_dir / f"data_G{iG}.pkl"
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        R = data["R"]
        BOLD = data["BOLD_R"]

        for j in range(ru.N_REGIONS):
            AC1_local[j, iG] = ru.AC1(R[TLEN:, j], n=20)
        AC1_GS[iG] = ru.global_AC1(R, tlen=TLEN)
        aval_GS[iG] = ru.global_avalanche_sizes(R, tlen=TLEN)

        # BOLD FC and dFC at the 50-ROI subset
        bold_sub = BOLD[BOLD_TLEN:, BOLD_REROI]   # shape (160, 50)
        FC_sim[iG] = np.corrcoef(bold_sub.T)
        dFC_sim[iG] = np.corrcoef(go_edge(bold_sub))

        print(f"  [{graph}] G[{iG:2d}]={ru.G_GRID[iG]:.4f}  AC1_GS={AC1_GS[iG]:+.3f}  "
              f"AC1_loc.mean={AC1_local[:, iG].mean():+.3f}  events={len(aval_GS[iG])}")

    # Compare to all 53 empirical subjects
    FCs_emp, dFCs_emp = _load_empirical()
    n_subj = len(FCs_emp)
    ut_FC = np.triu_indices(len(BOLD_REROI), k=1)
    ut_dFC = np.triu_indices(DFC_T, k=1)
    corr_FC = np.zeros((n_subj, ru.N_G))
    KS_FC = np.zeros((n_subj, ru.N_G))
    KS_dFC = np.zeros((n_subj, ru.N_G))
    for s_idx in range(n_subj):
        s_key = str(s_idx + 1)
        fc_e = FCs_emp[s_key][ut_FC]
        dfc_e = dFCs_emp[s_key][ut_dFC]
        for iG in range(ru.N_G):
            fc_s = FC_sim[iG][ut_FC]
            dfc_s = dFC_sim[iG][ut_dFC]
            corr_FC[s_idx, iG] = np.corrcoef(fc_e, fc_s)[0, 1]
            KS_FC[s_idx, iG] = stats.ks_2samp(fc_e, fc_s).statistic
            KS_dFC[s_idx, iG] = stats.ks_2samp(dfc_e, dfc_s).statistic

    print(f"  [{graph}] elapsed: {time.time() - t0:.1f}s")
    return {
        "graph": graph,
        "G_grid": ru.G_GRID,
        "W": W,
        "instr": instr,
        "AC1_local": AC1_local,
        "AC1_GS": AC1_GS,
        "aval_GS": aval_GS,
        "FC_sim": FC_sim,
        "dFC_sim": dFC_sim,
        "corr_FC": corr_FC,        # (n_subj, n_G)
        "KS_FC": KS_FC,            # (n_subj, n_G)
        "KS_dFC": KS_dFC,          # (n_subj, n_G)
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("graph", choices=list(GRAPHS))
    args = ap.parse_args()

    out = compute_metrics_for_graph(args.graph)
    out_path = DERIVED / f"C3_metrics_{args.graph}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

"""
Lean whole-brain simulator for the C.3 + B.2 structural-null experiment.

Differences from notebooks/5)Whole_brain_simulations.ipynb:
  - Accepts a custom 60x60 weights matrix (Allen surrogate); rest of the connectivity
    metadata (centres, tract_lengths, region_labels) is taken from the original.
  - Saves a LEAN pkl per G value: only `R` and `BOLD_R` (drops PSI, INPUT, BOLD_PSI),
    cutting per-file size from ~480 MB to ~173 MB.
  - Other parameters identical to the published critical-regime run.

Usage:
  python run_C3B2_simulate.py <surrogate_name> <out_dir>
    surrogate_name : one of {allen, S1, S3}; selects the weights .npy in derived_data/.
    out_dir        : absolute path where data_G%d.pkl files will be written.
"""

from __future__ import annotations

import argparse
import gc
import math
import pickle
import time
from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # revision/ (paths.py)
import paths  # noqa: E402
_sys.path.insert(0, str(paths.SRC))  # original src/ (functions.py, Utils.py, ...)
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import os as _os
_os.chdir(paths.SRC)  # src/functions_simulator.py reads ../data relative to the working directory
import functions_simulator as fun  # noqa: E402

PROJECT_ROOT = paths.REPO
DERIVED = paths.DERIVED
ALLEN_ZIP = paths.ALLEN_ZIP

# ===== Physical / numerical parameters (matches notebook 5 — critical regime) =====
A_CRIT = 0.973
SIGMA = 0.6725
OMEGA = 1.0
J = 1.25
NOISE_AMP = 0.0001          # `eta` parameter, weak noise on Psi
NOISE_SEED = 0
DT = 0.05                   # ms
SIM_LENGTH = 180000         # ms — full published run
G_GRID = np.linspace(0.0, 0.21, 16)
V_SPEED = np.inf            # null delays

CORTICAL_LABELS = np.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93,
    94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
])

WEIGHTS_FILES = {
    "allen": "W_allen_60.npy",
    "S1":    "W_surrogate_S1_shuffle.npy",
    "S2":    "W_surrogate_S2_sparse_allen.npy",
    "S3":    "W_surrogate_S3_sparse_random.npy",
}


def build_60region_connectivity(weights_60: np.ndarray):
    """Build a TVB Connectivity object on the 60 cortical ROIs with custom weights.

    Centres are dummy (irrelevant when V_SPEED == inf, which removes conduction delays).
    Tract-lengths are taken from the 60-region subset of the original Allen connectome.
    """
    Allen148 = fun.set_up_connectivity148(str(ALLEN_ZIP), V_SPEED)
    tracts_60 = Allen148.tract_lengths[np.ix_(CORTICAL_LABELS, CORTICAL_LABELS)]
    centres_60 = np.zeros((weights_60.shape[0], 3), dtype=np.float64)

    from tvb.simulator.lab import connectivity
    SC = connectivity.Connectivity(
        weights=weights_60.astype(np.float64),
        tract_lengths=tracts_60,
        speed=np.asarray(V_SPEED),
        centres=centres_60,
        region_labels=np.asarray([str(i) for i in range(weights_60.shape[0])], dtype="<U128"),
    )
    SC.configure()
    return SC


def run_one_G(SC, G: float, sim_length: int, sample_stride_R: int = 10,
              a_value=None) -> dict:
    """Run one simulation at coupling G; return dict with R + BOLD_R only.

    a_value : scalar or array-like. If None (default), uses A_CRIT (homogeneous critical).
              If array of length n_regions, uses per-region a (heterogeneous).
    """
    if a_value is None:
        a_value = A_CRIT
    t_b, B_r, B_psi, t, r, psi, _coupl = fun.simulate_network_conduct_seed_wholeBOLD_R(
        SC, J, a_value, SIGMA, OMEGA, G, NOISE_AMP, NOISE_SEED, V_SPEED, sim_length, DT,
        verbose=False,
    )
    return {
        "R":      r[::sample_stride_R, :].astype(np.float64),
        "BOLD_R": B_r.astype(np.float64),
        "G":      float(G),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("surrogate", choices=list(WEIGHTS_FILES))
    ap.add_argument("out_dir")
    ap.add_argument("--sim-length", type=int, default=SIM_LENGTH,
                    help="Simulation length in ms (default: full published run).")
    ap.add_argument("--g-indices", type=str, default=None,
                    help="Comma-separated G indices to run. Default: all 16.")
    ap.add_argument("--a-array", type=str, default=None,
                    help="Path to .npy file with per-region a values. "
                         "If omitted, uses homogeneous critical a = %g." % A_CRIT)
    args = ap.parse_args()

    weights_path = DERIVED / WEIGHTS_FILES[args.surrogate]
    W = np.load(weights_path)
    print(f"[setup] surrogate={args.surrogate}  weights={weights_path.name}  shape={W.shape}")

    a_value = A_CRIT
    a_label = "homogeneous"
    if args.a_array:
        a_value = np.load(args.a_array)
        a_label = f"heterogeneous (from {Path(args.a_array).name})"
        print(f"[setup] a array shape={a_value.shape}  range=[{a_value.min():.4f}, {a_value.max():.4f}]")

    SC = build_60region_connectivity(W)
    print(f"[setup] connectivity built: {SC.number_of_regions} regions, a: {a_label}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] writing to {out_dir}")

    # Persist run parameters next to the data
    with open(out_dir / "Fixed_parameters_hybrid.pkl", "wb") as f:
        pickle.dump({
            "J": J,
            "a": a_value if isinstance(a_value, float) else np.asarray(a_value),
            "sigma": SIGMA, "omega": OMEGA, "v": V_SPEED,
            "dt": DT, "sim_length": args.sim_length, "N": NOISE_AMP,
            "noise_seed": NOISE_SEED, "Gs": G_GRID,
            "surrogate": args.surrogate,
            "a_array_path": args.a_array,
        }, f)

    g_indices = list(range(len(G_GRID)))
    if args.g_indices:
        g_indices = [int(s) for s in args.g_indices.split(",")]

    for iG in g_indices:
        out_pkl = out_dir / f"data_G{iG}.pkl"
        if out_pkl.exists():
            print(f"[skip ] G[{iG:2d}]={G_GRID[iG]:.4f}  exists, skipping")
            continue
        t0 = time.time()
        result = run_one_G(SC, G_GRID[iG], args.sim_length, a_value=a_value)
        with open(out_pkl, "wb") as f:
            pickle.dump(result, f)
        print(f"[done ] G[{iG:2d}]={G_GRID[iG]:.4f}  R.shape={result['R'].shape}  "
              f"BOLD_R.shape={result['BOLD_R'].shape}  ({time.time()-t0:.1f}s)")
        del result
        gc.collect()


if __name__ == "__main__":
    main()

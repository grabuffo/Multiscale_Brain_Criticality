"""
Generate the structural-null surrogates of the Allen cortical connectome (Suppl. Fig. S5).

S1 (weight shuffle): take the 60x60 cortical Allen connectome,
random-permute the off-diagonal entries (diagonal kept at zero). Preserves the
exact weight distribution and density; destroys all topology.

S3 (sparse i.i.d. binarized): generate a 60x60 binary matrix
where each off-diagonal entry is 1 with probability p equal to the empirical
edge density of the Allen connectome. Edge density is defined as
fraction(non-zero off-diagonal entries) in the original. Diagonal kept at zero.

Both surrogates are reproducible (fixed numpy seed). The original Allen weights
are used for the simulator; we only modify the .weights array.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # src/ (paths.py, utilities)
import paths  # noqa: E402
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import os as _os
_os.chdir(paths.SRC)  # src/functions_simulator.py reads ../data relative to the working directory
import functions_simulator as fun  # noqa: E402

PROJECT_ROOT = paths.REPO
DERIVED = paths.DERIVED
DERIVED.mkdir(parents=True, exist_ok=True)

CORTICAL_LABELS = np.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93,
    94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
])


def load_allen_cortical_60(conn_zip: Path) -> np.ndarray:
    """Return the 60x60 cortical sub-connectome with diagonal zeroed and max-normalized."""
    Allen_SC_148 = fun.set_up_connectivity148(str(conn_zip), np.inf)
    W = Allen_SC_148.weights[np.ix_(CORTICAL_LABELS, CORTICAL_LABELS)].copy()
    np.fill_diagonal(W, 0.0)
    if W.max() > 0:
        W /= W.max()
    return W


def shuffle_offdiag(W: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """S1 — shuffle the off-diagonal entries of W. Preserves exact value distribution."""
    n = W.shape[0]
    off_mask = ~np.eye(n, dtype=bool)
    vals = W[off_mask].copy()
    rng.shuffle(vals)
    W_shuf = np.zeros_like(W)
    W_shuf[off_mask] = vals
    return W_shuf


def sparse_allen_backbone(W: np.ndarray, frac: float = 0.15) -> np.ndarray:
    """S2 — sparsified Allen: keep only the top `frac` of off-diagonal entries by weight.

    Preserves Allen topology (which connections exist) restricted to the strongest backbone,
    with the original weights. Discards the long tail of weak edges.
    """
    n = W.shape[0]
    off_mask = ~np.eye(n, dtype=bool)
    n_off = int(off_mask.sum())
    n_keep = int(round(frac * n_off))
    off_values = W[off_mask].copy()
    sorted_desc = np.argsort(off_values)[::-1]
    keep_idx = np.zeros(n_off, dtype=bool)
    keep_idx[sorted_desc[:n_keep]] = True
    out_off = np.where(keep_idx, off_values, 0.0)
    W_out = np.zeros_like(W)
    W_out[off_mask] = out_off
    return W_out


def sparse_iid_random(W: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """S3 — random sparse: i.i.d. random topology at density `frac`, weights drawn from
    Allen's top-`frac` distribution.

    Density matched to S2 (sparse_allen_backbone) and weight distribution matched to S2,
    so any difference between S2 and S3 is attributable to topology alone.
    """
    n = W.shape[0]
    off_mask = ~np.eye(n, dtype=bool)
    n_off = int(off_mask.sum())
    n_keep = int(round(frac * n_off))

    # Take the top `n_keep` Allen weights as the value pool
    off_values_sorted_desc = np.sort(W[off_mask])[::-1]
    top_weights = off_values_sorted_desc[:n_keep].copy()
    rng.shuffle(top_weights)

    # Choose `n_keep` random off-diagonal positions
    off_positions = np.argwhere(off_mask)
    chosen_idx = rng.choice(len(off_positions), size=n_keep, replace=False)

    W_out = np.zeros_like(W)
    for k, idx in enumerate(chosen_idx):
        i, j = off_positions[idx]
        W_out[i, j] = top_weights[k]
    return W_out


def report(name: str, W: np.ndarray) -> None:
    n = W.shape[0]
    off = ~np.eye(n, dtype=bool)
    nz = W[off] > 0
    print(f"--- {name} ---")
    print(f"  shape           : {W.shape}")
    print(f"  off-diag count  : {off.sum()}")
    print(f"  nonzero entries : {int(nz.sum())}  (density {nz.mean():.4f})")
    print(f"  weight stats    : min={W[off].min():.4f}  mean={W[off].mean():.4f}  max={W[off].max():.4f}")
    print(f"  in-strength     : min={W.sum(axis=0).min():.3f}  max={W.sum(axis=0).max():.3f}  mean={W.sum(axis=0).mean():.3f}")


SPARSE_FRAC = 0.15  # brain-realistic density target for S2 / S3


def main() -> None:
    conn_zip = paths.ALLEN_ZIP
    W_allen = load_allen_cortical_60(conn_zip)
    np.save(DERIVED / "W_allen_60.npy", W_allen)

    rng = np.random.default_rng(seed=20260509)
    W_S1 = shuffle_offdiag(W_allen, rng)
    np.save(DERIVED / "W_surrogate_S1_shuffle.npy", W_S1)

    W_S2 = sparse_allen_backbone(W_allen, frac=SPARSE_FRAC)
    np.save(DERIVED / "W_surrogate_S2_sparse_allen.npy", W_S2)

    W_S3 = sparse_iid_random(W_allen, frac=SPARSE_FRAC, rng=rng)
    np.save(DERIVED / "W_surrogate_S3_sparse_random.npy", W_S3)

    print("Saved:")
    for name in ("W_allen_60",
                 "W_surrogate_S1_shuffle",
                 "W_surrogate_S2_sparse_allen",
                 "W_surrogate_S3_sparse_random"):
        print(f"  {DERIVED / (name + '.npy')}")
    print()
    report("Allen (cortical 60)",                 W_allen)
    report("S1 — weight shuffle (full density)",  W_S1)
    report(f"S2 — sparse Allen backbone (top {SPARSE_FRAC*100:.0f}%)",     W_S2)
    report(f"S3 — sparse random (i.i.d., {SPARSE_FRAC*100:.0f}% density)", W_S3)


if __name__ == "__main__":
    main()

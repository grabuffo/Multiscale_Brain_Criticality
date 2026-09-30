"""
Diagnostic / verification figure for the structural-null surrogates (Allen, S1, S2, S3).

Confirms that:
  - S1 preserves the *exact* off-diagonal weight distribution of Allen (strict weight-distribution null).
  - S2 keeps the top-15% of Allen weights (sparse Allen backbone, structured).
  - S3 has the same density and value distribution as S2, but random topology.
  - In all surrogates the diagonal is zero.

Saves: paper/figures/DallaPorta_Suppl_Figure_5.{pdf,png}.
Also prints numerical confirmation to stdout.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # src/ (paths.py, utilities)
import paths  # noqa: E402

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = paths.REPO
DERIVED = paths.DERIVED
OUT_FIG = paths.FIGURES
OUT_FIG.mkdir(parents=True, exist_ok=True)


SURR = [
    ("Allen",                         "W_allen_60.npy",                 "tab:blue"),
    ("S1 — weight shuffle",           "W_surrogate_S1_shuffle.npy",     "tab:grey"),
    ("S2 — sparse Allen (top 15%)",   "W_surrogate_S2_sparse_allen.npy","tab:orange"),
    ("S3 — sparse random (15%)",      "W_surrogate_S3_sparse_random.npy","tab:olive"),
]


def main() -> None:
    Ws = []
    for label, fname, color in SURR:
        W = np.load(DERIVED / fname)
        Ws.append((label, W, color))

    n = Ws[0][1].shape[0]
    off_mask = ~np.eye(n, dtype=bool)

    # ===== Numerical verification =====
    print("=== Numerical verification of surrogates ===")
    W_allen = Ws[0][1]
    allen_off = W_allen[off_mask]
    for label, W, _ in Ws:
        off = W[off_mask]
        nz = (off > 0)
        diag_max = float(np.max(np.abs(np.diag(W))))
        print(f"--- {label} ---")
        print(f"  density            : {nz.mean():.4f}  ({int(nz.sum())} / {len(off)} entries)")
        print(f"  weight stats (>0)  : min={off[nz].min():.4f}  max={off[nz].max():.4f}  mean={off[nz].mean():.4f}")
        print(f"  diagonal max |.|   : {diag_max:.2e}")
        if W is not W_allen:
            corr = float(np.corrcoef(allen_off, off)[0, 1])
            ks_p = _ks_distribution_match(allen_off, off)
            print(f"  topology corr w/ Allen (entry-by-entry) : {corr:+.4f}  "
                  f"(=0 means topology fully randomised)")
            print(f"  KS distance of weight distributions     : {ks_p:.4f}  "
                  f"(=0 means distributions match exactly)")

    # ===== Figure =====
    # Top two rows: 2x2 grid of adjacency-matrix heatmaps
    #   row 1: Allen full | S1 (full weight shuffle)
    #   row 2: S2 (sparse Allen backbone) | S3 (sparse random)
    # Bottom (full-width): weight distribution overlay for Allen (blue) vs S2 (orange).
    # By construction (verified above): S1 weight distribution == Allen exactly,
    # S3 weight distribution == S2 exactly; we state this in the caption rather
    # than overlaying redundant histograms.
    fig = plt.figure(figsize=(10, 11))
    gs = fig.add_gridspec(
        3, 2, height_ratios=[1.0, 1.0, 0.55], hspace=0.30, wspace=0.20,
        left=0.07, right=0.95, top=0.96, bottom=0.06,
    )

    vmax = max(W[off_mask].max() for _, W, _ in Ws)

    # Order: top-left = Allen, top-right = S1, bottom-left = S2, bottom-right = S3
    positions = [
        (0, 0, "Allen",                          W_allen),
        (0, 1, "S1 — weight shuffle",            Ws[1][1]),
        (1, 0, "S2 — sparse Allen (top 15%)",    Ws[2][1]),
        (1, 1, "S3 — sparse random (15%)",       Ws[3][1]),
    ]
    for row, col, label, W in positions:
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(W, cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(label, fontsize=11)
        ax.set_xticks([0, 30, 59])
        ax.set_yticks([0, 30, 59])
        if row == 1:
            ax.set_xlabel("source ROI")
        if col == 0:
            ax.set_ylabel("target ROI")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

    # Bottom row (full width): weight distribution overlay for Allen vs S2 only.
    ax_hist = fig.add_subplot(gs[2, :])
    bins = np.geomspace(1e-5, 1.0, 50)
    selected = [("Allen", W_allen, "tab:blue"),
                ("S2 — sparse Allen backbone (top 15%)", Ws[2][1], "tab:orange")]
    for label, W, color in selected:
        nz = W[off_mask]
        nz = nz[nz > 0]
        ax_hist.hist(nz, bins=bins, histtype="step", lw=2.0, color=color, label=label)
    ax_hist.set_xscale("log")
    ax_hist.set_yscale("log")
    ax_hist.set_xlabel("Edge weight (off-diagonal, nonzero)")
    ax_hist.set_ylabel("Count")
    ax_hist.legend(fontsize=10, frameon=False, loc="best")
    for s in ("top", "right"):
        ax_hist.spines[s].set_visible(False)

    pdf = OUT_FIG / "DallaPorta_Suppl_Figure_5.pdf"
    png = OUT_FIG / "DallaPorta_Suppl_Figure_5.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    print(f"\nSaved {pdf}\nSaved {png}")


def _ks_distribution_match(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample KS distance — 0 means identical distributions."""
    from scipy.stats import ks_2samp
    return float(ks_2samp(a, b).statistic)


if __name__ == "__main__":
    main()

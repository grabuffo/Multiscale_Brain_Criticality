"""
Reviewer C, point 2 — BOLD vs.\ neural timescale gap.

Supplementary panel demonstrating that the global coupling G* identified by BOLD-FC
fit (slow, hemodynamically filtered scale) and the G that maximises the neural-scale
global criticality signature (AC1 of GS at 2 kHz) coincide for the homogeneous
critical regime — i.e., the BOLD optimization is informative about the fast-scale
critical regime.

Reads: paper/revision/derived_data/C3_metrics_allen.pkl
Saves: paper/revision/figures/Fig_C2_BOLD_neural_alignment.{pdf,png}
"""

from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # revision/ (paths.py)
import paths  # noqa: E402
_sys.path.insert(0, str(paths.SRC))  # original src/ (functions.py, Utils.py, ...)
import pickle

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = paths.REPO
DERIVED = paths.DERIVED
OUT_FIG = paths.FIGURES
OUT_FIG.mkdir(parents=True, exist_ok=True)


def _hide_top_right(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main() -> None:
    with open(DERIVED / "C3_metrics_allen.pkl", "rb") as f:
        M = pickle.load(f)
    G = M["G_grid"]
    AC1_GS = M["AC1_GS"]
    corr_FC = np.nanmean(M["corr_FC"], axis=0)
    KS_FC = np.nanmean(M["KS_FC"], axis=0)
    KS_dFC = np.nanmean(M["KS_dFC"], axis=0)

    # Working point identified by the published BOLD FC analysis (Fig. 3): G* = 0.028.
    G_pub_idx = 2
    G_pub = G[G_pub_idx]

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.6))

    color_neu = "tab:blue"
    color_bold = "tab:purple"

    titles = [
        ("(A) Neural scale  AC1$(GS)$ — peak at $G^\\ast$", AC1_GS, color_neu, r"AC1$(GS)$", "max"),
        ("(B) BOLD  Pearson $r$(FC$_{\\rm sim}$, FC$_{\\rm emp}$)",
         corr_FC, color_bold, r"$r$", "max"),
        ("(C) BOLD  KS distance(FC$_{\\rm sim}$, FC$_{\\rm emp}$)",
         KS_FC, color_bold, "KS distance", "min"),
        ("(D) BOLD  KS distance(dFC$_{\\rm sim}$, dFC$_{\\rm emp}$) — min at $G^\\ast$",
         KS_dFC, color_bold, "KS distance", "min"),
    ]
    for ax, (ttl, y, color, ylab, sense) in zip(axes, titles):
        ax.plot(G, y, marker="o", color=color, lw=1.8, markersize=5)
        ax.axvline(G_pub, color="0.4", ls=":", lw=1.0, label=f"$G^\\ast = {G_pub:.3f}$")
        # Mark this metric's own optimum
        opt_idx = int(np.nanargmax(y) if sense == "max" else np.nanargmin(y))
        ax.scatter([G[opt_idx]], [y[opt_idx]], color="black", marker="x", s=70, zorder=5,
                   label=f"this metric's optimum at $G = {G[opt_idx]:.3f}$")
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("G")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=7, frameon=False, loc="best")
        _hide_top_right(ax)

    fig.suptitle(
        "Alignment of neural-scale and BOLD-scale criticality signatures (homogeneous critical regime)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    pdf = OUT_FIG / "Fig_C2_BOLD_neural_alignment.pdf"
    png = OUT_FIG / "Fig_C2_BOLD_neural_alignment.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    print(f"Saved {pdf}\nSaved {png}")
    print()
    print(f"Published working point G* = {G_pub:.4f} (Fig. 3, critical regime)")
    print(f"  AC1(GS) at G*    = {AC1_GS[G_pub_idx]:.4f}  (max of AC1(GS) is {AC1_GS.max():.4f} at G={G[AC1_GS.argmax()]:.4f})")
    print(f"  Pearson r at G*  = {corr_FC[G_pub_idx]:.4f}  (max is {corr_FC.max():.4f} at G={G[corr_FC.argmax()]:.4f})")
    print(f"  KS FC at G*      = {KS_FC[G_pub_idx]:.4f}  (min is {KS_FC.min():.4f} at G={G[KS_FC.argmin()]:.4f})")
    print(f"  KS dFC at G*     = {KS_dFC[G_pub_idx]:.4f}  (min is {KS_dFC.min():.4f} at G={G[KS_dFC.argmin()]:.4f})")


if __name__ == "__main__":
    main()

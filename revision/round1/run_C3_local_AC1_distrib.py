"""
Two-panel comparison of local-AC1 statistics across Allen / S1 / S2 / S3 in the
critical regime: mean and variance of AC1(R_n) across the 60 cortical ROIs as a
function of G. Addresses the "Local Scales" half of the manuscript title
(complementary to the global-criticality and FC-fit comparisons of Suppl. Fig. S6).

Reads:  paper/revision/derived_data/C3_metrics_{allen,S1,S2,S3}.pkl
Saves:  paper/revision/figures/Fig_C3_local_AC1_distrib.{pdf,png}
"""

from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # revision/ (paths.py)
import paths  # noqa: E402
_sys.path.insert(0, str(paths.SRC))  # original src/ (functions.py, Utils.py, ...)
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import revision_utils as ru  # noqa: E402

PROJECT_ROOT = paths.REPO
DERIVED = paths.DERIVED
OUT_FIG = paths.FIGURES
OUT_FIG.mkdir(parents=True, exist_ok=True)

CRIT_G_INDEX = 2  # G* = 0.028

# Match the visual identity used in Fig_C3_comparison.
GRAPH_STYLE = {
    "allen": dict(color="tab:blue",   label=r"Allen homogeneous critical ($a = 0.973$)", marker="o", lw=1.8),
    "S1":    dict(color="tab:grey",   label=r"S1 --- weight-shuffled (full)",            marker="s", lw=1.5, ls="--"),
    "S2":    dict(color="tab:orange", label=r"S2 --- sparse Allen backbone (15\%)",       marker="D", lw=1.5, ls="-."),
    "S3":    dict(color="tab:olive",  label=r"S3 --- sparse random (15\%)",               marker="^", lw=1.5, ls=":"),
}


def _hide_top_right(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main() -> None:
    graphs = ["allen", "S1", "S2", "S3"]
    data = {}
    for g in graphs:
        with open(DERIVED / f"C3_metrics_{g}.pkl", "rb") as f:
            data[g] = pickle.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    G = data["allen"]["G_grid"]

    # Panel A: mean of AC1(R_n) across regions vs G
    ax = axes[0]
    for g in graphs:
        sty = GRAPH_STYLE[g]
        AC1_local = data[g]["AC1_local"]  # (60, 16)
        m = AC1_local.mean(axis=0)
        ax.plot(G, m, marker=sty["marker"], color=sty["color"], lw=sty["lw"],
                ls=sty.get("ls", "-"), label=sty["label"], markersize=4)
    ax.axhline(0, color="0.7", lw=0.5)
    ax.axvline(G[CRIT_G_INDEX], color="0.5", lw=0.7, ls=":")
    ax.set_xlabel("G")
    ax.set_ylabel(r"$\langle$AC1$(R_n)\rangle_n$")
    ax.set_title("(A) Mean local AC1 across regions")
    ax.legend(fontsize=8, frameon=False, loc="best")
    _hide_top_right(ax)

    # Panel B: variance of AC1(R_n) across regions vs G
    ax = axes[1]
    for g in graphs:
        sty = GRAPH_STYLE[g]
        AC1_local = data[g]["AC1_local"]
        v = AC1_local.var(axis=0)
        ax.plot(G, v, marker=sty["marker"], color=sty["color"], lw=sty["lw"],
                ls=sty.get("ls", "-"), label=sty["label"], markersize=4)
    ax.axvline(G[CRIT_G_INDEX], color="0.5", lw=0.7, ls=":")
    ax.set_xlabel("G")
    ax.set_ylabel(r"$\mathrm{Var}_n[\mathrm{AC1}(R_n)]$")
    ax.set_title("(B) Variance of local AC1 across regions")
    ax.legend(fontsize=8, frameon=False, loc="best")
    _hide_top_right(ax)

    fig.tight_layout()
    pdf = OUT_FIG / "Fig_C3_local_AC1_distrib.pdf"
    png = OUT_FIG / "Fig_C3_local_AC1_distrib.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    print(f"Saved {pdf}")
    print(f"Saved {png}")


if __name__ == "__main__":
    main()

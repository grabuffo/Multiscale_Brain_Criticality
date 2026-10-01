"""
Suppl. Fig. S9 (structured-heterogeneity test) - six-panel comparison
between Allen homogeneous critical, Allen homogeneous subcritical,
the in-strength-ranked subcritical gradient (gradient_v2), and the
random-shuffle null (gradient_random; same a values, randomly assigned
to regions; seed=2026).

Layout (2x3):
  Row 1: (A) AC1(GS) vs G; (B) avalanche P(S) at G*; (C) BOLD FC Pearson r vs G.
  Row 2: (D) per-region AC1 vs in-strength scatter at G*; (E) Spearman rho vs G;
         (F) BOLD FC KS distance vs G.

Reuses the panel functions from figS6_surrogate_dynamics.py; panel titles for D, E, F
are overridden so the labels match this 6-panel layout (E and F do not exist in
the S6 4-panel layout that figS6_surrogate_dynamics.py was simplified to).

Reads:  paper/data/derived/metrics_{graph}.pkl for graphs in --graphs
Saves:  paper/figures/DallaPorta_Suppl_Figure_9.{pdf,png}
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # src/ (paths.py, utilities)
import paths  # noqa: E402
import argparse
import sys

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import supplementary_utils as ru  # noqa: E402
from figS6_surrogate_dynamics import (  # noqa: E402
    GRAPH_STYLE,
    load_metrics,
    panel_AC1_GS,
    panel_global_avalanche,
    panel_BOLD_FC_corr,
    panel_AC1_vs_instrength,
    panel_BOLD_FC_KS,
    panel_rho_vs_G,
)


PROJECT_ROOT = paths.REPO
OUT_FIG = paths.FIGURES
OUT_FIG.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--graphs",
        default="allen,allen_sub,gradient_v2,gradient_random",
        help="Comma-separated list of conditions to overlay.",
    )
    ap.add_argument(
        "--output-name",
        default="DallaPorta_Suppl_Figure_9",
        help="Output basename in paper/figures/.",
    )
    args = ap.parse_args()
    graphs = [g.strip() for g in args.graphs.split(",")]

    M = load_metrics(graphs)
    if not M:
        raise RuntimeError("No metrics loaded; check that the .pkl caches exist.")

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.5))

    # Row 1
    panel_AC1_GS(axes[0, 0], M)
    panel_global_avalanche(axes[0, 1], M)
    panel_BOLD_FC_corr(axes[0, 2], M)

    # Row 2
    panel_AC1_vs_instrength(axes[1, 0], M)
    panel_rho_vs_G(axes[1, 1], M)
    panel_BOLD_FC_KS(axes[1, 2], M)

    # Override panel-letter labels so they match the 6-panel S9 layout.
    # figS6_surrogate_dynamics.py was simplified to 4 panels for S6, so panel_rho_vs_G
    # is labelled (D) there; here it must be (E).
    axes[0, 2].set_title("(C) BOLD FC fit (correlation)")
    axes[1, 1].set_title("(E) In-strength gradient vs. coupling")

    # Collapse the per-panel legends (all show the same conditions) into a single
    # figure-level legend at the top.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    for ax_row in axes:
        for ax in ax_row:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    if handles:
        fig.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=min(len(handles), 4),
            frameon=False,
            fontsize=9,
        )
    pdf = OUT_FIG / f"{args.output_name}.pdf"
    png = OUT_FIG / f"{args.output_name}.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf}")
    print(f"Saved {png}")


if __name__ == "__main__":
    main()

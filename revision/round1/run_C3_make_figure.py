"""
Comparison figure: Allen connectome vs. structural-null surrogates (S1, S2, S3)
for Reviewer B point 2 and Reviewer C point 3.

Panels (3x2 layout):
  Row 1 — global dynamical signatures:
    A) AC1(GS) vs G — does global criticality emerge with surrogate topology?
    B) Global avalanche P(S) at G* — does scale-free behaviour survive?
  Row 2 — empirical and per-region structural signatures:
    C) BOLD FC fit (Pearson r vs. empirical) vs G — does empirical FC remain reproducible?
    D) Spearman rho(in-strength, AC1) vs G — in-strength gradient strength across coupling.
  Row 3 — distribution of local dynamics across regions:
    E) Mean of AC1(R_n) across 60 ROIs vs G.
    F) Variance of AC1(R_n) across 60 ROIs vs G.

Reads: paper/revision/derived_data/C3_metrics_{graph}.pkl
Saves: paper/revision/figures/Fig_C3_comparison.{pdf,png}
"""

from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # revision/ (paths.py)
import paths  # noqa: E402
_sys.path.insert(0, str(paths.SRC))  # original src/ (functions.py, Utils.py, ...)
import argparse
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import revision_utils as ru  # noqa: E402

FIRST_SUB_SRC = paths.SRC
sys.path.insert(0, str(FIRST_SUB_SRC))
import Utils as fx  # noqa: E402


PROJECT_ROOT = paths.REPO
DERIVED = paths.DERIVED
OUT_FIG = paths.FIGURES
OUT_FIG.mkdir(parents=True, exist_ok=True)

# Visual identity per graph; allen always plotted in blue (matches the critical-regime
# convention used elsewhere in the manuscript), surrogates in muted/grey tones.
GRAPH_STYLE = {
    "allen":     dict(color="tab:blue",   label=r"Allen homogeneous critical ($a = 0.973$)", marker="o", lw=1.8),
    "S1":        dict(color="tab:grey",   label="S1 — weight-shuffled (full)",         marker="s", lw=1.5, ls="--"),
    "S2":        dict(color="tab:orange", label="S2 — sparse Allen backbone (15%)",    marker="D", lw=1.5, ls="-."),
    "S3":        dict(color="tab:olive",  label="S3 — sparse random (15%)",            marker="^", lw=1.5, ls=":"),
    "B4grad":    dict(color="tab:purple", label=r"Subcritical gradient ($a \in [0.974, 0.978]$)", marker="P", lw=1.6),
    "B4grad_v2":     dict(color="tab:orange", label=r"Subcritical gradient, hubs at $a_\mathrm{crit}$ ($a \in [0.973, 0.978]$)", marker="*", lw=1.6, ls="-"),
    "B4grad_random": dict(color="tab:red",    label=r"Subcritical gradient, randomly shuffled (same $a$ values, random assignment)", marker="x", lw=1.4, ls="--"),
    "allen_sub":     dict(color="tab:green",  label=r"Allen homogeneous subcritical ($a = 0.979$)",   marker="X", lw=1.5, ls="--"),
}

CRIT_G_INDEX = 2   # G* = 0.028 for the critical regime


def _hide_top_right(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def load_metrics(graphs):
    out = {}
    for g in graphs:
        path = DERIVED / f"C3_metrics_{g}.pkl"
        if not path.exists():
            print(f"  WARNING: {path} not found; skipping {g!r}")
            continue
        with open(path, "rb") as f:
            out[g] = pickle.load(f)
        print(f"  loaded {g}: AC1_local{out[g]['AC1_local'].shape}, AC1_GS{out[g]['AC1_GS'].shape}")
    return out


def panel_AC1_GS(ax, M):
    for g, m in M.items():
        sty = GRAPH_STYLE[g]
        ax.plot(m["G_grid"], m["AC1_GS"],
                marker=sty["marker"], color=sty["color"], lw=sty["lw"],
                ls=sty.get("ls", "-"), label=sty["label"], markersize=4)
    ax.set_xlabel("G")
    ax.set_ylabel(r"AC1(GS)")
    ax.set_title("(A) Global signal AC1 vs. coupling")
    ax.axvline(ru.G_GRID[CRIT_G_INDEX], color="0.5", lw=0.7, ls=":")
    ax.legend(fontsize=8, frameon=False, loc="best")
    _hide_top_right(ax)


def panel_global_avalanche(ax, M, iG=CRIT_G_INDEX):
    for g, m in M.items():
        sty = GRAPH_STYLE[g]
        events = np.asarray(m["aval_GS"][iG])
        if events.size < 5:
            continue
        fx.plot_pdf(events, ax=ax, color=sty["color"], lw=sty["lw"],
                    ls=sty.get("ls", "-"), label=sty["label"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$S$ (avalanche size)")
    ax.set_ylabel(r"$P(S)$")
    ax.set_title(f"(B) Global avalanche $P(S)$ at $G^\\ast$={ru.G_GRID[iG]:.3f}")
    ax.legend(fontsize=8, frameon=False, loc="best")
    _hide_top_right(ax)


def panel_AC1_vs_instrength(ax, M, iG=CRIT_G_INDEX):
    for g, m in M.items():
        sty = GRAPH_STYLE[g]
        instr = m["instr"]
        ac = m["AC1_local"][:, iG]
        rho, _ = spearmanr(instr, ac)
        label = f"{sty['label']} (ρ={rho:+.2f})"
        ax.scatter(instr, ac, color=sty["color"], s=22, alpha=0.75,
                   edgecolor="0.2", linewidth=0.3, label=label)
    ax.set_xlabel("In-strength")
    ax.set_ylabel(r"AC1$(R_n)$")
    ax.set_title(f"(D) Local AC1 vs. in-strength at $G^\\ast$={ru.G_GRID[iG]:.3f}")
    ax.legend(fontsize=7, frameon=False, loc="best")
    _hide_top_right(ax)


def panel_BOLD_FC_corr(ax, M):
    """Pearson r between simulated and empirical FC vs G (mean ± std across 53 subjects)."""
    for g, m in M.items():
        if "corr_FC" not in m:
            continue
        sty = GRAPH_STYLE[g]
        mean = np.nanmean(m["corr_FC"], axis=0)
        std = np.nanstd(m["corr_FC"], axis=0)
        ax.plot(m["G_grid"], mean,
                marker=sty["marker"], color=sty["color"], lw=sty["lw"],
                ls=sty.get("ls", "-"), label=sty["label"], markersize=4)
        ax.fill_between(m["G_grid"], mean - std, mean + std,
                        color=sty["color"], alpha=0.15)
    ax.axvline(ru.G_GRID[CRIT_G_INDEX], color="0.5", lw=0.7, ls=":")
    ax.set_xlabel("G")
    ax.set_ylabel(r"Pearson $r$(FC$_\mathrm{sim}$, FC$_\mathrm{emp}$)")
    ax.set_title("(C) BOLD FC fit")
    ax.legend(fontsize=8, frameon=False, loc="best")
    _hide_top_right(ax)


def panel_BOLD_FC_KS(ax, M):
    """KS distance between simulated and empirical FC distributions vs G."""
    for g, m in M.items():
        if "KS_FC" not in m:
            continue
        sty = GRAPH_STYLE[g]
        mean = np.nanmean(m["KS_FC"], axis=0)
        std = np.nanstd(m["KS_FC"], axis=0)
        ax.plot(m["G_grid"], mean,
                marker=sty["marker"], color=sty["color"], lw=sty["lw"],
                ls=sty.get("ls", "-"), label=sty["label"], markersize=4)
        ax.fill_between(m["G_grid"], mean - std, mean + std,
                        color=sty["color"], alpha=0.15)
    ax.axvline(ru.G_GRID[CRIT_G_INDEX], color="0.5", lw=0.7, ls=":")
    ax.set_xlabel("G")
    ax.set_ylabel("KS distance (FC)")
    ax.set_title("(F) BOLD FC fit (KS distance)")
    ax.legend(fontsize=8, frameon=False, loc="best")
    _hide_top_right(ax)


def panel_rho_vs_G(ax, M):
    for g, m in M.items():
        sty = GRAPH_STYLE[g]
        rho_G = np.array([
            spearmanr(m["instr"], m["AC1_local"][:, ig])[0]
            for ig in range(m["G_grid"].size)
        ])
        # Mask G=0 (regions identical → ρ is noise)
        ax.plot(m["G_grid"][1:], rho_G[1:],
                marker=sty["marker"], color=sty["color"], lw=sty["lw"],
                ls=sty.get("ls", "-"), label=sty["label"], markersize=4)
    ax.axhline(0, color="0.6", lw=0.5)
    ax.axvline(ru.G_GRID[CRIT_G_INDEX], color="0.5", lw=0.7, ls=":")
    ax.set_xlabel("G")
    ax.set_ylabel(r"Spearman $\rho$(in-strength, AC1)")
    ax.set_title("(D) In-strength gradient vs. coupling")
    ax.legend(fontsize=8, frameon=False, loc="best")
    _hide_top_right(ax)


def panel_local_AC1_mean(ax, M):
    """Mean of AC1(R_n) across regions vs G for each graph."""
    for g, m in M.items():
        sty = GRAPH_STYLE[g]
        AC1_local = m["AC1_local"]  # (60, n_G)
        mean_AC1 = AC1_local.mean(axis=0)
        ax.plot(m["G_grid"], mean_AC1,
                marker=sty["marker"], color=sty["color"], lw=sty["lw"],
                ls=sty.get("ls", "-"), label=sty["label"], markersize=4)
    ax.axhline(0, color="0.7", lw=0.5)
    ax.axvline(ru.G_GRID[CRIT_G_INDEX], color="0.5", lw=0.7, ls=":")
    ax.set_xlabel("G")
    ax.set_ylabel(r"$\langle$AC1$(R_n)\rangle_n$")
    ax.set_title("(E) Mean local AC1 across regions")
    ax.legend(fontsize=8, frameon=False, loc="best")
    _hide_top_right(ax)


def panel_local_AC1_var(ax, M):
    """Variance of AC1(R_n) across regions vs G for each graph."""
    for g, m in M.items():
        sty = GRAPH_STYLE[g]
        AC1_local = m["AC1_local"]
        var_AC1 = AC1_local.var(axis=0)
        ax.plot(m["G_grid"], var_AC1,
                marker=sty["marker"], color=sty["color"], lw=sty["lw"],
                ls=sty.get("ls", "-"), label=sty["label"], markersize=4)
    ax.axvline(ru.G_GRID[CRIT_G_INDEX], color="0.5", lw=0.7, ls=":")
    ax.set_xlabel("G")
    ax.set_ylabel(r"$\mathrm{Var}_n[\mathrm{AC1}(R_n)]$")
    ax.set_title("(F) Variance of local AC1 across regions")
    ax.legend(fontsize=8, frameon=False, loc="best")
    _hide_top_right(ax)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="allen,S1",
                    help="Comma-separated list of graphs to overlay (default: allen,S1).")
    ap.add_argument("--output-name", default="Fig_C3_comparison",
                    help="Output basename in paper/revision/figures/ (default: Fig_C3_comparison).")
    args = ap.parse_args()
    graphs = [g.strip() for g in args.graphs.split(",")]

    M = load_metrics(graphs)
    if "allen" not in M:
        raise RuntimeError("Allen baseline metrics missing; run run_C3_compute_metrics.py allen first.")

    fig, axes = plt.subplots(3, 2, figsize=(11.5, 12.8))
    panel_AC1_GS(axes[0, 0], M)             # A
    panel_global_avalanche(axes[0, 1], M)   # B
    panel_BOLD_FC_corr(axes[1, 0], M)       # C
    panel_rho_vs_G(axes[1, 1], M)           # D
    panel_local_AC1_mean(axes[2, 0], M)     # E
    panel_local_AC1_var(axes[2, 1], M)      # F
    fig.tight_layout()

    pdf_path = OUT_FIG / f"{args.output_name}.pdf"
    png_path = OUT_FIG / f"{args.output_name}.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()

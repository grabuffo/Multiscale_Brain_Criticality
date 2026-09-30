"""
Build the (G, in-strength) phase-diagram supplementary figure.

Layout (2 rows × 3 cols):
  Row 1 — Binned heatmap of AC1(R_n) over (G, in-strength bin), one column per regime.
  Row 2 — Quantitative summaries spanning regimes:
            (i)   In-strength–AC1 Spearman ρ as a function of G.
            (ii)  Per-region non-monotonicity score (critical regime) vs. in-strength.
            (iii) Cross-region variance of AC1 as a function of G.

Saves PDF and PNG to paper/revision/figures/.
"""

from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # revision/ (paths.py)
import paths  # noqa: E402
_sys.path.insert(0, str(paths.SRC))  # original src/ (functions.py, Utils.py, ...)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from revision_utils import (
    G_GRID,
    G_WORKING_INDEX,
    HIGHEST_INSTRENGTH_IDX,
    INSTRENGTH,
    REGIMES,
    cross_region_variance,
    instrength_AC1_correlation,
    non_monotonicity_score,
)


PROJECT_ROOT = paths.REPO
DERIVED = paths.DERIVED
OUT_FIG = paths.FIGURES
OUT_FIG.mkdir(parents=True, exist_ok=True)


REGIME_CMAPS = {"subcritical": "Greens", "critical": "Blues", "supercritical": "Reds"}
REGIME_LINE = {"subcritical": "tab:green", "critical": "tab:blue", "supercritical": "tab:red"}
REGIME_LABELS = {"subcritical": "Subcritical", "critical": "Critical", "supercritical": "Supercritical"}


def binned_heatmap(
    instr: np.ndarray, AC1_2d: np.ndarray, n_bins: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    bin_edges = np.quantile(instr, np.linspace(0, 1, n_bins + 1))
    bin_edges[-1] += 1e-9
    bin_idx = np.digitize(instr, bin_edges[1:-1])
    Z = np.full((n_bins, AC1_2d.shape[1]), np.nan)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            Z[b] = AC1_2d[mask].mean(axis=0)
    return bin_edges, Z


def _hide_top_right(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main() -> None:
    AC1_local = np.load(DERIVED / "AC1_local.npy")  # (3, 60, 16)
    assert AC1_local.shape == (3, 60, 16)

    rho = instrength_AC1_correlation(AC1_local)              # (3, 16)
    nm_crit = non_monotonicity_score(AC1_local, regime_idx=1)  # (60,)
    var_AC1 = cross_region_variance(AC1_local)               # (3, 16)

    fig = plt.figure(figsize=(11.5, 6.4))
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[1.0, 1.0],
        wspace=0.40, hspace=0.50,
        left=0.07, right=0.96, top=0.93, bottom=0.10,
    )

    n_bins = 8

    # ===== Row 1: binned heatmaps =====
    for ic, regime in enumerate(REGIMES):
        AC1_2d = AC1_local[ic]
        cmap = REGIME_CMAPS[regime]
        vmin = float(np.nanpercentile(AC1_2d, 2))
        vmax = float(np.nanpercentile(AC1_2d, 98))
        if vmin == vmax:
            vmax = vmin + 1e-9
        norm = Normalize(vmin=vmin, vmax=vmax)
        gw = G_WORKING_INDEX[regime]

        ax = fig.add_subplot(gs[0, ic])
        bin_edges, Z = binned_heatmap(INSTRENGTH, AC1_2d, n_bins=n_bins)
        dG = np.diff(G_GRID).mean()
        g_edges = np.concatenate([[G_GRID[0] - dG / 2], G_GRID + dG / 2])
        im = ax.pcolormesh(g_edges, bin_edges, Z, cmap=cmap, norm=norm, shading="flat")
        ax.axvline(G_GRID[gw], color="0.15", lw=0.9, ls="--")
        ax.set_title(REGIME_LABELS[regime])
        ax.set_xlabel("G")
        if ic == 0:
            ax.set_ylabel("In-strength (binned)")
        _hide_top_right(ax)
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label(r"$\langle$AC1$(R_n)\rangle_{\mathrm{bin}}$", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    # ===== Row 2: quantitative summaries =====

    # (i) Spearman ρ(G)
    ax_rho = fig.add_subplot(gs[1, 0])
    for ir, regime in enumerate(REGIMES):
        # Mask G=0 (regions are identical there, so ρ is just noise)
        ax_rho.plot(
            G_GRID[1:], rho[ir, 1:],
            marker="o", markersize=4, lw=1.6,
            color=REGIME_LINE[regime], label=REGIME_LABELS[regime],
        )
    ax_rho.axhline(0, color="0.6", lw=0.5)
    for regime in REGIMES:
        ax_rho.axvline(G_GRID[G_WORKING_INDEX[regime]],
                       color=REGIME_LINE[regime], ls=":", lw=0.6, alpha=0.6)
    ax_rho.set_xlabel("G")
    ax_rho.set_ylabel(r"Spearman $\rho$(in-strength, AC1)")
    ax_rho.set_title("(i) In-strength–AC1 gradient")
    ax_rho.legend(fontsize=8, frameon=False, loc="best")
    _hide_top_right(ax_rho)

    # (ii) Non-monotonicity score (critical) vs. in-strength
    ax_nm = fig.add_subplot(gs[1, 1])
    ax_nm.scatter(
        INSTRENGTH, nm_crit,
        s=28, color=REGIME_LINE["critical"], alpha=0.75,
        edgecolor="0.2", linewidth=0.3,
    )
    from scipy.stats import pearsonr, spearmanr
    r_p, _ = pearsonr(INSTRENGTH, nm_crit)
    r_s, _ = spearmanr(INSTRENGTH, nm_crit)
    ax_nm.text(
        0.04, 0.95,
        f"Pearson r = {r_p:+.2f}\nSpearman ρ = {r_s:+.2f}",
        transform=ax_nm.transAxes, va="top", ha="left", fontsize=8,
    )
    ax_nm.axhline(0, color="0.6", lw=0.5)
    ax_nm.set_xlabel("In-strength")
    ax_nm.set_ylabel(r"$\max_G \mathrm{AC1}(G) - \mathrm{AC1}(0)$")
    ax_nm.set_title("(ii) Non-monotonicity (critical regime)")
    _hide_top_right(ax_nm)

    # (iii) Cross-region variance of AC1 vs G
    ax_var = fig.add_subplot(gs[1, 2])
    for ir, regime in enumerate(REGIMES):
        ax_var.plot(
            G_GRID, var_AC1[ir],
            marker="o", markersize=4, lw=1.6,
            color=REGIME_LINE[regime], label=REGIME_LABELS[regime],
        )
    for regime in REGIMES:
        ax_var.axvline(G_GRID[G_WORKING_INDEX[regime]],
                       color=REGIME_LINE[regime], ls=":", lw=0.6, alpha=0.6)
    ax_var.set_xlabel("G")
    ax_var.set_ylabel(r"$\mathrm{Var}_n\,\mathrm{AC1}(R_n)$")
    ax_var.set_title("(iii) Cross-region heterogeneity")
    ax_var.legend(fontsize=8, frameon=False, loc="best")
    _hide_top_right(ax_var)

    pdf_path = OUT_FIG / "Fig_B3_phase_diagram.pdf"
    png_path = OUT_FIG / "Fig_B3_phase_diagram.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")

    np.save(DERIVED / "rho_instr_AC1.npy", rho)
    np.save(DERIVED / "non_monotonicity_score_crit.npy", nm_crit)
    np.save(DERIVED / "var_AC1_per_G.npy", var_AC1)
    print("Saved derived summary arrays.")


if __name__ == "__main__":
    main()

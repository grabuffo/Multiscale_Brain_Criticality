"""
Single combined supplementary figure for Reviewer B point 3.

Layout (4 rows x 3 cols, panels A-L):
  Row 1 (A, B, C) - binned heatmap of <AC1(R_n)> over (G, in-strength bin),
        per regime. Order: subcritical, critical, supercritical.
  Row 2 (D, E, F) - in-strength-AC1 Spearman rho as a function of G,
        one regime per panel. Order: subcritical, critical, supercritical
        (matches the column ordering of rows 1, 3, 4 so every column is the
        same regime throughout the figure).
  Row 3 (G, H, I) - PSD at the high-in-strength exemplar VISal, across G,
        per regime. Order: subcritical, critical, supercritical.
  Row 4 (J, K, L) - local avalanche P(S) at VISal, across G, per regime.
        Order: subcritical, critical, supercritical.

Saves PDF and PNG to paper/revision/figures/Fig_B3_combined_SI.{pdf,png}.
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
from matplotlib.colors import Normalize
from scipy.signal import welch

HERE = Path(__file__).resolve().parent
FIRST_SUB_SRC = paths.SRC
sys.path.insert(0, str(FIRST_SUB_SRC))
import Utils as fx                   # noqa: E402

from revision_utils import (         # noqa: E402
    DEFAULT_SIM_ROOT,
    G_GRID,
    G_WORKING_INDEX,
    HIGHEST_INSTRENGTH_IDX,
    HIGHEST_INSTRENGTH_NAME,
    INSTRENGTH,
    REGIME_DIRS,
    REGIMES,
    instrength_AC1_correlation,
    measure_events,
)


PROJECT_ROOT = paths.REPO
DERIVED = paths.DERIVED
OUT_FIG = paths.FIGURES
OUT_FIG.mkdir(parents=True, exist_ok=True)


REGIME_CMAPS = {"subcritical": "Greens", "critical": "Blues", "supercritical": "Reds"}
REGIME_LINE = {"subcritical": "tab:green", "critical": "tab:blue", "supercritical": "tab:red"}
REGIME_LABELS = {"subcritical": "Subcritical", "critical": "Critical", "supercritical": "Supercritical"}

GGs = [0, 2, 5, 15]
TLEN = 5000
FS = 2000
NPERSEG = 2048


def _shades(cmap_name, n):
    cmap = plt.get_cmap(cmap_name)
    return [cmap(0.3 + 0.7 * j / (n - 1)) for j in range(n)]


def _hide_top_right(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def binned_heatmap(instr, AC1_2d, n_bins=8):
    bin_edges = np.quantile(instr, np.linspace(0, 1, n_bins + 1))
    bin_edges[-1] += 1e-9
    bin_idx = np.digitize(instr, bin_edges[1:-1])
    Z = np.full((n_bins, AC1_2d.shape[1]), np.nan)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            Z[b] = AC1_2d[mask].mean(axis=0)
    return bin_edges, Z


def _load_R_at_region(sim_root: Path, regime: str, iG: int, region_idx: int) -> np.ndarray:
    pkl = sim_root / REGIME_DIRS[regime] / f"data_G{iG}.pkl"
    with open(pkl, "rb") as f:
        R = pickle.load(f)["R"]
    return R[TLEN:, region_idx]


PANEL_LETTERS = list("ABCDEFGHIJKL")


def _panel_letter(ax, letter: str) -> None:
    """Annotate a subplot with a bold uppercase panel letter in its top-left corner."""
    ax.text(
        -0.18, 1.06, letter, transform=ax.transAxes,
        fontsize=12, fontweight="bold", va="bottom", ha="left",
    )


def main(sim_root: Path = DEFAULT_SIM_ROOT) -> None:
    AC1_local = np.load(DERIVED / "AC1_local.npy")
    rho = instrength_AC1_correlation(AC1_local)

    # Load VISal time series and pre-compute PSD + avalanches per regime per G.
    region_idx = HIGHEST_INSTRENGTH_IDX
    PSD_local: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    SS_local: dict[int, dict[int, list]] = {}
    for ir, regime in enumerate(REGIMES):
        PSD_local[ir] = {}
        SS_local[ir] = {}
        for iG in GGs:
            sig = _load_R_at_region(sim_root, regime, iG, region_idx)
            freqs, psd = welch(sig, fs=FS, nperseg=NPERSEG)
            PSD_local[ir][iG] = (freqs, psd)
            durations, _ = measure_events(np.abs(sig), np.median(np.abs(sig)), dir=-1)
            SS_local[ir][iG] = durations

    fig = plt.figure(figsize=(12, 12.5))
    gs = fig.add_gridspec(
        4, 3,
        height_ratios=[1.0, 1.0, 1.0, 1.0],
        wspace=0.42, hspace=0.62,
        left=0.07, right=0.96, top=0.965, bottom=0.05,
    )

    n_bins = 8

    # Row-2 ordering: same as rows 1, 3, 4 (subcritical, critical, supercritical),
    # so each column corresponds to the same regime throughout the figure.
    ROW2_ORDER = list(REGIMES)  # ("subcritical", "critical", "supercritical")

    # ===== Row 1 (A, B, C): binned heatmaps =====
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
        _panel_letter(ax, PANEL_LETTERS[ic])

    # ===== Row 2 (D, E, F): in-strength-AC1 Spearman rho per regime =====
    # Common y-axis limits so the three regimes are directly comparable.
    rho_clean = rho[:, 1:]
    y_pad = 0.08 * (np.nanmax(rho_clean) - np.nanmin(rho_clean))
    rho_ylim = (np.nanmin(rho_clean) - y_pad, np.nanmax(rho_clean) + y_pad)

    for ic, regime in enumerate(ROW2_ORDER):
        ir = REGIMES.index(regime)
        ax = fig.add_subplot(gs[1, ic])
        ax.plot(
            G_GRID[1:], rho[ir, 1:],
            marker="o", markersize=4, lw=1.8,
            color=REGIME_LINE[regime],
        )
        ax.axhline(0, color="0.6", lw=0.5)
        ax.axvline(G_GRID[G_WORKING_INDEX[regime]],
                   color=REGIME_LINE[regime], ls=":", lw=0.9, alpha=0.8)
        ax.set_xlabel("G")
        if ic == 0:
            ax.set_ylabel(r"Spearman $\rho$(in-strength, AC1)")
        ax.set_ylim(rho_ylim)
        ax.set_title(REGIME_LABELS[regime])
        _hide_top_right(ax)
        _panel_letter(ax, PANEL_LETTERS[3 + ic])

    # ===== Row 3 (G, H, I): VISal PSD =====
    Gs_labels = [f"G={G_GRID[g]:.3f}" for g in GGs]
    f_min, f_max = 0.5, 100
    for i, regime in enumerate(REGIMES):
        ax = fig.add_subplot(gs[2, i])
        _hide_top_right(ax)
        shades = _shades(REGIME_CMAPS[regime], len(GGs))
        for j, g in enumerate(GGs):
            freqs, psd = PSD_local[i][g]
            psd_norm = psd / np.trapezoid(psd, freqs)
            ax.plot(freqs, psd_norm, color=shades[j], label=Gs_labels[j])
        ax.set_xlabel("Frequency (Hz)")
        ax.set_xlim(f_min, f_max)
        ax.set_ylim(2e-4, 0.2)
        ax.set_yscale("log")
        if i == 0:
            ax.set_ylabel("PSD (a.u.)")
        ax.legend(fontsize=7, frameon=False)
        if i == 1:
            ax.set_title("VISal — PSD")
        _panel_letter(ax, PANEL_LETTERS[6 + i])

    # ===== Row 4 (J, K, L): VISal avalanches =====
    s_min, s_max = 1e0, 1e3
    for i, regime in enumerate(REGIMES):
        ax = fig.add_subplot(gs[3, i])
        _hide_top_right(ax)
        shades = _shades(REGIME_CMAPS[regime], len(GGs))
        for j, g in enumerate(GGs):
            fx.plot_pdf(SS_local[i][g], ax=ax, color=shades[j], label=Gs_labels[j])
        ax.set_xlabel(r"$S$")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlim(s_min, s_max)
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel(r"$P(S)$")
        ax.legend(fontsize=7, frameon=False)
        if i == 1:
            ax.set_title("VISal — local avalanches")
        _panel_letter(ax, PANEL_LETTERS[9 + i])

    pdf_path = OUT_FIG / "Fig_B3_combined_SI.pdf"
    png_path = OUT_FIG / "Fig_B3_combined_SI.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()

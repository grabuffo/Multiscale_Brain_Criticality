"""
PSD and avalanche P(S) at VISal (Right Anterolateral visual area, cortical idx 10,
in-strength = 4.4064 — the highest-in-strength cortical region).

These panels are kept as a supplementary contrast to Fig. 5C (which still uses
retrosplenial-v): a high-in-strength exemplar that demonstrates how the in-strength
gradient identified in the (G, in-strength) phase diagram materialises in time-series
statistics — sharper PSD oscillatory peaks and more peaked avalanche distributions.

Style matches the original notebook 6 cells exactly: same colormaps, same G values,
same Welch settings, same avalanche-extraction call.

Saves two separate 1x3 panels (subcritical / critical / supercritical), suitable for
direct insertion into the supplementary materials.
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
from scipy import stats
from scipy.signal import welch

# Use plot_pdf from the original Utils (numpy-2 compatible since it doesn't call np.trapz).
HERE = Path(__file__).resolve().parent
FIRST_SUB_SRC = paths.SRC
sys.path.insert(0, str(FIRST_SUB_SRC))
import Utils as fx                    # noqa: E402

# measure_events is reimplemented locally in revision_utils with np.trapezoid (numpy 2.x).
from revision_utils import (           # noqa: E402
    DEFAULT_SIM_ROOT,
    G_GRID,
    HIGHEST_INSTRENGTH_IDX,
    HIGHEST_INSTRENGTH_NAME,
    REGIME_DIRS,
    REGIMES,
    measure_events,
)


PROJECT_ROOT = paths.REPO
OUT_FIG = paths.FIGURES
OUT_FIG.mkdir(parents=True, exist_ok=True)

# Match notebook 6: GGs = [0, 2, 5, 15], shading from 0.3 to 1.0 in the regime cmap.
GGs = [0, 2, 5, 15]
COLORMAPS = ["Greens", "Blues", "Reds"]
TLEN = 5000   # discard first 5000 samples (matches notebook 6, cell-34)
FS = 2000     # sampling rate, Hz
NPERSEG = 2048


def _shades(cmap_name, n):
    cmap = plt.get_cmap(cmap_name)
    return [cmap(0.3 + 0.7 * j / (n - 1)) for j in range(n)]


def _load_R_at_region(sim_root: Path, regime: str, iG: int, region_idx: int) -> np.ndarray:
    pkl = sim_root / REGIME_DIRS[regime] / f"data_G{iG}.pkl"
    with open(pkl, "rb") as f:
        R = pickle.load(f)["R"]   # (T, N)
    return R[TLEN:, region_idx]


def main(sim_root: Path = DEFAULT_SIM_ROOT) -> None:
    region_idx = HIGHEST_INSTRENGTH_IDX
    print(f"Re-plotting Fig. 5C at region {region_idx} ({HIGHEST_INSTRENGTH_NAME}).")

    # 1) Collect signal slices, PSDs and avalanche distributions
    PSD_local: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    SS_local: dict[int, dict[int, list]] = {}

    for ir, regime in enumerate(REGIMES):
        PSD_local[ir] = {}
        SS_local[ir] = {}
        for j, iG in enumerate(GGs):
            sig = _load_R_at_region(sim_root, regime, iG, region_idx)
            freqs, psd = welch(sig, fs=FS, nperseg=NPERSEG)
            PSD_local[ir][iG] = (freqs, psd)

            # avalanche extraction matching notebook 6 cell-34
            durations, _integrals = measure_events(
                np.abs(sig), np.median(np.abs(sig)), dir=-1
            )
            SS_local[ir][iG] = durations
            print(f"  {regime:13s}  G[{iG:2d}]={G_GRID[iG]:.4f}  PSD ok, {len(durations)} events")

    Gs_labels = [f"G={G_GRID[g]:.3f}" for g in GGs]

    # ===== Panel: Local Avalanches (matches notebook 6, cell-35 style) =====
    fig_av, axes_av = plt.subplots(1, 3, figsize=(10, 2.4), sharey=True)
    x_min, x_max = 1e0, 1e3
    for i in range(3):
        for spine in ("top", "right"):
            axes_av[i].spines[spine].set_visible(False)
        shades = _shades(COLORMAPS[i], len(GGs))
        for j, g in enumerate(GGs):
            fx.plot_pdf(SS_local[i][g], ax=axes_av[i], color=shades[j], label=Gs_labels[j])
        axes_av[i].set_xlabel(r"$S$")
        axes_av[i].set_yscale("log")
        axes_av[i].set_xscale("log")
        axes_av[i].set_xlim(x_min, x_max)
        axes_av[i].set_yticks([])
        if i == 0:
            axes_av[i].set_ylabel(r"$P(S)$")
        axes_av[i].legend(fontsize=7, frameon=False)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4)
    av_pdf = OUT_FIG / "Fig_B3_SI_VISal_avalanches.pdf"
    av_png = OUT_FIG / "Fig_B3_SI_VISal_avalanches.png"
    fig_av.savefig(av_pdf, dpi=300, bbox_inches="tight", transparent=True)
    fig_av.savefig(av_png, dpi=300, bbox_inches="tight")
    plt.close(fig_av)
    print(f"Saved {av_pdf}")
    print(f"Saved {av_png}")

    # ===== Panel: Local PSD (matches notebook 6, cell-37 style) =====
    fig_psd, axes_psd = plt.subplots(1, 3, figsize=(10, 2.4), sharey=True)
    f_min, f_max = 0.5, 100
    for i in range(3):
        for spine in ("top", "right"):
            axes_psd[i].spines[spine].set_visible(False)
        shades = _shades(COLORMAPS[i], len(GGs))
        for j, g in enumerate(GGs):
            freqs, psd = PSD_local[i][g]
            psd = psd / np.trapezoid(psd, freqs)   # area-based normalization (matches original)
            axes_psd[i].plot(freqs, psd, color=shades[j], label=Gs_labels[j])
        axes_psd[i].set_xlabel("Frequency (Hz)")
        axes_psd[i].set_xlim(f_min, f_max)
        axes_psd[i].set_ylim(2e-4, 0.2)
        axes_psd[i].set_yscale("log")
        if i == 0:
            axes_psd[i].set_ylabel("PSD (a.u.)")
        axes_psd[i].legend(fontsize=7, frameon=False)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4)
    psd_pdf = OUT_FIG / "Fig_B3_SI_VISal_PSD.pdf"
    psd_png = OUT_FIG / "Fig_B3_SI_VISal_PSD.png"
    fig_psd.savefig(psd_pdf, dpi=300, bbox_inches="tight", transparent=True)
    fig_psd.savefig(psd_png, dpi=300, bbox_inches="tight")
    plt.close(fig_psd)
    print(f"Saved {psd_pdf}")
    print(f"Saved {psd_png}")


if __name__ == "__main__":
    main()

"""
Corroborating indicators of long-range temporal correlations across the G sweep
for the critical regime (Suppl. Fig. S7).

Layout (2 rows x 2 columns):
  Row 1, on the global signal GS(t) = (1/N) sum_n z(R_n(t)):
    (A) AC1(GS)             (B) H_DFA(GS)
  Row 2, on per-region R_n(t), aggregated across the 60 cortical ROIs (mean +/- std):
    (C) AC1(R_n)            (D) H_DFA(R_n)

Vertical dotted line marks the working point G* = 0.028.

Reads:  paper/data/derived/{ac1_dfa_indicators.pkl, AC1_local.npy}
Saves:  paper/figures/DallaPorta_Suppl_Figure_7.{pdf,png}

(alpha_PSD remains cached in ac1_dfa_indicators.pkl but is not plotted here:
its absolute value on GS is supra-canonical, so the AC1 + DFA pair gives
a cleaner story aligned with the canonical critical references.)
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # src/ (paths.py, utilities)
import paths  # noqa: E402
import pickle

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = paths.REPO
DERIVED = paths.DERIVED
OUT_FIG = paths.FIGURES
OUT_FIG.mkdir(parents=True, exist_ok=True)

CRIT_G_INDEX = 2  # G* = 0.028
LINE_COLOR_GS = "tab:purple"      # global-signal row
LINE_COLOR_REG = "tab:blue"        # per-region row


def _hide_top_right(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _plot_GS(ax, G, y, color, title, ylabel, refs=None):
    ax.plot(G, y, color=color, lw=1.8, marker="o", markersize=4)
    if refs is not None:
        for level, label, ls in refs:
            ax.axhline(level, color="0.5", lw=0.8, ls=ls, label=label)
    ax.axvline(G[CRIT_G_INDEX], color="0.4", lw=0.7, ls=":")
    ax.set_xlabel("G")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    if refs is not None:
        ax.legend(fontsize=8, frameon=False, loc="best")
    _hide_top_right(ax)


def _plot_band(ax, G, X, color, title, ylabel, refs=None):
    """Mean +/- std across regions (per-region, aggregated)."""
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    ax.plot(G, mu, color=color, lw=1.8, marker="o", markersize=4)
    ax.fill_between(G, mu - sd, mu + sd, color=color, alpha=0.18)
    if refs is not None:
        for level, label, ls in refs:
            ax.axhline(level, color="0.5", lw=0.8, ls=ls, label=label)
    ax.axvline(G[CRIT_G_INDEX], color="0.4", lw=0.7, ls=":")
    ax.set_xlabel("G")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    if refs is not None:
        ax.legend(fontsize=8, frameon=False, loc="best")
    _hide_top_right(ax)


def main() -> None:
    with open(DERIVED / "ac1_dfa_indicators.pkl", "rb") as f:
        C = pickle.load(f)
    AC1_all = np.load(DERIVED / "AC1_local.npy")  # (3, 60, 16)
    AC1_crit = AC1_all[1]                          # critical regime, (60, 16)

    needed = {"AC1_GS", "H_DFA_GS"}
    if not needed.issubset(C):
        raise RuntimeError(
            f"ac1_dfa_indicators.pkl is missing GS keys {needed - set(C)}; "
            "run compute_ac1_dfa_global.py first."
        )

    G = C["G_grid"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), sharex="col")

    # ===== Row 1: global signal GS(t) =====
    _plot_GS(
        axes[0, 0], G, C["AC1_GS"], LINE_COLOR_GS,
        title="(A) AC1$(GS)$ — global signal",
        ylabel=r"AC1$(GS)$",
        refs=[(0.0, "white-noise reference", "-.")],
    )
    _plot_GS(
        axes[0, 1], G, C["H_DFA_GS"], LINE_COLOR_GS,
        title=r"(B) DFA scaling exponent $H_\mathrm{DFA}(GS)$",
        ylabel=r"$H_\mathrm{DFA}(GS)$",
        refs=[
            (0.5, r"white noise ($H = 0.5$)", ":"),
            (1.0, r"$1/f$ noise ($H \approx 1$)", "--"),
        ],
    )

    # ===== Row 2: per-region R_n(t), mean +/- std across 60 ROIs =====
    _plot_band(
        axes[1, 0], G, AC1_crit, LINE_COLOR_REG,
        title=r"(C) AC1$(R_n)$ — per-region (mean $\pm$ std, 60 ROIs)",
        ylabel=r"AC1$(R_n)$",
        refs=[(0.0, "white-noise reference", "-.")],
    )
    _plot_band(
        axes[1, 1], G, C["H_DFA"], LINE_COLOR_REG,
        title=r"(D) $H_\mathrm{DFA}(R_n)$ — per-region (mean $\pm$ std)",
        ylabel=r"$H_\mathrm{DFA}(R_n)$",
        refs=[
            (0.5, r"white noise ($H = 0.5$)", ":"),
            (1.0, r"$1/f$ noise ($H \approx 1$)", "--"),
        ],
    )

    fig.tight_layout()
    pdf = OUT_FIG / "DallaPorta_Suppl_Figure_7.pdf"
    png = OUT_FIG / "DallaPorta_Suppl_Figure_7.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    print(f"Saved {pdf}")
    print(f"Saved {png}")

    print()
    print("=== AC1 and DFA: global signal vs per-region median ===")
    print(
        f"{'G':>7s}  "
        f"{'AC1_GS':>8s}  {'AC1_reg':>8s}  "
        f"{'HDFA_GS':>8s}  {'HDFA_reg':>9s}"
    )
    for ig in range(len(G)):
        print(
            f"{G[ig]:>7.3f}  "
            f"{C['AC1_GS'][ig]:>+8.3f}  {np.nanmedian(AC1_crit[:, ig]):>+8.3f}  "
            f"{C['H_DFA_GS'][ig]:>8.3f}  {np.nanmedian(C['H_DFA'][:, ig]):>9.3f}"
        )


if __name__ == "__main__":
    main()

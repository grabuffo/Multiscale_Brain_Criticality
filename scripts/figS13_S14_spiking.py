"""
Suppl. Figs. S13 (whole-brain spiking validation) and S14 (two
coupled spiking populations). Style follows the other supplementary figures
(S6, S7, S9). Reads data/derived/supp_spiking.pkl (spiking_figure_data.py).
Output: figures/DallaPorta_Suppl_Figure_{13,14}.{svg,pdf,png}
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # src/ (paths.py, utilities)
import paths  # noqa: E402

D = pickle.load(open(paths.DERIVED / "supp_spiking.pkl", "rb"))
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
C_CRIT, C_SUB = "tab:blue", "tab:green"


def hide(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(paths.FIGURES / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ->", name)


def wb_series(reg, key, seed=0):
    rows = [r for r in D["wb"][reg] if r["seed"] == seed]
    return np.array([r["G"] for r in rows]), np.array([r[key] for r in rows])


def figS13():
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 13, "axes.labelsize": 14,
                         "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11})
    wb = D["wb"]
    fig = plt.figure(figsize=(18.0, 10.5))
    gsp = fig.add_gridspec(2, 6, wspace=2.0, hspace=0.4)
    axes = {(0, 0): fig.add_subplot(gsp[0, 0:2]), (0, 1): fig.add_subplot(gsp[0, 2:4]),
            (0, 2): fig.add_subplot(gsp[0, 4:6]), (1, 0): fig.add_subplot(gsp[1, 1:3]),
            (1, 1): fig.add_subplot(gsp[1, 3:5])}
    # (A) calibration of the isolated spiking population
    ax = axes[0, 0]
    cal = wb["calib"]
    ax.plot(cal["a"], cal["AC1"], "k-o", ms=3, lw=1.5, label="AC1 (1 ms)")
    ax.set_ylabel("AC1 (1 ms)")
    ax2 = ax.twinx()
    ax2.plot(cal["a"], cal["chi"], "-s", color="0.55", ms=3, lw=1.2, label=r"$\chi$")
    ax2.set_yscale("log"); ax2.set_ylabel(r"$\chi$")
    a_crit, a_sub = wb["critical"][0]["a"], wb["subcritical"][0]["a"]
    ax.axvline(a_crit, color=C_CRIT, ls="--", lw=1.2, label=f"critical, $a={a_crit:.3f}$")
    ax.axvline(a_sub, color=C_SUB, ls="--", lw=1.2, label=f"subcritical, $a={a_sub:.3f}$")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="lower left")
    ax.set_xlabel("$a$"); ax.set_title("(A) Isolated spiking population ($K=1000$, $G=0$)")
    ax.spines["top"].set_visible(False)
    # (B)-(E) global indicators vs G
    panels = [(axes[0, 1], "AC1", "(B) AC1$(GS)$ at 1 ms", "AC1$(GS)$", False),
              (axes[0, 2], "chi", r"(C) Susceptibility $\chi(GS)$", r"$\chi(GS)$", True),
              (axes[1, 0], "H", r"(D) DFA scaling exponent $H_\mathrm{DFA}(GS)$", r"$H_\mathrm{DFA}(GS)$", False),
              (axes[1, 1], "rho", r"(E) In-strength gradient vs. coupling", r"Spearman $\rho$(in-strength, $\chi(R_n)$)", False)]
    for ax, key, title, ylab, logy in panels:
        for reg, col in (("critical", C_CRIT), ("subcritical", C_SUB)):
            G, y = wb_series(reg, key)
            ax.plot(G, y, "-o", color=col, lw=1.8, ms=4,
                    label=("Spiking, critical" if reg == "critical" else "Spiking, subcritical"))
        G1, y1 = wb_series("critical", key, seed=1)
        ax.plot(G1, y1, "D", mfc="none", mec=C_CRIT, ms=8, mew=1.5, label="critical, independent run")
        if key == "H":
            ax.axhline(0.5, color="0.5", lw=0.8, ls=":", label=r"white noise ($H=0.5$)")
            ax.axhline(1.0, color="0.5", lw=0.8, ls="--", label=r"$1/f$ noise ($H\approx1$)")
        if key == "rho":
            ax.axhline(0, color="0.6", lw=0.5)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("G"); ax.set_ylabel(ylab); ax.set_title(title); hide(ax)
        if key == "H":
            ax.set_ylim(0.15, 1.08)
            ax.legend(frameon=False, loc="lower center", ncol=2, fontsize=10, columnspacing=1.0)
        else:
            ax.legend(frameon=False, loc="best")
    save(fig, "DallaPorta_Suppl_Figure_13")


def figS14():
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 13, "axes.labelsize": 14,
                         "xtick.labelsize": 12, "ytick.labelsize": 12})
    tp = D["tp"]
    a = tp["ff"]["a_par"]; a_c = a[3]
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 10.0))
    sc = tp["scan"]
    for ax, key, ylab, title, logy in ((axes[0, 0], "AC1", "AC1 (1 ms)", "(A) Isolated spiking population: AC1", False),
                                       (axes[1, 0], "chi", r"$\chi$", r"(D) Isolated spiking population: $\chi$", True)):
        ax.plot(sc["a"], sc[key], "k-o", ms=3, lw=1.5)
        ax.axvline(a_c, color=C_CRIT, ls="--", lw=1.2)
        for x in a:
            ax.axvline(x, color="0.85", lw=0.6, zorder=0)
        if logy:
            ax.set_yscale("log")
        ax.set_xlim(sc["a"][0] - 0.002, a[-1] + 0.002)
        ax.set_xlabel("$a$"); ax.set_ylabel(ylab); ax.set_title(title); hide(ax)
    for r, key, lab in ((0, "dAC1", r"$\Delta$AC1"), (1, "dlogchi", r"$\Delta\log_{10}\chi$")):
        for c, topo, t in ((1, "ff", "feedforward A$\\rightarrow$B"), (2, "fb", "mutual A$\\leftrightarrow$B")):
            ax = axes[r, c]
            M = tp[topo][key]
            v = np.nanmax(np.abs(M))
            im = ax.imshow(M.T, origin="lower", aspect="equal", cmap="Spectral_r", interpolation="gaussian",
                           extent=[a[0], a[-1], a[0], a[-1]], vmin=-v, vmax=v)
            ax.axhline(a_c, c="k", ls="--", lw=0.75); ax.axvline(a_c, c="k", ls="--", lw=0.75)
            ax.set_xticks([a[0], a_c, a[-1]]); ax.set_yticks([a[0], a_c, a[-1]])
            ax.set_xlabel(r"$a_A$"); ax.set_ylabel(r"$a_B$")
            ax.set_title(f"({'BCEF'[2 * r + c - 1]}) {lab}, {t}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save(fig, "DallaPorta_Suppl_Figure_14")


if __name__ == "__main__":
    figS13()
    figS14()

"""
Simple supplementary panels:
  S10  per-region local parameter a_n of the structured-heterogeneity simulation
       (derived_data/a_gradient_subcrit_v2.npy) against in-strength, with its random shuffle
  S12  TVB FirstOrderVolterra hemodynamic kernel used to compute BOLD
       (run_C2_local_BOLD_AC1.hrf_kernel)
Output: figures/DallaPorta_Suppl_Figure_10.{pdf,png}, DallaPorta_Suppl_Figure_12.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # src/ (paths.py, utilities)
import paths  # noqa: E402
from supplementary_utils import hrf_kernel, HRF_FS  # noqa: E402


def main():
    instr = np.load(paths.DERIVED / "instrength.npy")
    a = np.load(paths.DERIVED / "a_gradient_subcrit_v2.npy")
    a_rand = np.load(paths.DERIVED / "a_gradient_subcrit_random.npy")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].scatter(instr, a, color="tab:orange", s=20, label="in-strength-aligned gradient")
    ax[0].scatter(instr, a_rand, color="tab:red", marker="x", s=20, label="random shuffle")
    ax[0].axhline(0.973, color="tab:blue", ls=":", lw=1, label="critical ($a=0.973$)")
    ax[0].axhline(0.979, color="tab:green", ls=":", lw=1, label="subcritical ($a=0.979$)")
    ax[0].set_xlabel("In-strength"); ax[0].set_ylabel("$a_n$"); ax[0].legend(fontsize=7, frameon=False)
    ax[1].hist(a, bins=15, color="tab:orange", alpha=0.8)
    ax[1].set_xlabel("$a_n$"); ax[1].set_ylabel("Number of regions")
    for x in ax:
        for s in ("top", "right"):
            x.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(paths.FIGURES / f"DallaPorta_Suppl_Figure_10.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    h = hrf_kernel()
    t = np.arange(h.size) / HRF_FS
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(t, h / h.max(), color="k")
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("HRF (normalized)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(paths.FIGURES / f"DallaPorta_Suppl_Figure_12.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved DallaPorta_Suppl_Figure_10, DallaPorta_Suppl_Figure_12")


if __name__ == "__main__":
    main()

"""
Phase-randomized afferent control - figure for Reviewer C, point 1, sub-question (b).

Compares AC1 per region and AC1 of the global signal across four conditions, all at
the critical-regime working point G* = 0.028:
  - Uncoupled (G=0, no input)                                  - single TVB value
  - Coupled (original TVB sim, G=G*)                           - single TVB value
  - Open-loop replay with recorded coupling input               - n_reps replicates
  - Open-loop replay with phase-randomized coupling input       - n_reps replicates
        (independent IC + noise seeds; independent phase-randomization seeds)

Panel B shows mean +/- 1 SD across the n_reps replicates for the two open-loop
conditions, with the individual rep values overlaid as dots. The two TVB
reference conditions are shown as single bars without error bars.

Saves: paper/revision/figures/Fig_C1b1_phase_randomized.{pdf,png}
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

CONDITIONS = [
    # (label,                          per-region key,  GS key,           per-rep GS key,    color)
    ("uncoupled\n(G=0)",               "AC1_uncoupled", "AC1_GS_uncoupled", None,             "0.55"),
    ("coupled\n(G=G*, TVB)",           "AC1_coupled",   "AC1_GS_coupled",   None,             "tab:blue"),
    ("open-loop\nrecorded I",          "AC1_orig",      "AC1_GS_orig",      "AC1_GS_orig_reps", "tab:orange"),
    ("open-loop\nphase-rand. I",       "AC1_pr",        "AC1_GS_pr",        "AC1_GS_pr_reps",   "tab:red"),
]


def _hide_top_right(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main() -> None:
    with open(DERIVED / "C1b1_phase_randomized.pkl", "rb") as f:
        D = pickle.load(f)
    n_reps = int(D.get("n_reps", 1))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    # --- Panel A: per-region AC1 distributions (boxplot) ---
    ax = axes[0]
    data = [D[k] for _, k, _, _, _ in CONDITIONS]
    labels = [c[0] for c in CONDITIONS]
    colors = [c[4] for c in CONDITIONS]
    bp = ax.boxplot(data, tick_labels=labels, widths=0.55, showfliers=False, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set(facecolor=c, alpha=0.5, edgecolor=c)
    for median in bp["medians"]:
        median.set(color="black", lw=1.4)
    rng = np.random.default_rng(0)
    for i, (vals, c) in enumerate(zip(data, colors), start=1):
        x = rng.uniform(i - 0.12, i + 0.12, size=len(vals))
        ax.scatter(x, vals, color=c, alpha=0.45, s=8, edgecolor="none")
    ax.set_ylabel(r"AC1$(R_n)$  (per region)")
    ax.set_title("(A) Per-region AC1")
    ax.tick_params(axis="x", labelsize=8)
    _hide_top_right(ax)

    # --- Panel B: AC1 of global signal with error bars where reps are available ---
    ax = axes[1]
    bar_x = np.arange(len(CONDITIONS))

    means = []
    sds = []
    rep_arrs = []
    for _, _, gs_key, gs_reps_key, _ in CONDITIONS:
        means.append(float(D[gs_key]))
        if gs_reps_key is None:
            sds.append(0.0)
            rep_arrs.append(None)
        else:
            reps = np.asarray(D[gs_reps_key], dtype=np.float64)
            sds.append(float(reps.std(ddof=1)) if reps.size > 1 else 0.0)
            rep_arrs.append(reps)

    # bars (no yerr where SD is zero, i.e. single-TVB-value reference conditions)
    yerr_for_bars = [sd if sd > 0 else np.nan for sd in sds]
    for x, mean, sd, c in zip(bar_x, means, yerr_for_bars, colors):
        ax.bar(
            x, mean, color=c, width=0.6, alpha=0.65, edgecolor="0.2",
            yerr=sd if np.isfinite(sd) else None,
            capsize=4, error_kw=dict(elinewidth=1.4, ecolor="0.15"),
        )

    # overlay per-rep dots for replicated conditions
    rng2 = np.random.default_rng(1)
    for x, reps, c in zip(bar_x, rep_arrs, colors):
        if reps is None:
            continue
        jitter = rng2.uniform(-0.13, 0.13, size=reps.size)
        ax.scatter(x + jitter, reps, color="0.15", s=14, alpha=0.7, edgecolor="none", zorder=5)

    # annotate values
    for x, mean, sd in zip(bar_x, means, sds):
        if sd > 0:
            ax.text(x, mean + sd + 0.020, f"{mean:.3f}\n$\\pm$ {sd:.3f}",
                    ha="center", va="bottom", fontsize=8.5)
        else:
            ax.text(x, mean + 0.020, f"{mean:.3f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(bar_x)
    ax.set_xticklabels(labels, fontsize=8)
    y_top = max(m + s for m, s in zip(means, sds)) * 1.22
    ax.set_ylim(0, y_top)
    ax.set_ylabel(r"AC1$(GS)$  (global signal)")
    ax.set_title(f"(B) Global-signal AC1 (Fig.\\ 4D headline), $n = {n_reps}$ replicates")
    _hide_top_right(ax)

    fig.suptitle(
        f"Phase-randomized afferent control at $G^\\ast = {float(D['G_star']):.3f}$",
        fontsize=11, y=0.995,
    )
    fig.tight_layout()

    pdf = OUT_FIG / "Fig_C1b1_phase_randomized.pdf"
    png = OUT_FIG / "Fig_C1b1_phase_randomized.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    print(f"Saved {pdf}")
    print(f"Saved {png}")

    print()
    print(f"AC1(GS) summary (n={n_reps} replicates for open-loop conditions):")
    for (label, _, _, gs_reps_key, _), mean, sd in zip(CONDITIONS, means, sds):
        tag = f"+/- {sd:.4f} (n={n_reps})" if gs_reps_key is not None else "(no reps)"
        print(f"  {label.replace(chr(10), ' '):28s}  AC1(GS) = {mean:+.4f}  {tag}")


if __name__ == "__main__":
    main()

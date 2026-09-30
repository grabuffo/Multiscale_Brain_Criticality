"""
Figure 1 with panel D (avalanche statistics) redrawn from avalanche_statistics.py.

Panels A-C are taken unchanged from the published figure (DallaPorta_Figure_1.pdf,
rendered at 600 dpi). Panel D shows the size distribution P(S) (S = area between the
threshold and |R| over each excursion below the median, events with S >= 0.01), the
duration distribution P(T) (T = number of time steps) and S vs T in the critical regime,
with reference lines at the exponents fitted in the critical regime.
Output: figures/DallaPorta_Figure_1.png
"""
from pathlib import Path
import pickle
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # src/ (paths.py, utilities)
import paths  # noqa: E402
import Utils as fx  # noqa: E402

PUB_PDF = paths.FIG1_PDF
CD = pickle.load(open(paths.DERIVED / "avalanche_statistics.pkl", "rb"))
SET1 = plt.get_cmap("Set1")
COLS = [SET1(0), SET1(1), SET1(2)]
DPI = 600
X0, X1, Y0, Y1 = 0.338, 0.998, 0.628, 0.885        # panel-D box (fractions of the page)


def main():
    png = paths.DERIVED / "fig1_published_600dpi.png"
    if not png.exists():
        subprocess.run(["pdftocairo", "-png", "-r", str(DPI), "-singlefile", str(PUB_PDF),
                        str(png.with_suffix(""))], check=True)
    page = plt.imread(png)[..., :3]
    H, W = page.shape[:2]
    r0, r1, c0, c1 = int(Y0 * H), int(Y1 * H), int(X0 * W), int(X1 * W)
    w_in, h_in = (c1 - c0) / DPI, (r1 - r0) / DPI

    plt.rcParams.update({"font.size": 6.8})
    fig = plt.figure(figsize=(w_in, h_in), facecolor="white", dpi=DPI)
    # axes at the positions of the published panel D
    axes = [fig.add_axes([x, 0.243, w, 0.68]) for x, w in ((0.085, 0.215), (0.415, 0.215), (0.745, 0.215))]
    S = [x[x >= 1e-2] for x in CD["S"]]   # area between R and threshold (events with S >= 0.01)
    T = CD["T"]                             # duration (time steps)
    for j in range(3):
        fx.plot_pdf(S[j], ax=axes[0], color=COLS[j])
        fx.plot_pdf(T[j], ax=axes[1], color=COLS[j])

    def anchor(v, x0):
        """log-binned pdf value of the critical-regime data at x0 (to place a reference line)."""
        b = np.logspace(np.log10(v.min()), np.log10(v.max()), 30)
        h, e = np.histogram(v, b, density=True)
        c = np.sqrt(e[1:] * e[:-1])
        return np.exp(np.interp(np.log(x0), np.log(c[h > 0]), np.log(h[h > 0])))

    # reference lines at the exponents fitted in the critical regime (Methods), offset above the data
    for ax, lab, v, xs, expo in ((axes[0], "S", S[1], np.array([1.0, 200.0]), 1.4),
                                 (axes[1], "T", T[1], np.array([150.0, 6000.0]), 1.8)):
        c = 4 * anchor(v, xs[0]) * xs[0] ** expo
        ax.plot(xs, c * xs ** -expo, "k--")
        ax.text(xs[0] * 1.6, c * xs[0] ** -expo * 1.3, rf"$\propto {lab}^{{-{expo}}}$")
        ax.set_xlabel(lab); ax.set_ylabel(f"P({lab})")
    axes[2].scatter(T[1], CD["S"][1], s=1, color=COLS[1])
    axes[2].set_xscale("log"); axes[2].set_yscale("log")
    k = CD["T"][1] > 30
    c3 = 4 * np.exp(np.median(np.log(CD["S"][1][k].clip(1e-12)) - 2 * np.log(CD["T"][1][k])))
    x3 = np.array([30, 8000], dtype=float)
    axes[2].plot(x3, c3 * x3 ** 2.0, "k--"); axes[2].text(12, c3 * 40 ** 2 * 60, r"$\propto T^{2.0}$")
    axes[2].set_ylabel(r"$\langle S \rangle$"); axes[2].set_xlabel("T")
    fig.canvas.draw()
    panel = np.asarray(fig.canvas.buffer_rgba())[..., :3] / 255.0
    plt.close(fig)
    panel = panel[: r1 - r0, : c1 - c0]
    out = page.copy()
    out[r0:r0 + panel.shape[0], c0:c0 + panel.shape[1]] = panel
    plt.imsave(paths.FIGURES / "DallaPorta_Figure_1.png", out, dpi=DPI)
    print("  -> DallaPorta_Figure_1.png")


if __name__ == "__main__":
    main()

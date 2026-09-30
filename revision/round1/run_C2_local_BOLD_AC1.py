"""
Reviewer C, point 2 — does the local AC1 vs a profile survive HRF filtering?

For a single isolated NMM (decoupled, G = 0), we scan the parameter a along the
sigma = 0.677 slice of Fig. 1B (bottom panel) and compare:
  - AC1 of the raw mean-field R(t) at 2 kHz (matches Fig. 1B bottom)
  - AC1 of the BOLD-like signal obtained by convolving R(t) with the TVB
    FirstOrderVolterra HRF kernel used elsewhere in the manuscript, then
    sampling at TR = 1 s

If both profiles peak at the same a (the critical regime), the slow-coordination
signature survives hemodynamic filtering at the local-population level.
This complements the whole-brain cross-scale analysis already shown in
Suppl. Fig. S11.

Numerical scheme: stochastic-Heun, dt = 0.05 ms (20 kHz), matching the TVB
whole-brain runs. Integration is batched across the a grid: one Python step
advances all parameter values simultaneously.

Saves: paper/revision/derived_data/C2_local_BOLD_AC1.pkl
       paper/revision/figures/Fig_C2_local_BOLD_AC1.{pdf,png}
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # revision/ (paths.py)
import paths  # noqa: E402
_sys.path.insert(0, str(paths.SRC))  # original src/ (functions.py, Utils.py, ...)
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import revision_utils as ru  # noqa: E402

PROJECT_ROOT = paths.REPO
DERIVED = paths.DERIVED
DERIVED.mkdir(parents=True, exist_ok=True)
OUT_FIG = paths.FIGURES
OUT_FIG.mkdir(parents=True, exist_ok=True)

# ---- Local NMM parameters (match Fig. 1B bottom slice) ----
SIGMA = 0.6725             # exact value used in TVB whole-brain runs
J = 1.25
OMEGA = 1.0
ETA = 0.0001                # noise on Psi (matches the manuscript "N=0.0001")
DT = 0.05                   # ms
SIM_LENGTH_MS = 300_000     # 300 s per a; >> AC1 stabilisation length
DOWNSAMPLE = 10             # save R at 2 kHz (factor 10 vs 20 kHz integration)
BURN_IN_SAMPLES = 5_000     # at 2 kHz = 2.5 s discard

# a-grid spanning the supercritical, critical, subcritical anchors of Fig. 1B-C
# (red 0.967, blue 0.973, green 0.979 in the manuscript).
A_GRID = np.linspace(0.960, 0.985, 26)

# Anchor values reported as colored markers
A_ANCHORS = {"supercritical": 0.967, "critical": 0.973, "subcritical": 0.979}
A_COLORS = {"supercritical": "tab:red", "critical": "tab:blue", "subcritical": "tab:green"}

# ---- HRF kernel (TVB FirstOrderVolterra defaults) ----
TAU_S = 0.8                  # s
TAU_F = 0.4                  # s
HRF_LENGTH_S = 20.0          # TVB hrf_length default
HRF_FS = 250.0               # TVB internal HRF sampling
TR = 1.0                     # s (Bold monitor period = 1000 ms)

# AC1 lag conventions
AC1_LAG_R = 20               # 10 ms at 2 kHz, same lag as ru.global_AC1 default
AC1_LAG_BOLD = 1             # 1 s at TR = 1 s

# Boundary on R (matches TVB MF_LG_CouplingR state_variable_boundaries)
R_MIN = 1e-6


def hrf_kernel() -> np.ndarray:
    """TVB FirstOrderVolterra kernel: h(t) = (1/3) exp(-t/(2*tau_s)) sin(omega t)/omega."""
    omega_k = np.sqrt(1.0 / TAU_F - 1.0 / (4.0 * TAU_S * TAU_S))
    t = np.arange(0, HRF_LENGTH_S, 1.0 / HRF_FS)
    h = (1.0 / 3.0) * np.exp(-t / (2.0 * TAU_S)) * np.sin(omega_k * t) / omega_k
    return h


def integrate_local_batch(a_grid: np.ndarray, sim_length_ms: float, seed: int = 0):
    """Batched stochastic-Heun integration of n_a independent single-NMMs.

    All NMMs share sigma, J, omega, eta but each has its own value of a.
    Returns R_traj of shape (n_save, n_a) at 2 kHz.
    """
    n_a = a_grid.size
    n_steps = int(sim_length_ms / DT)
    n_save = n_steps // DOWNSAMPLE

    rng = np.random.default_rng(seed)
    R = rng.uniform(0.2, 0.8, size=n_a)
    Psi = rng.uniform(0.0, 2.0 * np.pi, size=n_a)

    sqrt_2eta_dt = np.sqrt(2.0 * ETA * DT)
    R_traj = np.empty((n_save, n_a), dtype=np.float64)
    save_idx = 0

    a = a_grid
    J_ = J
    sigma2 = SIGMA * SIGMA
    omega_ = OMEGA

    for t_step in range(n_steps):
        cos_p = np.cos(Psi)
        sin_p = np.sin(Psi)
        one_minus_R2 = 1.0 - R * R
        one_plus_R2 = 1.0 + R * R

        f_R = 0.5 * R * (J_ * one_minus_R2 - sigma2) - 0.5 * a * one_minus_R2 * cos_p
        f_P = omega_ + a * one_plus_R2 * sin_p / (2.0 * R)

        dW = rng.standard_normal(n_a)
        noise = sqrt_2eta_dt * dW

        R_pred = R + DT * f_R
        Psi_pred = Psi + DT * f_P + noise
        np.maximum(R_pred, R_MIN, out=R_pred)

        cos_pp = np.cos(Psi_pred)
        sin_pp = np.sin(Psi_pred)
        one_minus_R2p = 1.0 - R_pred * R_pred
        one_plus_R2p = 1.0 + R_pred * R_pred

        f_R_p = 0.5 * R_pred * (J_ * one_minus_R2p - sigma2) - 0.5 * a * one_minus_R2p * cos_pp
        f_P_p = omega_ + a * one_plus_R2p * sin_pp / (2.0 * R_pred)

        R = R + 0.5 * DT * (f_R + f_R_p)
        Psi = Psi + 0.5 * DT * (f_P + f_P_p) + noise
        np.maximum(R, R_MIN, out=R)

        if t_step % DOWNSAMPLE == 0 and save_idx < n_save:
            R_traj[save_idx] = R
            save_idx += 1

    return R_traj[:save_idx, :]


def R_to_BOLD(R_2k: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Resample R from 2 kHz to HRF_FS, convolve with h, sample at TR.

    Mirrors the TVB Bold monitor pipeline at the level of the kernel/sampling.
    """
    # 2 kHz -> 250 Hz: average over 8 samples per HRF_FS bin
    factor = int(round(2000.0 / HRF_FS))
    n_full = (R_2k.shape[0] // factor) * factor
    R_resampled = R_2k[:n_full].reshape(-1, factor, R_2k.shape[1]).mean(axis=1)

    # convolve along time axis (per-region)
    BOLD_full = np.empty_like(R_resampled)
    for j in range(R_resampled.shape[1]):
        BOLD_full[:, j] = fftconvolve(R_resampled[:, j], h, mode="full")[:R_resampled.shape[0]]

    # sample at TR by stepping every (HRF_FS * TR) points
    step_to_TR = int(round(HRF_FS * TR))
    return BOLD_full[::step_to_TR, :]


def main() -> None:
    h = hrf_kernel()
    print(f"[hrf  ] length = {h.size / HRF_FS:.2f} s, peak at t = {np.argmax(h) / HRF_FS:.2f} s")
    print(f"[grid ] n_a = {A_GRID.size}, range = [{A_GRID.min():.3f}, {A_GRID.max():.3f}]")
    print(f"[sim  ] T = {SIM_LENGTH_MS / 1000:.0f} s @ dt = {DT} ms (batched over a)")

    t0 = time.time()
    R_traj = integrate_local_batch(A_GRID, SIM_LENGTH_MS, seed=0)
    print(f"[int  ] R_traj.shape = {R_traj.shape}, elapsed {time.time() - t0:.0f} s")

    # Discard burn-in
    R_eq = R_traj[BURN_IN_SAMPLES:, :]
    print(f"[trim ] usable R length = {R_eq.shape[0] / 2000:.1f} s")

    # AC1(R) at 2 kHz, lag = 20 samples (consistent with rest of project)
    AC1_R = np.array([ru.AC1(R_eq[:, ja], n=AC1_LAG_R) for ja in range(A_GRID.size)])

    # Convert to BOLD via TVB FirstOrderVolterra kernel, sample at TR = 1 s
    BOLD = R_to_BOLD(R_eq, h)
    print(f"[bold ] BOLD.shape = {BOLD.shape} at TR = {TR} s")

    # AC1(BOLD) at TR=1s, lag 1 sample = 1 s
    AC1_BOLD = np.array([ru.AC1(BOLD[:, ja], n=AC1_LAG_BOLD) for ja in range(A_GRID.size)])

    # ---- Print summary ----
    print()
    print(f"{'a':>7s}  {'AC1(R)':>8s}  {'AC1(BOLD)':>10s}  flag")
    for ja, a in enumerate(A_GRID):
        flag = ""
        for name, av in A_ANCHORS.items():
            if abs(a - av) < 1e-3:
                flag = f"<-- {name}"
        print(f"{a:>7.4f}  {AC1_R[ja]:>+8.4f}  {AC1_BOLD[ja]:>+10.4f}  {flag}")
    a_star_R = A_GRID[int(np.argmax(AC1_R))]
    a_star_BOLD = A_GRID[int(np.argmax(AC1_BOLD))]
    print()
    print(f"a* (max AC1(R))    = {a_star_R:.4f}")
    print(f"a* (max AC1(BOLD)) = {a_star_BOLD:.4f}")

    # ---- Save ----
    out = {
        "a_grid":  A_GRID,
        "AC1_R":   AC1_R,
        "AC1_BOLD": AC1_BOLD,
        "sigma":   SIGMA,
        "J":       J,
        "omega":   OMEGA,
        "eta":     ETA,
        "dt":      DT,
        "sim_length_ms": SIM_LENGTH_MS,
        "tau_s":   TAU_S,
        "tau_f":   TAU_F,
        "TR":      TR,
        "AC1_lag_R":    AC1_LAG_R,
        "AC1_lag_BOLD": AC1_LAG_BOLD,
        "a_anchors":    A_ANCHORS,
    }
    pkl_path = DERIVED / "C2_local_BOLD_AC1.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(out, f)
    print(f"Saved {pkl_path}")

    # ---- Figure ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)

    for name, av in A_ANCHORS.items():
        for ax in axes:
            ax.axvline(av, color=A_COLORS[name], lw=0.9, ls=":", alpha=0.7,
                        label=f"{name} ($a = {av}$)")

    # Common y-axis range so the flatness of panel B is visually evident vs panel A's peak
    y_lo = min(AC1_R.min(), AC1_BOLD.min(), 0.0) - 0.05
    y_hi = max(AC1_R.max(), AC1_BOLD.max()) + 0.1

    axes[0].plot(A_GRID, AC1_R, color="0.15", lw=1.8, marker="o", markersize=4)
    axes[0].set_xlabel(r"$a$")
    axes[0].set_ylabel(r"AC1$(R)$  (raw, 2 kHz)")
    axes[0].set_title(r"(A) AC1 of $R(t)$ vs $a$ at $\sigma = $" + f"{SIGMA}")
    axes[0].set_ylim(y_lo, y_hi)
    axes[0].legend(fontsize=8, frameon=False, loc="best")
    for s in ("top", "right"):
        axes[0].spines[s].set_visible(False)

    axes[1].plot(A_GRID, AC1_BOLD, color="tab:purple", lw=1.8, marker="o", markersize=4)
    # White-noise reference: AC1(BOLD) at lag 1s for white-noise R, given the HRF kernel
    omega_k_local = np.sqrt(1.0 / TAU_F - 1.0 / (4.0 * TAU_S * TAU_S))
    lag_samples_local = int(round(HRF_FS * TR))
    h_local = (1.0 / 3.0) * np.exp(-(np.arange(int(HRF_LENGTH_S * HRF_FS)) / HRF_FS) / (2.0 * TAU_S)) \
               * np.sin(omega_k_local * (np.arange(int(HRF_LENGTH_S * HRF_FS)) / HRF_FS)) / omega_k_local
    ac1_white_ref = float(np.sum(h_local[:h_local.size - lag_samples_local] * h_local[lag_samples_local:]) / np.sum(h_local * h_local))
    axes[1].axhline(ac1_white_ref, color="0.4", lw=0.9, ls="--",
                     label=f"white-noise $R$ reference ({ac1_white_ref:.2f})")
    axes[1].set_xlabel(r"$a$")
    axes[1].set_ylabel(r"AC1$(\mathrm{BOLD})$  (TR $= 1$~s)")
    axes[1].set_title(r"(B) AC1 of BOLD signal vs $a$ (TVB FirstOrderVolterra kernel)")
    axes[1].set_ylim(y_lo, y_hi)
    axes[1].legend(fontsize=8, frameon=False, loc="best")
    for s in ("top", "right"):
        axes[1].spines[s].set_visible(False)

    fig.suptitle(
        f"Local AC1 across the $\\sigma = {SIGMA}$ slice (single NMM, decoupled, $G = 0$):\n"
        f"raw $R(t)$ vs.\\ HRF-convolved BOLD ($\\tau_s = {TAU_S}$~s, $\\tau_f = {TAU_F}$~s, TR $= {TR}$~s, sim length $= {SIM_LENGTH_MS/1000:.0f}$~s).",
        fontsize=10.0, y=1.04,
    )
    fig.tight_layout()
    pdf = OUT_FIG / "Fig_C2_local_BOLD_AC1.pdf"
    png = OUT_FIG / "Fig_C2_local_BOLD_AC1.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    print(f"Saved {pdf}")
    print(f"Saved {png}")


if __name__ == "__main__":
    main()

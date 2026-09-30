"""
Phase-randomized afferent control — direct test of the temporal-smoothing concern
raised in Reviewer C, point 1.

Procedure
---------
1. Load the published critical-regime Allen simulation at the working point G* = 0.028.
   Extract the recorded coupling input I_MF[t, n] (the "INPUT" array).
2. Replace each region's input column I_MF[:, n] with a Fourier phase-randomized surrogate
   I_MF_PR[:, n] that preserves the power spectrum of I_MF[:, n] but destroys its causal
   temporal structure.
3. Drive the *uncoupled* mean-field model with the surrogate input as the only forcing on
   Psi_n(t).  All local parameters (J, a, sigma, omega, noise eta) match the published
   critical-regime values.
4. Validation: also run the uncoupled mean-field with the ORIGINAL (non-randomized) input —
   this should reproduce R_n(t) from the coupled sim, validating the integrator.

If AC1 of the surrogate-driven simulation collapses toward the uncoupled (G=0) baseline,
the AC1 modulation in Fig. 5 is genuinely dynamical, not passive smoothing.
If AC1 stays near the coupled value, passive smoothing dominates.

Numerical scheme: stochastic Heun, dt = 0.5 ms (2 kHz), matching the saved INPUT rate.

Cache: paper/revision/derived_data/C1b1_phase_randomized.pkl
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # revision/ (paths.py)
import paths  # noqa: E402
_sys.path.insert(0, str(paths.SRC))  # original src/ (functions.py, Utils.py, ...)
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import revision_utils as ru  # noqa: E402

PROJECT_ROOT = paths.REPO
DERIVED = paths.DERIVED
DERIVED.mkdir(parents=True, exist_ok=True)

SIM_DIR = paths.SIM_ROOT / "Gscan_connectome_crit"

# Critical-regime parameters (match notebook 5)
J_LOCAL = 1.25
A_CRIT = 0.973
SIGMA = 0.6725
OMEGA = 1.0
NOISE_AMP = 0.0001  # eta on Psi
NOISE_SEED = 0
DT_SIM = 0.05        # ms — matches the original TVB simulation step (20 kHz)
INPUT_UPSAMPLE = 10  # saved INPUT is at 2 kHz, hold-upsample to 20 kHz
TLEN = 5000
G_INDEX_STAR = 2     # G* = 0.028

# Boundary on R (matches MF_LG_CouplingR.state_variable_boundaries)
R_MIN = 1e-6


def phase_randomize(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Fourier phase randomization preserving the power spectrum of x."""
    X = np.fft.rfft(x)
    n_freq = X.size
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_freq)
    phases[0] = 0.0
    if x.size % 2 == 0:
        phases[-1] = 0.0
    X_rand = np.abs(X) * np.exp(1j * phases)
    return np.fft.irfft(X_rand, n=x.size)


def integrate_uncoupled_with_forcing(
    I_forcing: np.ndarray,
    dt: float,
    J: float,
    a: float,
    sigma: float,
    omega: float,
    eta: float,
    seed: int,
    R0: np.ndarray | None = None,
    Psi0: np.ndarray | None = None,
):
    """Stochastic-Heun integration of the *uncoupled* mean-field model with forcing on Psi.

    Dynamics:
        dR/dt   = 0.5 R [J(1 - R^2) - sigma^2] - 0.5 a (1 - R^2) cos(Psi)
        dPsi/dt = omega + a (1 + R^2) sin(Psi) / (2R) + I_forcing(t) + eta * dW/dt

    Parameters
    ----------
    I_forcing : (n_steps, N) array — forcing applied to dPsi at each step.
    """
    n_steps, N = I_forcing.shape
    rng = np.random.default_rng(seed)
    if R0 is None:
        R0 = rng.uniform(0.01, 1.0, size=N)
    if Psi0 is None:
        Psi0 = rng.uniform(0.0, 2.0 * np.pi, size=N)

    R_traj = np.empty((n_steps, N), dtype=np.float64)
    Psi_traj = np.empty((n_steps, N), dtype=np.float64)
    R = R0.copy()
    Psi = Psi0.copy()
    R_traj[0] = R
    Psi_traj[0] = Psi

    # TVB Additive noise convention: gfun = sqrt(2*nsig), so per-step noise std = sqrt(2*nsig*dt).
    sqrt_2nsig_dt = np.sqrt(2.0 * eta * dt)

    for t in range(n_steps - 1):
        I_t = I_forcing[t]
        cos_p = np.cos(Psi)
        sin_p = np.sin(Psi)
        # f at current state
        f_R = 0.5 * R * (J * (1.0 - R * R) - sigma * sigma) - 0.5 * a * (1.0 - R * R) * cos_p
        f_P = omega + a * (1.0 + R * R) * sin_p / (2.0 * R) + I_t

        # Single stochastic increment shared between predictor and corrector
        dW = rng.standard_normal(N)
        noise_inc = sqrt_2nsig_dt * dW

        # Predictor (Heun)
        R_pred = R + dt * f_R
        Psi_pred = Psi + dt * f_P + noise_inc
        R_pred = np.maximum(R_pred, R_MIN)

        # f at predicted state
        cos_pp = np.cos(Psi_pred)
        sin_pp = np.sin(Psi_pred)
        f_R_p = (
            0.5 * R_pred * (J * (1.0 - R_pred * R_pred) - sigma * sigma)
            - 0.5 * a * (1.0 - R_pred * R_pred) * cos_pp
        )
        f_P_p = omega + a * (1.0 + R_pred * R_pred) * sin_pp / (2.0 * R_pred) + I_t

        # Corrector
        R = R + 0.5 * dt * (f_R + f_R_p)
        Psi = Psi + 0.5 * dt * (f_P + f_P_p) + noise_inc
        R = np.maximum(R, R_MIN)
        R_traj[t + 1] = R
        Psi_traj[t + 1] = Psi

    return R_traj, Psi_traj


def _ic_seed(rep: int) -> int:
    """Seed for sampling R0, Psi0 in replicate `rep`."""
    return 11_22_33 + rep


def _noise_seed(rep: int) -> int:
    """Seed for SDE noise in replicate `rep`."""
    return 100 + rep


def _pr_seed(rep: int) -> int:
    """Seed for Fourier phase-randomization of the input in replicate `rep`."""
    return 42 + rep


def _sample_ICs(rng: np.random.Generator, N: int):
    R0 = rng.uniform(0.01, 1.0, size=N)
    Psi0 = rng.uniform(0.0, 2.0 * np.pi, size=N)
    return R0, Psi0


def _run_open_loop(I_forcing_20k, R0, Psi0, noise_seed):
    """Integrate uncoupled MF with prescribed forcing, downsample to 2 kHz, return R_2k."""
    R_traj, _ = integrate_uncoupled_with_forcing(
        I_forcing=I_forcing_20k,
        dt=DT_SIM, J=J_LOCAL, a=A_CRIT, sigma=SIGMA, omega=OMEGA, eta=NOISE_AMP,
        seed=noise_seed, R0=R0, Psi0=Psi0,
    )
    return R_traj[::INPUT_UPSAMPLE, :]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reps", type=int, default=25,
                    help="Number of independent replicates for the open-loop and "
                         "phase-randomized conditions (default 25).")
    args = ap.parse_args()
    n_reps = args.n_reps

    iG_star = G_INDEX_STAR
    pkl = SIM_DIR / f"data_G{iG_star}.pkl"
    print(f"[load ] {pkl}")
    with open(pkl, "rb") as f:
        D = pickle.load(f)
    R_coupled = D["R"]                        # (T, 60) at 2 kHz
    INPUT_recorded_2k = D["INPUT"]            # (T, 60) coupling input at 2 kHz
    print(f"[shape] R_coupled={R_coupled.shape}  INPUT(2 kHz)={INPUT_recorded_2k.shape}")

    # Hold-upsample INPUT from 2 kHz to 20 kHz once; reused by all replicates.
    INPUT_recorded = np.repeat(INPUT_recorded_2k, INPUT_UPSAMPLE, axis=0)
    n_steps, N = INPUT_recorded.shape
    print(f"[sim  ] n_steps={n_steps}  N={N}  dt={DT_SIM} ms  total={n_steps*DT_SIM/1000:.1f} s")
    print(f"[reps ] n_reps={n_reps}")

    # ===== Reference values from the original TVB simulation (single values) =====
    AC1_coupled = np.array([ru.AC1(R_coupled[TLEN:, j], n=20) for j in range(N)])
    AC1_GS_coupled = ru.global_AC1(R_coupled, tlen=TLEN)

    R_g0 = pickle.load(open(SIM_DIR / "data_G0.pkl", "rb"))["R"]
    AC1_uncoupled = np.array([ru.AC1(R_g0[TLEN:, j], n=20) for j in range(N)])
    AC1_GS_uncoupled = ru.global_AC1(R_g0, tlen=TLEN)

    # ===== Replicates =====
    AC1_orig_reps = np.empty((n_reps, N), dtype=np.float64)
    AC1_pr_reps = np.empty((n_reps, N), dtype=np.float64)
    AC1_GS_orig_reps = np.empty(n_reps, dtype=np.float64)
    AC1_GS_pr_reps = np.empty(n_reps, dtype=np.float64)

    t_total = time.time()
    for rep in range(n_reps):
        # Initial conditions: fresh draw per rep
        rng_init = np.random.default_rng(_ic_seed(rep))
        R0, Psi0 = _sample_ICs(rng_init, N)

        # --- (1) Recorded-input replay: vary IC + noise; input is fixed ---
        t0 = time.time()
        R_orig_2k = _run_open_loop(INPUT_recorded, R0, Psi0, _noise_seed(rep))
        for j in range(N):
            AC1_orig_reps[rep, j] = ru.AC1(R_orig_2k[TLEN:, j], n=20)
        AC1_GS_orig_reps[rep] = ru.global_AC1(R_orig_2k, tlen=TLEN)
        t_orig = time.time() - t0

        # --- (2) Phase-randomized replay: vary IC + noise + phase-randomization seed ---
        rng_pr = np.random.default_rng(_pr_seed(rep))
        INPUT_PR_2k = np.empty_like(INPUT_recorded_2k)
        for n in range(N):
            INPUT_PR_2k[:, n] = phase_randomize(INPUT_recorded_2k[:, n], rng_pr)
        INPUT_PR = np.repeat(INPUT_PR_2k, INPUT_UPSAMPLE, axis=0)

        t0 = time.time()
        R_pr_2k = _run_open_loop(INPUT_PR, R0, Psi0, _noise_seed(rep))
        for j in range(N):
            AC1_pr_reps[rep, j] = ru.AC1(R_pr_2k[TLEN:, j], n=20)
        AC1_GS_pr_reps[rep] = ru.global_AC1(R_pr_2k, tlen=TLEN)
        t_pr = time.time() - t0

        elapsed = time.time() - t_total
        print(
            f"  rep {rep+1:2d}/{n_reps}: "
            f"AC1(GS) recorded={AC1_GS_orig_reps[rep]:+.4f} ({t_orig:.0f}s)  "
            f"phase-rand={AC1_GS_pr_reps[rep]:+.4f} ({t_pr:.0f}s)  "
            f"[cumulative {elapsed/60:.1f} min]"
        )

    # ===== Aggregates: mean across reps, for backward-compatible single-value keys =====
    AC1_orig = AC1_orig_reps.mean(axis=0)
    AC1_pr = AC1_pr_reps.mean(axis=0)
    AC1_GS_orig = float(AC1_GS_orig_reps.mean())
    AC1_GS_pr = float(AC1_GS_pr_reps.mean())

    print()
    print("=== AC1 of global signal GS(t) (headline of Fig. 4D) ===")
    print(f"  G=0 (uncoupled, original sim):    AC1(GS) = {AC1_GS_uncoupled:+.4f}")
    print(f"  G* coupled (original sim):        AC1(GS) = {AC1_GS_coupled:+.4f}")
    print(
        f"  G* uncoupled MF + recorded I:     "
        f"AC1(GS) = {AC1_GS_orig:+.4f} +/- {AC1_GS_orig_reps.std(ddof=1):.4f}  "
        f"(n={n_reps}; min={AC1_GS_orig_reps.min():+.4f}, max={AC1_GS_orig_reps.max():+.4f})"
    )
    print(
        f"  G* uncoupled MF + PHASE-RAND I:   "
        f"AC1(GS) = {AC1_GS_pr:+.4f} +/- {AC1_GS_pr_reps.std(ddof=1):.4f}  "
        f"(n={n_reps}; min={AC1_GS_pr_reps.min():+.4f}, max={AC1_GS_pr_reps.max():+.4f})"
    )

    out = {
        "G_star_index": iG_star,
        "G_star": ru.G_GRID[iG_star],
        "n_reps": n_reps,
        # Single-value reference conditions (from the original TVB simulation)
        "AC1_coupled":   AC1_coupled,
        "AC1_uncoupled": AC1_uncoupled,
        "AC1_GS_coupled":   AC1_GS_coupled,
        "AC1_GS_uncoupled": AC1_GS_uncoupled,
        # Backward-compatible mean-across-reps values
        "AC1_orig":      AC1_orig,
        "AC1_pr":        AC1_pr,
        "AC1_GS_orig":   AC1_GS_orig,
        "AC1_GS_pr":     AC1_GS_pr,
        # Per-replicate arrays
        "AC1_orig_reps":    AC1_orig_reps,     # (n_reps, 60)
        "AC1_pr_reps":      AC1_pr_reps,       # (n_reps, 60)
        "AC1_GS_orig_reps": AC1_GS_orig_reps,  # (n_reps,)
        "AC1_GS_pr_reps":   AC1_GS_pr_reps,    # (n_reps,)
    }
    out_path = DERIVED / "C1b1_phase_randomized.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()

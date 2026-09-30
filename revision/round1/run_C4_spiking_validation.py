"""
Reviewer C, point 4 — mean-field validity for G != 0.

Single-region spiking validation of the Ott--Antonsen mean-field reduction's
"R_m is a sufficient statistic for inter-regional input" assumption.

Test (option ii from discussion with the user):
  - simulate ONE spiking active-rotator network for a chosen region n;
  - feed it the inter-regional input it would have received in the closed-loop MF
    whole-brain simulation, computed online from the recorded R_m(t), Psi_m(t)
    of all other regions m != n at G = G*;
  - compare R_spike(t) against R_MF(t) (the MF prediction from the same run).

If the spiking R(t) tracks the MF R(t), the sufficient-statistic assumption is
supported even at intermediate coupling near criticality. If not, the discrepancy
quantifies the limit of the MF reduction.

Equations follow main.tex Eq. 1:
    dphi_i/dt = omega + a sin(phi_i) + (J/K) sum_j sin(phi_j - phi_i)
                + sigma eta_i(t) + I_i(t)
where I_i(t) is the inter-regional input.

Within-region coupling is rewritten in mean-field form for speed:
    (J/K) sum_j sin(phi_j - phi_i) = -J R sin(phi_i - Psi)
with R, Psi the local population's order parameter computed at every step.

Inter-regional input is rewritten in cos/sin form:
    I_i(t) = A(t) cos(phi_i) - B(t) sin(phi_i)
with
    A(t) = G sum_{m != n} W_nm R_m(t) sin(Psi_m(t))
    B(t) = G sum_{m != n} W_nm R_m(t) cos(Psi_m(t))

Numerics: stochastic Heun integration, dt = 0.05 ms (20 kHz, matching the
published TVB run). Recorded MF R_m, Psi_m at 2 kHz are hold-upsampled to 20 kHz
to match the integration rate.

Outputs (paper/revision/derived_data/):
  C4_spiking_<label>.pkl  - {'R_spike': R(t)@2kHz, 'meta': params, 'AC1': float}
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

# Critical-regime parameters as in functions_simulator.py (slightly different
# from the rounded 0.677 in main.tex; we use the value actually used in the runs).
A_CRIT = 0.973
A_SUPER = 0.967
A_SUB = 0.979
SIGMA = 0.6725
OMEGA = 1.0
J_LOCAL = 1.25
DT_SIM = 0.05  # ms

# Recorded MF data is at 2 kHz; spiking is integrated at 20 kHz; upsample factor.
RECORD_DT = 0.5  # ms (= 1/2 kHz)
UPSAMPLE = int(RECORD_DT / DT_SIM)  # = 10
assert UPSAMPLE * DT_SIM == RECORD_DT, "DT_SIM must divide RECORD_DT"


def integrate_spiking(
    K: int,
    n_steps: int,
    dt: float,
    a: float,
    sigma: float,
    omega: float,
    J: float,
    A_input: np.ndarray | None = None,
    B_input: np.ndarray | None = None,
    seed: int = 0,
    sample_stride: int = 10,
):
    """Stochastic-Heun integration of K active-rotators with mean-field local coupling
    and (optional) external time-varying inter-regional input.

    Parameters
    ----------
    K : population size.
    n_steps : number of integration steps at step size dt.
    A_input, B_input : arrays of length n_steps (or n_steps+1) giving the external
        forcing coefficients, applied as I_i(t) = A cos(phi_i) - B sin(phi_i).
        If None, no external input (G = 0 case).
    sample_stride : stride for saving R_traj (e.g., 10 for 2 kHz output from 20 kHz sim).

    Returns
    -------
    R_traj : ndarray, shape (n_steps // sample_stride,).
    """
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=K)

    # Manuscript convention: dW per step = sigma * sqrt(dt) * xi, xi ~ N(0, 1).
    sqrt_dt = np.sqrt(dt)

    n_save = n_steps // sample_stride
    R_traj = np.empty(n_save, dtype=np.float64)

    # Local-MF helper
    def local_mf(p):
        c = np.exp(1j * p).mean()
        return float(np.abs(c)), float(np.angle(c))

    # Drift function reused by predictor and corrector
    def drift(p, R_loc, Psi_loc, A_t, B_t):
        d = (
            omega
            + a * np.sin(p)
            - J * R_loc * np.sin(p - Psi_loc)
        )
        if A_t is not None:
            d = d + (A_t * np.cos(p) - B_t * np.sin(p))
        return d

    save_idx = 0
    for t in range(n_steps):
        R_loc, Psi_loc = local_mf(phi)
        if t % sample_stride == 0 and save_idx < n_save:
            R_traj[save_idx] = R_loc
            save_idx += 1

        if A_input is not None:
            A_t = A_input[t]
            B_t = B_input[t]
        else:
            A_t = None
            B_t = None

        # Predictor
        f1 = drift(phi, R_loc, Psi_loc, A_t, B_t)
        xi = rng.standard_normal(K)
        noise_inc = sigma * sqrt_dt * xi
        phi_p = phi + dt * f1 + noise_inc

        # Local MF at predicted state (recompute since phi changed)
        R_loc_p, Psi_loc_p = local_mf(phi_p)
        f2 = drift(phi_p, R_loc_p, Psi_loc_p, A_t, B_t)

        # Corrector (single Wiener increment shared with predictor — Heun)
        phi = phi + 0.5 * dt * (f1 + f2) + noise_inc

    return R_traj


def load_recorded_MF_input(region_idx: int, G: float, sim_pkl: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Load A(t), B(t) for the inter-regional drive on `region_idx` at coupling G.

    Reads R_m, Psi_m from the saved pkl (at 2 kHz) and returns A, B at 2 kHz.
    Note: returned arrays are at the recording rate; caller upsamples as needed.
    """
    W = np.load(DERIVED / "W_allen_60.npy")
    n = region_idx
    W_nm = W[n].copy()
    W_nm[n] = 0.0  # ensure no self-loop

    with open(sim_pkl, "rb") as f:
        D = pickle.load(f)
    R_full = D["R"]      # (T, 60), T at 2 kHz
    Psi_full = D["PSI"]

    sin_psi = np.sin(Psi_full)
    cos_psi = np.cos(Psi_full)
    A = G * (R_full * sin_psi) @ W_nm
    B = G * (R_full * cos_psi) @ W_nm
    return A, B, RECORD_DT


def hold_upsample(x: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(x, factor, axis=0)


def run_one(label: str, K: int, T_ms: float, a: float, A_input: np.ndarray | None,
            B_input: np.ndarray | None, seed: int = 0) -> dict:
    n_steps = int(T_ms / DT_SIM)
    if A_input is not None and len(A_input) < n_steps:
        raise ValueError(f"A_input length {len(A_input)} < n_steps {n_steps}")

    t0 = time.time()
    R_traj = integrate_spiking(
        K=K, n_steps=n_steps, dt=DT_SIM,
        a=a, sigma=SIGMA, omega=OMEGA, J=J_LOCAL,
        A_input=A_input, B_input=B_input,
        seed=seed, sample_stride=UPSAMPLE,
    )
    elapsed = time.time() - t0
    AC1 = float(ru.AC1(R_traj[5000:], n=20))  # match TLEN in revision_utils
    print(f"  [{label}] K={K} T={T_ms/1000:.1f}s elapsed={elapsed:.1f}s "
          f"R range [{R_traj.min():.3f}, {R_traj.max():.3f}]  AC1={AC1:+.4f}")
    return {"R_spike": R_traj.astype(np.float32), "AC1": AC1, "K": K, "T_ms": T_ms, "a": a,
            "elapsed_s": elapsed, "label": label}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["sanity_g0_crit", "sanity_g0_super",
                                       "sanity_g0_sub", "main_gstar"],
                    help="Which test to run.")
    ap.add_argument("--K", type=int, default=5000)
    ap.add_argument("--T-ms", type=float, default=60000.0)
    ap.add_argument("--region", type=int, default=10,
                    help="Region index for the main test (default: 10 = VISal, highest in-strength).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.which == "sanity_g0_crit":
        out = run_one("sanity_g0_crit", args.K, args.T_ms, A_CRIT, None, None, seed=args.seed)
        path = DERIVED / f"C4_spiking_sanity_g0_crit.pkl"
    elif args.which == "sanity_g0_super":
        out = run_one("sanity_g0_super", args.K, args.T_ms, A_SUPER, None, None, seed=args.seed)
        path = DERIVED / f"C4_spiking_sanity_g0_super.pkl"
    elif args.which == "sanity_g0_sub":
        out = run_one("sanity_g0_sub", args.K, args.T_ms, A_SUB, None, None, seed=args.seed)
        path = DERIVED / f"C4_spiking_sanity_g0_sub.pkl"
    elif args.which == "main_gstar":
        # Main test: G = G* = 0.028, region 10 (VISal), critical regime.
        sim_pkl = SIM_DIR / "data_G2.pkl"
        A_2k, B_2k, _ = load_recorded_MF_input(args.region, G=0.028, sim_pkl=sim_pkl)
        # Hold-upsample to 20 kHz (factor 10) to match dt=0.05 ms integration
        A_full = hold_upsample(A_2k, UPSAMPLE)
        B_full = hold_upsample(B_2k, UPSAMPLE)
        out = run_one("main_gstar", args.K, args.T_ms, A_CRIT, A_full, B_full, seed=args.seed)
        out["region_idx"] = args.region
        path = DERIVED / f"C4_spiking_main_gstar_region{args.region}.pkl"

    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()

"""
Utilities for the supplementary analyses (scripts/, notebook 8).

Kept separate from src/ so the original published code
remains untouched.
"""

from __future__ import annotations

import pickle
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # src/ (paths.py, utilities)
import paths  # noqa: E402

import numpy as np


# Path to the simulation outputs on the external drive.
# Override at call time if the data are moved/copied locally.
DEFAULT_SIM_ROOT = paths.SIM_ROOT

REGIMES = ("subcritical", "critical", "supercritical")
REGIME_DIRS = {
    "subcritical": "Gscan_connectome_sub",
    "critical": "Gscan_connectome_crit",
    "supercritical": "Gscan_connectome_super",
}

# G grid used in the published runs (notebook 5).
G_GRID = np.linspace(0.0, 0.21, 16)
N_G = len(G_GRID)
N_REGIONS = 60

# Working-point indices reported in the paper (notebook 6, cell-11: gadd=[15,2,5]).
G_WORKING_INDEX = {"subcritical": 15, "critical": 2, "supercritical": 5}

# In-strength vector, 60 cortical ROIs, hardcoded in notebook 6 cell-29.
INSTRENGTH = np.asarray([
    1.9737271, 1.95601406, 0.70958593, 2.09403356, 1.71593984,
    1.05118705, 0.6982905,  0.31607513, 1.9471671,  1.17863418,
    4.40639357, 3.98180306, 3.62618341, 1.61131354, 2.08169054,
    2.44109073, 0.55446726, 0.86706295, 1.91923054, 3.53381901,
    3.38335554, 2.04937983, 1.50577545, 1.16548838, 2.03478647,
    2.16127293, 2.33737489, 0.2039825,  0.45425794, 0.77431038,
    1.9737271, 1.95601406, 0.70958593, 2.09403356, 1.71593984,
    1.05118705, 0.6982905,  0.31607513, 1.9471671,  1.17863418,
    4.40639357, 3.98180306, 3.62618341, 1.61131354, 2.08169054,
    2.44109073, 0.55446726, 0.86706295, 1.91923054, 3.53381901,
    3.38335554, 2.04937983, 1.50577545, 1.16548838, 2.03478647,
    2.16127293, 2.33737489, 0.2039825,  0.45425794, 0.77431038,
])

# Index of the right retrosplenial area (ventral part) — the exemplar in Fig. 5C.
RETROSPLENIAL_IDX = 24

# Index of the highest-in-strength cortical region: Right Anterolateral visual area (VISal).
# In-strength = 4.4064 (max across all cortical ROIs). Exemplar in Suppl. Fig. S4.
HIGHEST_INSTRENGTH_IDX = 10
HIGHEST_INSTRENGTH_NAME = "Right Anterolateral visual area"


def make_subcritical_gradient_a(
    in_strength: np.ndarray,
    a_min: float = 0.974,
    a_max: float = 0.978,
) -> np.ndarray:
    """Build a per-region 'a' array gradient mapped to in-strength rank.

    Direction: highest-in-strength region -> a_min (mildly subcritical, near critical);
               lowest-in-strength region -> a_max (more subcritical).

    Linear in *rank* (not raw in-strength), to even out the heavy-tailed distribution.
    """
    instr = np.asarray(in_strength, dtype=np.float64)
    n = instr.size
    # ranks in [0, n-1]: 0 = lowest in-strength, n-1 = highest
    ranks = np.argsort(np.argsort(instr))
    # high rank -> low a; low rank -> high a
    a = a_max + (a_min - a_max) * ranks / (n - 1)
    return a


def find_events(time_series: np.ndarray, threshold: float, dir: int = 1):
    """Same as src/functions.find_events."""
    ts = np.asarray(time_series)
    if dir == 1:
        above = ts > threshold
    elif dir == -1:
        above = ts < threshold
    elif dir == 0:
        above = np.abs(ts) > threshold
    else:
        raise ValueError("dir must be -1, 0 or 1")
    start = np.where(above & ~np.roll(above, 1))[0]
    end = np.where(~above & np.roll(above, 1))[0]
    if above[0]:
        start = np.insert(start, 0, 0)
    if above[-1]:
        end = np.append(end, len(ts) - 1)
    if len(start) == len(end) + 1:
        start = start[1:]
    elif len(start) == len(end) - 1:
        end = end[1:]
    return start, end


def measure_events(time_series: np.ndarray, threshold: float, dir: int = 1):
    """Same as src/functions.measure_events but uses np.trapezoid (numpy 2.x)."""
    ts = np.asarray(time_series, dtype=np.float64)
    start, end = find_events(ts, threshold, dir)
    durations = []
    integrals = []
    for s, e in zip(start, end):
        durations.append(e - s + 1)
        integrals.append(np.trapezoid(ts[s : e + 1]))
    return durations, integrals


def AC1(ts: np.ndarray, n: int = 20) -> float:
    """Lag-n autocorrelation. Same definition as src/functions.py."""
    ts = np.asarray(ts, dtype=np.float64)
    if ts.size < 2:
        raise ValueError("Time series must contain at least two elements.")
    m = ts.mean()
    num = np.sum((ts[:-n] - m) * (ts[n:] - m))
    den = np.sum((ts - m) ** 2)
    return num / den if den != 0 else 0.0


def load_R(pkl_path: Path) -> np.ndarray:
    """Return the R(t) array of shape (T, n_regions)."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)["R"]


def instrength_AC1_correlation(
    AC1_local: np.ndarray, instr: np.ndarray = INSTRENGTH
) -> np.ndarray:
    """Spearman ρ between in-strength and AC1, per regime per G.

    Returns
    -------
    rho : ndarray, shape (3, N_G)
    """
    from scipy.stats import spearmanr

    rho = np.zeros((AC1_local.shape[0], AC1_local.shape[2]))
    for ir in range(AC1_local.shape[0]):
        for ig in range(AC1_local.shape[2]):
            rho[ir, ig], _ = spearmanr(instr, AC1_local[ir, :, ig])
    return rho


def non_monotonicity_score(AC1_local: np.ndarray, regime_idx: int = 1) -> np.ndarray:
    """Per-region score: max_G AC1 - AC1(G=0), in the chosen regime.

    Positive score = AC1 rises above its uncoupled baseline at some G > 0,
    i.e., the region exhibits a non-monotonic AC1(G) profile.
    """
    AC1 = AC1_local[regime_idx]            # (N_REGIONS, N_G)
    return AC1.max(axis=1) - AC1[:, 0]


def cross_region_variance(AC1_local: np.ndarray) -> np.ndarray:
    """Var_n[AC1_n(G)] per regime per G. Shape (3, N_G)."""
    return AC1_local.var(axis=1)


def global_signal(R: np.ndarray, tlen: int = 5000) -> np.ndarray:
    """Population-average global signal: mean across z-scored regions, after burn-in.

    Matches the published definition (notebook 6, cell-22): GS(t) = (1/N) Σ_n z(R_n(t)).
    """
    from scipy import stats

    R_trim = np.asarray(R[tlen:], dtype=np.float64)
    return stats.zscore(R_trim, axis=0).mean(axis=1)


def global_AC1(R: np.ndarray, tlen: int = 5000, lag: int = 20) -> float:
    """AC1 of the global signal (matches Fig. 4D)."""
    return AC1(global_signal(R, tlen=tlen), n=lag)


def global_avalanche_sizes(R: np.ndarray, tlen: int = 5000) -> list:
    """Avalanche durations of GS(t) (matches Fig. 4E definition: events of GS > median, dir=1)."""
    GS = global_signal(R, tlen=tlen)
    durations, _ = measure_events(GS, float(np.median(GS)), dir=1)
    return durations


def in_strength_from_weights(W: np.ndarray) -> np.ndarray:
    """Sum of incoming weights per node.

    Convention used in this project (verified against the hardcoded INSTRENGTH array
    in `notebooks/6)Whole_brain_RAW_analysis.ipynb`): W[i, j] is the
    weight from j to i, so in-strength of node i is W[i, :].sum() = W.sum(axis=1).
    """
    W = np.asarray(W).copy()
    np.fill_diagonal(W, 0.0)
    return W.sum(axis=1)


def compute_AC1_local(
    sim_root: Path = DEFAULT_SIM_ROOT,
    tlen: int = 5000,
    lag: int = 20,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute AC1(R_n) per region per G per regime.

    Returns
    -------
    AC1_local : ndarray, shape (3, N_REGIONS, N_G)
        Index order: regime (sub/crit/super), region, G.
    """
    AC1_local = np.zeros((len(REGIMES), N_REGIONS, N_G))
    for ir, regime in enumerate(REGIMES):
        folder = sim_root / REGIME_DIRS[regime]
        for iG in range(N_G):
            R = load_R(folder / f"data_G{iG}.pkl")
            R = R[tlen:, :N_REGIONS]
            for j in range(N_REGIONS):
                AC1_local[ir, j, iG] = AC1(R[:, j], n=lag)
            if verbose:
                print(f"  {regime:13s}  G[{iG:2d}] = {G_GRID[iG]:.4f}  done")
    return AC1_local


# ---- Hemodynamic kernel used by the TVB Bold monitor (FirstOrderVolterra defaults) ----
TAU_S = 0.8                  # s
TAU_F = 0.4                  # s
HRF_LENGTH_S = 20.0          # TVB hrf_length default
HRF_FS = 250.0               # TVB internal HRF sampling (Hz)
TR = 1.0                     # s (Bold monitor period = 1000 ms)


def hrf_kernel() -> np.ndarray:
    """TVB FirstOrderVolterra kernel: h(t) = (1/3) exp(-t/(2*tau_s)) sin(omega t)/omega."""
    omega_k = np.sqrt(1.0 / TAU_F - 1.0 / (4.0 * TAU_S * TAU_S))
    t = np.arange(0, HRF_LENGTH_S, 1.0 / HRF_FS)
    return (1.0 / 3.0) * np.exp(-t / (2.0 * TAU_S)) * np.sin(omega_k * t) / omega_k

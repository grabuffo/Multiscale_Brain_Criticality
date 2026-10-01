"""
Criticality indicators (Methods, "Criticality indicators").

- AC1: autocorrelation at a fixed lag (see simulators.AC1).
- susceptibility chi = Var of R(t) after a zero-phase 4th-order Butterworth low-pass at
  1/window (200 ms -> 5 Hz), which removes fast collective oscillations.
- DFA exponent H_DFA (nolds, order-1 detrending, log-spaced windows 16 ms - 8 s at 2 kHz).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

CHI_WIN_MS = (25, 50, 100, 200, 400)
CHI_MAIN_MS = 200
DFA_NVALS = np.unique(np.logspace(np.log10(32), np.log10(16000), 20).astype(int))


def lowpass(x, fs, window_ms, axis=0):
    sos = butter(4, 1000.0 / window_ms, btype="low", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=np.float64), axis=axis)


def chi(x, fs, window_ms=CHI_MAIN_MS, axis=0):
    """Susceptibility: variance of the low-passed signal (per column if 2-D)."""
    return lowpass(x, fs, window_ms, axis=axis).var(axis=axis)


def dfa_exponent(sig):
    import nolds
    sig = np.asarray(sig, dtype=np.float64)
    if sig.size < int(DFA_NVALS.max() * 4):
        return np.nan
    try:
        return float(nolds.dfa(sig, nvals=DFA_NVALS, overlap=True, order=1, fit_exp="poly"))
    except Exception:
        return np.nan

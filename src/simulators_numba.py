"""
Standalone numba simulators (no TVB dependency), used for the spiking-network simulations (notebook 9).

1. `simulate_mf` -- whole-brain mean-field model, Eqs. (4)-(6) of the manuscript:
   Ott--Antonsen local dynamics for each region, regions coupled through their
   mean-field phases,  dPsi_n += I^MF_n = G sum_m W_nm R_m sin(Psi_m - Psi_n).
   Numerics as in the TVB runs: stochastic Heun, dt = 0.05 ms, weak additive noise
   on Psi (std sqrt(2 eta dt)), output at 2 kHz (stride 10) or 20 kHz (stride 1).

2. `simulate_spiking_net` -- the microscopic model, Eqs. (1)-(2): M regions of K
   active rotators, coupled through the connectome. The inter-regional sum
   (1/K) sum_j sin(phi_jm - phi_in) = R_m sin(Psi_m - phi_in) is evaluated exactly
   through the presynaptic order parameters (cost O(MK) per step, no approximation).
   Per-rotator Gaussian noise sigma*sqrt(dt)*xi. Returns regional R(t) at 2 kHz.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

# ===== published critical-regime parameters =====
A_SUB, A_CRIT, A_SUPER = 0.979, 0.973, 0.967
SIGMA = 0.6725
OMEGA = 1.0
J_LOCAL = 1.25
ETA_MF = 0.0001
DT = 0.05          # ms
STRIDE = 10        # -> 2 kHz output
R_FLOOR = 1e-6


@njit(cache=True)
def _mf_drift(R, Psi, a, J, sigma, omega, W, G):
    M = R.shape[0]
    dR = np.empty(M)
    dPsi = np.empty(M)
    sinP = np.sin(Psi)
    cosP = np.cos(Psi)
    # I_sin = G * sum_m W_nm R_m sin(Psi_m - Psi_n)
    for n in range(M):
        hr = 0.0
        hi = 0.0
        for m in range(M):
            w = W[n, m]
            if w != 0.0 and m != n:
                hr += w * R[m] * cosP[m]
                hi += w * R[m] * sinP[m]
        hr *= G
        hi *= G
        I_sin = hi * cosP[n] - hr * sinP[n]   # G sum W R_m sin(Psi_m - Psi_n)
        r = R[n]
        r2 = r * r
        dR[n] = 0.5 * r * (J * (1.0 - r2) - sigma * sigma) \
            - 0.5 * a[n] * (1.0 - r2) * cosP[n]
        dPsi[n] = omega + (a[n] * (1.0 + r2) * sinP[n]) / (2.0 * r)
        dPsi[n] += I_sin
    return dR, dPsi


@njit(cache=True)
def _mf_run(W, G, a, J, sigma, omega, eta, dt, n_steps, stride, seed):
    np.random.seed(seed)
    M = W.shape[0]
    R = np.random.uniform(0.01, 1.0, M)
    Psi = np.random.uniform(0.0, 2.0 * np.pi, M)
    noise_std = np.sqrt(2.0 * eta * dt)   # TVB Additive convention
    n_save = n_steps // stride
    R_out = np.empty((n_save, M), dtype=np.float32)
    k = 0
    for t in range(n_steps):
        if t % stride == 0 and k < n_save:
            for n in range(M):
                R_out[k, n] = R[n]
            k += 1
        dR1, dPsi1 = _mf_drift(R, Psi, a, J, sigma, omega, W, G)
        xi = np.random.standard_normal(M) * noise_std
        Rp = R + dt * dR1
        Psip = Psi + dt * dPsi1 + xi
        for n in range(M):
            if Rp[n] < R_FLOOR:
                Rp[n] = R_FLOOR
        dR2, dPsi2 = _mf_drift(Rp, Psip, a, J, sigma, omega, W, G)
        for n in range(M):
            R[n] += 0.5 * dt * (dR1[n] + dR2[n])
            if R[n] < R_FLOOR:
                R[n] = R_FLOOR
            Psi[n] += 0.5 * dt * (dPsi1[n] + dPsi2[n]) + xi[n]
    return R_out


def simulate_mf(W, G, a=A_CRIT, T_ms=180000.0, dt=DT,
                stride=STRIDE, seed=0, J=J_LOCAL, sigma=SIGMA, omega=OMEGA,
                eta=ETA_MF):
    """R(t) at 2 kHz, shape (T_ms/dt/stride, M)."""
    M = W.shape[0]
    a_arr = np.full(M, a, dtype=np.float64) if np.isscalar(a) else np.asarray(a, dtype=np.float64)
    n_steps = int(round(T_ms / dt))
    return _mf_run(np.ascontiguousarray(W, dtype=np.float64), float(G), a_arr,
                   float(J), float(sigma), float(omega), float(eta), float(dt),
                   n_steps, int(stride), int(seed))


@njit(cache=True, parallel=True)
def _spk_run(W, G, K, a, J, sigma, omega, dt, n_steps, stride, seed):
    M = W.shape[0]
    np.random.seed(seed)
    phi = np.random.uniform(0.0, 2.0 * np.pi, M * K)
    sq = sigma * np.sqrt(dt)
    n_save = n_steps // stride
    R_out = np.empty((n_save, M), dtype=np.float32)

    zr = np.empty(M)
    zi = np.empty(M)
    zr_p = np.empty(M)
    zi_p = np.empty(M)
    hr = np.empty(M)
    hi = np.empty(M)
    hr_p = np.empty(M)
    hi_p = np.empty(M)
    f1 = np.empty(M * K)
    noise = np.empty(M * K)
    phi_p = np.empty(M * K)
    sph = np.empty(M * K)
    cph = np.empty(M * K)

    k_save = 0
    for t in range(n_steps):
        # sin/cos cache + order parameters z_n = mean_i e^{i phi}
        for n in prange(M):
            sr = 0.0
            si = 0.0
            for i in range(n * K, (n + 1) * K):
                s = np.sin(phi[i])
                c = np.cos(phi[i])
                sph[i] = s
                cph[i] = c
                sr += c
                si += s
            zr[n] = sr / K
            zi[n] = si / K
        if t % stride == 0 and k_save < n_save:
            for n in range(M):
                R_out[k_save, n] = np.sqrt(zr[n] * zr[n] + zi[n] * zi[n])
            k_save += 1
        # afferent field  H_n = J z_n + G sum_m W_nm z_m
        for n in range(M):
            ar = J * zr[n]
            ai = J * zi[n]
            for m in range(M):
                if m != n and W[n, m] != 0.0:
                    ar += G * W[n, m] * zr[m]
                    ai += G * W[n, m] * zi[m]
            hr[n] = ar
            hi[n] = ai
        # predictor: drift reuses cached sin/cos; Im(H e^{-i phi}) = hi*cos - hr*sin
        for n in prange(M):
            for i in range(n * K, (n + 1) * K):
                f1[i] = omega + a[n] * sph[i] + hi[n] * cph[i] - hr[n] * sph[i]
                noise[i] = sq * np.random.standard_normal()
                phi_p[i] = phi[i] + dt * f1[i] + noise[i]
        # corrector fields at predicted state (sin/cos cached for drift reuse)
        for n in prange(M):
            sr = 0.0
            si = 0.0
            for i in range(n * K, (n + 1) * K):
                s = np.sin(phi_p[i])
                c = np.cos(phi_p[i])
                sph[i] = s
                cph[i] = c
                sr += c
                si += s
            zr_p[n] = sr / K
            zi_p[n] = si / K
        for n in range(M):
            ar = J * zr_p[n]
            ai = J * zi_p[n]
            for m in range(M):
                if m != n and W[n, m] != 0.0:
                    ar += G * W[n, m] * zr_p[m]
                    ai += G * W[n, m] * zi_p[m]
            hr_p[n] = ar
            hi_p[n] = ai
        for n in prange(M):
            for i in range(n * K, (n + 1) * K):
                f2 = omega + a[n] * sph[i] + hi_p[n] * cph[i] - hr_p[n] * sph[i]
                phi[i] += 0.5 * dt * (f1[i] + f2) + noise[i]
    return R_out


def simulate_spiking_net(W, G, K=1000, a=A_CRIT, T_ms=60000.0, dt=DT,
                         stride=STRIDE, seed=0, J=J_LOCAL, sigma=SIGMA,
                         omega=OMEGA):
    """Coupled spiking network: regional R(t) at 2 kHz, shape (T, M)."""
    M = W.shape[0]
    a_arr = np.full(M, a, dtype=np.float64) if np.isscalar(a) else np.asarray(a, dtype=np.float64)
    n_steps = int(round(T_ms / dt))
    return _spk_run(np.ascontiguousarray(W, dtype=np.float64), float(G), int(K),
                    a_arr, float(J), float(sigma), float(omega), float(dt),
                    n_steps, int(stride), int(seed))


def AC1(ts, n=20):
    """Lag-n autocorrelation, same definition as supplementary_utils.AC1."""
    ts = np.asarray(ts, dtype=np.float64)
    m = ts.mean()
    num = np.sum((ts[:-n] - m) * (ts[n:] - m))
    den = np.sum((ts - m) ** 2)
    return num / den if den != 0 else 0.0

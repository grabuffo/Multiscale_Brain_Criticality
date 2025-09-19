#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: grabuffo
"""

import numpy as np

# Define the system of differential equations with noise for Psi
def system(t, y, a, sigma, J, omega):
    R, Psi = y
    
    dR = (
        0.5 * R * (J * (1 - R**2) - sigma**2) - 0.5 * a * (1 - R**2) * np.cos(Psi)
    )
    dPsi = (
        omega + (a * (1 + R**2) * np.sin(Psi)) / (2 * R)
    )
    
    return np.array([dR, dPsi])

# Heun stochastic integrator
def heun_stochastic(system, t_span, y0, dt, a, sigma, g, J, omega):
    t0, t1 = t_span
    t = t0
    y = np.array(y0)
    times = [t0]
    ys = [y0]

    while t < t1:
        # Generate noise for Psi
        noise = np.random.normal(0, g) * np.sqrt(dt)
        
        # Compute drift (deterministic part)
        drift = system(t, y, a, sigma, J, omega)
        
        # Predict step
        y_tilde = y + dt * drift + np.array([0, noise])  # Add noise to Psi only
        
        # Correct step
        drift_tilde = system(t + dt, y_tilde, a, sigma, J, omega)
        y = y + 0.5 * dt * (drift + drift_tilde) + np.array([0, noise])
        
        t += dt
        times.append(t)
        ys.append(y)
    
    return np.array(times), np.array(ys).T
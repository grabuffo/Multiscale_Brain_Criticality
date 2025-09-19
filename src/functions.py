#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: grabuffo
"""
import os
import glob
import math
import pickle
import gzip
import numpy as np
from scipy import stats
import matplotlib.pylab as plt
import matplotlib.cm as cm

def AC1(ts,n=20):
    """
    Computes autocorrelation at lag 1 for a given time series.

    Parameters:
        ts (array-like): The input time series (list, NumPy array, or pandas Series).

    Returns:
        float: Autocorrelation at lag 1.
    """
    ts = np.asarray(ts)
    if len(ts) < 2:
        raise ValueError("Time series must contain at least two elements.")

    ts_mean = np.mean(ts)
    numerator = np.sum((ts[:-n] - ts_mean) * (ts[n:] - ts_mean))
    denominator = np.sum((ts - ts_mean) ** 2)

    return numerator / denominator if denominator != 0 else 0.0

def retry_function(func, attempts=5, delay=1):
    for _ in range(attempts):
        try:
            return func()
        except Exception as e:
            print(f"Attempt failed: {e}")
            time.sleep(delay)
    # If all attempts fail, return None
    return None


def go_edge(tseries):
    nregions=tseries.shape[1]
    Blen=tseries.shape[0]
    nedges=int(nregions**2/2-nregions/2)
    iTriup= np.triu_indices(nregions,k=1) 
    gz=stats.zscore(tseries)
    Eseries = gz[:,iTriup[0]]*gz[:,iTriup[1]]
    return Eseries

# Single time series analysis (mean-field criticality analysis)

def find_events(time_series, threshold, dir=1):
    # Find where the time series exceeds the threshold
    if dir==1:
        above_threshold = time_series > threshold
    elif dir==-1:
        above_threshold = time_series < threshold
    elif dir==0:
        above_threshold = np.abs(time_series) > threshold
    # Find the indices where it transitions from below to above threshold
    start_indices = np.where(above_threshold & ~np.roll(above_threshold, 1))[0]
    # Find the indices where it transitions from above to below threshold
    end_indices = np.where(~above_threshold & np.roll(above_threshold, 1))[0]
    
    # Handle cases where the series starts or ends above threshold
    if above_threshold[0]:
        start_indices = np.insert(start_indices, 0, 0)
    if above_threshold[-1]:
        end_indices = np.append(end_indices, len(time_series) - 1)

    if len(start_indices)==len(end_indices)+1:
        start_indices=start_indices[1:]
    elif len(start_indices)==len(end_indices)-1:
        end_indices=end_indices[1:]
    
    return start_indices, end_indices

def measure_events(time_series, threshold, dir=1):
    start_indices, end_indices = find_events(time_series, threshold, dir)
    
    durations = []
    integrals = []
    
    for start, end in zip(start_indices, end_indices):
        duration = end - start + 1
        integral = np.trapz(time_series[start:end+1])
        durations.append(duration)
        integrals.append(integral)
    
    return durations, integrals
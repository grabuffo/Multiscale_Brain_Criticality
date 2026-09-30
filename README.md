# Multiscale_Brain_Criticality

This repository contains the code to reproduce the results of:  
**"The Connectome Modulates Critical Brain Dynamics Across Local and Global Scales"**  
Rabuffo, G.; Bozzo, P.; Nguyen, B.; Depannemeacker, D.; Pompili, M.; Gollo, L.; Fukai, T.; Sorrentino, P.; Dalla Porta, L.

![alt text](https://github.com/grabuffo/Multiscale_Brain_Criticality/blob/main/DallaPorta_Figure_1.png)

---

## Overview

This project introduces a **multiscale, connectome-based modeling framework** that unites the study of local and global brain criticality.  
By tuning neural mass models to **subcritical, critical, and supercritical regimes** and embedding them into the empirically derived **mouse connectome**, we explore how local and global dynamics interact to generate experimentally observed features of brain activity.

Key contributions of this work include:
- Demonstrating that **global signatures of criticality** (maximal autocorrelation, avalanche scaling, 1/f spectra) emerge only when local populations are tuned near criticality and coupled within an optimal range of global coupling.  
- Showing that **subcritical and supercritical regimes** also reproduce meaningful dynamics (e.g., oscillations, flattened spectra), suggesting that distinct brain regions may operate at different distances from criticality.  
- Revealing that **structural in-strength** shapes spatial gradients of timescales, reversing direction between subcritical and supercritical tuning.  
- Linking **global criticality** with improved correspondence to empirical mouse fMRI data (functional connectivity and dynamic FC).  

This framework highlights how local tuning, long-range interactions, and network topology jointly shape **scale-free, flexible dynamics across the connectome**.

---

## Data

The study uses structural and functional data from the **Allen Mouse Brain Atlas** and resting-state fMRI from 53 control mice under light anesthesia (medetomidine–isoflurane protocol) [Grandjean, 2020].  

- **Connectome**: tracer-based directed structural connectivity of the mouse brain (Oh et al., 2014; Melozzi et al., 2017).  
- **fMRI dataset**: [DOI:10.34973/1he1-5c70](https://doi.org/10.34973/1he1-5c70) (CC-BY 4.0).  

---

## Notebooks

The repository is organized into numbered Jupyter notebooks. Running them in order reproduces all simulations and figures from the paper.

| Notebook                           | Description                                                                 |
|------------------------------------|-----------------------------------------------------------------------------|
| **1) Empirical_data_processing**   | Preprocesses structural and functional datasets (Allen connectome, rsfMRI). |
| **2) Simulate_Local_NMM**          | Simulates isolated neural mass models in subcritical, critical, and supercritical regimes. |
| **3) Local_phase_space**           | Explores phase space structure, stability, and bifurcations of the local model. |
| **4) Two_coupled_populations**     | Examines how coupling shifts two regions toward or away from criticality depending on initial state. |
| **5) Whole_brain_simulations**     | Embeds local models into the empirical connectome and runs large-scale simulations across coupling strengths \(G\). |
| **6) Whole_brain_RAW_analysis**    | Analyzes raw neural activity: autocorrelations, metastability, avalanche statistics, timescale gradients. |
| **7) Whole_brain_BOLD_analysis**   | Transforms neural activity into BOLD signals (Balloon–Windkessel model), computes FC/dFC, and compares with empirical fMRI. |

---

## Reproducing Results

- The pipeline runs end-to-end: from **empirical preprocessing** (1) to **BOLD-level analysis and data comparison**.  
- Each notebook generates the figures corresponding to its stage of analysis.
- Avalanche analyses, power spectra, and autocorrelation timescales are reproduced in analysis notebooks.  

---

## Revision analyses (`revision/`)

Code for the analyses added during peer review. The original notebooks and `src/` are unchanged
apart from two NumPy-2 compatibility fixes (`np.inf`, `np.trapezoid`).

`revision/paths.py` defines all paths. Large whole-brain simulation outputs are not stored in the
repository: set the environment variable `MBC_SIM_ROOT` to the folder containing the simulation
folders (`Gscan_connectome_crit`, `Gscan_connectome_sub`, ...) produced by
`notebooks/5)Whole_brain_simulations.ipynb` and `revision/round1/run_C3B2_simulate.py`
(default: `<repo>/simulations`). Small inputs (cortical connectome, surrogates, in-strength,
local-parameter gradients) are in `revision/derived_data/`; outputs go to `revision/derived_data/`
and `revision/figures/`.

### `revision/round1/` — supplementary analyses (first revision)

| Supplementary figure | Script(s) |
|---|---|
| S4 — AC1 over (G, in-strength), in-strength gradient, VISal exemplar | `run_B3_compute_AC1.py` → `run_B3_combined_SI.py` (also `R1_B3_phase_diagram.ipynb`, `run_B3_make_figure.py`, `run_B3_VISal_panels.py`) |
| S5 — structural-null surrogates | `run_C3B2_make_surrogates.py` → `run_C3_make_surrogate_diag.py` |
| S6 — Allen vs surrogates | `run_C3B2_simulate.py` (TVB) → `run_C3_compute_metrics.py` → `run_C3_make_figure.py --graphs allen,S1,S2,S3` (see `R2_C3B2_structural_null.ipynb`) |
| S7 — AC1 and DFA vs G | `run_C1_compute_indicators.py`, `run_C1_compute_GS_indicators.py` → `run_C1_make_figure.py` |
| S8 — phase-randomized afferent control | `run_C1b1_phase_randomize.py` → `run_C1b1_make_figure.py` |
| S9 — structured heterogeneity | `run_C3_compute_metrics.py` (graphs `B4grad_v2`, `B4grad_random`, `allen_sub`) → `run_B4_make_figure.py` |
| S10, S12 — gradient setup, HRF kernel | `run_S10_S12_simple_panels.py` |
| S11 — BOLD vs neural-scale alignment | `run_C3_compute_metrics.py allen` → `run_C2_make_figure.py` |
| Local BOLD AC1 / local AC1 distributions (response material) | `run_C2_local_BOLD_AC1.py`, `run_C3_local_AC1_distrib.py`, `run_C3_surrogate_GS_DFA.py`, `run_C4_spiking_validation.py` |

### `revision/prr/` — spiking-network validation and avalanche statistics

Standalone numba code (no TVB): `simulators.py` (whole-brain mean field with phase coupling,
Eqs. 4–6; microscopic spiking network, Eqs. 1–2) and `indicators.py` (susceptibility, DFA).

| Figure | Script(s) | Run time |
|---|---|---|
| Fig. 1D — avalanche sizes/durations, exponents | `fig1_avalanches.py` → `make_fig1_panelD.py` | ~1 min |
| S13 — whole-brain spiking network (60 regions × 1000 neurons) | `spiking_wholebrain.py` (see usage at the end of the file) → `supp_spiking_figdata.py` → `supp_spiking_figures.py` | ~23 min per G value (14 cores) |
| S14 — two coupled spiking populations (5000 neurons each) | `two_pop_spiking.py --mode scan`, `--mode grid --topo ff`, `--mode grid --topo fb` → `supp_spiking_figdata.py` → `supp_spiking_figures.py` | ~20–50 min per run |
| Bistable window (Methods) | `bistability_check.py` | ~10 min |

Note: the spiking simulator draws random numbers inside parallel (numba `prange`) loops, so
repeated runs with the same seed are statistically equivalent but not bit-identical.

---

## Requirements

- Python 3.9+ (original notebooks: The Virtual Brain, NumPy < 2)
- Jupyter Notebook
- NumPy, SciPy, Pandas, Matplotlib, Seaborn, NetworkX
- Revision scripts: numba, nolds, powerlaw

See `requirements.txt`.

---

## Citation

If you use this code, please cite:  
**Rabuffo, G.; Bozzo, P.; Nguyen, B.; Depannemeacker, D.; Pompili, M.; Gollo, L.; Fukai, T.; Sorrentino, P.; Dalla Porta, L. The Connectome Modulates Critical Brain Dynamics Across Local and Global Scales.**


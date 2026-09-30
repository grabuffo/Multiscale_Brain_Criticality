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
| **8) Supplementary_analyses**      | Avalanche size/duration statistics (Fig. 1D), bistable window, and Supplementary Figures S4–S12 (in-strength gradients, structural-null surrogates, DFA, phase-randomized control, structured heterogeneity, BOLD vs neural-scale signatures). |
| **9) Spiking_network_simulations** | Validation with the full microscopic model: whole-brain network of spiking neurons (Suppl. Fig. S13) and two coupled spiking populations (Suppl. Fig. S14). |

---

## Code organization

- `notebooks/` — numbered notebooks, to be run in order.
- `src/` — model and analysis functions (TVB model and coupling, avalanche and FC/dFC utilities), plus `simulators_numba.py` (standalone numba implementation of the mean-field and spiking-network models), `indicators.py` (susceptibility, DFA) and `paths.py` (all paths).
- `scripts/` — scripts called by notebooks 8–9; each script states in its header which figure it produces. Long simulations can be run directly from a terminal (`python scripts/<name>.py`).
- `data/` — Allen connectome, empirical fMRI data, and `data/derived/` (cortical connectome, surrogate connectomes, in-strength, local-parameter gradients).

Whole-brain simulation outputs are large and are not stored in the repository. Notebooks 5 and 8 write/read them in the folder given by the environment variable `MBC_SIM_ROOT` (default: `<repo>/simulations`). Generated figures are written to `figures/`.

The spiking-network simulator draws random numbers inside parallel loops, so repeated runs with the same seed are statistically equivalent but not bit-identical.

## Reproducing Results

- The pipeline runs end-to-end: from **empirical preprocessing** (1) to **BOLD-level analysis and data comparison**.  
- Each notebook generates the figures corresponding to its stage of analysis.
- Avalanche analyses, power spectra, and autocorrelation timescales are reproduced in analysis notebooks.  

---

## Requirements

- Python 3.9+ (original notebooks: The Virtual Brain, NumPy < 2)
- Jupyter Notebook
- NumPy, SciPy, Pandas, Matplotlib, Seaborn, NetworkX
- Notebooks 8–9: numba, nolds, powerlaw

See `requirements.txt`.

---

## Citation

If you use this code, please cite:  
**Rabuffo, G.; Bozzo, P.; Nguyen, B.; Depannemeacker, D.; Pompili, M.; Gollo, L.; Fukai, T.; Sorrentino, P.; Dalla Porta, L. The Connectome Modulates Critical Brain Dynamics Across Local and Global Scales.**


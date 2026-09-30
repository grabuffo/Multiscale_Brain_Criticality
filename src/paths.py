"""
Paths used by the scripts in scripts/ and notebooks 8)-9).

Large whole-brain simulation outputs (TVB runs, ~170-480 MB per coupling value) are not stored
in the repository. Set the environment variable MBC_SIM_ROOT to the folder containing them
(folders Gscan_connectome_crit, Gscan_connectome_sub, ... produced by
notebooks/5)Whole_brain_simulations.ipynb and scripts/simulate_whole_brain_tvb.py);
by default they are expected in <repo>/simulations.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
SCRIPTS = REPO / "scripts"
DATA = REPO / "data"
EMP_DIR = DATA / "Empirical_fMRI"
ALLEN_ZIP = DATA / "Allen_148" / "Allen_148.zip"
FIG1_PDF = REPO / "DallaPorta_Figure_1.pdf"

DERIVED = DATA / "derived"            # small inputs (cortical connectome, surrogates, in-strength) + analysis outputs
FIGURES = REPO / "figures"            # generated figures
SIM_ROOT = Path(os.environ.get("MBC_SIM_ROOT", REPO / "simulations"))

for _d in (DERIVED, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

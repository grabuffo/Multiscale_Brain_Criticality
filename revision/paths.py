"""
Paths shared by the revision scripts (revision/round1 and revision/prr).

Large simulation outputs (whole-brain TVB runs, ~170-480 MB per coupling value) are not
stored in the repository. Set the environment variable MBC_SIM_ROOT to the folder that
contains them (the folders Gscan_connectome_crit, Gscan_connectome_sub, ... produced by
notebooks/5)Whole_brain_simulations.ipynb and revision/round1/run_C3B2_simulate.py);
by default they are expected in <repo>/simulations.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"                                   # original model / analysis code
DATA = REPO / "data"                                 # Allen connectome, empirical fMRI
EMP_DIR = DATA / "Empirical_fMRI"
ALLEN_ZIP = DATA / "Allen_148" / "Allen_148.zip"
FIG1_PDF = REPO / "DallaPorta_Figure_1.pdf"

REVISION = REPO / "revision"
DERIVED = REVISION / "derived_data"                  # small inputs (connectomes, in-strength) and outputs
FIGURES = REVISION / "figures"
SIM_ROOT = Path(os.environ.get("MBC_SIM_ROOT", REPO / "simulations"))

for _d in (DERIVED, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

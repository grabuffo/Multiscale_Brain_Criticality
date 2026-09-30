"""
Compute and cache AC1(R_n) over (regime, region, G) from the published whole-brain
simulations. Output: paper/revision/derived_data/AC1_local.npy
"""

from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # revision/ (paths.py)
import paths  # noqa: E402
_sys.path.insert(0, str(paths.SRC))  # original src/ (functions.py, Utils.py, ...)
import time

import numpy as np

from revision_utils import (
    G_GRID,
    INSTRENGTH,
    REGIMES,
    compute_AC1_local,
)


PROJECT_ROOT = paths.REPO
OUT_DIR = paths.DERIVED
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    t0 = time.time()
    AC1_local = compute_AC1_local(verbose=True)
    elapsed = time.time() - t0
    print(f"Total elapsed: {elapsed:.1f} s")
    print(f"AC1_local shape: {AC1_local.shape}")

    out_path = OUT_DIR / "AC1_local.npy"
    np.save(out_path, AC1_local)
    np.save(OUT_DIR / "G_grid.npy", G_GRID)
    np.save(OUT_DIR / "instrength.npy", INSTRENGTH)
    np.save(OUT_DIR / "regimes.npy", np.asarray(REGIMES))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

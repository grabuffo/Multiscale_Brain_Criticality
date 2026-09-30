"""
Whole-brain spiking-network validation (Suppl. Fig. S13).

All 60 cortical regions are populations of K active rotators (Eqs. 1-2), in the
spiking model's own hybrid-type regime (J = 1, sigma = 0.499, as in Suppl. Fig. S2 /
Buendia 2021), coupled through the Allen connectome (exact rewrite of the
inter-regional term through presynaptic order parameters; no mean-field step).

  calib : isolated populations on a fine a-grid at this K -> a_c = argmax of the
          200 ms susceptibility (the transition shifts slightly with K)
  run   : whole-brain G sweep at a_c ("critical") and a_c + DA_SUB ("subcritical")

Question: does global criticality (a peak of the global susceptibility / slow
fluctuations) emerge at intermediate G, and in which local regime?

Output: revision/derived_data/spk_wb/{calib_K<K>.pkl, wb_<regime>_G<iG>_K<K>.pkl}
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import paths  # noqa: E402
import simulators as r2  # noqa: E402
import indicators as ind  # noqa: E402

DERIVED_R2 = paths.DERIVED
OUT = paths.DERIVED / "spk_wb"
OUT.mkdir(parents=True, exist_ok=True)

J_SPK, SIGMA_SPK = 1.0, 0.499
A_CAL = np.round(np.arange(1.05, 1.0901, 0.0025), 4)
G_GRID = np.linspace(0.0, 0.21, 16)
G_IDX = (0, 1, 2, 3, 4, 6, 8, 12)          # 0, .014, .028, .042, .056, .084, .112, .168
DA_SUB = 0.005
DISCARD = 5000                              # 2.5 s at 2 kHz
CHI_WIN_MS = ind.CHI_WIN_MS


def metrics(R):
    Rc = R[DISCARD:].astype(np.float64)
    M = Rc.shape[1]
    mR = Rc.mean(axis=1)
    return {
        "R_mean": Rc.mean(axis=0),
        "chi_win_ms": np.array(CHI_WIN_MS),
        "chi_reg_w": np.array([ind.chi(Rc, 2000.0, w) for w in CHI_WIN_MS]),
        "chi_GS_w": np.array([ind.chi(mR, 2000.0, w) for w in CHI_WIN_MS]),
        "AC1_1ms": np.array([r2.AC1(Rc[:, n], n=2) for n in range(M)]),
        "AC1_1ms_GS": float(r2.AC1(mR, n=2)),
    }


def calib(K, T_ms):
    path = OUT / f"calib_K{K}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    W = np.zeros((len(A_CAL), len(A_CAL)))
    R = r2.simulate_spiking_net(W, 0.0, K=K, a=A_CAL, T_ms=T_ms, J=J_SPK, sigma=SIGMA_SPK)
    m = metrics(R)
    a_c = float(A_CAL[int(np.argmax(m["chi_reg_w"][CHI_WIN_MS.index(200)]))])
    out = {"a": A_CAL, "a_c": a_c, **m}
    pickle.dump(out, open(path, "wb"))
    print(f"[calib K={K}] a_c = {a_c:.4f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=1000)
    ap.add_argument("--T-ms", type=float, default=120000.0)
    ap.add_argument("--regimes", default="critical,subcritical")
    ap.add_argument("--G-values", default=None,
                    help="comma-separated G values (confirmation runs); default = published-grid indices G_IDX")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cal = calib(args.K, 60000.0)
    W = np.load(DERIVED_R2 / "W_allen_60.npy")
    a_of = {"critical": cal["a_c"], "subcritical": round(cal["a_c"] + DA_SUB, 4)}
    if args.G_values is None:
        jobs = [(G_GRID[iG], iG, f"G{iG}") for iG in G_IDX]
    else:
        jobs = [(float(g), None, f"g{float(g):.3f}") for g in args.G_values.split(",")]
    for reg in args.regimes.split(","):
        for G, iG, tag in jobs:
            stag = "" if args.seed == 0 else f"_s{args.seed}"
            path = OUT / f"wb_{reg}_{tag}_K{args.K}{stag}.pkl"
            if path.exists():
                continue
            t0 = time.time()
            R = r2.simulate_spiking_net(W, G, K=args.K, a=a_of[reg], T_ms=args.T_ms,
                                        J=J_SPK, sigma=SIGMA_SPK, seed=args.seed)
            out = {"regime": reg, "a": a_of[reg], "G": float(G), "iG": iG, "K": args.K, "seed": args.seed,
                   "J": J_SPK, "sigma": SIGMA_SPK, "R": R[DISCARD:].astype(np.float16), **metrics(R)}
            pickle.dump(out, open(path, "wb"))
            print(f"[{reg} a={a_of[reg]:.4f} s{args.seed}] G={G:.3f} chiGS200={out['chi_GS_w'][3]:.2e} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()

# Paper runs (Suppl. Fig. S13):
#   python spiking_wholebrain.py                                   # calibration + 8 G values, critical & subcritical
#   python spiking_wholebrain.py --regimes critical --G-values 0.021,0.035
#   python spiking_wholebrain.py --regimes critical --G-values 0.028 --seed 1   # independent run at the peak

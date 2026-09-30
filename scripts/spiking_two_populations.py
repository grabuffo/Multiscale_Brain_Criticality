"""
Two coupled populations of spiking neurons (Suppl. Fig. S14; spiking counterpart of Fig. 2G-H).

Populations of K active rotators with J = 1, sigma = 0.499 (Suppl. Fig. S2 / Buendia et al.),
coupled with G = 0.05, feedforward (A -> B, 'ff') or mutual (A <-> B, 'fb').
All systems of one condition are integrated in parallel as a single network with
block-diagonal connectivity (simulators.simulate_spiking_net).

  scan : isolated populations on a fine a-grid -> transition a_c (susceptibility peak)
  grid : 7x7 grid a_A, a_B = a_c + 0.01 k (k = -3..3) plus isolated references at each a

Output: data/derived/two_pop_spk/{scan_K<K>.pkl, grid_<topo>_K<K>.pkl}
Usage:  python spiking_two_populations.py --mode scan
        python spiking_two_populations.py --mode grid --topo ff
        python spiking_two_populations.py --mode grid --topo fb
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # src/ (paths.py, utilities)
import paths  # noqa: E402
import simulators_numba as sim  # noqa: E402
import indicators as ind  # noqa: E402

OUT = paths.DERIVED / "two_pop_spk"
OUT.mkdir(parents=True, exist_ok=True)

J, SIGMA, G = 1.0, 0.499, 0.05
A_SCAN = np.round(np.arange(0.97, 1.0801, 0.0025), 4)
DISCARD = 5000                              # 2.5 s at 2 kHz
FS = 2000.0


def metrics(R):
    Rc = R[DISCARD:].astype(np.float64)
    return {"R_mean": Rc.mean(0),
            "AC1": np.array([sim.AC1(Rc[:, n], n=2) for n in range(Rc.shape[1])]),   # 1 ms
            "chi": ind.chi(Rc, FS)}


def build_grid(topo, a_par):
    n = len(a_par)
    M = 2 * n * n + n
    W = np.zeros((M, M)); a = np.zeros(M); idx = {}
    k = 0
    for ia in range(n):
        for ib in range(n):
            A, B = k, k + 1
            a[A], a[B] = a_par[ia], a_par[ib]
            W[B, A] = 1.0                    # W[n, m]: m -> n
            if topo == "fb":
                W[A, B] = 1.0
            idx[(ia, ib)] = (A, B)
            k += 2
    for i in range(n):
        a[k] = a_par[i]; idx[("iso", i)] = k; k += 1
    return W, a, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("scan", "grid"), required=True)
    ap.add_argument("--topo", choices=("ff", "fb"), default="ff")
    ap.add_argument("--K", type=int, default=5000)
    ap.add_argument("--T-ms", type=float, default=60000.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.mode == "scan":
        path = OUT / f"scan_K{args.K}.pkl"
        W, a, idx, Gx = np.zeros((len(A_SCAN), len(A_SCAN))), A_SCAN.copy(), {}, 0.0
        a_par = None
    else:
        path = OUT / f"grid_{args.topo}_K{args.K}.pkl"
        scan = pickle.load(open(OUT / f"scan_K{args.K}.pkl", "rb"))
        a_c = scan["a"][int(np.argmax(scan["spk"]["chi"]))]
        a_par = np.round(a_c + 0.01 * np.arange(-3, 4), 4)
        W, a, idx = build_grid(args.topo, a_par)
        Gx = G
    t0 = time.time()
    R = sim.simulate_spiking_net(W, Gx, K=args.K, a=a, T_ms=args.T_ms, seed=args.seed, J=J, sigma=SIGMA)
    out = {"mode": args.mode, "topo": args.topo, "K": args.K, "G": Gx, "J": J, "sigma": SIGMA,
           "a": a, "a_par": a_par, "idx": idx, "spk": metrics(R), "R_spk": R[DISCARD:].astype(np.float16)}
    pickle.dump(out, open(path, "wb"))
    print(f"saved {path.name} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()

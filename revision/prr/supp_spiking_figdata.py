"""
Figure data for the spiking-validation supplementary figures (Suppl. Figs. S13-S14).

  S13  whole-brain spiking network at the spiking hybrid-type transition
       (spiking_wholebrain.py; 60 ROIs x K = 1000, J = 1, sigma = 0.499)
  S14  two coupled spiking populations around their own transition
       (two_pop_spiking.py; K = 5000, G = 0.05)

Indicators (see Methods, "Criticality indicators"): AC1 at 1 ms lag; susceptibility
chi = variance of R low-passed at 5 Hz (indicators.chi); DFA exponent.
Reads the outputs of spiking_wholebrain.py and two_pop_spiking.py from
revision/derived_data; writes revision/derived_data/supp_spiking.pkl.
"""
from __future__ import annotations

import glob
import pickle
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import paths  # noqa: E402
import simulators as r2  # noqa: E402
import indicators as ind  # noqa: E402

_dfa_exponent = ind.dfa_exponent
SRC = paths.DERIVED
DER_R2 = paths.DERIVED
OUT = paths.DERIVED / "supp_spiking.pkl"
FS = 2000.0


def wholebrain():
    instr = np.load(DER_R2 / "instrength.npy")
    out = {}
    for reg in ("critical", "subcritical"):
        rows = []
        for p in glob.glob(str(SRC / "spk_wb" / f"wb_{reg}_*_K1000*.pkl")):
            d = pickle.load(open(p, "rb"))
            R = d["R"].astype(float)
            gs = R.mean(axis=1)
            rows.append({"G": d["G"], "seed": d.get("seed", 0),
                         "AC1": r2.AC1(gs, n=2), "chi": ind.chi(gs, FS), "H": _dfa_exponent(gs),
                         "rho": spearmanr(instr, ind.chi(R, FS)).statistic, "a": d["a"]})
        rows.sort(key=lambda r: (r["seed"], r["G"]))
        out[reg] = rows
    cal = pickle.load(open(SRC / "spk_wb" / "calib_K1000.pkl", "rb"))
    out["calib"] = {"a": cal["a"], "a_c": cal["a_c"], "AC1": cal["AC1_1ms"],
                    "chi": cal["chi_reg_w"][list(cal["chi_win_ms"]).index(200)]}
    return out


def two_pop():
    scan = pickle.load(open(SRC / "two_pop_spk" / "scan_K5000.pkl", "rb"))
    Rs = scan["R_spk"].astype(float)
    out = {"scan": {"a": scan["a"], "AC1": np.array([r2.AC1(Rs[:, j], n=2) for j in range(Rs.shape[1])]),
                    "chi": ind.chi(Rs, FS)}}
    for topo in ("ff", "fb"):
        d = pickle.load(open(SRC / "two_pop_spk" / f"grid_{topo}_K5000.pkl", "rb"))
        R = d["R_spk"].astype(float)
        ac = np.array([r2.AC1(R[:, j], n=2) for j in range(R.shape[1])])
        ch = ind.chi(R, FS)
        idx, n = d["idx"], len(d["a_par"])
        iso_ac = np.array([ac[idx[("iso", i)]] for i in range(n)])
        iso_ch = np.array([ch[idx[("iso", i)]] for i in range(n)])
        tgt_ac = np.array([[ac[idx[(ia, ib)][1]] for ib in range(n)] for ia in range(n)])
        tgt_ch = np.array([[ch[idx[(ia, ib)][1]] for ib in range(n)] for ia in range(n)])
        out[topo] = {"a_par": d["a_par"], "dAC1": (tgt_ac - iso_ac[None, :]) / iso_ac[None, :],
                     "dlogchi": np.log10(tgt_ch / iso_ch[None, :])}
    return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    D = {"wb": wholebrain(), "tp": two_pop()}
    pickle.dump(D, open(OUT, "wb"))
    for reg in ("critical", "subcritical"):
        for r in D["wb"][reg]:
            print(f"{reg:11s} s{r['seed']} G={r['G']:.3f} AC1={r['AC1']:.3f} chi={r['chi']:.2e} H={r['H']:.2f} rho={r['rho']:+.2f}")
    print("saved", OUT)


if __name__ == "__main__":
    main()

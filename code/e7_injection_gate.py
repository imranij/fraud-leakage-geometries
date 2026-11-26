"""E7 -- the LeakBench injection gate.

The gate compares the injected deviation-type distribution with the real one in BPI 2019 at a
tolerance fixed in advance. BPI 2019 was not obtained, and configs/default.yaml leaves the
tolerance as not_prespecified, so the gate has neither a comparator nor a threshold: its distance
is NOT COMPUTED and every LeakBench result in this run stands as a benchmark property only.
What this script does compute is the injected side of the comparison, which is the half of the gate
that does not need the log.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")))

import numpy as np
import pandas as pd

import common as C
from src.leakbench.inject import inject, verify                      # noqa: E402

TYPE_MIX = {"DUP_EXACT": 1, "DUP_NEAR": 1, "WRONG_PAYEE": 1, "TRANSPOSE": 1,
            "DOUBLE_FX": 0.5, "SPLIT_PAY": 0.5}
RATES = [0.005, 0.01, 0.02]
SEEDS = [0, 1, 2]


def main():
    d = C.load_ulb().sort_values("Time").reset_index(drop=True)
    te = d.loc[int(0.8 * len(d)):].reset_index(drop=True)
    h = pd.DataFrame({"record_id": np.arange(len(te)),
                      "day": (te["Time"].values / 3600.0).astype(int),
                      "entity_id": (np.abs(te["V1"].values * 1000).astype(int) % 500),
                      "counterparty_id": (np.abs(te["V2"].values * 1000).astype(int) % 2000),
                      "amount": te["Amount"].values, "currency": "EUR",
                      "y": te["Class"].values.astype(int)})
    rows = []
    for rate in RATES:
        for hard in (False, True):
            for seed in SEEDS:
                v = verify(inject(h, rate=rate, seed=seed, type_mix=TYPE_MIX, hard=hard))
                tot = sum(v["by_type"].values())
                for t, c in v["by_type"].items():
                    rows.append({"rate": rate, "hard": hard, "seed": seed, "leak_type": t,
                                 "n": c, "share": c / tot})
    df = pd.DataFrame(rows)
    C.dump("e7_injected_type_distribution.csv", df)
    C.dump("e7_gate.json", {
        "status": "not computed",
        "reason": ("The gate is the distance between the injected deviation-type distribution and "
                   "the real one in BPI 2019, against a tolerance fixed in advance. BPI 2019 was not "
                   "obtained, so the real distribution does not exist for this run; and "
                   "configs/default.yaml records the tolerance as not_prespecified, so no threshold "
                   "was fixed either. Both halves of the gate are therefore absent."),
        "consequence": ("Every LeakBench result in this run is reported as a property of the "
                        "benchmark and bears on the dichotomy not at all."),
        "injected_mean_shares": df.groupby("leak_type")["share"].mean().round(4).to_dict()})


if __name__ == "__main__":
    main()

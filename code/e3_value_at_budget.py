"""E3 -- H4 value at budget, with the four comparators of the manuscript.

The pre-specified pairing is BPI 2019 (leakage) x IEEE-CIS/BAF (fraud). Neither side of that pairing
was obtainable. What is run instead, and labelled as such everywhere: the ULB credit-card file
supplies the fraud queue with its REAL labels and real transaction amounts, and the leakage queue is
supplied by LeakBench injection (src/leakbench/inject.py) into the same file. The leakage half is
therefore a benchmark property, not real leakage, and the E7 gate that would license reading it
otherwise cannot be evaluated without BPI 2019.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib", "src"))

import numpy as np
import pandas as pd

import common as C

CODE_SRC = os.path.join(C.PAPER, "lib", "src")
sys.path.insert(0, os.path.abspath(os.path.join(C.PAPER, "lib")))
from src.leakbench.inject import inject, verify, LEAK_TYPES          # noqa: E402
from src.triage.value_at_budget import vab_grid, value_at_budget, policy_oracle, \
    policy_single_queue, policy_value_based, policy_fraud_first, policy_fixed_split   # noqa: E402

TYPE_MIX = {"DUP_EXACT": 1, "DUP_NEAR": 1, "WRONG_PAYEE": 1, "TRANSPOSE": 1,
            "DOUBLE_FX": 0.5, "SPLIT_PAY": 0.5}
RATE = 0.01
SEED = 0

# loss-given-deviation model (S2): the share of the record's amount expected to be lost per type
LGD = {"DUP_EXACT": 1.0, "DUP_NEAR": 1.0, "WRONG_PAYEE": 0.6, "TRANSPOSE": 0.5,
       "DOUBLE_FX": 0.5, "SPLIT_PAY": 0.3, "NONE": 0.0}


def leak_rule_score(df: pd.DataFrame) -> np.ndarray:
    """Conformance stand-in on a log-less host: the duplicate / near-duplicate / split rules a
    payables control runs. This is a rule detector, not the alignment channel."""
    d = df.reset_index(drop=True)
    s = np.zeros(len(d))
    key = d["counterparty_id"].astype(str) + "|" + d["amount"].round(2).astype(str)
    s += key.duplicated(keep=False).values.astype(float) * 2.0
    amt_r = d["amount"].round(0)
    key2 = d["counterparty_id"].astype(str) + "|" + amt_r.astype(str)
    s += key2.duplicated(keep=False).values.astype(float)
    g = d.groupby(["entity_id", d["day"].astype(int)])["record_id"].transform("size").values
    s += (g >= 3).astype(float)
    return s + np.linspace(0, 1e-9, len(d))


def main():
    d = C.load_ulb().sort_values("Time").reset_index(drop=True)
    n = len(d)
    i_va = int(0.8 * n)
    feats = [c for c in d.columns if c != "Class"]
    Xtr_all = d.loc[:i_va - 1, feats].values
    ytr_all = d.loc[:i_va - 1, "Class"].values
    rng = np.random.default_rng(0)
    sub = np.sort(np.concatenate([np.flatnonzero(ytr_all == 1),
                                  rng.choice(np.flatnonzero(ytr_all == 0), 50000, replace=False)]))
    s_f_te, model = C.fit_gbdt(Xtr_all[sub], ytr_all[sub], d.loc[i_va:, feats].values, seed=SEED)
    te = d.loc[i_va:].reset_index(drop=True)

    harmonised = pd.DataFrame({
        "record_id": np.arange(len(te)),
        "day": (te["Time"].values / 3600.0).astype(int),        # hourly budget periods over the 2-day log
        "entity_id": (np.abs(te["V1"].values * 1000).astype(int) % 500),
        "counterparty_id": (np.abs(te["V2"].values * 1000).astype(int) % 2000),
        "amount": te["Amount"].values,
        "currency": "EUR",
        "y": te["Class"].values.astype(int)})
    inj = inject(harmonised, rate=RATE, seed=SEED, type_mix=TYPE_MIX)
    C.dump("e7_leakbench_injection_verify.json",
           {**verify(inj), "host": "ULB credit-card (test split)", "rate": RATE, "seed": SEED,
            "type_mix": TYPE_MIX})

    p_f = np.zeros(len(inj))
    base = inj.injected == 0
    p_f[base.values] = s_f_te[inj.loc[base, "record_id"].values]
    # injected records inherit the fraud score of the record they were copied from
    src = inj.loc[~base, "source_record_id"].values.astype(int)
    p_f[(~base).values] = s_f_te[src]

    p_l_raw = leak_rule_score(inj)
    p_l = (p_l_raw - p_l_raw.min()) / (p_l_raw.max() - p_l_raw.min() + 1e-12)
    v_l = inj["amount"].values * inj["leak_type"].map(LGD).fillna(0.5).values
    # a case not flagged by the typology still carries an expected leakage value of zero in EV
    v_l = np.where(inj["leak_type"].values == "NONE", inj["amount"].values * 0.5, v_l)

    frame = pd.DataFrame({"day": inj["day"].values, "p_f": p_f, "p_l": p_l,
                          "amount": inj["amount"].values, "v_l": v_l, "y": inj["y"].values})
    out, piv = vab_grid(frame, k_fracs=(0.005, 0.01, 0.02, 0.05), lgfs=(0.5, 1.0),
                        recs=(0.2, 0.5, 0.8, 1.0), min_share={"F": 0.1, "L": 0.1})
    C.dump("tab_vab_full_grid.csv", out)
    pv = piv.reset_index()
    C.dump("tab_vab_pivot.csv", pv)

    prim = out[(out.k_frac == 0.01) & (out.lgf == 1.0) & (out.recoverability == 0.5)]
    p = prim.set_index("policy")
    two, single, orc = p.loc["two_queue"], p.loc["single_queue"], p.loc["oracle"]
    grid1 = pv[pv.k_frac == 0.01]
    summary = {
        "host": "ULB credit-card test split with LeakBench-injected leakage (benchmark property)",
        "budget": 0.01, "lambda_F": 1.0, "rho_L": 0.5,
        "two_queue_VaB_total": float(two.VaB_total), "single_queue_VaB_total": float(single.VaB_total),
        "oracle_VaB_total": float(orc.VaB_total),
        "fraud_first_VaB_total": float(p.loc["fraud_first"].VaB_total),
        "fixed_50_VaB_total": float(p.loc["fixed_50"].VaB_total),
        "H4_primary_difference": float(two.VaB_total - single.VaB_total),
        "share_of_oracle_gap_closed": float((two.VaB_total - single.VaB_total) /
                                            max(orc.VaB_total - single.VaB_total, 1e-12)),
        "n_grid_points_at_K1pct": int(len(grid1)),
        "n_winning_grid_points_at_K1pct": int(grid1["two_queue_wins"].sum()),
        "worst_single_queue_advantage_at_K1pct": float(
            max(0.0, (grid1["single_queue"] - grid1["two_queue"]).max())),
        "two_queue_VaB_F": float(two.VaB_F), "two_queue_VaB_L": float(two.VaB_L),
        "single_queue_VaB_F": float(single.VaB_F), "single_queue_VaB_L": float(single.VaB_L),
        "oracle_VaB_F": float(orc.VaB_F), "oracle_VaB_L": float(orc.VaB_L)}
    C.dump("h4_primary_endpoint.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()

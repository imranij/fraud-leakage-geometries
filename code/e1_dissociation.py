"""E1 -- H1 dissociation, the side of it that the obtained data support.

Fraud side (H1(c)): Rules, IF, AE, GBDT, FTT-substitute on every obtained fraud dataset,
value-weighted recall at a 1% false-positive rate, five seeds, paired tests.

Leakage side (H1(a),(b)): NOT COMPUTED on real data -- the BPI Challenge 2019 log could not be
obtained (see DATA_PROVENANCE.md). What is computed instead is LeakBench injection on an obtained
host (ULB), which is a BENCHMARK PROPERTY and is written to its own file; the E7 gate that would
license reading it as evidence for the dichotomy cannot be evaluated without the real log.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import common as C

FPR = 0.01


def eval_dataset(name, Xtr, ytr, Xte, yte, vte, df_te_for_rules=None, value_col=None,
                 entity_col=None, run_ftt=True):
    rows = []
    # unsupervised / rule detectors do not use labels; still run per seed where stochastic
    for seed in C.SEEDS:
        sc = {}
        sc["GBDT (records)"] = C.fit_gbdt(Xtr, ytr, Xte, seed=seed)[0]
        sc["IF"] = C.fit_isolation_forest(Xtr, Xte, seed=seed)
        sc["AE"] = C.fit_autoencoder(Xtr, Xte, seed=seed)
        if run_ftt:
            sc["FTT"] = C.FTTLite(seed=seed).fit_predict(Xtr, ytr, Xte)
        if df_te_for_rules is not None:
            sc["Rules"] = C.rules_score(df_te_for_rules, value_col, entity_col, None)
        for det, s in sc.items():
            rows.append({"dataset": name, "detector": det, "seed": seed,
                         "vw_recall_at_1pct_fpr": C.value_weighted_recall_at_fpr(yte, s, vte, FPR),
                         "count_recall_at_1pct_fpr": C.count_recall_at_fpr(yte, s, FPR),
                         "count_recall_at_0.1pct_fpr": C.count_recall_at_fpr(yte, s, 0.001),
                         "pr_auc": C.pr_auc(yte, s)})
        print(f"  {name} seed {seed} done", flush=True)
    return pd.DataFrame(rows)


def main():
    all_rows = []

    # ---------------- vehicle claims: split by year (1994-95 train, 1996 test)
    cc = C.load_carclaims()
    X, y, val, year, cols = C.carclaims_matrix(cc)
    tr_m = year < 1996
    Xtr, ytr = X[tr_m].values, y[tr_m]
    te_m = ~tr_m
    Xte, yte, vte = X[te_m].values, y[te_m], val[te_m]
    cc_te = cc[te_m].reset_index(drop=True).assign(_value=vte)
    print(f"Vehicle: train {Xtr.shape} pos {ytr.sum()} | test {Xte.shape} pos {yte.sum()}", flush=True)
    r = eval_dataset("Vehicle", Xtr, ytr, Xte, yte, vte, cc_te, "_value", "Make")
    all_rows.append(r); C.dump("tab_dissociation_fraud_raw_Vehicle.csv", r)

    # ---------------- Medicare: provider level, split by mean claim date within provider
    m, F, y, val, feat = C.medicare_provider_features()
    cut = np.quantile(m["start_mean"].values, 0.7)
    tr_m = m["start_mean"].values <= cut
    Xtr, ytr = F[tr_m].values, y[tr_m]
    Xte, yte, vte = F[~tr_m].values, y[~tr_m], val[~tr_m]
    md_te = m[~tr_m].reset_index(drop=True)
    print(f"Medicare: train {Xtr.shape} pos {ytr.sum()} | test {Xte.shape} pos {yte.sum()}", flush=True)
    r = eval_dataset("Medicare", Xtr, ytr, Xte, yte, vte, md_te, "amt_sum", None)
    all_rows.append(r); C.dump("tab_dissociation_fraud_raw_Medicare.csv", r)

    # ---------------- ULB: chronological 70/10/20 over its two days (S9 split rule)
    d = C.load_ulb()
    d = d.sort_values("Time").reset_index(drop=True)
    n = len(d)
    i_tr, i_va = int(0.7 * n), int(0.8 * n)
    feats = [c for c in d.columns if c not in ("Class",)]
    Xtr = d.loc[:i_va - 1, feats].values
    ytr = d.loc[:i_va - 1, "Class"].values
    te = d.loc[i_va:].reset_index(drop=True)
    Xte, yte, vte = te[feats].values, te["Class"].values, te["Amount"].values
    print(f"ULB: train {Xtr.shape} pos {ytr.sum()} | test {Xte.shape} pos {yte.sum()}", flush=True)
    rng = np.random.default_rng(0)
    pos_i = np.flatnonzero(ytr == 1)
    neg_i = rng.choice(np.flatnonzero(ytr == 0), 50000, replace=False)
    sub = np.sort(np.concatenate([pos_i, neg_i]))
    print(f"ULB: training subsample {len(sub)} rows (all {len(pos_i)} positives kept) -- "
          f"computational budget, logged in DEVIATIONS.md", flush=True)
    r = eval_dataset("ULB", Xtr[sub], ytr[sub], Xte, yte, vte, te, "Amount", None)
    all_rows.append(r); C.dump("tab_dissociation_fraud_raw_ULB.csv", r)

    res = pd.concat(all_rows, ignore_index=True)
    C.dump("tab_dissociation_fraud_raw.csv", res)

    agg = (res.groupby(["dataset", "detector"])["vw_recall_at_1pct_fpr"]
           .agg(["mean", "std", "count"]).reset_index())
    agg2 = (res.groupby(["dataset", "detector"])[["count_recall_at_1pct_fpr", "pr_auc"]]
            .mean().reset_index())
    agg = agg.merge(agg2, on=["dataset", "detector"])
    C.dump("tab_dissociation_fraud_summary.csv", agg)

    # H1(c): GBDT minus best of {IF, AE, CONF}; CONF does not apply (no traces on fraud data)
    tests = []
    for ds, g in res.groupby("dataset"):
        piv = g.pivot_table(index="seed", columns="detector", values="vw_recall_at_1pct_fpr")
        unsup = [c for c in ["IF", "AE", "Rules"] if c in piv.columns]
        best_name = piv[unsup].mean().idxmax()
        t = C.paired_tests(piv["GBDT (records)"].values, piv[best_name].values)
        t.update({"dataset": ds, "comparison": "GBDT - best unsupervised", "best_unsupervised": best_name,
                  "gbdt_mean": float(piv["GBDT (records)"].mean()), "best_mean": float(piv[best_name].mean())})
        tests.append(t)
        if "FTT" in piv.columns:
            t2 = C.paired_tests(piv["FTT"].values, piv["GBDT (records)"].values)
            t2.update({"dataset": ds, "comparison": "FTT - GBDT", "best_unsupervised": "",
                       "gbdt_mean": float(piv["GBDT (records)"].mean()), "best_mean": float(piv["FTT"].mean())})
            tests.append(t2)
    C.dump("tab_h1c_paired_tests.csv", pd.DataFrame(tests))

    # ---------------- leakage side: explicitly not computed on real data
    C.dump("h1_leakage_real_log.json", {
        "status": "not computed",
        "reason": ("BPI Challenge 2019 could not be obtained: the canonical host (data.4tu.nl) is "
                   "blocked at the egress gateway and no GitHub mirror carrying the log was found "
                   "(the log is ~695 MB, above GitHub's file-size limit; every candidate repository "
                   "inspected documents the log as a local download, not a committed file). "
                   "H1(a) and H1(b), the leakage-population description, the process-drift analysis "
                   "and the E7 gate therefore have no real-log input."),
        "affects": ["H1(a)", "H1(b)", "leakage population (Table S5)", "E7 gate",
                    "H3(c) stationarity and drift", "H4 leakage queue on real data",
                    "H5 leakage case set"]})


if __name__ == "__main__":
    main()

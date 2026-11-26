"""e14_baf_dissociation.py -- the BAF column of Table 3 (tab:dissociation).

Same H1(c) protocol as e1_dissociation.py's vehicle-claims and Medicare arms: the same five
detectors (Rules, IF, AE, GBDT records, FTT-substitute), the same five seeds, value-weighted
recall at a 1% false-positive rate, the same paired tests.

Split (temporal, as configs/default.yaml specifies and as the BAF paper itself uses):
  train = months 0-5 (794,989 rows), test = months 6-7. `month` is the split variable and is
  NOT a feature.

VALUE PROXY (deviation, logged in DEVIATIONS.md 7.1).  Bank Account Fraud carries no monetary
amount: it is an application file, and no field records a loss.  The value proxy adopted here is
`proposed_credit_limit`, the credit line the application asks for -- the exposure the institution
carries if a fraudulent application is approved.  It is a proxy for exposure, not an observed
loss.  Count-based recall is computed beside it for every cell and reported in RESULTS_REPORT.md.
"""
from __future__ import annotations
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import common as C

FPR = 0.01
RAWP = os.path.join(C.RAW, "baf", "Base.csv")
VALUE_COL = "proposed_credit_limit"
CAT = ["payment_type", "employment_status", "housing_status", "source", "device_os"]
TRAIN_NEG_SUBSAMPLE = 120_000          # computational budget; all positives kept


def load_baf():
    d = pd.read_csv(RAWP, engine="c")
    assert d.shape == (1_000_000, 32), d.shape
    assert int(d.fraud_bool.sum()) == 11_029, int(d.fraud_bool.sum())
    return d


def main():
    t0 = time.time()
    d = load_baf()
    print(f"BAF loaded {d.shape} fraud={int(d.fraud_bool.sum())} ({d.fraud_bool.mean():.4%}) "
          f"{time.time()-t0:.0f}s", flush=True)
    y = d["fraud_bool"].to_numpy(np.int8)
    month = d["month"].to_numpy(np.int8)
    value = d[VALUE_COL].to_numpy(np.float64)

    X = d.drop(columns=["fraud_bool", "month"]).copy()
    for c in CAT:
        X[c] = X[c].astype("category").cat.codes
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes
    feats = list(X.columns)
    Xa = X.to_numpy(np.float32)
    del X
    rules_df = d[[VALUE_COL, "device_os", "zip_count_4w", "velocity_4w",
                  "bank_branch_count_8w"]].copy()
    del d

    tr = month <= 5
    te = ~tr
    Xte, yte, vte = Xa[te], y[te].astype(int), value[te]
    rd_te = rules_df[te].reset_index(drop=True)
    # rules detector: the same shape as e1's -- large amount, zero amount, velocity, duplicate
    s_rules = np.zeros(len(rd_te))
    v = rd_te[VALUE_COL].to_numpy(float)
    s_rules += (v >= np.quantile(v, 0.99)).astype(float)
    s_rules += (v == 0).astype(float)
    cnt = rd_te.groupby("device_os")[VALUE_COL].transform("size").to_numpy()
    s_rules += (cnt >= np.quantile(cnt, 0.99)).astype(float)
    s_rules += rd_te[[VALUE_COL, "device_os"]].duplicated(keep=False).to_numpy().astype(float)
    s_rules = s_rules + np.linspace(0, 1e-9, len(rd_te))

    rng = np.random.default_rng(0)
    pos_i = np.flatnonzero(tr & (y == 1))
    neg_i = rng.choice(np.flatnonzero(tr & (y == 0)), TRAIN_NEG_SUBSAMPLE, replace=False)
    sub = np.sort(np.concatenate([pos_i, neg_i]))
    Xtr, ytr = Xa[sub], y[sub].astype(int)
    print(f"train {Xtr.shape} pos {ytr.sum()} (all months 0-5 positives kept, "
          f"{TRAIN_NEG_SUBSAMPLE} negatives subsampled) | test {Xte.shape} pos {yte.sum()}",
          flush=True)

    rows = []
    for seed in C.SEEDS:
        sc = {}
        t = time.time(); sc["GBDT (records)"] = C.fit_gbdt(Xtr, ytr, Xte, seed=seed)[0]
        print(f"   seed {seed} gbdt {time.time()-t:.0f}s", flush=True)
        t = time.time(); sc["IF"] = C.fit_isolation_forest(Xtr, Xte, seed=seed)
        print(f"   seed {seed} if {time.time()-t:.0f}s", flush=True)
        t = time.time(); sc["AE"] = C.fit_autoencoder(Xtr, Xte, seed=seed)
        print(f"   seed {seed} ae {time.time()-t:.0f}s", flush=True)
        t = time.time(); sc["FTT"] = C.FTTLite(seed=seed).fit_predict(Xtr, ytr, Xte)
        print(f"   seed {seed} ftt {time.time()-t:.0f}s", flush=True)
        sc["Rules"] = s_rules
        for det, s in sc.items():
            rows.append({"dataset": "BAF", "detector": det, "seed": seed,
                         "vw_recall_at_1pct_fpr": C.value_weighted_recall_at_fpr(yte, s, vte, FPR),
                         "count_recall_at_1pct_fpr": C.count_recall_at_fpr(yte, s, FPR),
                         "count_recall_at_0.1pct_fpr": C.count_recall_at_fpr(yte, s, 0.001),
                         "pr_auc": C.pr_auc(yte, s)})
        C.dump("tab_dissociation_fraud_raw_BAF.csv", pd.DataFrame(rows))
        print(f"  BAF seed {seed} done {time.time()-t0:.0f}s", flush=True)

    res = pd.DataFrame(rows)
    C.dump("tab_dissociation_fraud_raw_BAF.csv", res)

    agg = (res.groupby(["dataset", "detector"])["vw_recall_at_1pct_fpr"]
           .agg(["mean", "std", "count"]).reset_index())
    agg2 = (res.groupby(["dataset", "detector"])[["count_recall_at_1pct_fpr", "pr_auc"]]
            .mean().reset_index())
    C.dump("tab_dissociation_fraud_summary_BAF.csv", agg.merge(agg2, on=["dataset", "detector"]))

    tests = []
    piv = res.pivot_table(index="seed", columns="detector", values="vw_recall_at_1pct_fpr")
    pivc = res.pivot_table(index="seed", columns="detector", values="count_recall_at_1pct_fpr")
    unsup = [c for c in ["IF", "AE", "Rules"] if c in piv.columns]
    best = piv[unsup].mean().idxmax()
    t1 = C.paired_tests(piv["GBDT (records)"].values, piv[best].values)
    t1.update({"dataset": "BAF", "comparison": "GBDT - best unsupervised", "best_unsupervised": best,
               "gbdt_mean": float(piv["GBDT (records)"].mean()), "best_mean": float(piv[best].mean()),
               "metric": "vw_recall_at_1pct_fpr"})
    tests.append(t1)
    bestc = pivc[unsup].mean().idxmax()
    t1c = C.paired_tests(pivc["GBDT (records)"].values, pivc[bestc].values)
    t1c.update({"dataset": "BAF", "comparison": "GBDT - best unsupervised", "best_unsupervised": bestc,
                "gbdt_mean": float(pivc["GBDT (records)"].mean()), "best_mean": float(pivc[bestc].mean()),
                "metric": "count_recall_at_1pct_fpr"})
    tests.append(t1c)
    t2 = C.paired_tests(piv["FTT"].values, piv["GBDT (records)"].values)
    t2.update({"dataset": "BAF", "comparison": "FTT - GBDT", "best_unsupervised": "",
               "gbdt_mean": float(piv["GBDT (records)"].mean()), "best_mean": float(piv["FTT"].mean()),
               "metric": "vw_recall_at_1pct_fpr"})
    tests.append(t2)
    C.dump("tab_h1c_paired_tests_BAF.csv", pd.DataFrame(tests))

    C.dump("baf_value_proxy.json", {
        "dataset": "Bank Account Fraud (NeurIPS 2022), Base variant",
        "rows": 1000000, "cols": 32, "positives": 11029, "positive_rate": 0.011029,
        "split": "temporal: train months 0-5, test months 6-7 (month is not a feature)",
        "n_train_rows_available": int(tr.sum()), "n_train_rows_used": int(len(sub)),
        "n_test_rows": int(te.sum()), "n_test_positives": int(yte.sum()),
        "value_proxy_field": VALUE_COL,
        "value_proxy_definition": ("the credit line the application requests, taken as the exposure "
                                   "the institution carries if a fraudulent application is approved. "
                                   "BAF records no monetary loss; this is a proxy for exposure, not "
                                   "an observed amount."),
        "value_proxy_distinct_values": 12,
        "value_proxy_mean_all": float(value.mean()),
        "value_proxy_mean_positives": float(value[y == 1].mean()),
        "value_proxy_mean_negatives": float(value[y == 0].mean()),
        "test_positive_value_total": float(vte[yte == 1].sum()),
        "count_recall_reported_beside": True})
    print(f"BAF arm complete in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

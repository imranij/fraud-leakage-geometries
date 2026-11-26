"""e20_ieee_dissociation.py -- the IEEE-CIS column of Table 3 (tab:dissociation), and the cached
feature matrices the IEEE-CIS adversary (e21) and the IEEE-CIS triage arm (e22) reuse.

Same H1(c) protocol as e1_dissociation.py (vehicle, Medicare) and e14_baf_dissociation.py (BAF):
the same five detectors (Rules, IF, AE, GBDT records, FTT-substitute), the same five seeds, value-
weighted recall at a 1% false-positive rate, count-based recall reported beside it, the same paired
tests for the H1(c) row.

VALUE.  Unlike every other fraud column in the table, IEEE-CIS carries a REAL monetary field,
`TransactionAmt` (the transaction amount in an unstated currency; median 68.77, max 31,937).  The
value-weighted recall of this column is therefore a genuine value metric on observed amounts, not a
proxy: the BAF column is weighted by `proposed_credit_limit` (an exposure proxy; DEVIATIONS 7.1),
the vehicle column by the midpoint of a coded price band (DEVIATIONS 7.2), and the Medicare column
by total reimbursement at provider level.

SPLIT (temporal).  `TransactionDT` is a time DELTA in seconds from an unstated reference, not a
timestamp; it is used purely as an ordering variable.  Rows are ordered by TransactionDT and cut at
its 75th percentile -- the same 75/25 temporal proportion as the BAF arm (months 0-5 / 6-7).  The
cut value in seconds and in days-from-the-first-transaction is recorded in
results/ieee_cis_design.json.  TransactionDT is NOT a feature.

IDENTITY JOIN.  train_identity.csv covers 144,233 of the 590,540 transactions (24.42%).  The 75.58%
without an identity record are KEPT, not dropped: the join is a left join on TransactionID, the 40
identity columns are missing for those rows, and an explicit binary feature `has_identity` records
whether the row joined.  Missingness is then handled by the single imputation rule below, so no
transaction is discarded for want of an identity row.

MISSINGNESS.  Columns that are entirely missing within the training window are dropped (recorded).
Every remaining missing entry is replaced by the TRAINING-window median of its column.  The same
imputed matrix is given to all five detectors, so the comparison is on equal information: LightGBM
would otherwise exploit native NaN handling that the isolation forest, the autoencoder and the
neural control cannot use.
"""
from __future__ import annotations
import sys, os, time, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import common as C

FPR = 0.01
DIR = os.path.join(C.RAW, "ieee_cis")
TXN = os.path.join(DIR, "train_transaction.csv")
IDN = os.path.join(DIR, "train_identity.csv")
CACHE = os.path.join(C.RES, "cache")
os.makedirs(CACHE, exist_ok=True)
VALUE_COL = "TransactionAmt"
TRAIN_NEG_SUBSAMPLE = 120_000          # identical computational budget to the BAF arm
SPLIT_Q = 0.75

TXN_OBJ = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
           "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]
IDN_OBJ = ["id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29", "id_30", "id_31",
           "id_33", "id_34", "id_35", "id_36", "id_37", "id_38", "DeviceType", "DeviceInfo"]


def build():
    """One pass over the 652 MB transaction file; returns the arrays the run needs."""
    t0 = time.time()
    hdr = pd.read_csv(TXN, nrows=0).columns.tolist()
    dt = {c: np.float32 for c in hdr}
    dt.update({"TransactionID": np.int64, "isFraud": np.int8, "TransactionDT": np.int64})
    for c in TXN_OBJ:
        dt[c] = "object"
    d = pd.read_csv(TXN, dtype=dt, engine="c")
    print(f"transactions {d.shape} in {time.time()-t0:.0f}s", flush=True)
    assert d.shape == (590_540, 394), d.shape
    assert int(d.isFraud.sum()) == 20_663, int(d.isFraud.sum())

    ih = pd.read_csv(IDN, nrows=0).columns.tolist()
    idt = {c: np.float32 for c in ih}
    idt["TransactionID"] = np.int64
    for c in IDN_OBJ:
        idt[c] = "object"
    idn = pd.read_csv(IDN, dtype=idt, engine="c")
    assert idn.shape[0] == 144_233, idn.shape
    print(f"identity {idn.shape}", flush=True)

    d = d.merge(idn, on="TransactionID", how="left")
    del idn
    n_id = int(d["id_01"].notna().sum()) if "id_01" in d.columns else 0
    d["has_identity"] = (d["DeviceType"].notna() | d["id_01"].notna()).astype(np.float32)
    n_joined = int(d["has_identity"].sum())
    print(f"joined {d.shape}; has_identity {n_joined} ({n_joined/len(d):.4%})", flush=True)

    y = d["isFraud"].to_numpy(np.int8)
    tdt = d["TransactionDT"].to_numpy(np.int64)
    value = d[VALUE_COL].to_numpy(np.float64)

    # temporal split -- TransactionDT is an ordering variable only, never a feature
    cut = float(np.quantile(tdt, SPLIT_Q))
    tr = tdt <= cut
    te = ~tr

    # rules detector inputs (interpretable fields only)
    rules_df = pd.DataFrame({VALUE_COL: d[VALUE_COL].astype(float),
                             "card1": d["card1"].astype("float32")})

    # feature matrix
    X = d.drop(columns=["isFraud", "TransactionDT", "TransactionID"])
    del d
    cats = [c for c in TXN_OBJ + IDN_OBJ if c in X.columns]
    for c in cats:
        X[c] = X[c].astype("category").cat.codes.astype(np.float32).replace(-1.0, np.nan)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes.astype(np.float32)

    # drop columns entirely missing (or constant) within the TRAINING window
    tr_notna = X[tr].notna().sum(axis=0)
    all_nan_train = [c for c in X.columns if tr_notna[c] == 0]
    X = X.drop(columns=all_nan_train)
    nun = X[tr].nunique(dropna=True)
    constant_train = [c for c in X.columns if nun[c] <= 1]
    X = X.drop(columns=constant_train)
    feats = list(X.columns)
    print(f"features {len(feats)}; dropped all-NaN-in-train {len(all_nan_train)}, "
          f"constant-in-train {len(constant_train)}", flush=True)

    med = X[tr].median(numeric_only=True)
    n_missing_cells = int(X.isna().to_numpy().sum())
    X = X.fillna(med).fillna(0.0)
    Xa = X.to_numpy(np.float32)
    del X

    meta = {"all_nan_in_train": all_nan_train, "constant_in_train": constant_train,
            "n_features": len(feats), "n_missing_cells_imputed": n_missing_cells,
            "n_identity_rows": n_id, "n_joined": n_joined, "cut_dt": cut,
            "tdt_min": int(tdt.min()), "tdt_max": int(tdt.max())}
    return Xa, y, value, tdt, tr, te, feats, rules_df, meta


def main():
    t0 = time.time()
    Xa, y, value, tdt, tr, te, feats, rules_df, meta = build()

    Xte, yte, vte = Xa[te], y[te].astype(int), value[te]
    rng = np.random.default_rng(0)
    pos_i = np.flatnonzero(tr & (y == 1))
    neg_all = np.flatnonzero(tr & (y == 0))
    neg_i = rng.choice(neg_all, min(TRAIN_NEG_SUBSAMPLE, len(neg_all)), replace=False)
    sub = np.sort(np.concatenate([pos_i, neg_i]))
    Xtr, ytr = Xa[sub], y[sub].astype(int)
    del Xa

    design = {
        "dataset": "IEEE-CIS Fraud Detection (Vesta / IEEE-CIS 2019)",
        "rows": 590540, "cols_transaction": 394, "cols_identity": 41,
        "positives": int(y.sum()), "positive_rate": float(y.mean()),
        "identity_rows": 144233,
        "identity_coverage": float(meta["n_joined"] / 590540),
        "identity_handling": ("left join on TransactionID; the 75.58% of transactions with no "
                              "identity record are KEPT, flagged by a binary `has_identity` "
                              "feature, and their 40 identity columns imputed by the training-"
                              "window median like any other missing entry. No row is dropped."),
        "split": (f"temporal on TransactionDT (a time delta in seconds, used as an ordering "
                  f"variable only): train = rows with TransactionDT <= the 75th percentile "
                  f"({meta['cut_dt']:.0f} s = day {(meta['cut_dt']-meta['tdt_min'])/86400:.1f} "
                  f"from the first transaction), test = the remainder. Same 75/25 temporal "
                  f"proportion as the BAF arm. TransactionDT is not a feature."),
        "transaction_dt_min_s": meta["tdt_min"], "transaction_dt_max_s": meta["tdt_max"],
        "span_days": (meta["tdt_max"] - meta["tdt_min"]) / 86400.0,
        "cut_dt_s": meta["cut_dt"],
        "n_train_rows_available": int(tr.sum()), "n_train_rows_used": int(len(sub)),
        "n_train_positives": int(ytr.sum()),
        "n_test_rows": int(te.sum()), "n_test_positives": int(yte.sum()),
        "test_positive_rate": float(yte.mean()),
        "train_positive_rate": float(y[tr].mean()),
        "value_field": VALUE_COL,
        "value_is_real_money": True,
        "value_definition": ("TransactionAmt, the observed transaction amount. This is a REAL "
                             "monetary field, so the value-weighted recall of this column is a "
                             "genuine value metric -- unlike the BAF column (proposed_credit_limit, "
                             "an exposure proxy) and the vehicle column (price-band midpoints)."),
        "value_median_all": float(np.median(value)), "value_max_all": float(value.max()),
        "value_mean_all": float(value.mean()),
        "value_mean_positives": float(value[y == 1].mean()),
        "value_mean_negatives": float(value[y == 0].mean()),
        "test_positive_value_total": float(vte[yte == 1].sum()),
        "test_total_value": float(vte.sum()),
        "n_features": meta["n_features"],
        "n_cols_dropped_all_nan_in_train": len(meta["all_nan_in_train"]),
        "cols_dropped_all_nan_in_train": meta["all_nan_in_train"],
        "n_cols_dropped_constant_in_train": len(meta["constant_in_train"]),
        "cols_dropped_constant_in_train": meta["constant_in_train"],
        "n_missing_cells_imputed": meta["n_missing_cells_imputed"],
        "missingness_rule": ("columns entirely missing in the training window are dropped; every "
                             "other missing entry is set to the training-window median. The same "
                             "imputed matrix is given to all five detectors so that the comparison "
                             "is on equal information."),
        "train_negatives_subsampled_to": TRAIN_NEG_SUBSAMPLE,
        "count_recall_reported_beside": True,
    }
    C.dump("ieee_cis_design.json", design)
    print(json.dumps({k: design[k] for k in
                      ("n_train_rows_used", "n_train_positives", "n_test_rows",
                       "n_test_positives", "n_features", "span_days")}, indent=1), flush=True)

    # cache for e21 (adversary) and e22 (triage)
    np.save(os.path.join(CACHE, "ieee_Xtr.npy"), Xtr)
    np.save(os.path.join(CACHE, "ieee_ytr.npy"), ytr.astype(np.int8))
    np.save(os.path.join(CACHE, "ieee_Xte.npy"), Xte)
    np.save(os.path.join(CACHE, "ieee_yte.npy"), yte.astype(np.int8))
    np.save(os.path.join(CACHE, "ieee_vte.npy"), vte)
    np.save(os.path.join(CACHE, "ieee_tdt_te.npy"), tdt[te])
    json.dump(feats, open(os.path.join(CACHE, "ieee_feats.json"), "w"))
    print(f"cached matrices {time.time()-t0:.0f}s", flush=True)

    # ------------------------------------------------------------------ Rules
    rd_te = rules_df[te].reset_index(drop=True)
    s_rules = np.zeros(len(rd_te))
    v = rd_te[VALUE_COL].to_numpy(float)
    s_rules += (v >= np.quantile(v, 0.99)).astype(float)         # unusually large amount
    s_rules += (v == 0).astype(float)                            # zero-value record
    cnt = rd_te.groupby("card1")[VALUE_COL].transform("size").to_numpy()
    s_rules += (cnt >= np.quantile(cnt, 0.99)).astype(float)     # velocity on the card
    s_rules += rd_te[[VALUE_COL, "card1"]].duplicated(keep=False).to_numpy().astype(float)
    s_rules = s_rules + np.linspace(0, 1e-9, len(rd_te))
    del rd_te, rules_df

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
            rows.append({"dataset": "IEEE-CIS", "detector": det, "seed": seed,
                         "vw_recall_at_1pct_fpr": C.value_weighted_recall_at_fpr(yte, s, vte, FPR),
                         "count_recall_at_1pct_fpr": C.count_recall_at_fpr(yte, s, FPR),
                         "count_recall_at_0.1pct_fpr": C.count_recall_at_fpr(yte, s, 0.001),
                         "pr_auc": C.pr_auc(yte, s)})
        C.dump("tab_dissociation_fraud_raw_IEEE.csv", pd.DataFrame(rows))
        print(f"  IEEE-CIS seed {seed} done {time.time()-t0:.0f}s", flush=True)

    res = pd.DataFrame(rows)
    C.dump("tab_dissociation_fraud_raw_IEEE.csv", res)
    agg = (res.groupby(["dataset", "detector"])["vw_recall_at_1pct_fpr"]
           .agg(["mean", "std", "count"]).reset_index())
    agg2 = (res.groupby(["dataset", "detector"])[
        ["count_recall_at_1pct_fpr", "count_recall_at_0.1pct_fpr", "pr_auc"]]
        .mean().reset_index())
    C.dump("tab_dissociation_fraud_summary_IEEE.csv", agg.merge(agg2, on=["dataset", "detector"]))

    tests = []
    piv = res.pivot_table(index="seed", columns="detector", values="vw_recall_at_1pct_fpr")
    pivc = res.pivot_table(index="seed", columns="detector", values="count_recall_at_1pct_fpr")
    unsup = [c for c in ["IF", "AE", "Rules"] if c in piv.columns]
    for metric, P in (("vw_recall_at_1pct_fpr", piv), ("count_recall_at_1pct_fpr", pivc)):
        best = P[unsup].mean().idxmax()
        t1 = C.paired_tests(P["GBDT (records)"].values, P[best].values)
        t1.update({"dataset": "IEEE-CIS", "comparison": "GBDT - best unsupervised",
                   "best_unsupervised": best,
                   "gbdt_mean": float(P["GBDT (records)"].mean()),
                   "best_mean": float(P[best].mean()), "metric": metric})
        tests.append(t1)
    t2 = C.paired_tests(piv["FTT"].values, piv["GBDT (records)"].values)
    t2.update({"dataset": "IEEE-CIS", "comparison": "FTT - GBDT", "best_unsupervised": "",
               "gbdt_mean": float(piv["GBDT (records)"].mean()),
               "best_mean": float(piv["FTT"].mean()), "metric": "vw_recall_at_1pct_fpr"})
    tests.append(t2)
    C.dump("tab_h1c_paired_tests_IEEE.csv", pd.DataFrame(tests))
    print(f"IEEE-CIS arm complete in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

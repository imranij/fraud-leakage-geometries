"""e15_h1a_diagnosis.py -- diagnosis of the H1(a) anomaly on BPI 2019.

The 2026-09-20 run reported, on `3-way match, invoice after GR`, isolation forest at 81.2% and
autoencoder at 75.9% VALUE-WEIGHTED recall at a 1% FPR -- above supervised GBDT and an order of
magnitude above the fraud side.  This script re-runs exactly e12's protocol (same flow types,
same 70/30 stratified splits, same seeds, same detectors, same fits) and records, for every cell:

  * value-weighted recall at 1% FPR   (the reported quantity)
  * COUNT-based recall at 1% FPR      (the same threshold, positives counted not weighted)
  * Spearman rho between the detector score and the case value on the test split
  * the value-weighted recall of a detector that ranks on VALUE ALONE (the null that would
    explain the anomaly)
  * the value-weighted recall of a RANDOM ranker (finite-sample floor)

plus, per flow type, the concentration of positive value: the share of total positive value held
by the top 1%, 5% and 10% of positive cases, the Gini coefficient of positive value, and the
share of positive value in the largest vendor / company strata.
"""
import os, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__)); PAPER = os.path.dirname(HERE)
CACHE = os.path.join(PAPER, "results", "cache"); RES = os.path.join(PAPER, "results")
SEEDS = [0, 1, 2, 3, 4]
FLOWS = ["3-way match, invoice after GR", "3-way match, invoice before GR", "2-way match"]
FPR = 0.01

lab = pd.read_pickle(os.path.join(CACHE, "labels.pkl"))
cases = pd.read_pickle(os.path.join(CACHE, "cases.pkl"))
acts = pd.read_csv(os.path.join(CACHE, "activities.csv"))
N = len(lab)
ev = np.load(os.path.join(CACHE, "events.npz"))
ci, ac, ts = ev["case_idx"], ev["act_code"], ev["ts"]
o = np.lexsort((ts, ci)); ci_s, ac_s = ci[o], ac[o]
bounds = np.searchsorted(ci_s, np.arange(N + 1))
NACT = int(acts.code.max()) + 1


def thr_at_fpr(y, score, fpr=FPR):
    return np.quantile(np.asarray(score, float)[~np.asarray(y).astype(bool)], 1 - fpr)


def both_recalls(y, score, value, fpr=FPR):
    y = np.asarray(y).astype(bool); score = np.asarray(score, float)
    value = np.nan_to_num(np.abs(np.asarray(value, float)))
    if y.sum() == 0 or (~y).sum() == 0:
        return np.nan, np.nan
    thr = thr_at_fpr(y, score, fpr)
    det = y & (score > thr)
    tot = value[y].sum()
    vw = float(value[det].sum() / tot) if tot > 0 else np.nan
    cnt = float(det.sum() / y.sum())
    return vw, cnt


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() <= 0:
        return np.nan
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


# ------------------------------------------------ record-only design matrix (identical to e12)
rec = pd.DataFrame({
    "value": np.nan_to_num(cases["value_absmax"].to_numpy()),
    "log_value": np.log1p(np.nan_to_num(np.abs(cases["value_absmax"].to_numpy()))),
    "month": pd.to_datetime(cases["t_first"].to_numpy(), unit="us", utc=True).month,
    "gr_based": (cases["gr_based_inv_verif"].to_numpy() == "True").astype(int),
    "gr_flag": (cases["goods_receipt_flag"].to_numpy() == "True").astype(int),
})
for col in ["vendor", "spend_area", "doc_type", "item_type", "company"]:
    v = cases[col].astype("category")
    rec[col + "_code"] = v.cat.codes
    freq = v.value_counts(normalize=True)
    rec[col + "_freq"] = v.map(freq).astype(float).to_numpy()
REC_COLS = list(rec.columns)

vend = cases["vendor"].to_numpy(); val = np.nan_to_num(cases["value_absmax"].to_numpy())
key = pd.Series(vend).astype(str) + "|" + pd.Series(np.round(val, 2)).astype(str)
dup = key.duplicated(keep=False).to_numpy().astype(float)
vcount = pd.Series(vend).map(pd.Series(vend).value_counts()).to_numpy()
rules_score = dup + 0.5 * (vcount > np.quantile(vcount, 0.99)).astype(float) \
              + 0.25 * (val > np.quantile(val[val > 0], 0.99)).astype(float)


def seq_of(j):
    return ac_s[bounds[j]:bounds[j + 1]]


def tseq_scores(train_idx, eval_idx, seed):
    Xs, ys = [], []
    rng = np.random.default_rng(seed)
    tr = train_idx if len(train_idx) <= 20000 else rng.choice(train_idx, 20000, replace=False)
    for j in tr:
        s = seq_of(j)
        for k in range(len(s)):
            Xs.append((s[k - 2] if k >= 2 else NACT, s[k - 1] if k >= 1 else NACT)); ys.append(s[k])
    X = np.array(Xs); y = np.array(ys)
    Xo = np.zeros((len(X), 2 * (NACT + 1)), dtype=np.float32)
    Xo[np.arange(len(X)), X[:, 0]] = 1; Xo[np.arange(len(X)), (NACT + 1) + X[:, 1]] = 1
    m = MLPClassifier(hidden_layer_sizes=(64,), max_iter=8, random_state=seed).fit(Xo, y)
    classes = {c: i for i, c in enumerate(m.classes_)}
    out = np.zeros(len(eval_idx)); Bx, Bidx = [], []
    for n_, j in enumerate(eval_idx):
        s = seq_of(j)
        for k in range(len(s)):
            Bx.append((s[k - 2] if k >= 2 else NACT, s[k - 1] if k >= 1 else NACT, s[k])); Bidx.append(n_)
    Bx = np.array(Bx); Bidx = np.array(Bidx)
    Xo2 = np.zeros((len(Bx), 2 * (NACT + 1)), dtype=np.float32)
    Xo2[np.arange(len(Bx)), Bx[:, 0]] = 1; Xo2[np.arange(len(Bx)), (NACT + 1) + Bx[:, 1]] = 1
    P = m.predict_proba(Xo2)
    tgt = np.array([classes.get(t, -1) for t in Bx[:, 2]])
    p = np.where(tgt >= 0, P[np.arange(len(P)), np.clip(tgt, 0, P.shape[1] - 1)], 1e-9)
    np.add.at(out, Bidx, -np.log(np.clip(p, 1e-9, 1)))
    return out / np.bincount(Bidx, minlength=len(eval_idx)).clip(1)


# ------------------------------------------------ concentration of positive value
conc = []
for fl in FLOWS + ["ALL FLOWS"]:
    idx = (np.arange(N) if fl == "ALL FLOWS"
           else np.where(lab["flow_type"].to_numpy() == fl)[0])
    y = lab["pos_B"].to_numpy()[idx].astype(bool)
    v = np.nan_to_num(np.abs(lab["value"].to_numpy()[idx]))
    pv = np.sort(v[y])[::-1]
    tot = pv.sum()
    row = {"flow_type": fl, "n_cases": len(idx), "n_pos": int(y.sum()),
           "pos_value_total": float(tot),
           "gini_positive_value": gini(v[y]),
           "gini_all_value": gini(v),
           "mean_value_positive": float(v[y].mean()) if y.sum() else np.nan,
           "mean_value_negative": float(v[~y].mean()) if (~y).sum() else np.nan,
           "median_value_positive": float(np.median(v[y])) if y.sum() else np.nan,
           "median_value_negative": float(np.median(v[~y])) if (~y).sum() else np.nan}
    for q in (0.01, 0.05, 0.10):
        k = max(1, int(np.ceil(q * len(pv))))
        row[f"share_of_positive_value_in_top_{int(q*100)}pct_of_positives"] = \
            float(pv[:k].sum() / tot) if tot > 0 else np.nan
    # share of TOTAL positive value that lies in the top 1% of ALL cases by value
    for q in (0.01, 0.05):
        thr = np.quantile(v, 1 - q)
        row[f"share_of_positive_value_in_top_{int(q*100)}pct_of_ALL_cases_by_value"] = \
            float(v[y & (v >= thr)].sum() / tot) if tot > 0 else np.nan
        row[f"share_of_positives_in_top_{int(q*100)}pct_of_ALL_cases_by_value"] = \
            float((y & (v >= thr)).sum() / y.sum()) if y.sum() else np.nan
    # stratum concentration
    for col in ("vendor", "company", "spend_area"):
        s = cases[col].to_numpy()[idx]
        if y.sum():
            g = pd.DataFrame({"s": s[y], "v": v[y]}).groupby("s")["v"].agg(["sum", "size"])
            g = g.sort_values("sum", ascending=False)
            row[f"top1_{col}_share_of_positive_value"] = float(g["sum"].iloc[0] / tot) if tot > 0 else np.nan
            row[f"top5_{col}_share_of_positive_value"] = float(g["sum"].iloc[:5].sum() / tot) if tot > 0 else np.nan
            row[f"n_{col}_with_positives"] = int(len(g))
    conc.append(row)
pd.DataFrame(conc).to_csv(os.path.join(RES, "h1a_value_concentration.csv"), index=False)
print(pd.DataFrame(conc)[["flow_type", "n_pos", "gini_positive_value",
                          "share_of_positive_value_in_top_1pct_of_positives",
                          "mean_value_positive", "mean_value_negative"]].to_string(index=False), flush=True)

# ------------------------------------------------ per-detector diagnosis
rows = []
for fl in FLOWS:
    idx = np.where(lab["flow_type"].to_numpy() == fl)[0]
    y = lab["pos_B"].to_numpy()[idx]
    v = np.nan_to_num(np.abs(lab["value"].to_numpy()[idx]))
    if y.sum() < 5:
        print(f"[{fl}] skipped, {int(y.sum())} positives", flush=True)
        continue
    print(f"[{fl}] n={len(idx)} pos={int(y.sum())}", flush=True)
    Xrec = np.nan_to_num(rec.iloc[idx][REC_COLS].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    for seed in SEEDS:
        tr, te = train_test_split(np.arange(len(idx)), test_size=0.3, random_state=seed, stratify=y)
        ytr, yte, vte = y[tr], y[te], v[te]
        sc = StandardScaler().fit(Xrec[tr])
        Ztr = np.nan_to_num(sc.transform(Xrec[tr]), nan=0.0, posinf=0.0, neginf=0.0)
        Zte = np.nan_to_num(sc.transform(Xrec[te]), nan=0.0, posinf=0.0, neginf=0.0)
        scores = {}
        scores["Rules"] = rules_score[idx][te]
        scores["IF"] = -IsolationForest(n_estimators=100, random_state=seed).fit(Ztr[~ytr]).score_samples(Zte)
        aem = MLPRegressor(hidden_layer_sizes=(16,), max_iter=12, random_state=seed).fit(Ztr[~ytr], Ztr[~ytr])
        scores["AE"] = ((aem.predict(Zte) - Zte) ** 2).mean(1)
        scores["TSEQ"] = tseq_scores(idx[tr][~ytr], idx[te], seed)
        scores["GBDT (records)"] = lgb.LGBMClassifier(
            n_estimators=120, learning_rate=0.10, random_state=seed,
            class_weight="balanced", verbose=-1).fit(Xrec[tr], ytr).predict_proba(Xrec[te])[:, 1]
        # the null that would explain the anomaly, and the floor
        scores["VALUE ALONE (null)"] = vte.astype(float)
        scores["RANDOM (floor)"] = np.random.default_rng(seed).random(len(te))
        for det, s in scores.items():
            vw, cnt = both_recalls(yte, s, vte)
            rho = stats.spearmanr(s, vte).statistic if len(np.unique(s)) > 1 else np.nan
            thr = thr_at_fpr(yte, s)
            flagged = s > thr
            rows.append({"flow_type": fl, "detector": det, "seed": seed,
                         "vw_recall_at_1pct_fpr": vw, "count_recall_at_1pct_fpr": cnt,
                         "spearman_score_vs_value": float(rho),
                         "mean_value_of_flagged": float(vte[flagged].mean()) if flagged.sum() else np.nan,
                         "mean_value_of_all_test": float(vte.mean()),
                         "n_flagged": int(flagged.sum()), "n_test": len(te), "n_pos_test": int(yte.sum()),
                         "n_pos_flagged": int((flagged & yte.astype(bool)).sum())})
        pd.DataFrame(rows).to_csv(os.path.join(RES, "h1a_diagnosis_raw.csv"), index=False)
        print(f"   seed {seed} done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "h1a_diagnosis_raw.csv"), index=False)
s = (df.groupby(["flow_type", "detector"])
     [["vw_recall_at_1pct_fpr", "count_recall_at_1pct_fpr", "spearman_score_vs_value",
       "mean_value_of_flagged", "mean_value_of_all_test", "n_pos_flagged", "n_pos_test"]]
     .agg(["mean", "std"]))
s.columns = ["_".join(c) for c in s.columns]
s = s.reset_index()
s.to_csv(os.path.join(RES, "h1a_diagnosis_summary.csv"), index=False)
print(s[["flow_type", "detector", "vw_recall_at_1pct_fpr_mean", "count_recall_at_1pct_fpr_mean",
         "spearman_score_vs_value_mean", "n_pos_flagged_mean", "n_pos_test_mean"]].to_string(index=False))

"""e12_leakage_detectors.py -- the leakage columns of Table 3 (tab:dissociation).

Value-weighted recall at a 1% false-positive rate, mean +- SD over five seeds, per flow type,
for: Rules, IF, AE, TSEQ, CONF, GBDT (records), GBDT+phi, FTT.
Labels: outcome-derived, primary window B (e10).  CONF and GBDT+phi need e11's alignment deltas.
"""
import os, json, sys, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__)); PAPER = os.path.dirname(HERE)
CACHE = os.path.join(PAPER, "results", "cache"); RES = os.path.join(PAPER, "results")
SEEDS = [0, 1, 2, 3, 4]
FLOWS = ["3-way match, invoice after GR", "3-way match, invoice before GR", "2-way match"]
PRIMARY_NOISE = 0.2
FPR = 0.01

lab = pd.read_pickle(os.path.join(CACHE, "labels.pkl"))
cases = pd.read_pickle(os.path.join(CACHE, "cases.pkl"))
acts = pd.read_csv(os.path.join(CACHE, "activities.csv"))
CODE2ACT = dict(zip(acts.code, acts.activity))
N = len(lab)

ev = np.load(os.path.join(CACHE, "events.npz"))
ci, ac, ts = ev["case_idx"], ev["act_code"], ev["ts"]
o = np.lexsort((ts, ci)); ci_s, ac_s = ci[o], ac[o]
bounds = np.searchsorted(ci_s, np.arange(N + 1))

def vwr_at_fpr(y, score, value, fpr=FPR):
    """value-weighted recall among positives at the score threshold giving `fpr` on negatives"""
    y = np.asarray(y).astype(bool); score = np.asarray(score, float); value = np.asarray(value, float)
    value = np.nan_to_num(np.abs(value))
    neg = score[~y]
    if len(neg) == 0 or y.sum() == 0:
        return np.nan
    thr = np.quantile(neg, 1 - fpr)
    det = y & (score > thr)
    tot = value[y].sum()
    return float(value[det].sum() / tot) if tot > 0 else np.nan

# ---------------- record-only design matrix (no trace information) -----------
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

# ---------------- Rules baseline: exact-duplicate + velocity ------------------
vend = cases["vendor"].to_numpy(); val = np.nan_to_num(cases["value_absmax"].to_numpy())
key = pd.Series(vend).astype(str) + "|" + pd.Series(np.round(val, 2)).astype(str)
dup = key.duplicated(keep=False).to_numpy().astype(float)
vcount = pd.Series(vend).map(pd.Series(vend).value_counts()).to_numpy()
rules_score = dup + 0.5 * (vcount > np.quantile(vcount, 0.99)).astype(float) \
              + 0.25 * (val > np.quantile(val[val > 0], 0.99)).astype(float)

# ---------------- TSEQ: next-event model (surprisal) --------------------------
def seq_of(j):
    return ac_s[bounds[j]:bounds[j + 1]]

NACT = int(acts.code.max()) + 1
def tseq_scores(train_idx, eval_idx, seed):
    """MLP next-event predictor on (prev2, prev1) one-hot; score = mean surprisal of the trace."""
    Xs, ys = [], []
    rng = np.random.default_rng(seed)
    tr = train_idx if len(train_idx) <= 20000 else rng.choice(train_idx, 20000, replace=False)
    for j in tr:
        s = seq_of(j)
        for k in range(len(s)):
            p1 = s[k - 1] if k >= 1 else NACT
            p2 = s[k - 2] if k >= 2 else NACT
            Xs.append((p2, p1)); ys.append(s[k])
    X = np.array(Xs); y = np.array(ys)
    Xo = np.zeros((len(X), 2 * (NACT + 1)), dtype=np.float32)
    Xo[np.arange(len(X)), X[:, 0]] = 1; Xo[np.arange(len(X)), (NACT + 1) + X[:, 1]] = 1
    m = MLPClassifier(hidden_layer_sizes=(64,), max_iter=8, random_state=seed)
    m.fit(Xo, y)
    classes = {c: i for i, c in enumerate(m.classes_)}
    out = np.zeros(len(eval_idx))
    Bx, Bidx = [], []
    for n_, j in enumerate(eval_idx):
        s = seq_of(j)
        for k in range(len(s)):
            p1 = s[k - 1] if k >= 1 else NACT
            p2 = s[k - 2] if k >= 2 else NACT
            Bx.append((p2, p1, s[k])); Bidx.append(n_)
    Bx = np.array(Bx); Bidx = np.array(Bidx)
    Xo2 = np.zeros((len(Bx), 2 * (NACT + 1)), dtype=np.float32)
    Xo2[np.arange(len(Bx)), Bx[:, 0]] = 1; Xo2[np.arange(len(Bx)), (NACT + 1) + Bx[:, 1]] = 1
    P = m.predict_proba(Xo2)
    tgt = np.array([classes.get(t, -1) for t in Bx[:, 2]])
    p = np.where(tgt >= 0, P[np.arange(len(P)), np.clip(tgt, 0, P.shape[1] - 1)], 1e-9)
    sur = -np.log(np.clip(p, 1e-9, 1))
    np.add.at(out, Bidx, sur)
    cnt = np.bincount(Bidx, minlength=len(eval_idx)).clip(1)
    return out / cnt

# ---------------- alignment deltas (CONF, phi) --------------------------------
delta_map = {}
dpath = os.path.join(RES, "conf_alignment_delta.csv")
if os.path.exists(dpath):
    dd = pd.read_csv(dpath)
    dd = dd[dd.noise_threshold == PRIMARY_NOISE]
    for fl, g in dd.groupby("flow_type"):
        delta_map[fl] = g.set_index("case_idx")[["delta", "n_log_moves", "n_model_moves", "trace_len"]]
    print("alignment deltas loaded:", {k: len(v) for k, v in delta_map.items()}, flush=True)
else:
    print("NO alignment deltas -- CONF / GBDT+phi will be skipped", flush=True)

TYPE_COLS = [c for c in lab.columns if c.startswith("T_")]

rows = []
for fl in FLOWS:
    idx_all = np.where(lab["flow_type"].to_numpy() == fl)[0]
    dm = delta_map.get(fl)
    # the evaluation set is the aligned sample when CONF is in play, so every detector is
    # scored on exactly the same cases (equal-information requirement)
    idx = np.array(sorted(set(idx_all) & set(dm.index))) if dm is not None else idx_all
    y = lab["pos_B"].to_numpy()[idx]
    v = np.nan_to_num(np.abs(lab["value"].to_numpy()[idx]))
    print(f"[{fl}] eval n={len(idx)} pos={int(y.sum())} ({y.mean():.3%})", flush=True)
    if y.sum() < 5:
        rows.append({"flow_type": fl, "detector": "ALL", "seed": -1,
                     "vw_recall_at_1pct_fpr": np.nan, "n_eval": len(idx), "n_pos": int(y.sum()),
                     "note": "too few positives to evaluate"})
        continue
    Xrec = np.nan_to_num(rec.iloc[idx][REC_COLS].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if dm is not None:
        D = dm.loc[idx]
        phi = np.column_stack([D["delta"].to_numpy(), D["n_log_moves"].to_numpy(),
                               D["n_model_moves"].to_numpy(), D["trace_len"].to_numpy(),
                               lab.iloc[idx][TYPE_COLS].to_numpy(np.float32)])
    for seed in SEEDS:
        tr, te = train_test_split(np.arange(len(idx)), test_size=0.3, random_state=seed, stratify=y)
        ytr, yte = y[tr], y[te]
        vte = v[te]
        def rec_row(det, sc):
            rows.append({"flow_type": fl, "detector": det, "seed": seed,
                         "vw_recall_at_1pct_fpr": vwr_at_fpr(yte, sc, vte),
                         "n_eval": len(te), "n_pos": int(yte.sum()), "note": ""})
        rec_row("Rules", rules_score[idx][te])
        sc = StandardScaler().fit(Xrec[tr])
        Ztr = np.nan_to_num(sc.transform(Xrec[tr]), nan=0.0, posinf=0.0, neginf=0.0)
        Zte = np.nan_to_num(sc.transform(Xrec[te]), nan=0.0, posinf=0.0, neginf=0.0)
        iso = IsolationForest(n_estimators=100, random_state=seed).fit(Ztr[~ytr])
        rec_row("IF", -iso.score_samples(Zte))
        aem = MLPRegressor(hidden_layer_sizes=(16,), max_iter=12, random_state=seed).fit(Ztr[~ytr], Ztr[~ytr])
        rec_row("AE", ((aem.predict(Zte) - Zte) ** 2).mean(1))
        rec_row("TSEQ", tseq_scores(idx[tr][~ytr], idx[te], seed))
        g = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.10, random_state=seed,
                               class_weight="balanced", verbose=-1).fit(Xrec[tr], ytr)
        rec_row("GBDT (records)", g.predict_proba(Xrec[te])[:, 1])
        if dm is not None:
            rec_row("CONF", phi[:, 0][te])          # alignment cost delta, label-free
            Xp = np.column_stack([Xrec, phi])
            gp = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.10, random_state=seed,
                                    class_weight="balanced", verbose=-1).fit(Xp[tr], ytr)
            rec_row("GBDT+$\\phi$", gp.predict_proba(Xp[te])[:, 1])
        pd.DataFrame(rows).to_csv(os.path.join(RES, "tab_dissociation_leakage_raw.csv"), index=False)
        print(f"   seed {seed} done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "tab_dissociation_leakage_raw.csv"), index=False)
s = (df[df.seed >= 0].groupby(["flow_type", "detector"])["vw_recall_at_1pct_fpr"]
     .agg(["mean", "std", "count"]).reset_index())
s.to_csv(os.path.join(RES, "tab_dissociation_leakage_summary.csv"), index=False)
print(s.to_string(index=False))

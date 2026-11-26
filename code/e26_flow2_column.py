"""e26_flow2_column.py -- the `3-way match, invoice before GR` leakage column of Table 3 under the
FULL protocol: all 221,010 cases of the flow, no negative sampling.

What was there before.  `e12_leakage_detectors.py` never reached this flow (its output has only
`3-way match, invoice after GR`).  `e17_conformance_replay.py` did reach it, but on a frame of
3,396 positives plus a seeded sample of 30,000 negatives -- a negative sample, not the population
(`results/conf_replay_sampling_frame.csv`).  A value-weighted recall at a fixed false-positive
rate is not invariant to the negative sample: the 1% threshold is a quantile of the negative score
distribution, and 30,000 of 217,614 negatives estimate that quantile with a different tail than
the population has.  This script runs the population.

Protocol, unchanged from the other leakage columns: five seeds, 70/30 stratified splits,
value-weighted recall at a 1% false-positive rate, count-based recall beside it, the same
record-only design matrix, the same rules baseline, the same IF / AE / TSEQ / GBDT detectors.
CONF and GBDT+phi use TOKEN-BASED REPLAY fitness against the same Inductive-Miner-infrequent
normative model discovered from the same reference set at noise threshold 0.2 -- the substitution
of DEVIATIONS.md 7.6, computed here on every distinct trace variant of the full population and
mapped to every case, so coverage is 100%.  Those two rows are labelled with the substitution and
are NOT alignment-derived.

Value-concentration controls (RESULTS_REPORT 0.2).  On this intent a detector that ranks on case
value alone is a strong competitor on the value-weighted metric, because positive value is
extremely concentrated.  The VALUE ALONE null and the RANDOM floor are therefore scored in the
same runs, and the concentration statistics for the full population are written beside the table.

Outputs carry the `flow2_` prefix.  No existing results file is touched.
"""
from __future__ import annotations
import os, json, time, warnings
from collections import Counter
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.algo.conformance.tokenreplay import algorithm as treplay

from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__)); PAPER = os.path.dirname(HERE)
CACHE = os.path.join(PAPER, "results", "cache"); RES = os.path.join(PAPER, "results")
FLOW = "3-way match, invoice before GR"
SEED, SEEDS = 0, [0, 1, 2, 3, 4]
PRIMARY_NOISE, FPR = 0.2, 0.01

ev = np.load(os.path.join(CACHE, "events.npz"))
ci, ac, ts = ev["case_idx"], ev["act_code"], ev["ts"]
acts = pd.read_csv(os.path.join(CACHE, "activities.csv"))
CODE2ACT = dict(zip(acts.code, acts.activity))
lab = pd.read_pickle(os.path.join(CACHE, "labels.pkl"))
cases = pd.read_pickle(os.path.join(CACHE, "cases.pkl"))
N = len(lab)
o = np.lexsort((ts, ci)); ci_s, ac_s = ci[o], ac[o]
bounds = np.searchsorted(ci_s, np.arange(N + 1))
CORR = {"Cancel Invoice Receipt", "Cancel Goods Receipt", "Vendor creates debit memo",
        "Cancel Subsequent Invoice", "Delete Purchase Order Item", "Set Payment Block",
        "Remove Payment Block", "Change Price", "Change Quantity"}
CORR_CODES = {c for c, a in CODE2ACT.items() if a in CORR}
CLEAR = int(acts.loc[acts.activity == "Clear Invoice", "code"].iloc[0])
TSZERO = pd.Timestamp("2018-01-01", tz="UTC")
NACT = int(acts.code.max()) + 1


def to_log(variants):
    L = EventLog()
    for v in variants:
        t = Trace()
        for a in v:
            e = Event(); e["concept:name"] = CODE2ACT[a]; e["time:timestamp"] = TSZERO
            t.append(e)
        L.append(t)
    return L


def vwr(y, score, value, fpr=FPR):
    y = np.asarray(y).astype(bool); score = np.asarray(score, float)
    value = np.nan_to_num(np.abs(np.asarray(value, float)))
    if y.sum() == 0 or (~y).sum() == 0:
        return np.nan, np.nan
    thr = np.quantile(score[~y], 1 - fpr)
    det = y & (score > thr)
    tot = value[y].sum()
    return (float(value[det].sum() / tot) if tot > 0 else np.nan), float(det.sum() / y.sum())


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() <= 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


# ------------------------------------------------------------------ record-only matrix (e12/e17)
rec = pd.DataFrame({
    "value": np.nan_to_num(cases["value_absmax"].to_numpy()),
    "log_value": np.log1p(np.nan_to_num(np.abs(cases["value_absmax"].to_numpy()))),
    "month": pd.to_datetime(cases["t_first"].to_numpy(), unit="us", utc=True).month,
    "gr_based": (cases["gr_based_inv_verif"].to_numpy() == "True").astype(int),
    "gr_flag": (cases["goods_receipt_flag"].to_numpy() == "True").astype(int)})
for col in ["vendor", "spend_area", "doc_type", "item_type", "company"]:
    v = cases[col].astype("category")
    rec[col + "_code"] = v.cat.codes
    rec[col + "_freq"] = v.map(v.value_counts(normalize=True)).astype(float).to_numpy()
REC_COLS = list(rec.columns)
TYPE_COLS = [c for c in lab.columns if c.startswith("T_")]

# rules baseline, computed on the whole case table exactly as in e12
vend = cases["vendor"].to_numpy(); val_all = np.nan_to_num(cases["value_absmax"].to_numpy())
key = pd.Series(vend).astype(str) + "|" + pd.Series(np.round(val_all, 2)).astype(str)
dup = key.duplicated(keep=False).to_numpy().astype(float)
vcount = pd.Series(vend).map(pd.Series(vend).value_counts()).to_numpy()
rules_score = dup + 0.5 * (vcount > np.quantile(vcount, 0.99)).astype(float) \
              + 0.25 * (val_all > np.quantile(val_all[val_all > 0], 0.99)).astype(float)


def seq_of(j):
    return ac_s[bounds[j]:bounds[j + 1]]


def tseq_scores(train_idx, eval_idx, seed):
    """e12's next-event model: MLP on (prev2, prev1) one-hot, score = mean trace surprisal."""
    rng = np.random.default_rng(seed)
    tr = train_idx if len(train_idx) <= 20000 else rng.choice(train_idx, 20000, replace=False)
    Xs, ys = [], []
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
            Bx.append((s[k - 2] if k >= 2 else NACT, s[k - 1] if k >= 1 else NACT, s[k]))
            Bidx.append(n_)
    Bx = np.array(Bx); Bidx = np.array(Bidx)
    sur = np.empty(len(Bx))
    for a in range(0, len(Bx), 200_000):
        b = min(a + 200_000, len(Bx))
        Xo2 = np.zeros((b - a, 2 * (NACT + 1)), dtype=np.float32)
        Xo2[np.arange(b - a), Bx[a:b, 0]] = 1
        Xo2[np.arange(b - a), (NACT + 1) + Bx[a:b, 1]] = 1
        P = m.predict_proba(Xo2)
        tgt = np.array([classes.get(t, -1) for t in Bx[a:b, 2]])
        p = np.where(tgt >= 0, P[np.arange(len(P)), np.clip(tgt, 0, P.shape[1] - 1)], 1e-9)
        sur[a:b] = -np.log(np.clip(p, 1e-9, 1))
    np.add.at(out, Bidx, sur)
    return out / np.bincount(Bidx, minlength=len(eval_idx)).clip(1)


def main():
    t00 = time.time()
    flow = lab["flow_type"].to_numpy(); posB = lab["pos_B"].to_numpy()
    idx = np.where(flow == FLOW)[0]                   # THE FULL POPULATION, no sampling
    y = posB[idx].astype(bool)
    v = np.nan_to_num(np.abs(lab["value"].to_numpy()[idx]))
    print(f"[{FLOW}] FULL population n={len(idx):,} pos={int(y.sum())} ({y.mean():.4%})", flush=True)

    # ---------------- value concentration on the full population (the Table 3 caveat)
    pv = v[y]
    conc = {"flow_type": FLOW, "n_cases": int(len(idx)), "n_positives": int(y.sum()),
            "gini_positive_value": gini(pv),
            "mean_positive_value": float(pv.mean()), "median_positive_value": float(np.median(pv)),
            "mean_value_all_cases": float(v.mean()),
            "top1pct_of_positives_share_of_positive_value":
                float(np.sort(pv)[::-1][:max(1, int(0.01 * len(pv)))].sum() / pv.sum()),
            "top5pct_of_positives_share_of_positive_value":
                float(np.sort(pv)[::-1][:max(1, int(0.05 * len(pv)))].sum() / pv.sum()),
            "total_positive_value": float(pv.sum())}
    top1_cases = np.argsort(v)[::-1][:max(1, int(0.01 * len(v)))]
    conc["share_of_positive_value_in_top1pct_of_cases_by_value"] = float(v[top1_cases][y[top1_cases]].sum() / pv.sum())
    conc["share_of_positives_in_top1pct_of_cases_by_value"] = float(y[top1_cases].sum() / y.sum())
    json.dump(conc, open(os.path.join(RES, "flow2_value_concentration.json"), "w"), indent=1, default=float)
    print(json.dumps(conc, indent=1), flush=True)

    # ---------------- token replay over every distinct variant of the full population
    t0 = time.time()
    seqs = {j: tuple(seq_of(j).tolist()) for j in idx}
    ref = [j for j in idx if seqs[j] and seqs[j][-1] == CLEAR and not (set(seqs[j]) & CORR_CODES)]
    vc = Counter(seqs[j] for j in ref)
    ref_log = to_log([vv for vv, c in vc.items() for _ in range(min(c, 20))])
    net, im, fm = pm4py.discover_petri_net_inductive(ref_log, noise_threshold=PRIMARY_NOISE)
    var = {}
    for j in idx:
        var.setdefault(seqs[j], []).append(j)
    vlist = list(var.keys())
    print(f"   variants={len(vlist)} ref_cases={len(ref)} ref_variants={len(vc)} "
          f"({time.time()-t0:.0f}s)", flush=True)
    res = treplay.apply(to_log(vlist), net, im, fm, variant=treplay.Variants.TOKEN_REPLAY,
                        parameters={"show_progress_bar": False})
    print(f"   token replay done ({time.time()-t0:.0f}s)", flush=True)
    d = {}
    for vv, r in zip(vlist, res):
        miss = float(r.get("missing_tokens", np.nan)); rem = float(r.get("remaining_tokens", np.nan))
        fit = float(r.get("trace_fitness", np.nan))
        for j in var[vv]:
            d[j] = (miss + rem, miss, rem, fit, len(vv))
    D = np.array([d[j] for j in idx], float)
    pd.DataFrame({"case_idx": idx, "delta_replay": D[:, 0], "missing_tokens": D[:, 1],
                  "remaining_tokens": D[:, 2], "trace_fitness": D[:, 3], "trace_len": D[:, 4]}
                 ).to_csv(os.path.join(RES, "flow2_replay_delta.csv"), index=False)
    json.dump({"flow_type": FLOW, "n_flow_cases": int(len(idx)), "n_frame": int(len(idx)),
               "sampled_negatives": False, "n_variants": int(len(vlist)),
               "coverage_of_frame": 1.0, "n_reference_cases": int(len(ref)),
               "n_reference_variants": int(len(vc)), "noise_threshold": PRIMARY_NOISE,
               "seed": SEED, "seconds": round(time.time() - t0, 1)},
              open(os.path.join(RES, "flow2_sampling_frame.json"), "w"), indent=1)

    Xrec = np.nan_to_num(rec.iloc[idx][REC_COLS].to_numpy(np.float32), nan=0., posinf=0., neginf=0.)
    phi = np.column_stack([D[:, 0], D[:, 1], D[:, 2], D[:, 4],
                           lab.iloc[idx][TYPE_COLS].to_numpy(np.float32)])
    Xp = np.column_stack([Xrec, phi])
    rules = rules_score[idx]

    rows = []
    for seed in SEEDS:
        t0 = time.time()
        tr, te = train_test_split(np.arange(len(idx)), test_size=0.3, random_state=seed, stratify=y)
        ytr, yte, vte = y[tr], y[te], v[te]
        sc = StandardScaler().fit(Xrec[tr])
        Ztr = np.nan_to_num(sc.transform(Xrec[tr]), nan=0., posinf=0., neginf=0.)
        Zte = np.nan_to_num(sc.transform(Xrec[te]), nan=0., posinf=0., neginf=0.)
        S = {}
        S["Rules"] = rules[te]
        S["IF"] = -IsolationForest(n_estimators=100, random_state=seed).fit(Ztr[~ytr]).score_samples(Zte)
        S["AE"] = ((MLPRegressor(hidden_layer_sizes=(16,), max_iter=12, random_state=seed)
                    .fit(Ztr[~ytr], Ztr[~ytr]).predict(Zte) - Zte) ** 2).mean(1)
        S["TSEQ"] = tseq_scores(idx[tr][~ytr], idx[te], seed)
        S["GBDT (records)"] = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.10,
            random_state=seed, class_weight="balanced", verbose=-1
            ).fit(Xrec[tr], ytr).predict_proba(Xrec[te])[:, 1]
        S["CONF (token replay)"] = phi[:, 0][te]
        S["GBDT+phi (token replay)"] = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.10,
            random_state=seed, class_weight="balanced", verbose=-1
            ).fit(Xp[tr], ytr).predict_proba(Xp[te])[:, 1]
        S["VALUE ALONE (null)"] = vte
        S["RANDOM (floor)"] = np.random.default_rng(1000 + seed).random(len(te))
        for det, s in S.items():
            a, b = vwr(yte, s, vte)
            rows.append({"flow_type": FLOW, "detector": det, "seed": seed,
                         "vw_recall_at_1pct_fpr": a, "count_recall_at_1pct_fpr": b,
                         "spearman_score_value": float(stats.spearmanr(s, vte).statistic),
                         "n_eval": len(te), "n_pos": int(yte.sum())})
        pd.DataFrame(rows).to_csv(os.path.join(RES, "flow2_column_raw.csv"), index=False)
        print(f"   seed {seed} done ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "flow2_column_raw.csv"), index=False)
    s = (df.groupby("detector")[["vw_recall_at_1pct_fpr", "count_recall_at_1pct_fpr",
                                 "spearman_score_value"]].agg(["mean", "std"]))
    s.columns = ["_".join(c) for c in s.columns]
    s = s.reset_index()
    s.to_csv(os.path.join(RES, "flow2_column_summary.csv"), index=False)
    print(s.to_string(index=False), flush=True)

    # paired comparisons that Table 3 and its note need
    p = df.pivot_table(index="seed", columns="detector", values="vw_recall_at_1pct_fpr")
    pc = df.pivot_table(index="seed", columns="detector", values="count_recall_at_1pct_fpr")
    tests = []
    pairs = [("CONF (token replay)", "GBDT+phi (token replay)", "H1(b), token-replay substitute"),
             ("GBDT (records)", "IF", "supervised vs isolation forest"),
             ("IF", "VALUE ALONE (null)", "isolation forest vs the value-only null"),
             ("AE", "VALUE ALONE (null)", "autoencoder vs the value-only null"),
             ("GBDT (records)", "VALUE ALONE (null)", "supervised vs the value-only null")]
    for a_, b_, what in pairs:
        for metric, pp in (("vw_recall_at_1pct_fpr", p), ("count_recall_at_1pct_fpr", pc)):
            a, b = pp[a_].to_numpy(), pp[b_].to_numpy()
            dd = a - b
            tests.append({"flow_type": FLOW, "comparison": what, "left": a_, "right": b_,
                          "metric": metric, "left_mean": float(a.mean()), "right_mean": float(b.mean()),
                          "mean_diff": float(dd.mean()), "sd_diff": float(dd.std(ddof=1)),
                          "t_p": float(stats.ttest_rel(a, b).pvalue),
                          "sign_pos": int((dd > 0).sum()), "n": len(dd)})
    pd.DataFrame(tests).to_csv(os.path.join(RES, "flow2_paired_tests.csv"), index=False)
    print(pd.DataFrame(tests).to_string(index=False), flush=True)

    # the comparison this script exists for: population vs the earlier negative sample
    try:
        old = pd.read_csv(os.path.join(RES, "tab_dissociation_leakage_replay_summary.csv"))
        old = old[old.flow_type == FLOW]
        cmp_ = []
        for _, r in old.iterrows():
            new = s[s.detector == r.detector]
            if len(new):
                cmp_.append({"detector": r.detector,
                             "sampled_frame_vw": float(r.vw_recall_at_1pct_fpr_mean),
                             "population_vw": float(new.vw_recall_at_1pct_fpr_mean.iloc[0]),
                             "sampled_frame_count": float(r.count_recall_at_1pct_fpr_mean),
                             "population_count": float(new.count_recall_at_1pct_fpr_mean.iloc[0])})
        pd.DataFrame(cmp_).to_csv(os.path.join(RES, "flow2_population_vs_sample.csv"), index=False)
        print(pd.DataFrame(cmp_).to_string(index=False), flush=True)
    except Exception as e:
        print("comparison to the sampled frame skipped:", e, flush=True)
    print(f"total {time.time()-t00:.0f}s", flush=True)


if __name__ == "__main__":
    main()

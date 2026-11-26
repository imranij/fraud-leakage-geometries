"""e17_conformance_replay.py -- H1(b) made tractable: token-based replay fitness in place of
optimal alignments, with the sampling frame recorded.

WHY.  e11_conformance.py discovers the normative model per flow type and then asks PM4Py for
OPTIMAL ALIGNMENTS.  The log of that run (logs/e11_conf.log) shows what happens: of 600 trace
variants submitted for `3-way match, invoice after GR`, the first ~200 complete in 5 seconds and
the remainder cost 10-15 seconds EACH (A* and Dijkstra alike), projecting to well over an hour for
one flow type and one noise threshold.  Optimal alignment of this log is not affordable in the
budget and is declared infeasible here, with that evidence.

WHAT IS COMPUTED INSTEAD.  Token-based replay fitness (Rozinat & van der Aalst), per trace variant,
against the same Inductive-Miner-infrequent normative model discovered from the same reference set
at the same primary noise threshold 0.2.  The conformance statistic is
    delta_replay = missing tokens + remaining tokens
which plays the role the alignment cost delta plays in Section 4.2.  This is a SUBSTITUTION, not a
sample restriction: token replay is a different conformance estimator from optimal alignment.  It
is logged in DEVIATIONS.md 7.6 and the CONF / GBDT+phi / H1(b) cells inherit it, which is why those
cells are NOT written into the manuscript (the sentence names alignment-derived features).

SAMPLING FRAME.  Flow `3-way match, invoice after GR`: all 15,182 cases, no sampling.
Flow `3-way match, invoice before GR`: all outcome-labelled positives plus a seeded random sample
of N_NEG negatives (the flow has 221,010 cases).  Flow `2-way match`: all 1,044 cases.
Replay is computed once per distinct trace variant and mapped back to every case with that variant,
so coverage of the frame is 100%.
"""
import os, json, time, warnings
from collections import Counter
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.algo.conformance.tokenreplay import algorithm as treplay

from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__)); PAPER = os.path.dirname(HERE)
CACHE = os.path.join(PAPER, "results", "cache"); RES = os.path.join(PAPER, "results")
SEED, SEEDS = 0, [0, 1, 2, 3, 4]
PRIMARY_NOISE = 0.2
N_NEG = 30_000
FPR = 0.01
FLOWS = ["3-way match, invoice after GR", "3-way match, invoice before GR", "2-way match"]

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


# ---------------- record-only matrix (identical to e12) ----------------------
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

flow = lab["flow_type"].to_numpy(); posB = lab["pos_B"].to_numpy()
rng = np.random.default_rng(SEED)
frame_rec, delta_rows, rows = [], [], []
t00 = time.time()

for fl in FLOWS:
    t0 = time.time()
    idx_all = np.where(flow == fl)[0]
    if len(idx_all) > 60_000:
        p = idx_all[posB[idx_all]]
        n = rng.choice(idx_all[~posB[idx_all]], min(N_NEG, int((~posB[idx_all]).sum())), replace=False)
        idx = np.sort(np.concatenate([p, n])); sampled = True
    else:
        idx = idx_all; sampled = False
    seqs = {j: tuple(ac_s[bounds[j]:bounds[j + 1]].tolist()) for j in idx_all}
    ref = [j for j in idx_all if seqs[j] and seqs[j][-1] == CLEAR and not (set(seqs[j]) & CORR_CODES)]
    vc = Counter(seqs[j] for j in ref)
    ref_log = to_log([v for v, c in vc.items() for _ in range(min(c, 20))]) if vc else to_log([])
    net, im, fm = pm4py.discover_petri_net_inductive(ref_log, noise_threshold=PRIMARY_NOISE)

    var = {}
    for j in idx:
        var.setdefault(seqs[j], []).append(j)
    vlist = list(var.keys())
    print(f"[{fl}] frame={len(idx)} (sampled={sampled}) variants={len(vlist)} "
          f"ref={len(ref)} ({time.time()-t0:.0f}s)", flush=True)
    res = treplay.apply(to_log(vlist), net, im, fm,
                        variant=treplay.Variants.TOKEN_REPLAY,
                        parameters={"show_progress_bar": False})
    print(f"   token replay done {time.time()-t0:.0f}s", flush=True)
    d_by_case = {}
    for v, r in zip(vlist, res):
        miss = float(r.get("missing_tokens", np.nan)); rem = float(r.get("remaining_tokens", np.nan))
        fit = float(r.get("trace_fitness", np.nan))
        for j in var[v]:
            d_by_case[j] = (miss + rem, miss, rem, fit, len(v))
    for j, (dl, mi, re_, fi, L) in d_by_case.items():
        delta_rows.append({"flow_type": fl, "noise_threshold": PRIMARY_NOISE, "case_idx": int(j),
                           "delta_replay": dl, "missing_tokens": mi, "remaining_tokens": re_,
                           "trace_fitness": fi, "trace_len": L})
    frame_rec.append({"flow_type": fl, "n_flow_cases": int(len(idx_all)), "n_frame": int(len(idx)),
                      "sampled_negatives": bool(sampled), "n_neg_sampled": int(N_NEG) if sampled else None,
                      "n_variants": int(len(vlist)), "coverage_of_frame": 1.0,
                      "n_reference_cases": int(len(ref)), "n_reference_variants": int(len(vc)),
                      "seed": SEED, "seconds": round(time.time() - t0, 1)})
    pd.DataFrame(delta_rows).to_csv(os.path.join(RES, "conf_replay_delta.csv"), index=False)
    pd.DataFrame(frame_rec).to_csv(os.path.join(RES, "conf_replay_sampling_frame.csv"), index=False)

    # ------------- the leakage columns, with CONF and GBDT+phi from the replay statistic
    y = posB[idx]
    v = np.nan_to_num(np.abs(lab["value"].to_numpy()[idx]))
    if y.sum() < 5:
        print(f"   [{fl}] only {int(y.sum())} positives -- not evaluable", flush=True)
        continue
    Xrec = np.nan_to_num(rec.iloc[idx][REC_COLS].to_numpy(np.float32), nan=0., posinf=0., neginf=0.)
    D = np.array([d_by_case[j] for j in idx], float)          # delta, miss, rem, fit, len
    phi = np.column_stack([D[:, 0], D[:, 1], D[:, 2], D[:, 4],
                           lab.iloc[idx][TYPE_COLS].to_numpy(np.float32)])
    Xp = np.column_stack([Xrec, phi])
    for seed in SEEDS:
        tr, te = train_test_split(np.arange(len(idx)), test_size=0.3, random_state=seed, stratify=y)
        ytr, yte, vte = y[tr], y[te], v[te]
        sc = StandardScaler().fit(Xrec[tr])
        Ztr = np.nan_to_num(sc.transform(Xrec[tr]), nan=0., posinf=0., neginf=0.)
        Zte = np.nan_to_num(sc.transform(Xrec[te]), nan=0., posinf=0., neginf=0.)
        S = {}
        S["CONF (token replay)"] = phi[:, 0][te]
        S["IF"] = -IsolationForest(n_estimators=100, random_state=seed).fit(Ztr[~ytr]).score_samples(Zte)
        S["AE"] = ((MLPRegressor(hidden_layer_sizes=(16,), max_iter=12, random_state=seed)
                    .fit(Ztr[~ytr], Ztr[~ytr]).predict(Zte) - Zte) ** 2).mean(1)
        S["GBDT (records)"] = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.10,
            random_state=seed, class_weight="balanced", verbose=-1
            ).fit(Xrec[tr], ytr).predict_proba(Xrec[te])[:, 1]
        S["GBDT+phi (token replay)"] = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.10,
            random_state=seed, class_weight="balanced", verbose=-1
            ).fit(Xp[tr], ytr).predict_proba(Xp[te])[:, 1]
        for det, s in S.items():
            a, b = vwr(yte, s, vte)
            rows.append({"flow_type": fl, "detector": det, "seed": seed,
                         "vw_recall_at_1pct_fpr": a, "count_recall_at_1pct_fpr": b,
                         "n_eval": len(te), "n_pos": int(yte.sum())})
        pd.DataFrame(rows).to_csv(os.path.join(RES, "tab_dissociation_leakage_replay_raw.csv"), index=False)
        print(f"   seed {seed} done ({time.time()-t0:.0f}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "tab_dissociation_leakage_replay_raw.csv"), index=False)
s = (df.groupby(["flow_type", "detector"])[["vw_recall_at_1pct_fpr", "count_recall_at_1pct_fpr"]]
     .agg(["mean", "std"]))
s.columns = ["_".join(c) for c in s.columns]
s = s.reset_index()
s.to_csv(os.path.join(RES, "tab_dissociation_leakage_replay_summary.csv"), index=False)
print(s.to_string(index=False))

# H1(b): CONF - GBDT+phi, paired over seeds, per flow type
from scipy import stats
tests = []
for fl, g in df.groupby("flow_type"):
    p = g.pivot_table(index="seed", columns="detector", values="vw_recall_at_1pct_fpr")
    pc = g.pivot_table(index="seed", columns="detector", values="count_recall_at_1pct_fpr")
    if "CONF (token replay)" not in p or "GBDT+phi (token replay)" not in p:
        continue
    for metric, pp in (("vw_recall_at_1pct_fpr", p), ("count_recall_at_1pct_fpr", pc)):
        a = pp["CONF (token replay)"].to_numpy(); b = pp["GBDT+phi (token replay)"].to_numpy()
        d = a - b
        tests.append({"flow_type": fl, "metric": metric,
                      "conf_mean": float(a.mean()), "gbdt_phi_mean": float(b.mean()),
                      "mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)),
                      "t_p": float(stats.ttest_rel(a, b).pvalue),
                      "sign_pos": int((d > 0).sum()), "n": len(d),
                      "estimator": "token-based replay fitness (SUBSTITUTION for optimal alignment)"})
pd.DataFrame(tests).to_csv(os.path.join(RES, "tab_h1b_paired_tests_replay.csv"), index=False)
print(pd.DataFrame(tests).to_string(index=False))
print(f"total {time.time()-t00:.0f}s")

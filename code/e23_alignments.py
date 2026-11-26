"""e23_alignments.py -- H1(b) on OPTIMAL ALIGNMENTS, on a declared random sample of trace variants.

WHY THIS FILE EXISTS.  Section 4.2 and Supplementary S2 specify the conformance channel as the A*
alignment of PM4Py against an Inductive-Miner-infrequent normative model, with unit cost for a log
move and a model move and zero for a synchronous move.  e11_conformance.py showed that the full
variant set is not affordable: its log records the first ~200 variants of one flow type completing
in five seconds and the remainder costing 10-15 SECONDS EACH.  e17_conformance_replay.py therefore
fell back on token-based replay, which is a DIFFERENT ESTIMATOR, and that is why the CONF,
GBDT+phi and H1(b) cells stayed open.

WHAT THIS FILE DOES INSTEAD.  It keeps the estimator -- optimal alignment, same model, same noise
threshold, same cost function -- and restricts the POPULATION to a declared probability sample of
trace variants.  A sample analysed with the specified estimator is a sample, not a substitution.

SAMPLE DESIGN (recorded in results/conf_align_frame.csv).  Within each flow type:
  * Sampling frame: every distinct trace variant of the flow type, with its case frequency.
  * Stratum C (certainty / take-all): the N_CERTAINTY most frequent variants, inclusion
    probability 1.  These are the short, common traces; they are cheap and they carry most cases.
  * Stratum R (random): a simple random sample WITHOUT replacement of N_RANDOM variants from the
    remaining variants, drawn with numpy.random.default_rng(SEED), SEED = 0.  Inclusion
    probability N_RANDOM / (n_variants - N_CERTAINTY), recorded per flow type.
  * Stratum R is attempted in the seeded random order under a wall-clock budget.  Variants the
    budget does not reach are recorded as NOT ATTEMPTED (they are not silently dropped, and the
    realised inclusion probability is recorded beside the designed one).
  * TIMEOUT: every alignment is given at most TIMEOUT_S seconds (PM4Py's max_align_time_trace).
    A variant that exceeds it is CENSORED: it is counted, its identity and length are written to
    results/conf_align_censored.csv, and it contributes no alignment cost.  It is never dropped
    without record.

COVERAGE.  Reported as (a) number of variants aligned, (b) the share of the flow type's cases
those variants represent, (c) the share of the flow type's outcome-labelled positives they
represent.  The detection numbers below are computed on the EVALUATED SUBPOPULATION -- the cases
whose variant was successfully aligned -- and that subpopulation's positive rate is reported
against the flow type's, because a variant sample is not a case sample.

MODEL.  Identical to e11_conformance.py and e17_conformance_replay.py: the reference set is the
cases of the flow type whose trace ends in "Clear Invoice" with no correction activity; the
Inductive Miner (infrequent) is run on that reference log at the PRIMARY noise threshold 0.2.

DETECTION.  Same detectors, seeds, split and metric as e17_conformance_replay.py, so that the two
estimators are directly comparable:
    CONF (alignments)      = the alignment cost delta
    GBDT+phi (alignments)  = LightGBM on the record features PLUS phi(sigma) built from the
                             alignment: cost, number of log moves, number of model moves, trace
                             length, and the six typed-deviation indicators
    GBDT (records), IF, AE = record-only, exactly as in e17
Five seeds, stratified 70/30 split per seed, value-weighted recall at a 1% false-positive rate
with count-based recall beside it.  H1(b) is the paired difference CONF - GBDT+phi.
"""
from __future__ import annotations
import os, sys, json, time, warnings
from collections import Counter
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.algo.conformance.alignments.petri_net.variants import state_equation_a_star as sea

from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); PAPER = os.path.dirname(HERE)
CACHE = os.path.join(PAPER, "results", "cache"); RES = os.path.join(PAPER, "results")

SEED = 0
SEEDS = [0, 1, 2, 3, 4]
PRIMARY_NOISE = 0.2
FPR = 0.01
TIMEOUT_S = float(os.environ.get("ALIGN_TIMEOUT", "5.0"))
N_CERTAINTY = int(os.environ.get("ALIGN_CERTAINTY", "250"))
N_RANDOM = int(os.environ.get("ALIGN_RANDOM", "300"))
BUDGET_C = float(os.environ.get("ALIGN_BUDGET_C", "360"))     # s, certainty stratum, per flow
BUDGET_R = float(os.environ.get("ALIGN_BUDGET_R", "600"))     # s, random stratum, per flow
FLOWS = ["3-way match, invoice after GR", "3-way match, invoice before GR", "2-way match"]

# ------------------------------------------------------------------ load cache
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
print("cache loaded, sequences indexed", flush=True)


def to_trace(v):
    t = Trace()
    for a in v:
        e = Event(); e["concept:name"] = CODE2ACT[a]; e["time:timestamp"] = TSZERO
        t.append(e)
    return t


def to_log(variants):
    L = EventLog()
    for v in variants:
        L.append(to_trace(v))
    return L


# record-only matrix -- byte-identical construction to e17_conformance_replay.py
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

flow = lab["flow_type"].to_numpy()
posB = lab["pos_B"].to_numpy()


def vwr(y, score, value, fpr=FPR):
    y = np.asarray(y).astype(bool); score = np.asarray(score, float)
    value = np.nan_to_num(np.abs(np.asarray(value, float)))
    if y.sum() == 0 or (~y).sum() == 0:
        return np.nan, np.nan
    thr = np.quantile(score[~y], 1 - fpr)
    det = y & (score > thr)
    tot = value[y].sum()
    return (float(value[det].sum() / tot) if tot > 0 else np.nan), float(det.sum() / y.sum())


frame_rows, delta_rows, censor_rows, det_rows = [], [], [], []


def write_all():
    pd.DataFrame(frame_rows).to_csv(os.path.join(RES, "conf_align_frame.csv"), index=False)
    if delta_rows:
        pd.DataFrame(delta_rows).to_csv(os.path.join(RES, "conf_align_delta.csv"), index=False)
    pd.DataFrame(censor_rows).to_csv(os.path.join(RES, "conf_align_censored.csv"), index=False)
    if det_rows:
        pd.DataFrame(det_rows).to_csv(os.path.join(RES, "conf_align_detection_raw.csv"), index=False)


t00 = time.time()
for fl in FLOWS:
    t0 = time.time()
    idx_all = np.where(flow == fl)[0]
    seqs = {j: tuple(ac_s[bounds[j]:bounds[j + 1]].tolist()) for j in idx_all}

    # ---- normative model: same reference set, same miner, same primary noise threshold as e11/e17
    ref = [j for j in idx_all if seqs[j] and seqs[j][-1] == CLEAR and not (set(seqs[j]) & CORR_CODES)]
    vc_ref = Counter(seqs[j] for j in ref)
    ref_log = to_log([v for v, c in vc_ref.items() for _ in range(min(c, 20))]) if vc_ref else to_log([])
    net, im, fm = pm4py.discover_petri_net_inductive(ref_log, noise_threshold=PRIMARY_NOISE)
    print(f"[{fl}] cases={len(idx_all)} pos={int(posB[idx_all].sum())} "
          f"ref={len(ref)} ref_variants={len(vc_ref)} "
          f"net={len(net.places)}p/{len(net.transitions)}t ({time.time()-t0:.0f}s)", flush=True)

    # ---- sampling frame ----------------------------------------------------
    var2cases = {}
    for j in idx_all:
        var2cases.setdefault(seqs[j], []).append(j)
    allvars = sorted(var2cases.keys(), key=lambda v: (-len(var2cases[v]), len(v)))
    n_var = len(allvars)
    nC = min(N_CERTAINTY, n_var)
    certainty = allvars[:nC]
    tail = allvars[nC:]
    rng = np.random.default_rng(SEED)
    nR = min(N_RANDOM, len(tail))
    perm = rng.permutation(len(tail))
    random_sample = [tail[i] for i in perm[:nR]]
    p_incl_R = (nR / len(tail)) if tail else float("nan")
    print(f"   frame: {n_var} variants | certainty {nC} | tail {len(tail)} -> random {nR} "
          f"(p_incl={p_incl_R:.4f}, seed={SEED})", flush=True)

    # ---- alignments ---------------------------------------------------------
    params = {sea.Parameters.PARAM_MAX_ALIGN_TIME_TRACE: TIMEOUT_S,
              "show_progress_bar": False}
    res = {}
    n_attempt = {"C": 0, "R": 0}; n_censor = {"C": 0, "R": 0}; n_ok = {"C": 0, "R": 0}
    for stratum, vlist, budget in (("C", certainty, BUDGET_C), ("R", random_sample, BUDGET_R)):
        ts0 = time.time()
        for k, v in enumerate(vlist):
            if time.time() - ts0 > budget:
                print(f"   stratum {stratum}: budget {budget:.0f}s reached after {k}/{len(vlist)} "
                      f"variants attempted", flush=True)
                break
            n_attempt[stratum] += 1
            ta = time.time()
            try:
                r = sea.apply(to_trace(v), net, im, fm, parameters=dict(params))
            except Exception as e:
                r = None
                print(f"      ALIGN ERROR len={len(v)}: {repr(e)[:120]}", flush=True)
            el = time.time() - ta
            if r is None or r.get("cost") is None:
                n_censor[stratum] += 1
                censor_rows.append({"flow_type": fl, "stratum": stratum, "trace_len": len(v),
                                    "n_cases_with_variant": len(var2cases[v]),
                                    "seconds_spent": round(el, 2), "timeout_s": TIMEOUT_S,
                                    "reason": "alignment exceeded the per-variant timeout"})
                continue
            n_ok[stratum] += 1
            al = r.get("alignment", []) or []
            nlog = sum(1 for a in al if a[1] == ">>" or a[1] is None)
            nmod = sum(1 for a in al if a[0] == ">>" or a[0] is None)
            res[v] = (float(r["cost"]), nlog, nmod, len(v),
                      float(r.get("fitness", np.nan)), stratum, el)
        print(f"   stratum {stratum}: attempted {n_attempt[stratum]} aligned {n_ok[stratum]} "
              f"censored {n_censor[stratum]} in {time.time()-ts0:.0f}s", flush=True)
        write_all()

    # ---- coverage -----------------------------------------------------------
    ev_idx = np.sort(np.concatenate([np.array(var2cases[v], int) for v in res])) if res \
        else np.array([], int)
    n_cases_cov = len(ev_idx)
    n_pos_cov = int(posB[ev_idx].sum()) if len(ev_idx) else 0
    for v, (cost, nlog, nmod, L, fit, stratum, el) in res.items():
        for j in var2cases[v]:
            delta_rows.append({"flow_type": fl, "case_idx": int(j), "stratum": stratum,
                               "noise_threshold": PRIMARY_NOISE, "delta_alignment": cost,
                               "n_log_moves": nlog, "n_model_moves": nmod, "trace_len": L,
                               "alignment_fitness": fit})
    frame_rows.append({
        "flow_type": fl, "estimator": "optimal alignment (PM4Py A*, state-equation heuristic)",
        "noise_threshold": PRIMARY_NOISE, "sampling_seed": SEED,
        "n_flow_cases": int(len(idx_all)), "n_flow_positives": int(posB[idx_all].sum()),
        "n_variants_in_frame": int(n_var),
        "n_certainty_designed": int(nC), "n_certainty_attempted": int(n_attempt["C"]),
        "n_certainty_aligned": int(n_ok["C"]), "n_certainty_censored": int(n_censor["C"]),
        "n_tail_variants": int(len(tail)),
        "n_random_designed": int(nR), "n_random_attempted": int(n_attempt["R"]),
        "n_random_aligned": int(n_ok["R"]), "n_random_censored": int(n_censor["R"]),
        "p_inclusion_random_designed": p_incl_R,
        "p_inclusion_random_realised": (n_attempt["R"] / len(tail)) if tail else float("nan"),
        "timeout_s": TIMEOUT_S,
        "n_variants_aligned_total": int(len(res)),
        "share_of_variants_aligned": len(res) / n_var,
        "n_cases_covered": int(n_cases_cov),
        "share_of_cases_covered": n_cases_cov / len(idx_all),
        "n_positives_covered": int(n_pos_cov),
        "share_of_positives_covered": (n_pos_cov / max(int(posB[idx_all].sum()), 1)),
        "positive_rate_in_covered_subpopulation": (n_pos_cov / n_cases_cov) if n_cases_cov else float("nan"),
        "positive_rate_in_flow": float(posB[idx_all].mean()),
        "n_reference_cases": int(len(ref)), "n_reference_variants": int(len(vc_ref)),
        "seconds": round(time.time() - t0, 1)})
    write_all()
    print(f"   COVERAGE {fl}: {len(res)} variants aligned, {n_cases_cov}/{len(idx_all)} cases "
          f"({n_cases_cov/len(idx_all):.1%}), {n_pos_cov}/{int(posB[idx_all].sum())} positives",
          flush=True)

    # ---- detection on the evaluated subpopulation ---------------------------
    if n_pos_cov < 5:
        print(f"   [{fl}] only {n_pos_cov} positives in the evaluated subpopulation -- "
              f"detection numbers not evaluable, reported as such", flush=True)
        continue
    y = posB[ev_idx]
    val = np.nan_to_num(np.abs(lab["value"].to_numpy()[ev_idx]))
    Xrec = np.nan_to_num(rec.iloc[ev_idx][REC_COLS].to_numpy(np.float32),
                         nan=0., posinf=0., neginf=0.)
    D = np.array([res[seqs[j]][:4] for j in ev_idx], float)       # cost, nlog, nmod, len
    phi = np.column_stack([D, lab.iloc[ev_idx][TYPE_COLS].to_numpy(np.float32)])
    Xp = np.column_stack([Xrec, phi])
    for seed in SEEDS:
        tr, te = train_test_split(np.arange(len(ev_idx)), test_size=0.3,
                                  random_state=seed, stratify=y)
        ytr, yte, vte = y[tr], y[te], val[te]
        sc = StandardScaler().fit(Xrec[tr])
        Ztr = np.nan_to_num(sc.transform(Xrec[tr]), nan=0., posinf=0., neginf=0.)
        Zte = np.nan_to_num(sc.transform(Xrec[te]), nan=0., posinf=0., neginf=0.)
        S = {}
        S["CONF (alignments)"] = phi[:, 0][te]
        S["IF"] = -IsolationForest(n_estimators=100, random_state=seed
                                   ).fit(Ztr[~ytr]).score_samples(Zte)
        S["AE"] = ((MLPRegressor(hidden_layer_sizes=(16,), max_iter=12, random_state=seed)
                    .fit(Ztr[~ytr], Ztr[~ytr]).predict(Zte) - Zte) ** 2).mean(1)
        S["GBDT (records)"] = lgb.LGBMClassifier(
            n_estimators=120, learning_rate=0.10, random_state=seed,
            class_weight="balanced", verbose=-1).fit(Xrec[tr], ytr).predict_proba(Xrec[te])[:, 1]
        S["GBDT+phi (alignments)"] = lgb.LGBMClassifier(
            n_estimators=120, learning_rate=0.10, random_state=seed,
            class_weight="balanced", verbose=-1).fit(Xp[tr], ytr).predict_proba(Xp[te])[:, 1]
        for det, s in S.items():
            a, b = vwr(yte, s, vte)
            det_rows.append({"flow_type": fl, "detector": det, "seed": seed,
                             "vw_recall_at_1pct_fpr": a, "count_recall_at_1pct_fpr": b,
                             "n_eval": len(te), "n_pos": int(yte.sum())})
        write_all()
        print(f"   seed {seed} done ({time.time()-t0:.0f}s)", flush=True)
    print(f"[{fl}] complete {time.time()-t0:.0f}s (total {time.time()-t00:.0f}s)", flush=True)

# ---------------------------------------------------------------- summaries
write_all()
if det_rows:
    df = pd.DataFrame(det_rows)
    s = (df.groupby(["flow_type", "detector"])[
        ["vw_recall_at_1pct_fpr", "count_recall_at_1pct_fpr"]].agg(["mean", "std"]))
    s.columns = ["_".join(c) for c in s.columns]
    s.reset_index().to_csv(os.path.join(RES, "conf_align_detection_summary.csv"), index=False)
    print(s.reset_index().to_string(index=False), flush=True)

    tests = []
    for fl, g in df.groupby("flow_type"):
        for metric in ("vw_recall_at_1pct_fpr", "count_recall_at_1pct_fpr"):
            p = g.pivot_table(index="seed", columns="detector", values=metric)
            if "CONF (alignments)" not in p or "GBDT+phi (alignments)" not in p:
                continue
            a = p["CONF (alignments)"].to_numpy(); b = p["GBDT+phi (alignments)"].to_numpy()
            d = a - b
            row = {"flow_type": fl, "metric": metric, "comparison": "CONF - GBDT+phi",
                   "conf_mean": float(a.mean()), "gbdt_phi_mean": float(b.mean()),
                   "mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)),
                   "t_p": float(stats.ttest_rel(a, b).pvalue),
                   "wilcoxon_p": float(stats.wilcoxon(a, b).pvalue) if np.any(d != 0) else float("nan"),
                   "sign_pos": int((d > 0).sum()), "n": len(d),
                   "estimator": "optimal alignment (A*), declared random sample of trace variants"}
            tests.append(row)
            # record-only comparison, part (a)
            if "GBDT (records)" in p:
                b2 = p["GBDT (records)"].to_numpy(); d2 = a - b2
                tests.append({"flow_type": fl, "metric": metric,
                              "comparison": "CONF - GBDT (records)",
                              "conf_mean": float(a.mean()), "gbdt_phi_mean": float(b2.mean()),
                              "mean_diff": float(d2.mean()), "sd_diff": float(d2.std(ddof=1)),
                              "t_p": float(stats.ttest_rel(a, b2).pvalue),
                              "wilcoxon_p": float(stats.wilcoxon(a, b2).pvalue) if np.any(d2 != 0) else float("nan"),
                              "sign_pos": int((d2 > 0).sum()), "n": len(d2),
                              "estimator": "optimal alignment (A*), declared random sample of trace variants"})
    pd.DataFrame(tests).to_csv(os.path.join(RES, "conf_align_h1b_paired_tests.csv"), index=False)
    print(pd.DataFrame(tests).to_string(index=False), flush=True)
print(f"TOTAL {time.time()-t00:.0f}s", flush=True)

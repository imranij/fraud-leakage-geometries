"""e27_bpi2017.py -- the BPI Challenge 2017 process-generality check.

What the manuscript asks of this log, and nothing else.  Section 5 (and Table 2): "BPI Challenge
2017, a loan-application log of 1,202,267 events over 31,509 applications, serves ONLY as a
process-generality check ... It asks whether the channel can discover and align a second,
service-industry process at the reported cost, and it appears in no dissociation or
value-at-budget result."  So the endpoints computed here are exactly:

  (1) the published counts, verified against the file;
  (2) DISCOVERY of a normative model -- Inductive Miner infrequent at the same primary noise
      threshold 0.2 as the BPI 2019 channel -- with its fitness (token-based replay) and precision
      (ETC), and the wall-clock cost of each.  One pass over this log costs about 256 s, so the
      four-point noise grid does not fit the remaining budget: the grid is ordered primary-first
      and stops when the budget is spent.  A COMPLETED noise = 0.0 pass from an earlier run of
      this same script is kept in logs/e27_bpi2017_noise0_run.log: fitness 0.9916, precision
      0.1280, 57 places, 78 transitions, 15,930 evaluation variants;
  (3) ALIGNMENT at the reported cost: optimal alignments (PM4Py, Dijkstra without heuristics, the
      variant e11 used on BPI 2019) on distinct trace variants under a hard time budget, reporting
      how many variants completed, the per-variant cost distribution, and the projected cost of
      the whole log -- the same evidence that was recorded for BPI 2019 in logs/e11_conf.log.

No detector is fitted, no leakage label is derived, no value-at-budget number is produced: the
manuscript excludes this log from both.

Reference set.  The BPI 2019 channel discovers its normative model from the cases that reach the
normal terminal activity with no correction activity in them.  The same rule is applied here, with
the loan process's own vocabulary: an application is in the reference set if it reaches A_Pending
(the offer is activated) and contains neither A_Cancelled nor A_Denied.  A first version of the
rule also excluded O_Cancelled, O_Refused and O_Returned and returned ZERO reference cases -- in
BPI 2017 an application carries several offers and offer-level returns and cancellations are part
of the normal process.  That correction was made before any fitness or precision number existed
(the empty run is in logs/e27_bpi2017.log) and the rule has not moved since.

Activity labels are `concept:name`, as in e9/e11 for BPI 2019; the `lifecycle:transition` column is
kept out of the label, and the variant count under a complete-events-only reading is reported
beside it as a sensitivity.

Memory: the 200 MB flat CSV is read in chunks with explicit dtypes.
Outputs carry the `bpi2017_` prefix.  No existing results file is touched.
"""
from __future__ import annotations
import os, sys, json, time, warnings, hashlib
from collections import Counter
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

import pm4py
from pm4py.util import constants as pm4c
try:
    pm4c.SHOW_PROGRESS_BAR = False
except Exception:
    pass
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.algo.evaluation.replay_fitness import algorithm as rf
from pm4py.algo.evaluation.precision import algorithm as prec
from pm4py.algo.conformance.alignments.petri_net import algorithm as ali

HERE = os.path.dirname(os.path.abspath(__file__)); PAPER = os.path.dirname(HERE)
RES = os.path.join(PAPER, "results")
CSV = os.path.join(PAPER, "data", "raw", "bpi2017", "bpi2017_events.csv")
XES = os.path.join(PAPER, "data", "raw", "bpi2017", "BPI_Challenge_2017.xes.gz")

PUBLISHED_EVENTS, PUBLISHED_CASES = 1_202_267, 31_509
GRID = [0.2, 0.4, 0.1, 0.0]   # PRIMARY FIRST: the budget below may stop the grid early
PRIMARY_NOISE = 0.2
GRID_BUDGET = 60.0           # seconds; one pass costs ~256 s on this log (discovery 31 s,
                             # token-replay fitness 118 s, ETC precision 107 s), so this
                             # admits the PRIMARY threshold only. Logged in the addendum.
ALIGN_BUDGET = 150.0         # seconds for the alignment block
ALIGN_MAX_VARIANTS = 400     # distinct variants submitted, most frequent first
ALIGN_RANDOM_VARIANTS = 300  # distinct variants sampled uniformly, to measure the tail too
PREC_VARIANTS = 2000
TSZERO = pd.Timestamp("2016-01-01", tz="UTC")
REF_END = "A_Pending"
REF_EXCLUDE = {"A_Cancelled", "A_Denied"}
# First rule tried (recorded, because changing a rule after seeing an outcome has to be visible):
# REF_EXCLUDE also held O_Cancelled, O_Refused and O_Returned.  That rule returns ZERO reference
# cases, because in BPI 2017 an application carries several OFFERS and it is routine for a
# successful application to have offers that are returned or cancelled: those are offer-level
# events of the normal process, not application-level failures.  The rule was corrected to the
# application-level outcome only -- contains A_Pending, contains neither A_Cancelled nor A_Denied
# -- before any fitness or precision number was computed.  The zero-reference run is in
# logs/e27_bpi2017.log.

USECOLS = ["case:concept:name", "concept:name", "time:timestamp", "lifecycle:transition",
           "case:ApplicationType", "case:LoanGoal", "case:RequestedAmount", "EventOrigin"]
DTYPES = {"case:concept:name": "str", "concept:name": "category",
          "lifecycle:transition": "category", "case:ApplicationType": "category",
          "case:LoanGoal": "category", "case:RequestedAmount": "float32",
          "EventOrigin": "category"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def to_log(variants, code2act):
    L = EventLog()
    for v in variants:
        t = Trace()
        for a in v:
            e = Event(); e["concept:name"] = code2act[a]; e["time:timestamp"] = TSZERO
            t.append(e)
        L.append(t)
    return L


def main():
    t00 = time.time()
    case_ids, act_ids = {}, {}
    ev_case, ev_act, ev_ts, ev_life = [], [], [], []
    case_attr = []
    n_rows = 0
    for ch in pd.read_csv(CSV, usecols=USECOLS, dtype=DTYPES, chunksize=300_000):
        n_rows += len(ch)
        cids = ch["case:concept:name"].to_numpy()
        acts_ = ch["concept:name"].astype(str).to_numpy()
        life = ch["lifecycle:transition"].astype(str).to_numpy()
        atyp = ch["case:ApplicationType"].astype(str).to_numpy()
        goal = ch["case:LoanGoal"].astype(str).to_numpy()
        amt = ch["case:RequestedAmount"].to_numpy()
        ts = pd.to_datetime(ch["time:timestamp"], format="ISO8601", utc=True).astype("int64").to_numpy()
        ci = np.empty(len(ch), np.int32); ai = np.empty(len(ch), np.int16)
        li = np.empty(len(ch), np.int8)
        lmap = {"complete": 0, "start": 1, "schedule": 2, "withdraw": 3, "ate_abort": 4,
                "suspend": 5, "resume": 6}
        for k in range(len(ch)):
            c = cids[k]
            j = case_ids.get(c)
            if j is None:
                j = len(case_attr); case_ids[c] = j
                case_attr.append((c, atyp[k], goal[k], float(amt[k])))
            ci[k] = j
            a = acts_[k]
            b = act_ids.get(a)
            if b is None:
                b = len(act_ids); act_ids[a] = b
            ai[k] = b
            li[k] = lmap.get(life[k], 9)
        ev_case.append(ci); ev_act.append(ai); ev_ts.append(ts); ev_life.append(li)
        print(f"  {n_rows:,} rows, {len(case_attr):,} cases, {len(act_ids)} activities "
              f"({time.time()-t00:.0f}s)", flush=True)
    ev_case = np.concatenate(ev_case); ev_act = np.concatenate(ev_act)
    ev_ts = np.concatenate(ev_ts); ev_life = np.concatenate(ev_life)
    code2act = {v: k for k, v in act_ids.items()}
    n_events, n_cases = int(len(ev_case)), int(len(case_attr))
    print(f"TOTAL events={n_events:,} cases={n_cases:,} activities={len(act_ids)}", flush=True)

    verify = {"file": os.path.basename(CSV), "sha256_csv": sha256(CSV),
              "sha256_xes_gz": sha256(XES),
              "n_events_file": n_events, "n_cases_file": n_cases,
              "n_events_published": PUBLISHED_EVENTS, "n_cases_published": PUBLISHED_CASES,
              "events_match": n_events == PUBLISHED_EVENTS,
              "cases_match": n_cases == PUBLISHED_CASES,
              "n_activities": len(act_ids),
              "activities": sorted(act_ids.keys())}
    json.dump(verify, open(os.path.join(RES, "bpi2017_verification.json"), "w"), indent=1)
    print(json.dumps({k: verify[k] for k in ["n_events_file", "n_cases_file", "events_match",
                                             "cases_match", "n_activities"]}, indent=1), flush=True)

    # ------------------------------------------------ traces, variants, reference set
    o = np.lexsort((ev_ts, ev_case))
    ci_s, ac_s, li_s = ev_case[o], ev_act[o], ev_life[o]
    bounds = np.searchsorted(ci_s, np.arange(n_cases + 1))
    seqs = [tuple(ac_s[bounds[j]:bounds[j + 1]].tolist()) for j in range(n_cases)]
    seqs_complete = [tuple(ac_s[bounds[j]:bounds[j + 1]][li_s[bounds[j]:bounds[j + 1]] == 0].tolist())
                     for j in range(n_cases)]
    all_var = Counter(seqs)
    all_var_c = Counter(seqs_complete)
    excl_codes = {act_ids[a] for a in REF_EXCLUDE if a in act_ids}
    end_code = act_ids.get(REF_END)
    ref = [j for j in range(n_cases)
           if end_code is not None and end_code in seqs[j] and not (set(seqs[j]) & excl_codes)]
    assert len(ref) > 0, "reference set is empty -- the reference rule does not match this log"
    vc = Counter(seqs[j] for j in ref)
    print(f"variants(all events)={len(all_var):,} variants(complete only)={len(all_var_c):,} "
          f"reference cases={len(ref):,} reference variants={len(vc):,} "
          f"({time.time()-t00:.0f}s)", flush=True)

    ref_log = to_log([v for v, c in vc.items() for _ in range(min(c, 20))], code2act)
    eval_variants = [v for v, _ in all_var.most_common()]
    eval_log = to_log(eval_variants, code2act)

    # ------------------------------------------------ discovery + fitness/precision grid
    grid_rows = []
    t1 = time.time()
    SKIP_GRID = os.environ.get("E27_SKIP_GRID") == "1"
    for noise in (() if SKIP_GRID else GRID):
        if time.time() - t1 > GRID_BUDGET:
            grid_rows.append({"noise_threshold": noise, "fitness": np.nan, "precision": np.nan,
                              "error": "time budget"})
            print(f"   GRID budget hit at noise={noise}", flush=True)
            continue
        row = {"noise_threshold": noise}
        try:
            td = time.time()
            net, im, fm = pm4py.discover_petri_net_inductive(ref_log, noise_threshold=noise)
            row["discovery_seconds"] = round(time.time() - td, 2)
            row["n_places"] = len(net.places); row["n_transitions"] = len(net.transitions)
            tf = time.time()
            f = rf.apply(eval_log, net, im, fm, variant=rf.Variants.TOKEN_BASED,
                         parameters={"show_progress_bar": False})
            row["fitness"] = float(f.get("log_fitness", np.nan))
            row["fitness_seconds"] = round(time.time() - tf, 2)
            tp = time.time()
            row["precision"] = float(prec.apply(to_log(eval_variants[:PREC_VARIANTS], code2act),
                                                net, im, fm,
                                                variant=prec.Variants.ETCONFORMANCE_TOKEN,
                                                parameters={"show_progress_bar": False}))
            row["precision_seconds"] = round(time.time() - tp, 2)
            row["n_eval_variants"] = len(eval_log)
            row["n_precision_variants"] = min(PREC_VARIANTS, len(eval_variants))
        except Exception as e:
            row.update({"fitness": np.nan, "precision": np.nan, "error": repr(e)[:200]})
        grid_rows.append(row)
        print("   ", row, flush=True)
        pd.DataFrame(grid_rows).to_csv(os.path.join(RES, "bpi2017_noise_grid.csv"), index=False)
    if not SKIP_GRID:
        pd.DataFrame(grid_rows).to_csv(os.path.join(RES, "bpi2017_noise_grid.csv"), index=False)
    else:
        grid_rows = pd.read_csv(os.path.join(RES, "bpi2017_noise_grid.csv")).to_dict("records")

    # ------------------------------------------------ alignment at the reported cost
    align = {"variant": "PM4Py optimal alignment, Dijkstra without heuristics (the variant e11 "
                        "used on BPI 2019)",
             "noise_threshold": PRIMARY_NOISE, "budget_seconds": ALIGN_BUDGET,
             "n_variants_submitted": 0, "n_variants_completed": 0}
    try:
        net, im, fm = pm4py.discover_petri_net_inductive(ref_log, noise_threshold=PRIMARY_NOISE)
        all_var_keys = list(all_var.keys())
        sub = [v for v, _ in all_var.most_common(ALIGN_MAX_VARIANTS)]
        align["n_variants_submitted"] = len(sub)
        align["cases_covered_by_submitted_variants"] = int(sum(all_var[v] for v in sub))
        per = []
        ta = time.time()
        done = 0
        for v in sub:
            if time.time() - ta > ALIGN_BUDGET:
                break
            t2 = time.time()
            r = ali.apply(to_log([v], code2act), net, im, fm,
                          variant=ali.Variants.VERSION_DIJKSTRA_NO_HEURISTICS,
                          parameters={"show_progress_bar": False})
            dt = time.time() - t2
            rr = r[0] if isinstance(r, list) and r else None
            per.append({"trace_len": len(v), "n_cases_with_variant": int(all_var[v]),
                        "seconds": dt,
                        "cost": float(rr.get("cost", np.nan)) if rr else np.nan,
                        "fitness": float(rr.get("fitness", np.nan)) if rr else np.nan})
            done += 1
        align["n_variants_completed"] = done
        align["seconds_used"] = round(time.time() - ta, 1)
        pp = pd.DataFrame(per)
        pp.to_csv(os.path.join(RES, "bpi2017_alignment_cost.csv"), index=False)
        if len(pp):
            align.update({
                "median_seconds_per_variant": float(pp.seconds.median()),
                "mean_seconds_per_variant": float(pp.seconds.mean()),
                "max_seconds_per_variant": float(pp.seconds.max()),
                "p90_seconds_per_variant": float(pp.seconds.quantile(0.9)),
                "cases_covered_by_completed_variants": int(pp.n_cases_with_variant.sum()),
                "share_of_cases_covered_by_completed_variants":
                    float(pp.n_cases_with_variant.sum() / n_cases),
                "mean_alignment_cost": float(np.nanmean(pp.cost)),
                "mean_alignment_fitness": float(np.nanmean(pp.fitness)),
                "projected_hours_for_all_variants":
                    float(pp.seconds.mean() * len(all_var) / 3600.0)})
    except Exception as e:
        align["error"] = repr(e)[:300]
    # The head of the variant distribution is the easy part: on BPI 2019 the first ~200 variants
    # aligned in seconds and the tail cost 10-15 s each.  A UNIFORM sample of variants is therefore
    # measured too, so the projection to the whole log is not an extrapolation from the head alone.
    try:
        rs = np.random.default_rng(0)
        pick = [all_var_keys[i] for i in rs.choice(len(all_var_keys),
                                                   min(ALIGN_RANDOM_VARIANTS, len(all_var_keys)),
                                                   replace=False)]
        per2, ta2, done2 = [], time.time(), 0
        for v in pick:
            if time.time() - ta2 > ALIGN_BUDGET:
                break
            t2 = time.time()
            r = ali.apply(to_log([v], code2act), net, im, fm,
                          variant=ali.Variants.VERSION_DIJKSTRA_NO_HEURISTICS,
                          parameters={"show_progress_bar": False})
            rr = r[0] if isinstance(r, list) and r else None
            per2.append({"trace_len": len(v), "n_cases_with_variant": int(all_var[v]),
                         "seconds": time.time() - t2,
                         "cost": float(rr.get("cost", np.nan)) if rr else np.nan,
                         "fitness": float(rr.get("fitness", np.nan)) if rr else np.nan})
            done2 += 1
        pp2 = pd.DataFrame(per2)
        pp2.to_csv(os.path.join(RES, "bpi2017_alignment_cost_random.csv"), index=False)
        align["random_sample"] = {
            "n_variants_submitted": len(pick), "n_variants_completed": done2,
            "seconds_used": round(time.time() - ta2, 1),
            "median_seconds_per_variant": float(pp2.seconds.median()) if len(pp2) else None,
            "mean_seconds_per_variant": float(pp2.seconds.mean()) if len(pp2) else None,
            "max_seconds_per_variant": float(pp2.seconds.max()) if len(pp2) else None,
            "p90_seconds_per_variant": float(pp2.seconds.quantile(0.9)) if len(pp2) else None,
            "mean_trace_len": float(pp2.trace_len.mean()) if len(pp2) else None,
            "mean_alignment_fitness": float(np.nanmean(pp2.fitness)) if len(pp2) else None,
            "projected_hours_for_all_variants":
                float(pp2.seconds.mean() * len(all_var) / 3600.0) if len(pp2) else None}
        print("   random-variant alignment:", json.dumps(align["random_sample"], default=float),
              flush=True)
    except Exception as e:
        align["random_sample"] = {"error": repr(e)[:300]}
    json.dump(align, open(os.path.join(RES, "bpi2017_alignment_cost.json"), "w"), indent=1, default=float)
    print(json.dumps(align, indent=1, default=float), flush=True)

    prim = [r for r in grid_rows if r["noise_threshold"] == PRIMARY_NOISE]
    summary = {
        "log": "BPI Challenge 2017 (loan applications)",
        "role_in_the_manuscript": "process-generality check only; no dissociation, no value at budget",
        "verification": {k: verify[k] for k in ["n_events_file", "n_cases_file",
                                                "n_events_published", "n_cases_published",
                                                "events_match", "cases_match", "n_activities"]},
        "n_variants_all_events": len(all_var), "n_variants_complete_only": len(all_var_c),
        "reference_set_rule": f"contains {REF_END} and none of {sorted(REF_EXCLUDE)}",
        "n_reference_cases": len(ref), "n_reference_variants": len(vc),
        "primary_noise_threshold": PRIMARY_NOISE,
        "fitness_at_primary": prim[0].get("fitness") if prim else None,
        "precision_at_primary": prim[0].get("precision") if prim else None,
        "discovery_seconds_at_primary": prim[0].get("discovery_seconds") if prim else None,
        "alignment": align,
        "total_seconds": round(time.time() - t00, 1)}
    json.dump(summary, open(os.path.join(RES, "bpi2017_generality_check.json"), "w"),
              indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float), flush=True)


if __name__ == "__main__":
    main()

"""e11_conformance.py -- the CONF channel on BPI 2019: Inductive Miner (infrequent) normative
model per flow type, alignment cost delta per case (PM4Py A*), fitness/precision across the
noise-threshold grid {0, 0.1, 0.2, 0.4}.  Primary noise threshold FIXED IN ADVANCE at 0.2.

Reference set (Sec. 4.2): cases whose trace ends in a cleared invoice with no subsequent
correction activity.  The modal trace variant and its coverage are recorded; a single-variant
reference set makes the noise threshold inoperative, so discovery uses the whole reference set.

Alignments on 251,734 cases are not affordable in the budget, so alignment is computed once per
distinct TRACE VARIANT within a per-flow-type sample (all outcome-labelled positives plus a
seeded random sample of negatives) and mapped back to cases.  Recorded as a deviation.
"""
import os, json, time, sys
from collections import Counter
import numpy as np
import pandas as pd

import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.algo.conformance.alignments.petri_net import algorithm as ali
from pm4py.algo.evaluation.replay_fitness import algorithm as rf
from pm4py.algo.evaluation.precision import algorithm as prec

HERE = os.path.dirname(os.path.abspath(__file__)); PAPER = os.path.dirname(HERE)
CACHE = os.path.join(PAPER, "results", "cache"); RES = os.path.join(PAPER, "results")

SEED = 0
PRIMARY_NOISE = 0.2
GRID = [0.0, 0.1, 0.2, 0.4]
N_NEG_SAMPLE = 3000
MAX_VARIANTS = 60
EVAL_VARIANTS = 300
PREC_VARIANTS = 120
ALIGN_BUDGET = 60.0     # seconds per flow type for the alignment block
GRID_BUDGET = 45.0       # seconds per flow type for the fitness/precision grid
FLOWS = ["3-way match, invoice after GR", "3-way match, invoice before GR", "2-way match"]

ev = np.load(os.path.join(CACHE, "events.npz"))
ci, ac, ts = ev["case_idx"], ev["act_code"], ev["ts"]
acts = pd.read_csv(os.path.join(CACHE, "activities.csv"))
CODE2ACT = dict(zip(acts.code, acts.activity))
lab = pd.read_pickle(os.path.join(CACHE, "labels.pkl"))
N = len(lab)

o = np.lexsort((ts, ci))
ci_s, ac_s = ci[o], ac[o]
bounds = np.searchsorted(ci_s, np.arange(N + 1))
print("sequences indexed", flush=True)

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
            e = Event()
            e["concept:name"] = CODE2ACT[a]
            e["time:timestamp"] = TSZERO
            t.append(e)
        L.append(t)
    return L


rng = np.random.default_rng(SEED)
flow = lab["flow_type"].to_numpy()
posB = lab["pos_B"].to_numpy()

grid_rows, all_delta = [], []
summary = {"primary_noise_threshold": PRIMARY_NOISE, "seed": SEED,
           "n_neg_sample": N_NEG_SAMPLE, "max_variants_aligned": MAX_VARIANTS, "flows": {}}

for fl in FLOWS:
    t0 = time.time()
    idx = np.where(flow == fl)[0]
    seqs = {j: tuple(ac_s[bounds[j]:bounds[j + 1]].tolist()) for j in idx}
    ref = [j for j in idx if seqs[j] and seqs[j][-1] == CLEAR and not (set(seqs[j]) & CORR_CODES)]
    vc = Counter(seqs[j] for j in ref)
    modal = vc.most_common(1)[0] if vc else (None, 0)
    print(f"[{fl}] n={len(idx)} ref={len(ref)} ref_variants={len(vc)} modal_cov={modal[1]}", flush=True)

    p = idx[posB[idx]]
    n = idx[~posB[idx]]
    if len(n):
        n = rng.choice(n, size=min(N_NEG_SAMPLE, len(n)), replace=False)
    samp = np.concatenate([p, n])
    svar = {}
    for j in samp:
        svar.setdefault(seqs[j], []).append(j)
    svar_list = sorted(svar.keys(), key=lambda v: -len(svar[v]))
    covered = svar_list[:MAX_VARIANTS]
    cov_cases = sum(len(svar[v]) for v in covered)
    print(f"   sample={len(samp)} variants={len(svar)} aligning={len(covered)} "
          f"covering {cov_cases}/{len(samp)} cases", flush=True)

    ref_log = to_log([v for v, c in vc.items() for _ in range(min(c, 20))]) if vc else to_log([])
    eval_variants = sorted({seqs[j] for j in idx}, key=len)[:EVAL_VARIANTS]
    eval_log = to_log(eval_variants)
    align_log = to_log(covered)

    # ---- alignments first: this is what CONF and phi(sigma) need -------------
    for noise in [PRIMARY_NOISE]:
        if time.time() - t0 > ALIGN_BUDGET:
            print(f"   ALIGN budget hit, skipping noise={noise}", flush=True)
            continue
        try:
            net, im, fm = pm4py.discover_petri_net_inductive(ref_log, noise_threshold=noise)
            res = ali.apply(align_log, net, im, fm,
                            variant=ali.Variants.VERSION_DIJKSTRA_NO_HEURISTICS)
            for v, r in zip(covered, res):
                if r is None:
                    continue
                cost = float(r.get("cost", np.nan))
                al = r.get("alignment", [])
                nlog = sum(1 for a in al if a[1] == ">>" or a[1] is None)
                nmod = sum(1 for a in al if a[0] == ">>" or a[0] is None)
                for j in svar[v]:
                    all_delta.append({"flow_type": fl, "noise_threshold": noise, "case_idx": int(j),
                                      "delta": cost, "n_log_moves": nlog, "n_model_moves": nmod,
                                      "trace_len": len(v)})
            pd.DataFrame(all_delta).to_csv(os.path.join(RES, "conf_alignment_delta.csv"), index=False)
            print(f"   aligned noise={noise} at {time.time()-t0:.0f}s, rows={len(all_delta)}", flush=True)
        except Exception as e:
            print(f"   ALIGN FAIL noise={noise}: {repr(e)[:200]}", flush=True)

    # ---- fitness / precision grid ------------------------------------------
    t1 = time.time()
    for noise in GRID:
        if time.time() - t1 > GRID_BUDGET:
            print(f"   GRID budget hit, skipping noise={noise}", flush=True)
            grid_rows.append({"flow_type": fl, "noise_threshold": noise, "fitness": np.nan,
                              "precision": np.nan, "error": "time budget"})
            continue
        try:
            net, im, fm = pm4py.discover_petri_net_inductive(ref_log, noise_threshold=noise)
            f = rf.apply(eval_log, net, im, fm, variant=rf.Variants.TOKEN_BASED)
            pr = prec.apply(to_log(eval_variants[:PREC_VARIANTS]), net, im, fm,
                            variant=prec.Variants.ETCONFORMANCE_TOKEN)
            row = {"flow_type": fl, "noise_threshold": noise,
                   "fitness": float(f.get("log_fitness", np.nan)), "precision": float(pr),
                   "n_places": len(net.places), "n_transitions": len(net.transitions),
                   "n_eval_variants": len(eval_log)}
        except Exception as e:
            row = {"flow_type": fl, "noise_threshold": noise, "fitness": np.nan,
                   "precision": np.nan, "error": repr(e)[:200]}
        grid_rows.append(row)
        print("   ", row, flush=True)
        pd.DataFrame(grid_rows).to_csv(os.path.join(RES, "conf_noise_grid.csv"), index=False)

    summary["flows"][fl] = {"n_cases": int(len(idx)), "n_reference": int(len(ref)),
                            "n_reference_variants": int(len(vc)),
                            "modal_variant_coverage": int(modal[1]),
                            "modal_variant": [CODE2ACT[a] for a in modal[0]] if modal[0] else None,
                            "n_sampled": int(len(samp)), "n_variants_sampled": int(len(svar)),
                            "n_variants_aligned": int(len(covered)),
                            "sample_cases_covered": int(cov_cases),
                            "seconds": round(time.time() - t0, 1)}
    json.dump(summary, open(os.path.join(RES, "conf_summary.json"), "w"), indent=1)

print(json.dumps(summary, indent=1))

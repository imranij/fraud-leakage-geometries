"""E4 -- H3(a),(b): iterated best-response adversary on the obtained datasets with mutable fields.

Datasets: vehicle claims (coded policy and claim-timing fields) and Medicare (provider-level claim
aggregates). ULB is excluded by the manuscript (PCA-transformed); PaySim, IEEE-CIS and BAF were not
obtained. The leakage half of H3, part (c), needs BPI 2019 and is not computed.

Grid (S4): plausibility quantile q in {0.01, 0.05, 0.10, 0.25}; cost kappa in {0, 0.1, 0.5};
labelling delay Delta in {30, 90, 180} days (a round is one delay, so Delta maps rounds to calendar
time); T = 10 rounds; frozen and retrained detectors. Primary setting fixed before computing:
q = 0.05, kappa = 0.1, Delta = 90 days.

DEVIATION (computational budget, logged): the full 4x3 (q, kappa) grid is replaced by the cross --
every q at the primary kappa and every kappa at the primary q -- and three seeds are run at the
primary cell only. Partial results are written after every cell.

RETRAINED-ARM EVALUATION ORDER (--retrain-eval, added 2026-09-21; DEVIATIONS.md 8.7, B.2.1)
-------------------------------------------------------------------------------------------
`legacy` reproduces the run of 2026-09-19 exactly: the adversary moves, the round-(t+1) records
are appended to the training set, the detector is refitted, and that refitted detector then
scores the very records it was just fitted on. The retrained arm was therefore evaluated
in-sample, which is the defect logged as DEVIATIONS.md 8.7.

`fixed` (the default) follows the protocol as the Methods state it:

    "In round t the adversary observes the deployed detector f_{t-1} through score queries
     only ... The institution either retrains f_t on the labelled outcomes of round t after a
     labelling delay Delta, or does not retrain at all, which is the frozen detector of
     Hypothesis 3(a); a round is one labelling delay, because the adversary can respond only
     to what has been caught and labelled."
    (main_submission.tex, Section "Adaptive adversary")

Read literally: the detector deployed in round t is f_{t-1}; the round-t records are labelled
one round later, so the detector fitted on them, f_t, is deployed from round t+1. The loop
therefore refits BEFORE the adversary moves, and the refitted f_t scores the NEW round-(t+1)
records that the adversary produced against it. Round 0 is the pre-attack baseline and holds
no adversarial records, so the first refit uses the round-1 records and f_1 is first deployed
at round 2; R_0 and R_1 are consequently identical in the two arms, by construction.

The frozen arm is untouched by the switch: with retrain=False neither branch runs and the call
sequence is the one of 2026-09-19, so the frozen rows are reproduced bit for bit.

Outputs. In `fixed` mode every file is written under the prefix `fig5_durability_retrained_fixed_`
so that no result file of the 2026-09-19 run is overwritten. `legacy` keeps the original names.
"""
from __future__ import annotations

import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import common as C

FIXED_PREFIX = "fig5_durability_retrained_fixed_"

METHODS_SENTENCE = (
    "In round t the adversary observes the deployed detector f_{t-1} through score queries only, "
    "with no gradient access ... The institution either retrains f_t on the labelled outcomes of "
    "round t after a labelling delay Delta, or does not retrain at all, which is the frozen "
    "detector of Hypothesis 3(a); a round is one labelling delay, because the adversary can "
    "respond only to what has been caught and labelled.")


def out_name(name: str, retrain_eval: str) -> str:
    """Result-file name under the active evaluation order. `fixed` never reuses a legacy name."""
    if retrain_eval == "legacy":
        return name
    stem = name[len("fig5_durability_"):] if name.startswith("fig5_durability_") else name
    return FIXED_PREFIX + stem

QS = [0.01, 0.05, 0.10, 0.25]
KAPPAS = [0.0, 0.1, 0.5]
DELTAS = [30, 90, 180]
T = 10
PRIMARY = {"q": 0.05, "kappa": 0.1, "delta": 90}

VEHICLE_MUTABLE = ["Deductible", "DriverRating", "Days_Policy_Accident", "Days_Policy_Claim",
                   "PastNumberOfClaims", "NumberOfSuppliments", "AgeOfVehicle",
                   "WeekOfMonth", "WeekOfMonthClaimed", "AddressChange_Claim"]
MEDICARE_MUTABLE = ["n_claims", "amt_mean", "amt_max", "amt_std", "ded_mean",
                    "dur_mean", "dur_max", "claims_per_bene", "inpatient_share"]


def best_response(Xf, mut_idx, cop, thresh, score_fn, kappa, scale, n_iter=1, n_grid=7):
    """Query-only coordinate descent on the mutable coordinates (Eq. 2)."""
    x0 = Xf.copy()
    x = Xf.copy()
    grid = np.linspace(-1.5, 1.5, n_grid)
    for _ in range(n_iter):
        for j in mut_idx:
            best = x.copy()
            best_obj = score_fn(x) + kappa * (np.abs(x - x0) / scale).mean(1)
            for g in grid:
                if g == 0:
                    continue
                cand = x.copy()
                cand[:, j] = x[:, j] + g * scale[j]
                ok = cop.loglik(cand) >= thresh
                o = score_fn(cand) + kappa * (np.abs(cand - x0) / scale).mean(1)
                take = ok & (o < best_obj)
                best[take] = cand[take]
                best_obj = np.where(take, o, best_obj)
            x = best
    return x


def run_dataset(name, Xtr, ytr, Xte, yte, vte, mut_cols, cols, seeds=(0, 1, 2),
                retrain_eval="fixed", cells_mode="cross"):
    mut_idx = [cols.index(c) for c in mut_cols if c in cols]
    scale = Xtr.std(0) + 1e-9
    legit_tr = Xtr[ytr == 0]
    cop = C.GaussianCopula(legit_tr[np.random.default_rng(0).choice(
        len(legit_tr), min(len(legit_tr), 8000), replace=False)])
    ll_legit = cop.loglik(Xte[yte == 0])
    rows = []
    if cells_mode == "primary":
        cells = [(PRIMARY["q"], PRIMARY["kappa"])]
    else:
        cells = [(q, PRIMARY["kappa"]) for q in QS] + \
                [(PRIMARY["q"], k) for k in KAPPAS if k != PRIMARY["kappa"]]
    for q, kappa in cells:
        thresh = float(np.quantile(ll_legit, q))
        for retrain in (False, True):
            use = seeds if (q == PRIMARY["q"] and kappa == PRIMARY["kappa"]) else seeds[:1]
            for seed in use:
                _, model = C.fit_gbdt(Xtr, ytr, Xtr[:1], seed=seed)
                score_fn = lambda z: model.predict_proba(z)[:, 1]
                Xf, vf, Xn = Xte[yte == 1].copy(), vte[yte == 1].copy(), Xte[yte == 0]
                tr_X, tr_y = Xtr.copy(), ytr.copy()
                # Instrumentation for the in-sample question of DEVIATIONS.md 8.7: `seen_adv`
                # holds the adversarial records already appended to the training set, `prev`
                # the previous round's records, so each row records how far the adversary
                # moved and how much of what is scored the detector was fitted on.
                seen_adv, prev = set(), None
                for t in range(T + 1):
                    sn, sf = score_fn(Xn), score_fn(Xf)
                    keys = [r.tobytes() for r in np.ascontiguousarray(Xf)]
                    n_in_train = sum(k in seen_adv for k in keys)
                    n_moved = 0 if prev is None else int((np.abs(Xf - prev) > 0).any(1).sum())
                    s_all = np.r_[sn, sf]
                    y_all = np.r_[np.zeros(len(Xn), int), np.ones(len(Xf), int)]
                    v_all = np.r_[np.zeros(len(Xn)), vf]
                    thr = np.quantile(sn, 0.99)
                    rows.append({"dataset": name, "q": q, "kappa": kappa, "seed": seed,
                                 "retrain": retrain, "round": t,
                                 "retrain_eval": retrain_eval,
                                 "vw_recall": C.value_weighted_recall_at_fpr(y_all, s_all, v_all, 0.01),
                                 "count_recall": C.count_recall_at_fpr(y_all, s_all, 0.01),
                                 "value_stolen": float(vf[sf < thr].sum()),
                                 "n_fraud_records": len(Xf),
                                 "n_moved_since_previous_round": n_moved,
                                 "n_scored_in_training_set": n_in_train,
                                 "share_scored_in_training_set": n_in_train / max(len(Xf), 1),
                                 "mean_score_on_fraud_records": float(sf.mean())})
                    if t == T:
                        break
                    prev = Xf.copy()
                    if retrain and retrain_eval == "fixed" and t >= 1:
                        seen_adv.update(keys)
                        # The round-t records have just been scored by the deployed detector
                        # f_{t-1}.  Their labels arrive after one labelling delay, which is one
                        # round, so they fit f_t, which is deployed in round t+1.  Refitting here,
                        # BEFORE the adversary moves, makes the adversary of round t+1 respond to
                        # f_t (as the Methods require) and makes f_t score records it has not seen.
                        tr_X = np.vstack([tr_X, Xf])
                        tr_y = np.r_[tr_y, np.ones(len(Xf), int)]
                        _, model = C.fit_gbdt(tr_X, tr_y, tr_X[:1], seed=seed)
                        score_fn = lambda z, m=model: m.predict_proba(z)[:, 1]
                    Xf = best_response(Xf, mut_idx, cop, thresh, score_fn, kappa, scale)
                    if retrain and retrain_eval == "legacy":
                        # 2026-09-19 order, kept for reproduction only: the records appended here
                        # are the round-(t+1) records, and they are scored at the top of the next
                        # iteration by the detector just fitted on them (DEVIATIONS.md 8.7),
                        # which is why `share_scored_in_training_set` is 1.0 from round 1 here.
                        seen_adv.update(r.tobytes() for r in np.ascontiguousarray(Xf))
                        tr_X = np.vstack([tr_X, Xf])
                        tr_y = np.r_[tr_y, np.ones(len(Xf), int)]
                        _, model = C.fit_gbdt(tr_X, tr_y, tr_X[:1], seed=seed)
                        score_fn = lambda z, m=model: m.predict_proba(z)[:, 1]
                print(f"  {name} q={q} k={kappa} retrain={retrain} seed={seed} done", flush=True)
                C.dump(out_name(f"fig5_durability_partial_{name}.csv", retrain_eval),
                       pd.DataFrame(rows))
    return pd.DataFrame(rows)


def summarise(df, retrain_eval="fixed"):
    prim = []
    for ds, g in df.groupby("dataset"):
        sel = g[(g["q"] == PRIMARY["q"]) & (g["kappa"] == PRIMARY["kappa"])]
        for retrain in (False, True):
            s = sel[sel.retrain == retrain]
            if s.empty:
                continue
            p = s.pivot_table(index="seed", columns="round", values="value_stolen")
            r = s.pivot_table(index="seed", columns="round", values="vw_recall")
            tmax = int(max(p.columns))
            ratio = p[tmax] / p[0].replace(0, np.nan)
            prim.append({"dataset": ds, "retrain": retrain, "T": tmax,
                         "V_T_over_V_0_mean": float(ratio.mean()),
                         "V_T_over_V_0_sd": float(ratio.std(ddof=1)) if len(ratio) > 1 else float("nan"),
                         "R_0_mean": float(r[0].mean()), "R_T_mean": float(r[tmax].mean()),
                         "R_drop": float(r[0].mean() - r[tmax].mean()),
                         "monotone_nonincreasing": bool(np.all(np.diff(r.mean(0).values) <= 1e-12)),
                         "n_seeds": int(len(p)),
                         "retrain_eval": retrain_eval,
                         "calendar_years_at_delta90": tmax * PRIMARY["delta"] / 365.0})
    C.dump(out_name("h3_primary_endpoint.csv", retrain_eval), pd.DataFrame(prim))

    spread = []
    for ds, g in df.groupby("dataset"):
        last = g[g["round"] == g["round"].max()]
        base = last.groupby(["q", "kappa", "retrain"])["vw_recall"].mean().reset_index()
        spread.append({"dataset": ds,
                       "sd_across_q": float(base[base.kappa == PRIMARY["kappa"]]
                                            .groupby("q")["vw_recall"].mean().std(ddof=1)),
                       "sd_across_kappa": float(base[base.q == PRIMARY["q"]]
                                                .groupby("kappa")["vw_recall"].mean().std(ddof=1)),
                       "sd_across_retrain": float(base.groupby("retrain")["vw_recall"].mean().std(ddof=1)),
                       "note": "Delta enters only as the round-to-calendar mapping, so it produces "
                               "no spread in R_t by round index."})
    C.dump(out_name("h3_spread_by_free_quantity.csv", retrain_eval), pd.DataFrame(spread))

    C.dump(out_name("h3_leakage_side.json", retrain_eval), {
        "status": "not computed",
        "reason": ("H3(c) -- stationarity of the deviation-type mix between halves of the log, its "
                   "permutation reference, and conformance recall on outcome labels across replays "
                   "-- requires the BPI Challenge 2019 log, which could not be obtained."),
        "delta_grid_days": DELTAS, "rounds": T,
        "calendar_mapping_years": {str(d): T * d / 365.0 for d in DELTAS}})


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--retrain-eval", choices=("fixed", "legacy"), default="fixed",
                    help="order in which the retrained detector is refitted and scored; "
                         "'fixed' scores the adversary's new records out of sample (default), "
                         "'legacy' reproduces the in-sample run of 2026-09-19")
    ap.add_argument("--datasets", default="medicare,vehicle",
                    help="comma-separated subset of {medicare, vehicle}")
    ap.add_argument("--cells", choices=("cross", "primary"), default="cross",
                    help="'cross' is the S4 reduced grid; 'primary' is the primary cell only")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    ev = a.retrain_eval
    want = [d.strip().lower() for d in a.datasets.split(",") if d.strip()]
    if ev == "fixed":
        C.dump(FIXED_PREFIX + "config.json", {
            "retrain_eval": ev,
            "defect_repaired": "DEVIATIONS.md 8.7 (retrained arm scored in-sample)",
            "methods_sentence_followed": METHODS_SENTENCE,
            "protocol": ("round t: the deployed detector f_{t-1} scores the adversary's round-t "
                         "records; those records are then labelled (one round = one labelling "
                         "delay) and fit f_t; the adversary of round t+1 best-responds to f_t; "
                         "f_t scores the new round-(t+1) records."),
            "first_refit_at_round": 1, "first_round_the_arms_can_differ": 2,
            "frozen_arm_changed": False,
            "q_grid": QS, "kappa_grid": KAPPAS, "delta_grid_days": DELTAS, "rounds": T,
            "primary": PRIMARY, "cells": a.cells, "datasets": want,
            "command": "python3 e4_durability.py --retrain-eval %s --datasets %s --cells %s"
                       % (ev, ",".join(want), a.cells)})

    out = []
    if "medicare" in want:
        m, F, ym, valm, feat = C.medicare_provider_features()
        cut = np.quantile(m["start_mean"].values, 0.7)
        trm = m["start_mean"].values <= cut
        out.append(run_dataset("Medicare", F[trm].values, ym[trm], F[~trm].values, ym[~trm],
                               valm[~trm], MEDICARE_MUTABLE, feat,
                               retrain_eval=ev, cells_mode=a.cells))
        C.dump(out_name("fig5_durability_raw.csv", ev), pd.concat(out, ignore_index=True))
        summarise(pd.concat(out, ignore_index=True), ev)

    if "vehicle" in want:
        cc = C.load_carclaims()
        X, y, val, year, cols = C.carclaims_matrix(cc)
        tr = year < 1996
        out.append(run_dataset("Vehicle", X[tr].values, y[tr], X[~tr].values, y[~tr], val[~tr],
                               VEHICLE_MUTABLE, cols, retrain_eval=ev, cells_mode=a.cells))
        df = pd.concat(out, ignore_index=True)
        C.dump(out_name("fig5_durability_raw.csv", ev), df)
        summarise(df, ev)


if __name__ == "__main__":
    main()

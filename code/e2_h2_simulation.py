"""E2 -- H2 cross-channel transfer, run as the simulation study of Supplementary Material S3.

Every number this script writes is a property of the generative model below and of nothing else.

Generative model (S3), with the one substitution forced by data availability recorded in
DEVIATIONS.md: S3 fits the legitimate attribute distributions to BPI 2019, which could not be
obtained, so the record model is a Gaussian copula with empirical marginals fitted to the
*legitimate records of the ULB credit-card file* (real data), and the process model is the
four-flow-type purchase-to-pay net documented for BPI 2019 by its organisers, written out here
rather than discovered from the log. The rest of S3 is followed exactly: fraudulent records are
produced by the adversary at round 0 inside the density-based plausibility set; fraudulent traces
enter the process at the weak-control step with probability pi in {0, 0.25, 0.5, 0.75}; the
positive rate is 1%; five seeds; the learner is gradient-boosted trees with and without phi(sigma);
the endpoint is recall at a 1% false-positive rate.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import common as C

PIS = [0.0, 0.25, 0.5, 0.75]
N_CASES = 30000
POS_RATE = 0.01
ACTS = ["PO_create", "PO_approve", "GR", "IR", "Clear", "Pay"]
NORMATIVE = {
    "3way_after_GR": ["PO_create", "PO_approve", "GR", "IR", "Clear", "Pay"],
    "3way_before_GR": ["PO_create", "PO_approve", "IR", "GR", "Clear", "Pay"],
    "2way": ["PO_create", "PO_approve", "IR", "Clear", "Pay"],
    "consignment": ["PO_create", "GR", "Clear", "Pay"],
}
FLOW_P = [0.45, 0.25, 0.22, 0.08]
# The weak-control step is the step with the highest atypical-role execution rate (S3). In this
# net it is the goods receipt where one exists, and the invoice receipt otherwise.
WEAK = {"3way_after_GR": "GR", "3way_before_GR": "GR", "2way": "IR", "consignment": "GR"}


def align_cost(trace, model_seq):
    """Levenshtein alignment cost with unit log moves and unit model moves (S2's cost function)."""
    n, m = len(trace), len(model_seq)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (trace[i - 1] != model_seq[j - 1]) * 2)
        prev = cur
    return prev[m]


def gen_trace(flow, rng, enter_weak=False):
    seq = list(NORMATIVE[flow])
    atypical_role = 0
    if enter_weak:
        w = WEAK[flow]
        if w in seq:
            i = seq.index(w)
            if i > 0:
                seq.pop(i - 1)              # the control step before the weak step is skipped
            atypical_role = 1
    # background process noise, identical for legitimate and fraudulent cases
    if rng.random() < 0.08:
        i = rng.integers(0, len(seq))
        seq.insert(i, seq[i])               # a repeated step
    if rng.random() < 0.05 and len(seq) > 3:
        i = int(rng.integers(1, len(seq) - 1))
        seq[i], seq[i + 1] = seq[i + 1], seq[i]
    return seq, atypical_role


def phi(seq, flow, atypical_role):
    f = {f"cnt_{a}": seq.count(a) for a in ACTS}
    f["n_events"] = len(seq)
    f["n_repeat"] = len(seq) - len(set(seq))
    f["align_cost"] = align_cost(seq, NORMATIVE[flow])
    f["atypical_role"] = atypical_role
    f["skipped_pred"] = int(len(seq) < len(NORMATIVE[flow]))
    dfr = set(zip(seq, seq[1:]))
    nrm = set(zip(NORMATIVE[flow], NORMATIVE[flow][1:]))
    f["novel_dfr"] = len(dfr - nrm)
    return f


def build_record_models():
    """Legitimate and base-fraud record models, fitted to real ULB records."""
    d = C.load_ulb()
    cols = [f"V{i}" for i in range(1, 10)] + ["Amount"]
    legit = d.loc[d.Class == 0, cols].sample(n=40000, random_state=0).values
    fraud = d.loc[d.Class == 1, cols].values
    return C.GaussianCopula(legit), legit, fraud, cols


def adversary_round0(x0, cop, thresh, pilot_score, kappa=0.1, n_iter=2, rng=None):
    """One round of Eq. (2): coordinate descent inside the density-based plausibility set."""
    x = x0.copy()
    d = x.shape[1]
    scale = np.std(x0, axis=0) + 1e-9
    grid = np.linspace(-1.0, 1.0, 9)
    for _ in range(n_iter):
        for j in range(d):
            base = pilot_score(x) + kappa * np.abs(x - x0).mean(1) / (scale.mean() + 1e-9)
            best = x.copy()
            best_obj = base
            for g in grid:
                if g == 0:
                    continue
                cand = x.copy()
                cand[:, j] = x[:, j] + g * scale[j]
                ok = cop.loglik(cand) >= thresh
                obj = pilot_score(cand) + kappa * np.abs(cand - x0).mean(1) / (scale.mean() + 1e-9)
                take = ok & (obj < best_obj)
                best[take] = cand[take]
                best_obj = np.where(take, obj, best_obj)
            x = best
    return x


def run():
    cop, legit, fraud_base, cols = build_record_models()
    thresh = np.quantile(cop.loglik(legit[:8000]), 0.05)   # plausibility quantile q = 0.05

    # pilot detector the round-0 adversary queries (trained once, outside the evaluation loop)
    rng0 = np.random.default_rng(12345)
    pil_n = 8000
    pil_legit = legit[rng0.choice(len(legit), pil_n, replace=False)]
    pil_fraud = fraud_base[rng0.choice(len(fraud_base), pil_n // 20, replace=True)]
    Xp = np.vstack([pil_legit, pil_fraud])
    yp = np.r_[np.zeros(len(pil_legit)), np.ones(len(pil_fraud))]
    _, pilot = C.fit_gbdt(Xp, yp, Xp[:1], seed=0)
    pilot_score = lambda z: pilot.predict_proba(z)[:, 1]

    rows = []
    for pi in PIS:
        for seed in C.SEEDS:
            rng = np.random.default_rng(1000 * seed + int(pi * 100))
            n_pos = int(round(POS_RATE * N_CASES))
            n_neg = N_CASES - n_pos
            flows = rng.choice(list(NORMATIVE), N_CASES, p=FLOW_P)
            y = np.r_[np.zeros(n_neg, int), np.ones(n_pos, int)]

            # records
            Xl = legit[rng.choice(len(legit), n_neg, replace=True)]
            x0 = fraud_base[rng.choice(len(fraud_base), n_pos, replace=True)]
            Xf = adversary_round0(x0, cop, thresh, pilot_score, kappa=0.1, rng=rng)
            X = np.vstack([Xl, Xf])

            # traces
            feats = []
            for i in range(N_CASES):
                is_fraud = y[i] == 1
                enter = bool(is_fraud and rng.random() < pi)
                seq, ar = gen_trace(flows[i], rng, enter_weak=enter)
                feats.append(phi(seq, flows[i], ar))
            P = pd.DataFrame(feats)
            P["flow"] = pd.Categorical(flows, categories=list(NORMATIVE)).codes

            order = rng.permutation(N_CASES)
            X, y, P = X[order], y[order], P.iloc[order].reset_index(drop=True)
            ntr = int(0.7 * N_CASES)
            Xrec = pd.DataFrame(X, columns=cols)
            Xall = pd.concat([Xrec.reset_index(drop=True), P], axis=1)

            s_rec = C.fit_gbdt(Xrec.values[:ntr], y[:ntr], Xrec.values[ntr:], seed=seed)[0]
            s_all = C.fit_gbdt(Xall.values[:ntr], y[:ntr], Xall.values[ntr:], seed=seed)[0]
            yte = y[ntr:]
            vte = np.abs(Xrec["Amount"].values[ntr:]) + 1.0
            r_rec = C.count_recall_at_fpr(yte, s_rec, 0.01)
            r_all = C.count_recall_at_fpr(yte, s_all, 0.01)
            rows.append({"pi": pi, "seed": seed,
                         "recall_records_only": r_rec, "recall_with_phi": r_all,
                         "gain": r_all - r_rec,
                         "vw_recall_records_only": C.value_weighted_recall_at_fpr(yte, s_rec, vte, 0.01),
                         "vw_recall_with_phi": C.value_weighted_recall_at_fpr(yte, s_all, vte, 0.01)})
            print(f"  pi={pi} seed={seed} gain={r_all - r_rec:+.4f}", flush=True)

    df = pd.DataFrame(rows)
    C.dump("fig4_h2_transfer_raw.csv", df)
    agg = df.groupby("pi")[["recall_records_only", "recall_with_phi", "gain"]].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    C.dump("fig4_h2_transfer_summary.csv", agg)

    g0 = df.loc[df.pi == 0.0, "gain"].values
    g5 = df.loc[df.pi == 0.5, "gain"].values
    prim = C.paired_tests(g5, g0)
    prim.update({"endpoint": "H2 primary: gain at pi=0.5 minus gain at pi=0 (unpaired across pi, "
                             "paired by seed index)",
                 "gain_at_pi_0.5": float(g5.mean()), "gain_at_pi_0": float(g0.mean()),
                 "gain_at_pi_0.75": float(df.loc[df.pi == 0.75, "gain"].mean()),
                 "gain_at_pi_0.25": float(df.loc[df.pi == 0.25, "gain"].mean()),
                 "sd_gain_at_pi_0.5": float(g5.std(ddof=1)), "sd_gain_at_pi_0": float(g0.std(ddof=1))})
    C.dump("h2_primary_endpoint.json", prim)


if __name__ == "__main__":
    run()

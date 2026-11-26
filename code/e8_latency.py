"""E8 -- throughput per channel, in cases per second, on the run's hardware.

The learning channel is timed on the real ULB test split. The conformance channel is timed with
PM4Py's alignment on a log simulated from the H2 generative model, because BPI 2019 was not
obtained: the alignment figure is therefore a throughput measurement on a synthetic log of the same
shape, not on the 251,734-case log the deployment claim refers to, and it is labelled so.
"""
from __future__ import annotations

import sys, os, platform, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import common as C
import e2_h2_simulation as H2

N_ALIGN = 400


def main():
    d = C.load_ulb().sort_values("Time").reset_index(drop=True)
    feats = [c for c in d.columns if c != "Class"]
    i_va = int(0.8 * len(d))
    Xtr, ytr = d.loc[:i_va - 1, feats].values, d.loc[:i_va - 1, "Class"].values
    Xte = d.loc[i_va:, feats].values
    rows = []

    _, model = C.fit_gbdt(Xtr, ytr, Xtr[:1], seed=0)
    t0 = time.perf_counter(); model.predict_proba(Xte); dt = time.perf_counter() - t0
    rows.append({"channel": "learning (GBDT scoring)", "n_cases": len(Xte), "seconds": dt,
                 "cases_per_second": len(Xte) / dt, "input": "ULB test split (real)"})

    from sklearn.ensemble import IsolationForest
    m = IsolationForest(n_estimators=100, random_state=0, n_jobs=2).fit(Xtr)
    t0 = time.perf_counter(); m.score_samples(Xte); dt = time.perf_counter() - t0
    rows.append({"channel": "isolation forest scoring", "n_cases": len(Xte), "seconds": dt,
                 "cases_per_second": len(Xte) / dt, "input": "ULB test split (real)"})

    t0 = time.perf_counter(); C.fit_autoencoder(Xtr[:40000], Xte, seed=0); dt = time.perf_counter() - t0
    rows.append({"channel": "autoencoder scoring (incl. fit on 40k)", "n_cases": len(Xte),
                 "seconds": dt, "cases_per_second": len(Xte) / dt, "input": "ULB test split (real)"})

    # conformance channel: PM4Py A* alignments on a simulated purchase-to-pay log
    import pm4py
    rng = np.random.default_rng(0)
    recs = []
    for i in range(N_ALIGN):
        flow = "3way_after_GR"
        seq, _ = H2.gen_trace(flow, rng, enter_weak=bool(rng.random() < 0.2))
        t = pd.Timestamp("2018-01-01")
        for j, a in enumerate(seq):
            recs.append({"case:concept:name": f"c{i}", "concept:name": a,
                         "time:timestamp": t + pd.Timedelta(hours=j)})
    log = pd.DataFrame(recs)
    ref = log[log["case:concept:name"].isin([f"c{i}" for i in range(80)])]
    t0 = time.perf_counter()
    net, im, fm = pm4py.discover_petri_net_inductive(ref, noise_threshold=0.2)
    disc = time.perf_counter() - t0
    t0 = time.perf_counter()
    al = pm4py.conformance_diagnostics_alignments(log, net, im, fm)
    dt = time.perf_counter() - t0
    fit = float(np.mean([a["fitness"] for a in al if a is not None]))
    rows.append({"channel": "conformance (PM4Py A* alignment)", "n_cases": N_ALIGN, "seconds": dt,
                 "cases_per_second": N_ALIGN / dt,
                 "input": "simulated P2P log (BPI 2019 not obtained)"})
    rows.append({"channel": "conformance (inductive miner discovery)", "n_cases": 80,
                 "seconds": disc, "cases_per_second": 80 / disc,
                 "input": "simulated P2P log (BPI 2019 not obtained)"})

    df = pd.DataFrame(rows)
    df["projected_seconds_for_251734_cases"] = 251734 / df["cases_per_second"]
    C.dump("tab_latency.csv", df)
    C.dump("e8_environment.json", {"platform": platform.platform(), "python": platform.python_version(),
                                   "cpu_count": os.cpu_count(),
                                   "mean_alignment_fitness_on_simulated_log": fit})
    print(df.to_string(), flush=True)


if __name__ == "__main__":
    main()

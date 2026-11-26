"""e28_e7_ieee.py -- the E7 injection gate on an IEEE-CIS host.

The gate (experiment E7) puts the LeakBench seeded deviation-type distribution against the
deviation-type distribution the real BPI 2019 log carries.  It has been run on a ULB substitute
host (`results/e7_gate.json`) and on PaySim (`results/e7_gate_paysim.json`), failing both.
IEEE-CIS is the third host `configs/default.yaml` names and was obtained on 2026-09-20; this
script completes the set.  Nothing about the gate's construction changes: same seeded type mix,
same rates, same hard/easy variants, same three seeds, same tolerance commitment file, same real
comparator (`results/leakage_deviation_typology.csv`, flow_type == ALL).

Host construction reuses the cached IEEE-CIS test split of e20 (`results/cache/ieee_*.npy`), so
the 652 MB transaction CSV is not re-read.  Harmonised fields:
    day              = TransactionDT // 86400 (TransactionDT is a delta in seconds; day index only)
    entity_id        = card1   (the payment instrument; the closest thing IEEE-CIS has to a payer)
    counterparty_id  = P_emaildomain code (the closest thing IEEE-CIS has to a payee)
    amount           = TransactionAmt (a real monetary field)
    y                = isFraud
Outputs carry the `e7ieee_` prefix.  No existing results file is touched.
"""
from __future__ import annotations
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import common as C

sys.path.insert(0, os.path.abspath(os.path.join(C.PAPER, "lib")))
from src.leakbench.inject import inject, verify                    # noqa: E402

CACHE = os.path.join(C.PAPER, "results", "cache")
TYPE_MIX = {"DUP_EXACT": 1, "DUP_NEAR": 1, "WRONG_PAYEE": 1, "TRANSPOSE": 1,
            "DOUBLE_FX": 0.5, "SPLIT_PAY": 0.5}
RATES = [0.005, 0.01, 0.02]
GATE_SEEDS = [0, 1, 2]

COARSE = {"DUP_EXACT": "duplicate-like", "DUP_NEAR": "duplicate-like",
          "SPLIT_PAY": "duplicate-like", "TRANSPOSE": "value-like", "DOUBLE_FX": "value-like",
          "WRONG_PAYEE": "party-like", "duplicate invoice": "duplicate-like",
          "repeated clearing": "duplicate-like", "value mismatch": "value-like",
          "atypical role": "party-like", "skipped goods receipt": "control-step",
          "invoice before purchase order": "control-step"}


def jsd(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a[a > 0] * np.log2(a[a > 0] / b[a > 0])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def build_host():
    feats = json.load(open(os.path.join(CACHE, "ieee_feats.json")))
    i_card1, i_pmail = feats.index("card1"), feats.index("P_emaildomain")
    X = np.load(os.path.join(CACHE, "ieee_Xte.npy"), mmap_mode="r")
    card1 = np.asarray(X[:, i_card1], dtype=np.float64)
    pmail = np.asarray(X[:, i_pmail], dtype=np.float64)
    del X
    tdt = np.load(os.path.join(CACHE, "ieee_tdt_te.npy"))
    amt = np.load(os.path.join(CACHE, "ieee_vte.npy"))
    y = np.load(os.path.join(CACHE, "ieee_yte.npy"))
    harm = pd.DataFrame({
        "record_id": np.arange(len(tdt)),
        "day": (tdt // 86400).astype(float),
        "entity_id": card1.astype(np.int64),
        "counterparty_id": np.nan_to_num(pmail, nan=-1).astype(np.int64),
        "amount": amt.astype(float),
        "currency": "USD",
        "y": y.astype(int)})
    return harm


def main():
    t0 = time.time()
    harm = build_host()
    print(f"host = {len(harm):,} IEEE-CIS test records, fraud {int(harm.y.sum())} "
          f"({harm.y.mean():.4%}), {harm.day.nunique()} day periods "
          f"[{int(harm.day.min())}..{int(harm.day.max())}]", flush=True)

    rows = []
    for rate in RATES:
        for hard in (False, True):
            for seed in GATE_SEEDS:
                v = verify(inject(harm, rate=rate, seed=seed, type_mix=TYPE_MIX, hard=hard))
                tot = sum(v["by_type"].values())
                for t, c in v["by_type"].items():
                    rows.append({"rate": rate, "hard": hard, "seed": seed, "leak_type": t,
                                 "n": c, "share": c / tot})
                print(f"   injected rate={rate} hard={hard} seed={seed} "
                      f"n={v['n_injected']} ({time.time()-t0:.0f}s)", flush=True)
    inj_df = pd.DataFrame(rows)
    C.dump("e7ieee_injected_type_distribution.csv", inj_df)

    inj_share = inj_df.groupby("leak_type")["n"].sum(); inj_share /= inj_share.sum()
    real = pd.read_csv(os.path.join(C.RES, "leakage_deviation_typology.csv"))
    real = real[(real.flow_type == "ALL") & (real.deviation_type != "ANY (union)")]
    real_share = real.set_index("deviation_type")["n_cases"]; real_share /= real_share.sum()
    allt = sorted(set(inj_share.index) | set(real_share.index))
    d_lit = jsd([inj_share.get(t, 0.0) for t in allt], [real_share.get(t, 0.0) for t in allt])
    co = lambda s: {COARSE[k]: sum(float(v) for kk, v in s.items() if COARSE[kk] == COARSE[k])
                    for k in s.index}
    ci_, cr_ = co(inj_share), co(real_share)
    ck = sorted(set(ci_) | set(cr_))
    d_co = jsd([ci_.get(k, 0.0) for k in ck], [cr_.get(k, 0.0) for k in ck])
    TOL = json.load(open(os.path.join(C.RES, "e7_gate_tolerance_commitment.json")))["tolerance_jsd"]
    per = pd.DataFrame([{"type": t, "injected_share": float(inj_share.get(t, 0.0)),
                         "real_share": float(real_share.get(t, 0.0)),
                         "abs_diff": abs(float(inj_share.get(t, 0.0)) - float(real_share.get(t, 0.0)))}
                        for t in allt]).sort_values("abs_diff", ascending=False)
    C.dump("e7ieee_gate_type_comparison.csv", per)

    prev = {}
    for k, f in (("ULB", "e7_gate.json"), ("PaySim", "e7_gate_paysim.json")):
        p = os.path.join(C.RES, f)
        if os.path.exists(p):
            g = json.load(open(p))
            prev[k] = {"jsd_literal_union_support": g["jsd_literal_union_support"],
                       "jsd_coarse_mapping": g["jsd_coarse_mapping"], "verdict": g["verdict"]}

    out = {
        "host": "IEEE-CIS Fraud Detection, test split of the e20 temporal split "
                f"({len(harm)} transactions, TransactionDT > its 75th percentile)",
        "n_records_host": int(len(harm)), "n_fraud_host": int(harm.y.sum()),
        "n_day_periods": int(harm.day.nunique()),
        "rates": RATES, "seeds": GATE_SEEDS, "hard_variants": [False, True],
        "type_mix": TYPE_MIX,
        "tolerance_jsd": TOL, "jsd_literal_union_support": d_lit, "jsd_coarse_mapping": d_co,
        "shared_support_literal": sorted(set(inj_share.index) & set(real_share.index)),
        "verdict_literal": "FAIL" if d_lit > TOL else "PASS",
        "verdict_coarse": "FAIL" if d_co > TOL else "PASS",
        "verdict": "FAIL" if min(d_lit, d_co) > TOL else "PASS",
        "most_divergent_types": per.head(3)["type"].tolist(),
        "most_divergent_detail": per.head(3).to_dict("records"),
        "injected_vocabulary": sorted(inj_share.index.tolist()),
        "real_vocabulary": sorted(real_share.index.tolist()),
        "coarse_injected": ci_, "coarse_real": cr_,
        "previous_hosts": prev,
        "reading": ("The three hosts the design names have now all been run. The literal distance "
                    "is 1.000 on every host because the seeded vocabulary and the BPI 2019 "
                    "vocabulary are disjoint; the coarse mapping distance is a property of the "
                    "type mix and the real log, not of the host. The gate fails on all three."),
        "seconds": round(time.time() - t0, 1)}
    C.dump("e7ieee_gate.json", out)
    print(json.dumps({k: out[k] for k in ["jsd_literal_union_support", "jsd_coarse_mapping",
                                          "tolerance_jsd", "verdict", "shared_support_literal",
                                          "most_divergent_types", "previous_hosts"]}, indent=1))


if __name__ == "__main__":
    main()

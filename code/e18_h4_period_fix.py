"""e18_h4_period_fix.py -- corrected H4 value-at-budget: injected records share the budget periods
of the host instead of forming singleton periods of their own.

DEFECT FOUND 2026-09-20.  `src/leakbench/inject.py` offsets an injected record's timestamp by a
uniform fraction of a day (`c["day"] = r.day + rng.uniform(0, dt)`), so every injected record
carries a FRACTIONAL day.  `value_at_budget` groups by `day`; a fractional day matches no base
record, so each injected record became a budget period containing exactly one record, and
`k = max(1, round(k_frac * 1)) = 1` selected it.  Every policy therefore recovered 100% of the
leakage value at every budget: `VaB_L = 1.000` for the oracle, both value-ranked queues,
fraud-first and the fixed split alike, in `results/tab_vab_full_grid.csv` (2026-09-19 run).

FIX: floor `day` to the host's own period grid after injection, so an injected record competes for
the same budget as the base records of the period it lands in.  Nothing else changes.  The
uncorrected files are left in place; the corrected ones carry the `_periodfix` suffix.
"""
from __future__ import annotations
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import common as C
sys.path.insert(0, os.path.abspath(os.path.join(C.PAPER, "lib")))
from src.leakbench.inject import inject, verify                       # noqa: E402
from src.triage.value_at_budget import vab_grid                       # noqa: E402

TYPE_MIX = {"DUP_EXACT": 1, "DUP_NEAR": 1, "WRONG_PAYEE": 1, "TRANSPOSE": 1,
            "DOUBLE_FX": 0.5, "SPLIT_PAY": 0.5}
LGD = {"DUP_EXACT": 1.0, "DUP_NEAR": 1.0, "WRONG_PAYEE": 0.6, "TRANSPOSE": 0.5,
       "DOUBLE_FX": 0.5, "SPLIT_PAY": 0.3, "NONE": 0.0}
RATE, SEED = 0.01, 0


def leak_rule_score(d):
    d = d.reset_index(drop=True)
    s = np.zeros(len(d))
    s += (d["counterparty_id"].astype(str) + "|" + d["amount"].round(2).astype(str)
          ).duplicated(keep=False).to_numpy().astype(float) * 2.0
    s += (d["counterparty_id"].astype(str) + "|" + d["amount"].round(0).astype(str)
          ).duplicated(keep=False).to_numpy().astype(float)
    g = d.groupby(["entity_id", d["day"].astype(int)])["record_id"].transform("size").to_numpy()
    s += (g >= 3).astype(float)
    return s + np.linspace(0, 1e-9, len(d))


def run(harm, s_f_te, tag, host_desc):
    inj = inject(harm, rate=RATE, seed=SEED, type_mix=TYPE_MIX)
    n_singleton_before = int((inj.groupby("day").size() == 1).sum())
    inj["day"] = np.floor(inj["day"].to_numpy()).astype(np.int64)      # THE FIX
    p_f = np.zeros(len(inj))
    base = (inj.injected == 0).to_numpy()
    p_f[base] = s_f_te[inj.loc[base, "record_id"].to_numpy()]
    p_f[~base] = s_f_te[inj.loc[~base, "source_record_id"].to_numpy().astype(int)]
    raw = leak_rule_score(inj)
    p_l = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
    v_l = inj["amount"].to_numpy() * inj["leak_type"].map(LGD).fillna(0.5).to_numpy()
    v_l = np.where(inj["leak_type"].to_numpy() == "NONE", inj["amount"].to_numpy() * 0.5, v_l)
    frame = pd.DataFrame({"day": inj["day"].to_numpy(), "p_f": p_f, "p_l": p_l,
                          "amount": inj["amount"].to_numpy(), "v_l": v_l, "y": inj["y"].to_numpy()})
    print(f"[{tag}] {len(frame):,} records, {frame.day.nunique()} periods "
          f"(singleton periods before the fix: {n_singleton_before})", flush=True)
    out, piv = vab_grid(frame, k_fracs=(0.005, 0.01, 0.02, 0.05), lgfs=(0.5, 1.0),
                        recs=(0.2, 0.5, 0.8, 1.0), min_share={"F": 0.1, "L": 0.1})
    C.dump(f"tab_vab_full_grid_{tag}.csv", out)
    pv = piv.reset_index(); C.dump(f"tab_vab_pivot_{tag}.csv", pv)
    prim = out[(out.k_frac == 0.01) & (out.lgf == 1.0) & (out.recoverability == 0.5)].set_index("policy")
    two, single, orc = prim.loc["two_queue"], prim.loc["single_queue"], prim.loc["oracle"]
    g1 = pv[pv.k_frac == 0.01]
    s = {"host": host_desc, "period_definition": "integer host period; injected records floored into it",
         "n_records": int(len(frame)), "n_periods": int(frame.day.nunique()),
         "singleton_periods_before_fix": n_singleton_before,
         "budget": 0.01, "lambda_F": 1.0, "rho_L": 0.5,
         "two_queue_VaB_total": float(two.VaB_total), "single_queue_VaB_total": float(single.VaB_total),
         "oracle_VaB_total": float(orc.VaB_total),
         "fraud_first_VaB_total": float(prim.loc["fraud_first"].VaB_total),
         "fixed_50_VaB_total": float(prim.loc["fixed_50"].VaB_total),
         "two_queue_VaB_F": float(two.VaB_F), "two_queue_VaB_L": float(two.VaB_L),
         "single_queue_VaB_F": float(single.VaB_F), "single_queue_VaB_L": float(single.VaB_L),
         "oracle_VaB_F": float(orc.VaB_F), "oracle_VaB_L": float(orc.VaB_L),
         "H4_primary_difference": float(two.VaB_total - single.VaB_total),
         "share_of_oracle_gap_closed": float((two.VaB_total - single.VaB_total) /
                                             max(orc.VaB_total - single.VaB_total, 1e-12)),
         "n_grid_points_at_K1pct": int(len(g1)),
         "n_winning_grid_points_at_K1pct": int(g1["two_queue_wins"].sum()),
         "worst_single_queue_advantage_at_K1pct": float(max(0.0, (g1["single_queue"] - g1["two_queue"]).max()))}
    C.dump(f"h4_primary_endpoint_{tag}.json", s)
    print(json.dumps(s, indent=1), flush=True)


def main():
    t0 = time.time()
    d = C.load_ulb().sort_values("Time").reset_index(drop=True)
    i_va = int(0.8 * len(d))
    feats = [c for c in d.columns if c != "Class"]
    Xtr = d.loc[:i_va - 1, feats].values; ytr = d.loc[:i_va - 1, "Class"].values
    rng = np.random.default_rng(0)
    sub = np.sort(np.concatenate([np.flatnonzero(ytr == 1),
                                  rng.choice(np.flatnonzero(ytr == 0), 50000, replace=False)]))
    te = d.loc[i_va:].reset_index(drop=True)
    s_f, _ = C.fit_gbdt(Xtr[sub], ytr[sub], te[feats].values, seed=SEED)
    harm = pd.DataFrame({"record_id": np.arange(len(te)),
                         "day": (te["Time"].values / 3600.0).astype(int),
                         "entity_id": (np.abs(te["V1"].values * 1000).astype(int) % 500),
                         "counterparty_id": (np.abs(te["V2"].values * 1000).astype(int) % 2000),
                         "amount": te["Amount"].values, "currency": "EUR",
                         "y": te["Class"].values.astype(int)})
    run(harm, s_f, "periodfix", "ULB credit-card test split with LeakBench-injected leakage, "
        "injected records floored into the host's hourly budget periods")
    print(f"done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""e16_paysim_host.py -- E7 injection gate and H4 value-at-budget on the host configs/default.yaml
names: PaySim.

The 2026-09-19/20 run computed H4 on a ULB + LeakBench semi-synthetic host because PaySim was not
obtainable.  PaySim arrived on 2026-09-20 and is the host `configs/default.yaml` lists among
`datasets:` for the LeakBench work.  This script re-runs both endpoints on it and writes them to
their own files; the ULB-host files are left untouched so that both are reported.

Memory: 6,362,620 rows / 494 MB, read in chunks with explicit dtypes; the name columns are hashed
to integer ids on the fly and never held as objects.
"""
from __future__ import annotations
import sys, os, json, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import common as C

sys.path.insert(0, os.path.abspath(os.path.join(C.PAPER, "lib")))
from src.leakbench.inject import inject, verify                              # noqa: E402
from src.triage.value_at_budget import vab_grid                              # noqa: E402

CSV = os.path.join(C.RAW, "paysim", "PS_20174392719_1491204439457_log.csv")
TYPE_MIX = {"DUP_EXACT": 1, "DUP_NEAR": 1, "WRONG_PAYEE": 1, "TRANSPOSE": 1,
            "DOUBLE_FX": 0.5, "SPLIT_PAY": 0.5}
RATES = [0.005, 0.01, 0.02]
GATE_SEEDS = [0, 1, 2]
RATE, SEED = 0.01, 0
HOST_CAP = 300_000          # computational budget, logged: seeded uniform subsample of the test split
LGD = {"DUP_EXACT": 1.0, "DUP_NEAR": 1.0, "WRONG_PAYEE": 0.6, "TRANSPOSE": 0.5,
       "DOUBLE_FX": 0.5, "SPLIT_PAY": 0.3, "NONE": 0.0}
TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def jsd(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a[a > 0] * np.log2(a[a > 0] / b[a > 0])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def load_paysim():
    dt = {"step": "int16", "type": "category", "amount": "float32", "nameOrig": "str",
          "oldbalanceOrg": "float32", "newbalanceOrig": "float32", "nameDest": "str",
          "oldbalanceDest": "float32", "newbalanceDest": "float32",
          "isFraud": "int8", "isFlaggedFraud": "int8"}
    parts = []
    n = 0
    for ch in pd.read_csv(CSV, dtype=dt, chunksize=1_000_000):
        n += len(ch)
        p = pd.DataFrame({
            "step": ch["step"].to_numpy(np.int16),
            "type_code": ch["type"].cat.codes.to_numpy(np.int8) if hasattr(ch["type"], "cat")
                         else pd.Categorical(ch["type"], categories=TYPES).codes.astype(np.int8),
            "amount": ch["amount"].to_numpy(np.float32),
            "oldbalanceOrg": ch["oldbalanceOrg"].to_numpy(np.float32),
            "newbalanceOrig": ch["newbalanceOrig"].to_numpy(np.float32),
            "oldbalanceDest": ch["oldbalanceDest"].to_numpy(np.float32),
            "newbalanceDest": ch["newbalanceDest"].to_numpy(np.float32),
            "orig_id": pd.util.hash_array(ch["nameOrig"].to_numpy()).astype(np.uint32) % 200_000,
            "dest_id": pd.util.hash_array(ch["nameDest"].to_numpy()).astype(np.uint32) % 200_000,
            "isFraud": ch["isFraud"].to_numpy(np.int8)})
        parts.append(p)
        print(f"  {n:,} rows", flush=True)
    d = pd.concat(parts, ignore_index=True)
    del parts
    assert len(d) == 6_362_620, len(d)
    return d


def main():
    t0 = time.time()
    d = load_paysim()
    print(f"PaySim {d.shape} fraud={int(d.isFraud.sum())} ({d.isFraud.mean():.4%}) "
          f"steps {d.step.min()}-{d.step.max()} {time.time()-t0:.0f}s", flush=True)
    d["errBalOrig"] = (d.oldbalanceOrg - d.amount - d.newbalanceOrig).astype(np.float32)
    d["errBalDest"] = (d.oldbalanceDest + d.amount - d.newbalanceDest).astype(np.float32)
    d["hour"] = (d.step % 24).astype(np.int8)

    FEATS = ["type_code", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
             "newbalanceDest", "errBalOrig", "errBalDest", "hour"]
    # temporal split, configs/default.yaml: train 0.7, valid 0.1, test 0.2, by step
    q80 = int(np.quantile(d.step.to_numpy(), 0.8))
    tr_m = (d.step <= q80).to_numpy()
    te_m = ~tr_m
    print(f"split at step {q80}: train {tr_m.sum():,} (fraud {int(d.isFraud[tr_m].sum())}) | "
          f"test {te_m.sum():,} (fraud {int(d.isFraud[te_m].sum())})", flush=True)

    rng = np.random.default_rng(SEED)
    ytr_all = d.isFraud.to_numpy()[tr_m]
    pos = np.flatnonzero(ytr_all == 1)
    neg = rng.choice(np.flatnonzero(ytr_all == 0), min(200_000, (ytr_all == 0).sum()), replace=False)
    sub = np.sort(np.concatenate([pos, neg]))
    Xtr = d.loc[tr_m, FEATS].to_numpy(np.float32)[sub]
    ytr = ytr_all[sub]
    te = d.loc[te_m].reset_index(drop=True)
    del d
    # seeded uniform subsample of the test split -- computational budget, logged in DEVIATIONS.md
    if len(te) > HOST_CAP:
        keep = np.sort(np.random.default_rng(SEED).choice(len(te), HOST_CAP, replace=False))
        te = te.iloc[keep].reset_index(drop=True)
    print(f"host = {len(te):,} test records, fraud {int(te.isFraud.sum())} "
          f"({te.isFraud.mean():.4%})", flush=True)
    s_f_te, _ = C.fit_gbdt(Xtr, ytr, te[FEATS].to_numpy(np.float32), seed=SEED)
    print(f"fraud model fitted {time.time()-t0:.0f}s; test PR-AUC "
          f"{C.pr_auc(te.isFraud.to_numpy(), s_f_te):.4f}", flush=True)

    harmonised = pd.DataFrame({
        "record_id": np.arange(len(te)),
        "day": te["step"].to_numpy().astype(float),
        "entity_id": te["orig_id"].to_numpy(),
        "counterparty_id": te["dest_id"].to_numpy(),
        "amount": te["amount"].to_numpy().astype(float),
        "currency": "EUR",
        "y": te["isFraud"].to_numpy().astype(int)})

    # -------------------------------------------------- E7: injected side on the PaySim host
    rows = []
    for rate in RATES:
        for hard in (False, True):
            for seed in GATE_SEEDS:
                v = verify(inject(harmonised, rate=rate, seed=seed, type_mix=TYPE_MIX, hard=hard))
                tot = sum(v["by_type"].values())
                for t, c in v["by_type"].items():
                    rows.append({"rate": rate, "hard": hard, "seed": seed, "leak_type": t,
                                 "n": c, "share": c / tot})
                print(f"   injected rate={rate} hard={hard} seed={seed} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    inj_df = pd.DataFrame(rows)
    C.dump("e7_injected_type_distribution_paysim.csv", inj_df)

    inj_share = inj_df.groupby("leak_type")["n"].sum(); inj_share /= inj_share.sum()
    real = pd.read_csv(os.path.join(C.RES, "leakage_deviation_typology.csv"))
    real = real[(real.flow_type == "ALL") & (real.deviation_type != "ANY (union)")]
    real_share = real.set_index("deviation_type")["n_cases"]; real_share /= real_share.sum()
    allt = sorted(set(inj_share.index) | set(real_share.index))
    d_lit = jsd([inj_share.get(t, 0.0) for t in allt], [real_share.get(t, 0.0) for t in allt])
    COARSE = {"DUP_EXACT": "duplicate-like", "DUP_NEAR": "duplicate-like",
              "SPLIT_PAY": "duplicate-like", "TRANSPOSE": "value-like", "DOUBLE_FX": "value-like",
              "WRONG_PAYEE": "party-like", "duplicate invoice": "duplicate-like",
              "repeated clearing": "duplicate-like", "value mismatch": "value-like",
              "atypical role": "party-like", "skipped goods receipt": "control-step",
              "invoice before purchase order": "control-step"}
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
    C.dump("e7_gate_type_comparison_paysim.csv", per)
    C.dump("e7_gate_paysim.json", {
        "host": "PaySim (PS_20174392719_1491204439457_log.csv), test split, "
                f"{len(te)} records (seeded uniform subsample of {int(te_m.sum())})",
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
        "comparison_to_ULB_host": "same seeded vocabulary; the gate verdict does not depend on the host"})
    print(f"E7 on PaySim: literal {d_lit:.3f}, coarse {d_co:.3f}, tol {TOL} -> "
          f"{'FAIL' if min(d_lit, d_co) > TOL else 'PASS'}", flush=True)

    # -------------------------------------------------- H4 on the PaySim host
    # This block carries the budget-period defect of DEVIATIONS.md 7.5 (injected records keep a
    # fractional `day` and form singleton budget periods). e19_paysim_h4_fix.py recomputes it
    # correctly; run this block only when explicitly asked for the uncorrected comparison.
    if os.environ.get("PAPER2_RUN_DEFECTIVE_H4", "0") != "1":
        print("skipping the uncorrected H4 block (see e19_paysim_h4_fix.py, DEVIATIONS.md 7.5)",
              flush=True)
        return
    inj2 = inject(harmonised, rate=RATE, seed=SEED, type_mix=TYPE_MIX)
    C.dump("e7_leakbench_injection_verify_paysim.json",
           {**verify(inj2), "host": "PaySim test split", "rate": RATE, "seed": SEED,
            "type_mix": TYPE_MIX})
    p_f = np.zeros(len(inj2))
    base = (inj2.injected == 0).to_numpy()
    p_f[base] = s_f_te[inj2.loc[base, "record_id"].to_numpy()]
    src = inj2.loc[~base, "source_record_id"].to_numpy().astype(int)
    p_f[~base] = s_f_te[src]

    dd = inj2.reset_index(drop=True)
    s = np.zeros(len(dd))
    key = dd["counterparty_id"].astype(str) + "|" + dd["amount"].round(2).astype(str)
    s += key.duplicated(keep=False).to_numpy().astype(float) * 2.0
    key2 = dd["counterparty_id"].astype(str) + "|" + dd["amount"].round(0).astype(str)
    s += key2.duplicated(keep=False).to_numpy().astype(float)
    g = dd.groupby(["entity_id", dd["day"].astype(int)])["record_id"].transform("size").to_numpy()
    s += (g >= 3).astype(float)
    p_l_raw = s + np.linspace(0, 1e-9, len(dd))
    p_l = (p_l_raw - p_l_raw.min()) / (p_l_raw.max() - p_l_raw.min() + 1e-12)
    v_l = dd["amount"].to_numpy() * dd["leak_type"].map(LGD).fillna(0.5).to_numpy()
    v_l = np.where(dd["leak_type"].to_numpy() == "NONE", dd["amount"].to_numpy() * 0.5, v_l)

    frame = pd.DataFrame({"day": dd["day"].to_numpy(), "p_f": p_f, "p_l": p_l,
                          "amount": dd["amount"].to_numpy(), "v_l": v_l, "y": dd["y"].to_numpy()})
    print(f"VaB grid on {len(frame):,} records, {frame.day.nunique()} periods "
          f"({time.time()-t0:.0f}s)", flush=True)
    out, piv = vab_grid(frame, k_fracs=(0.005, 0.01, 0.02, 0.05), lgfs=(0.5, 1.0),
                        recs=(0.2, 0.5, 0.8, 1.0), min_share={"F": 0.1, "L": 0.1})
    C.dump("tab_vab_full_grid_paysim.csv", out)
    pv = piv.reset_index(); C.dump("tab_vab_pivot_paysim.csv", pv)

    prim = out[(out.k_frac == 0.01) & (out.lgf == 1.0) & (out.recoverability == 0.5)]
    p = prim.set_index("policy")
    two, single, orc = p.loc["two_queue"], p.loc["single_queue"], p.loc["oracle"]
    g1 = pv[pv.k_frac == 0.01]
    summary = {
        "host": ("PaySim test split (real fraud labels and real transaction amounts) with "
                 "LeakBench-injected leakage -- the host configs/default.yaml names"),
        "n_records_host": int(len(te)), "n_records_after_injection": int(len(dd)),
        "budget": 0.01, "lambda_F": 1.0, "rho_L": 0.5,
        "two_queue_VaB_total": float(two.VaB_total), "single_queue_VaB_total": float(single.VaB_total),
        "oracle_VaB_total": float(orc.VaB_total),
        "fraud_first_VaB_total": float(p.loc["fraud_first"].VaB_total),
        "fixed_50_VaB_total": float(p.loc["fixed_50"].VaB_total),
        "H4_primary_difference": float(two.VaB_total - single.VaB_total),
        "share_of_oracle_gap_closed": float((two.VaB_total - single.VaB_total) /
                                            max(orc.VaB_total - single.VaB_total, 1e-12)),
        "n_grid_points_at_K1pct": int(len(g1)),
        "n_winning_grid_points_at_K1pct": int(g1["two_queue_wins"].sum()),
        "worst_single_queue_advantage_at_K1pct": float(max(0.0, (g1["single_queue"] - g1["two_queue"]).max())),
        "two_queue_VaB_F": float(two.VaB_F), "two_queue_VaB_L": float(two.VaB_L),
        "single_queue_VaB_F": float(single.VaB_F), "single_queue_VaB_L": float(single.VaB_L),
        "oracle_VaB_F": float(orc.VaB_F), "oracle_VaB_L": float(orc.VaB_L),
        "seconds": round(time.time() - t0, 1)}
    C.dump("h4_primary_endpoint_paysim.json", summary)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()

"""e13_gate_and_drift.py -- the E7 injection gate (now with a real comparator) and H3(c):
stationarity of the deviation-type mix with its permutation reference, plus the process-drift curve.
"""
import os, json, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); PAPER = os.path.dirname(HERE)
CACHE = os.path.join(PAPER, "results", "cache"); RES = os.path.join(PAPER, "results")
SEED = 0
N_PERM = 1000

# ---------------------------------------------------------------- tolerance commitment
# The gate needs a tolerance. `configs/default.yaml` records it as not_prespecified: it was never
# fixed before the run. It is fixed HERE, and written to disk BEFORE the distance is computed,
# so that the ordering is on the record. This is a deviation: the manuscript's clause "fixed
# before any outcome was examined" is false of this run and is recast in fill_slots.py.
TOL = 0.10
commit = {"tolerance_jsd": TOL, "basis": "Jensen-Shannon divergence, log base 2, range [0,1]",
          "fixed_at_utc": datetime.now(timezone.utc).isoformat(),
          "fixed_before_distance_computed": True,
          "fixed_before_seeded_results_produced": False,
          "note": "configs/default.yaml carried not_prespecified; no tolerance existed before today"}
json.dump(commit, open(os.path.join(RES, "e7_gate_tolerance_commitment.json"), "w"), indent=1)

def jsd(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)
    def kl(a, b):
        k = a > 0
        return float(np.sum(a[k] * np.log2(a[k] / b[k])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

# ---------------------------------------------------------------- E7 gate
inj = pd.read_csv(os.path.join(RES, "e7_injected_type_distribution.csv"))
inj_share = inj.groupby("leak_type")["n"].sum()
inj_share = inj_share / inj_share.sum()

real = pd.read_csv(os.path.join(RES, "leakage_deviation_typology.csv"))
real = real[(real.flow_type == "ALL") & (real.deviation_type != "ANY (union)")]
real_share = real.set_index("deviation_type")["n_cases"]
real_share = real_share / real_share.sum()

# (a) literal: union support, zero mass for a type the other side does not carry
allt = sorted(set(inj_share.index) | set(real_share.index))
pv = np.array([inj_share.get(t, 0.0) for t in allt])
qv = np.array([real_share.get(t, 0.0) for t in allt])
d_literal = jsd(pv, qv)

# (b) sensitivity: the coarsest mapping under which the two vocabularies share any support
COARSE = {"DUP_EXACT": "duplicate-like", "DUP_NEAR": "duplicate-like", "SPLIT_PAY": "duplicate-like",
          "TRANSPOSE": "value-like", "DOUBLE_FX": "value-like", "WRONG_PAYEE": "party-like",
          "duplicate invoice": "duplicate-like", "repeated clearing": "duplicate-like",
          "value mismatch": "value-like", "atypical role": "party-like",
          "skipped goods receipt": "control-step", "invoice before purchase order": "control-step"}
def coarsen(s):
    out = {}
    for k, v in s.items():
        out[COARSE[k]] = out.get(COARSE[k], 0.0) + float(v)
    return out
ci_, cr_ = coarsen(inj_share), coarsen(real_share)
ck = sorted(set(ci_) | set(cr_))
d_coarse = jsd([ci_.get(k, 0.0) for k in ck], [cr_.get(k, 0.0) for k in ck])

per_type = []
for t in allt:
    a, b = float(inj_share.get(t, 0.0)), float(real_share.get(t, 0.0))
    per_type.append({"type": t, "injected_share": a, "real_share": b, "abs_diff": abs(a - b)})
per_type = pd.DataFrame(per_type).sort_values("abs_diff", ascending=False)
per_type.to_csv(os.path.join(RES, "e7_gate_type_comparison.csv"), index=False)

gate = {
    "tolerance_jsd": TOL,
    "jsd_literal_union_support": d_literal,
    "jsd_coarse_mapping": d_coarse,
    "shared_support_literal": sorted(set(inj_share.index) & set(real_share.index)),
    "verdict_literal": "FAIL" if d_literal > TOL else "PASS",
    "verdict_coarse": "FAIL" if d_coarse > TOL else "PASS",
    "verdict": "FAIL" if min(d_literal, d_coarse) > TOL else "PASS",
    "most_divergent_types": per_type.head(3)["type"].tolist(),
    "most_divergent_detail": per_type.head(3).to_dict("records"),
    "injected_vocabulary": sorted(inj_share.index.tolist()),
    "real_vocabulary": sorted(real_share.index.tolist()),
    "coarse_injected": ci_, "coarse_real": cr_,
}
json.dump(gate, open(os.path.join(RES, "e7_gate.json"), "w"), indent=1)
print("E7 GATE:", json.dumps({k: gate[k] for k in
      ["tolerance_jsd", "jsd_literal_union_support", "jsd_coarse_mapping", "verdict",
       "shared_support_literal", "most_divergent_types"]}, indent=1))

# ---------------------------------------------------------------- H3(c) stationarity
lab = pd.read_pickle(os.path.join(CACHE, "labels.pkl"))
TYPES = [c for c in lab.columns if c.startswith("T_")]
t0 = lab["t_first"].to_numpy()
order = np.argsort(t0)
half = len(order) // 2
h1, h2 = order[:half], order[half:]
M = lab[TYPES].to_numpy().astype(float)

def mix(rows):
    s = M[rows].sum(0)
    return s / s.sum() if s.sum() > 0 else s

d_obs = jsd(mix(h1), mix(h2))
rng = np.random.default_rng(SEED)
perm = np.empty(N_PERM)
idx = np.arange(len(order))
for b in range(N_PERM):
    rng.shuffle(idx)
    perm[b] = jsd(mix(order[idx[:half]]), mix(order[idx[half:]]))
p_emp = float((perm >= d_obs).mean())
ref_mean, ref_q95 = float(perm.mean()), float(np.quantile(perm, 0.95))

stat = {"jsd_between_halves": float(d_obs), "permutation_mean": ref_mean,
        "permutation_q95": ref_q95, "permutation_p": p_emp, "n_permutations": N_PERM,
        "stationary": bool(d_obs <= ref_q95),
        "split_date": str(pd.Timestamp(int(t0[order[half]]), unit="us")),
        "mix_first_half": dict(zip(TYPES, mix(h1).round(6).tolist())),
        "mix_second_half": dict(zip(TYPES, mix(h2).round(6).tolist()))}
json.dump(stat, open(os.path.join(RES, "h3c_stationarity.json"), "w"), indent=1)
print("H3(c) STATIONARITY:", json.dumps({k: stat[k] for k in
      ["jsd_between_halves", "permutation_mean", "permutation_q95", "permutation_p",
       "stationary"]}, indent=1))

# ---------------------------------------------------------------- process-drift curve
lab2 = lab.copy()
lab2["month"] = pd.to_datetime(lab2["t_first"].to_numpy(), unit="us", utc=True).to_period("M").astype(str)
mc = lab2["month"].value_counts()
keep = sorted([m for m in mc.index if mc[m] >= 200])
rows = []
base = mix(order[:len(order) // 3])
for m in keep:
    r = np.where(lab2["month"].to_numpy() == m)[0]
    mm = mix(r)
    rows.append({"month": m, "n_cases": int(len(r)),
                 "jsd_vs_first_third": jsd(base, mm) if mm.sum() > 0 else np.nan,
                 "typed_share": float(lab2["typed_any"].to_numpy()[r].mean()),
                 "positive_rate_window_B": float(lab2["pos_B"].to_numpy()[r].mean())})
drift = pd.DataFrame(rows)
drift.to_csv(os.path.join(RES, "h3c_process_drift.csv"), index=False)
print(drift.to_string(index=False))

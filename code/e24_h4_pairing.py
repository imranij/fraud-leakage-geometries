"""e24_h4_pairing.py -- H4 value-at-budget on the pre-specified pairing: BPI Challenge 2019
(leakage) x IEEE-CIS (fraud), paired by day.

Why this file exists.  The design's H4 host is the real leakage log paired with a real fraud file.
Neither half was in hand on 2026-09-19, so H4 was computed on a ULB + LeakBench semi-synthetic
host (`e3_value_at_budget.py`, corrected by `e18_h4_period_fix.py`) and, later, on PaySim
(`e19_paysim_h4_fix.py`).  BPI 2019 arrived on 2026-09-20 (morning) and IEEE-CIS on 2026-09-20
(afternoon).  This is the pairing the design specifies, computed for the first time.

Construction.

* Fraud queue.  IEEE-CIS, the cached temporal test split of `e20_ieee_dissociation.py`
  (147,635 transactions, 5,100 fraudulent, TransactionDT above its 75th percentile).  p_F is a
  LightGBM fitted on the training window only, exactly the e20 fit.  Value = TransactionAmt, an
  observed monetary amount.  y = 1 on fraud.
* Leakage queue.  BPI 2019 purchase-order items, outcome-derived labels (window B, `e10`), split
  temporally on the case's first event at the 75th percentile of that timestamp -- the same 75/25
  temporal proportion as the fraud side.  p_L is a LightGBM on the record-only case attributes of
  `e12_leakage_detectors.py`, fitted on the training window only.  Value = the case's cumulative
  net worth (EUR).  y = 2 on an outcome-labelled positive.
* Scoring across channels.  A card transaction carries no purchase-order trace and a
  purchase-order item carries no card features, so each record is scored by the channel that
  applies to it: p_L = 0 on fraud records and p_F = 0 on leakage records.  This is what makes the
  two queues two queues.
* Pairing by day.  The two files are not contemporaneous and IEEE-CIS publishes TransactionDT as
  a delta in seconds with no calendar origin, so calendar pairing is impossible.  The streams are
  paired by DAY RANK: the fraud test window's 53 distinct days are the 53 budget periods, and the
  leakage test window's days are distributed onto that grid in time order in proportion to their
  number (leakage day of rank r goes to period floor(r * 53 / n_leakage_days)).  Every record on
  both sides is used and every period holds records of both intents.
* Period fix.  The defect of `e18_h4_period_fix.py` -- injected records receiving a fractional
  `day` and so each becoming a singleton budget period with VaB_L = 1.000 everywhere -- cannot
  recur here, because nothing is injected: both sides are real records floored onto an integer
  period grid.  The script asserts it anyway: zero singleton periods, and VaB_L not identically 1.

Value scale.  The fraud side is in USD (median transaction 68.77) and the leakage side in EUR
(median case 534, mean 6,198, maximum 28,994,530).  The policies rank on expected value, so the
relative scale of the two sides is part of the pairing and not a nuisance.  The PRIMARY run reads
the two at face value (1 EUR = 1 USD, no conversion), which is the pairing as specified.  A
SECONDARY run, declared here before either was computed, rescales the leakage values so that the
total leakage value available equals the total fraud value available, which is the scale-free
version of the same comparison.  Both are reported whatever they show.

Outputs carry the `h4pair_` prefix.  No existing results file is touched.
"""
from __future__ import annotations
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import common as C

sys.path.insert(0, os.path.abspath(os.path.join(C.PAPER, "lib")))
from src.triage.value_at_budget import vab_grid                    # noqa: E402

CACHE = os.path.join(C.PAPER, "results", "cache")
SEED = 0
K_FRACS = (0.005, 0.01, 0.02, 0.05)
LGFS = (0.5, 1.0)
RECS = (0.2, 0.5, 0.8, 1.0)
MIN_SHARE = {"F": 0.1, "L": 0.1}


# ------------------------------------------------------------------ fraud side
def fraud_side():
    Xtr = np.load(os.path.join(CACHE, "ieee_Xtr.npy"), mmap_mode="r")
    ytr = np.load(os.path.join(CACHE, "ieee_ytr.npy"))
    t0 = time.time()
    _, model = C.fit_gbdt(np.asarray(Xtr), ytr, np.asarray(Xtr[:1]), seed=SEED)
    del Xtr
    print(f"  IEEE GBDT fitted {time.time()-t0:.0f}s", flush=True)
    Xte = np.load(os.path.join(CACHE, "ieee_Xte.npy"), mmap_mode="r")
    n = Xte.shape[0]
    p = np.empty(n, dtype=np.float64)
    for a in range(0, n, 25000):
        b = min(a + 25000, n)
        p[a:b] = model.predict_proba(np.asarray(Xte[a:b]))[:, 1]
    del Xte
    yte = np.load(os.path.join(CACHE, "ieee_yte.npy")).astype(int)
    vte = np.load(os.path.join(CACHE, "ieee_vte.npy")).astype(float)
    tdt = np.load(os.path.join(CACHE, "ieee_tdt_te.npy"))
    day = (tdt // 86400).astype(np.int64)
    print(f"  IEEE test {n:,} rows, {int(yte.sum())} fraud, PR-AUC "
          f"{C.pr_auc(yte, p):.4f}, {len(np.unique(day))} days", flush=True)
    return pd.DataFrame({"raw_day": day, "p_f": p, "amount": vte, "y": yte}), float(C.pr_auc(yte, p))


# ------------------------------------------------------------------ leakage side
def leakage_side():
    lab = pd.read_pickle(os.path.join(CACHE, "labels.pkl"))
    cases = pd.read_pickle(os.path.join(CACHE, "cases.pkl"))
    rec = pd.DataFrame({
        "value": np.nan_to_num(cases["value_absmax"].to_numpy()),
        "log_value": np.log1p(np.nan_to_num(np.abs(cases["value_absmax"].to_numpy()))),
        "month": pd.to_datetime(cases["t_first"].to_numpy(), unit="us", utc=True).month,
        "gr_based": (cases["gr_based_inv_verif"].to_numpy() == "True").astype(int),
        "gr_flag": (cases["goods_receipt_flag"].to_numpy() == "True").astype(int)})
    for col in ["vendor", "spend_area", "doc_type", "item_type", "company"]:
        v = cases[col].astype("category")
        rec[col + "_code"] = v.cat.codes
        freq = v.value_counts(normalize=True)
        rec[col + "_freq"] = v.map(freq).astype(float).to_numpy()
    X = np.nan_to_num(rec.to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = lab["pos_B"].to_numpy().astype(int)
    t = lab["t_first"].to_numpy()
    cut = float(np.quantile(t, 0.75))
    tr, te = t <= cut, t > cut
    print(f"  BPI2019 cases {len(lab):,}; train {int(tr.sum()):,} ({int(y[tr].sum())} pos) / "
          f"test {int(te.sum()):,} ({int(y[te].sum())} pos); split at "
          f"{pd.Timestamp(int(cut), unit='us')}", flush=True)
    t0 = time.time()
    p, _ = C.fit_gbdt(X[tr], y[tr], X[te], seed=SEED)
    print(f"  BPI2019 GBDT fitted {time.time()-t0:.0f}s, test PR-AUC {C.pr_auc(y[te], p):.4f}",
          flush=True)
    # labels.pkl stores t_first in MICROSECONDS since the epoch (verified: max 1547735520000000
    # -> 2019-01-17, the log's last day). Calendar day index = us // us-per-day.
    day = (t[te] // 86_400_000_000).astype(np.int64)
    val = np.abs(np.nan_to_num(lab["value"].to_numpy()[te])).astype(float)
    return (pd.DataFrame({"raw_day": day, "p_l": p, "v_l": val, "y": (y[te] * 2).astype(int),
                          "flow_type": lab["flow_type"].to_numpy()[te]}),
            float(C.pr_auc(y[te], p)), str(pd.Timestamp(int(cut), unit="us")))


# ------------------------------------------------------------------ pairing
def pair_by_day(F, L):
    fdays = np.sort(F["raw_day"].unique())
    ldays = np.sort(L["raw_day"].unique())
    P = len(fdays)
    f_period = {d: i for i, d in enumerate(fdays)}
    l_period = {d: min(int(np.floor(r * P / len(ldays))), P - 1) for r, d in enumerate(ldays)}
    F = F.assign(day=F["raw_day"].map(f_period).astype(int))
    L = L.assign(day=L["raw_day"].map(l_period).astype(int))
    return F, L, P, len(fdays), len(ldays)


def build_frame(F, L, leak_scale=1.0):
    a = pd.DataFrame({"day": F["day"].to_numpy(), "p_f": F["p_f"].to_numpy(),
                      "p_l": 0.0, "amount": F["amount"].to_numpy(),
                      "v_l": 0.0, "y": F["y"].to_numpy()})
    b = pd.DataFrame({"day": L["day"].to_numpy(), "p_f": 0.0, "p_l": L["p_l"].to_numpy(),
                      "amount": L["v_l"].to_numpy() * leak_scale,
                      "v_l": L["v_l"].to_numpy() * leak_scale, "y": L["y"].to_numpy()})
    return pd.concat([a, b], ignore_index=True)


def evaluate(frame, tag, meta):
    sizes = frame.groupby("day").size()
    n_singleton = int((sizes == 1).sum())
    print(f"[{tag}] {len(frame):,} records, {frame.day.nunique()} periods, "
          f"min/median/max period size {sizes.min()}/{int(sizes.median())}/{sizes.max()}, "
          f"singleton periods {n_singleton}", flush=True)
    out, piv = vab_grid(frame, k_fracs=K_FRACS, lgfs=LGFS, recs=RECS, min_share=MIN_SHARE)
    C.dump(f"h4pair_full_grid_{tag}.csv", out)
    pv = piv.reset_index()
    C.dump(f"h4pair_pivot_{tag}.csv", pv)

    vabl = out[out.policy.isin(["oracle", "single_queue", "two_queue", "fraud_first", "fixed_50"])]
    vabl_identically_one = bool(np.allclose(vabl["VaB_L"].to_numpy(), 1.0))
    prim = out[(out.k_frac == 0.01) & (out.lgf == 1.0) & (out.recoverability == 0.5)].set_index("policy")
    two, single, orc = prim.loc["two_queue"], prim.loc["single_queue"], prim.loc["oracle"]
    g1 = pv[pv.k_frac == 0.01]
    s = {"tag": tag, **meta,
         "n_records": int(len(frame)), "n_periods": int(frame.day.nunique()),
         "singleton_periods": n_singleton,
         "min_period_size": int(sizes.min()), "max_period_size": int(sizes.max()),
         "VaB_L_identically_one": vabl_identically_one,
         "VaB_L_min_over_grid": float(vabl["VaB_L"].min()),
         "VaB_L_max_over_grid": float(vabl["VaB_L"].max()),
         "budget": 0.01, "lambda_F": 1.0, "rho_L": 0.5,
         "two_queue_VaB_total": float(two.VaB_total),
         "single_queue_VaB_total": float(single.VaB_total),
         "oracle_VaB_total": float(orc.VaB_total),
         "fraud_first_VaB_total": float(prim.loc["fraud_first"].VaB_total),
         "fixed_50_VaB_total": float(prim.loc["fixed_50"].VaB_total),
         "two_queue_VaB_F": float(two.VaB_F), "two_queue_VaB_L": float(two.VaB_L),
         "single_queue_VaB_F": float(single.VaB_F), "single_queue_VaB_L": float(single.VaB_L),
         "oracle_VaB_F": float(orc.VaB_F), "oracle_VaB_L": float(orc.VaB_L),
         "fraud_first_VaB_F": float(prim.loc["fraud_first"].VaB_F),
         "fraud_first_VaB_L": float(prim.loc["fraud_first"].VaB_L),
         "fixed_50_VaB_F": float(prim.loc["fixed_50"].VaB_F),
         "fixed_50_VaB_L": float(prim.loc["fixed_50"].VaB_L),
         "two_queue_captured_share_of_available": float(two.VaB_total / 2.0),
         "single_queue_captured_share_of_available": float(single.VaB_total / 2.0),
         "oracle_captured_share_of_available": float(orc.VaB_total / 2.0),
         "H4_primary_difference": float(two.VaB_total - single.VaB_total),
         "share_of_oracle_gap_closed": float((two.VaB_total - single.VaB_total) /
                                             max(orc.VaB_total - single.VaB_total, 1e-12)),
         "n_grid_points_at_K1pct": int(len(g1)),
         "n_winning_grid_points_at_K1pct": int(g1["two_queue_wins"].sum()),
         "worst_single_queue_advantage_at_K1pct": float(
             max(0.0, (g1["single_queue"] - g1["two_queue"]).max())),
         "alert_precision_two_queue": float(two.alert_precision),
         "alert_precision_single_queue": float(single.alert_precision)}
    C.dump(f"h4pair_primary_endpoint_{tag}.json", s)
    print(json.dumps(s, indent=1), flush=True)
    return s


def main():
    t00 = time.time()
    F, pr_f = fraud_side()
    L, pr_l, split_at = leakage_side()
    F, L, P, nfd, nld = pair_by_day(F, L)
    tot_f = float(F.loc[F.y == 1, "amount"].sum())
    tot_l = float(L.loc[L.y == 2, "v_l"].sum())
    comp = (pd.concat([F.groupby("day").size().rename("n_fraud_records"),
                       L.groupby("day").size().rename("n_leakage_records"),
                       F.groupby("day")["y"].apply(lambda z: int((z == 1).sum())).rename("n_fraud_pos"),
                       L.groupby("day")["y"].apply(lambda z: int((z == 2).sum())).rename("n_leak_pos")],
                      axis=1).fillna(0).astype(int).reset_index())
    C.dump("h4pair_period_composition.csv", comp)
    meta = {"fraud_host": "IEEE-CIS test split (e20 temporal split), value = TransactionAmt (USD)",
            "leakage_host": "BPI Challenge 2019 purchase-order items, outcome labels window B, "
                            "test window = case first event after its 75th percentile, "
                            "value = cumulative net worth (EUR)",
            "pairing": f"by day rank: {nfd} fraud days are the budget periods; {nld} leakage days "
                       f"distributed onto them in time order in proportion",
            "leakage_split_at": split_at,
            "fraud_model_test_pr_auc": pr_f, "leakage_model_test_pr_auc": pr_l,
            "total_fraud_value_available": tot_f, "total_leakage_value_available": tot_l,
            "period_fix": "no injection; both sides are real records on an integer period grid"}
    print(f"pairing: {P} periods; total fraud value {tot_f:,.0f} USD, "
          f"total leakage value {tot_l:,.0f} EUR", flush=True)

    s_primary = evaluate(build_frame(F, L, 1.0), "primary",
                         {**meta, "value_scale": "face value, 1 EUR = 1 USD (the pairing as specified)"})
    scale = tot_f / tot_l
    s_scaled = evaluate(build_frame(F, L, scale), "equalscale",
                        {**meta, "value_scale": f"leakage values multiplied by {scale:.6g} so that "
                                                "total leakage value equals total fraud value"})
    C.dump("h4pair_comparison.json", {
        "primary": {k: s_primary[k] for k in
                    ["two_queue_VaB_total", "single_queue_VaB_total", "oracle_VaB_total",
                     "two_queue_VaB_F", "two_queue_VaB_L", "single_queue_VaB_F",
                     "single_queue_VaB_L", "oracle_VaB_F", "oracle_VaB_L",
                     "H4_primary_difference", "share_of_oracle_gap_closed",
                     "n_winning_grid_points_at_K1pct", "worst_single_queue_advantage_at_K1pct",
                     "VaB_L_identically_one"]},
        "equal_scale": {k: s_scaled[k] for k in
                        ["two_queue_VaB_total", "single_queue_VaB_total", "oracle_VaB_total",
                         "two_queue_VaB_F", "two_queue_VaB_L", "single_queue_VaB_F",
                         "single_queue_VaB_L", "oracle_VaB_F", "oracle_VaB_L",
                         "H4_primary_difference", "share_of_oracle_gap_closed",
                         "n_winning_grid_points_at_K1pct", "worst_single_queue_advantage_at_K1pct",
                         "VaB_L_identically_one"]},
        "leak_scale_factor": scale})
    print(f"done {time.time()-t00:.0f}s", flush=True)


if __name__ == "__main__":
    main()

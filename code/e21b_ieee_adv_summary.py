"""e21b -- summarise whatever cells of the IEEE-CIS adversary grid have completed.

e21 writes results/fig5_durability_partial_IEEE.csv after every cell and the summary files only at
the end. This script builds the same summary files from the partial, so the primary endpoint is on
disk even if the full grid is cut short by the run's time budget. It reports which cells are
complete.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, common as C

PRIMARY = {"q": 0.05, "kappa": 0.1, "delta": 90}
T = 10
MUT = ("TransactionAmt;dist1;card4;card6;P_emaildomain;R_emaildomain;"
       "DeviceType;DeviceInfo;id_30;id_31")


def main():
    src = ("fig5_durability_raw_IEEE.csv"
           if os.path.exists(os.path.join(C.RES, "fig5_durability_raw_IEEE.csv"))
           else "fig5_durability_partial_IEEE.csv")
    df = pd.read_csv(os.path.join(C.RES, src))
    done = df.groupby(["q", "kappa", "retrain", "seed"])["round"].max()
    complete = done[done == T]
    print(f"source {src}: {len(complete)} of 16 (q, kappa, retrain, seed) runs reached round {T}")
    print(complete.reset_index().to_string())
    keep = df.set_index(["q", "kappa", "retrain", "seed"]).loc[complete.index].reset_index()

    prim = []
    sel = keep[(keep["q"] == PRIMARY["q"]) & (keep["kappa"] == PRIMARY["kappa"])]
    for retrain in (False, True):
        s = sel[sel.retrain == retrain]
        if s.empty:
            continue
        p = s.pivot_table(index="seed", columns="round", values="value_stolen")
        r = s.pivot_table(index="seed", columns="round", values="vw_recall")
        rc = s.pivot_table(index="seed", columns="round", values="count_recall")
        ratio = p[T] / p[0].replace(0, np.nan)
        prim.append({"dataset": "IEEE-CIS", "retrain": retrain, "T": T,
                     "V_T_over_V_0_mean": float(ratio.mean()),
                     "V_T_over_V_0_sd": float(ratio.std(ddof=1)) if len(ratio) > 1 else float("nan"),
                     "R_0_mean": float(r[0].mean()), "R_T_mean": float(r[T].mean()),
                     "R_drop": float(r[0].mean() - r[T].mean()),
                     "count_R_0_mean": float(rc[0].mean()), "count_R_T_mean": float(rc[T].mean()),
                     "monotone_nonincreasing": bool(np.all(np.diff(r.mean(axis=0).values) <= 1e-12)),
                     "n_seeds": int(len(p)),
                     "calendar_years_at_delta90": T * PRIMARY["delta"] / 365.0,
                     "V_at_two_years_over_V_0": float((p.mean(axis=0) / p[0].mean()).iloc[8]),
                     "mutable_fields": MUT,
                     "value_field": "TransactionAmt (real monetary amount)",
                     "source": src})
    C.dump("h3_primary_endpoint_IEEE.csv", pd.DataFrame(prim))

    last = keep[keep["round"] == T]
    base = last.groupby(["q", "kappa", "retrain"])["vw_recall"].mean().reset_index()
    qs = base[base.kappa == PRIMARY["kappa"]].groupby("q")["vw_recall"].mean()
    ks = base[base.q == PRIMARY["q"]].groupby("kappa")["vw_recall"].mean()
    rs = base.groupby("retrain")["vw_recall"].mean()
    row = {"dataset": "IEEE-CIS",
           "n_q_levels": int(len(qs)), "n_kappa_levels": int(len(ks)),
           "sd_across_q": float(qs.std(ddof=1)) if len(qs) > 1 else float("nan"),
           "sd_across_kappa": float(ks.std(ddof=1)) if len(ks) > 1 else float("nan"),
           "sd_across_retrain": float(rs.std(ddof=1)) if len(rs) > 1 else float("nan"),
           "note": "Delta enters only as the round-to-calendar mapping, so it produces no spread "
                   "in R_t by round index."}
    # The manuscript's "three free quantities" are q, kappa and Delta -- NOT the retraining
    # switch, which is a separate arm of the design. Delta enters only as the round-to-calendar
    # mapping and therefore contributes exactly zero spread in R_t by round index.
    row["sd_across_delta"] = 0.0
    sds = {"the plausibility quantile": row["sd_across_q"],
           "the cost parameter": row["sd_across_kappa"],
           "the labelling delay": row["sd_across_delta"]}
    varied = (len(qs) > 1) and (len(ks) > 1)
    ok = {k: v for k, v in sds.items() if v == v}
    row["dominant_free_quantity"] = max(ok, key=ok.get) if (varied and len(ok) == 3) else ""
    row["dominant_margin"] = (float(sorted(ok.values())[-1] - sorted(ok.values())[-2])
                              if varied and len(ok) == 3 else float("nan"))
    row["all_three_varied"] = bool(varied)
    row["caveat"] = ("the retraining switch produces a larger spread (%.4f) than any of the three "
                     "free quantities, but it is not one of them and the retrained arm is affected "
                     "by the in-sample defect of DEVIATIONS 8.7" % row["sd_across_retrain"])
    C.dump("h3_spread_by_free_quantity_IEEE.csv", pd.DataFrame([row]))
    print(pd.DataFrame(prim).to_string())
    print(pd.DataFrame([row]).to_string())


if __name__ == "__main__":
    main()

"""e25_kappa_cells.py -- the kappa cells of the H3 adversary grid on Medicare, so that the
`dominant free quantity` slot can be decided across all three free quantities.

The 2026-09-19 run (`e4_durability.py`) left the Medicare grid with only the q arm complete:
`results/h3_spread_by_free_quantity.csv` carries `sd_across_q` and an EMPTY `sd_across_kappa`.
This script completes the cross -- q in {0.01, 0.05, 0.10, 0.25} at the primary kappa = 0.1, and
kappa in {0, 0.1, 0.5} at the primary q = 0.05 -- with THREE seeds at every cell rather than one,
so that the spread across a free quantity can be read against the seed noise at a fixed cell.

FROZEN ARM ONLY.  `e4_durability.py` retrains the detector on the adversary's own perturbed
records and then scores those same records (DEVIATIONS.md 8.7): the retrained numbers are
in-sample and invalid, so nothing here is computed from them.

Delta (the labelling delay) enters only as the round-to-calendar mapping and produces no spread in
R_t by round index; it is reported as an exact zero, not as an unmeasured quantity.

Everything else -- the copula plausibility set, the best-response search, T = 10 rounds, the
mutable field list, the value field -- is `e4_durability.py` unchanged.  Outputs carry the
`kappa_` prefix; no existing results file is touched.
"""
from __future__ import annotations
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import common as C
from e4_durability import best_response, MEDICARE_MUTABLE, T, PRIMARY, QS, KAPPAS, DELTAS

SEEDS = (0, 1, 2)


def run_cells(name, Xtr, ytr, Xte, yte, vte, mut_cols, cols, cells, seeds=SEEDS):
    mut_idx = [cols.index(c) for c in mut_cols if c in cols]
    scale = Xtr.std(0) + 1e-9
    legit_tr = Xtr[ytr == 0]
    cop = C.GaussianCopula(legit_tr[np.random.default_rng(0).choice(
        len(legit_tr), min(len(legit_tr), 8000), replace=False)])
    ll_legit = cop.loglik(Xte[yte == 0])
    rows = []
    t0 = time.time()
    for q, kappa in cells:
        thresh = float(np.quantile(ll_legit, q))
        for seed in seeds:
            _, model = C.fit_gbdt(Xtr, ytr, Xtr[:1], seed=seed)
            score_fn = lambda z: model.predict_proba(z)[:, 1]
            Xf, vf, Xn = Xte[yte == 1].copy(), vte[yte == 1].copy(), Xte[yte == 0]
            for t in range(T + 1):
                sn, sf = score_fn(Xn), score_fn(Xf)
                s_all = np.r_[sn, sf]
                y_all = np.r_[np.zeros(len(Xn), int), np.ones(len(Xf), int)]
                v_all = np.r_[np.zeros(len(Xn)), vf]
                thr = np.quantile(sn, 0.99)
                rows.append({"dataset": name, "q": q, "kappa": kappa, "seed": seed,
                             "retrain": False, "round": t,
                             "vw_recall": C.value_weighted_recall_at_fpr(y_all, s_all, v_all, 0.01),
                             "count_recall": C.count_recall_at_fpr(y_all, s_all, 0.01),
                             "value_stolen": float(vf[sf < thr].sum())})
                if t == T:
                    break
                Xf = best_response(Xf, mut_idx, cop, thresh, score_fn, kappa, scale)
            print(f"  {name} q={q} kappa={kappa} seed={seed} frozen done "
                  f"({time.time()-t0:.0f}s)", flush=True)
            C.dump(f"kappa_cells_raw_{name}.csv", pd.DataFrame(rows))
    return pd.DataFrame(rows)


def summarise(df, name):
    last = df[df["round"] == df["round"].max()]
    # cell means over seeds, and the seed noise at each cell
    cell = last.groupby(["q", "kappa"])["vw_recall"].agg(["mean", "std", "count"]).reset_index()
    C.dump(f"kappa_cell_means_{name}.csv", cell)

    q_arm = cell[cell.kappa == PRIMARY["kappa"]].sort_values("q")
    k_arm = cell[cell.q == PRIMARY["q"]].sort_values("kappa")
    sd_q = float(q_arm["mean"].std(ddof=1))
    sd_k = float(k_arm["mean"].std(ddof=1))
    sd_d = 0.0
    rng_q = float(q_arm["mean"].max() - q_arm["mean"].min())
    rng_k = float(k_arm["mean"].max() - k_arm["mean"].min())
    # noise reference: the pooled within-cell SD of the seed means, and the SD of a cell mean
    within = last.groupby(["q", "kappa"])["vw_recall"].std(ddof=1)
    n_per = last.groupby(["q", "kappa"])["vw_recall"].count()
    pooled_within = float(np.sqrt(np.nanmean(within.to_numpy() ** 2)))
    se_cell_mean = float(pooled_within / np.sqrt(float(n_per.mean())))

    order = sorted([("the plausibility quantile", sd_q), ("the cost parameter", sd_k),
                    ("the labelling delay", sd_d)], key=lambda z: -z[1])
    margin = order[0][1] - order[1][1]
    # "within noise of each other" = the gap between the top two spreads is smaller than the
    # standard error of a cell mean, the smallest difference this design can resolve.
    dominant = order[0][0] if margin > se_cell_mean else "no dominant quantity"
    out = {
        "dataset": name, "arm": "frozen only (DEVIATIONS 8.7: the retrained arm is in-sample)",
        "seeds_per_cell": int(n_per.min()), "endpoint": "value-weighted recall at 1% FPR, round T=10",
        "q_grid": QS, "kappa_grid": KAPPAS, "delta_grid_days": DELTAS,
        "q_arm": q_arm.to_dict("records"), "kappa_arm": k_arm.to_dict("records"),
        "sd_across_q": sd_q, "sd_across_kappa": sd_k, "sd_across_delta": sd_d,
        "range_across_q": rng_q, "range_across_kappa": rng_k, "range_across_delta": 0.0,
        "pooled_within_cell_seed_sd": pooled_within,
        "se_of_a_cell_mean": se_cell_mean,
        "ranking": [{"quantity": a, "sd": b} for a, b in order],
        "margin_between_top_two": margin,
        "dominant_free_quantity": dominant,
        "delta_note": ("Delta enters only as the round-to-calendar mapping (a round is one "
                       "labelling delay), so it produces no spread in R_t by round index: its "
                       "spread is an exact zero by construction, not an unmeasured quantity."),
        "all_three_varied": True}
    C.dump(f"kappa_dominant_quantity_{name}.json", out)
    print(json.dumps({k: out[k] for k in
                      ["sd_across_q", "sd_across_kappa", "sd_across_delta", "range_across_q",
                       "range_across_kappa", "pooled_within_cell_seed_sd", "se_of_a_cell_mean",
                       "margin_between_top_two", "dominant_free_quantity"]}, indent=1))
    return out


def main():
    t0 = time.time()
    m, F, ym, valm, feat = C.medicare_provider_features()
    cut = np.quantile(m["start_mean"].values, 0.7)
    trm = m["start_mean"].values <= cut
    print(f"Medicare providers {len(m)}, flagged {int(ym.sum())}; "
          f"train {int(trm.sum())} / test {int((~trm).sum())}", flush=True)
    cells = [(q, PRIMARY["kappa"]) for q in QS] + \
            [(PRIMARY["q"], k) for k in KAPPAS if k != PRIMARY["kappa"]]
    df = run_cells("Medicare", F[trm].values, ym[trm], F[~trm].values, ym[~trm], valm[~trm],
                   MEDICARE_MUTABLE, feat, cells)
    C.dump("kappa_cells_raw_Medicare.csv", df)
    summarise(df, "Medicare")
    print(f"done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

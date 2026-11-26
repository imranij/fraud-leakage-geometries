"""e21_ieee_adversary.py -- H3(a),(b) on IEEE-CIS with the RESTRICTED mutable set.

Supplementary Material S6 ("Restrictions"): "IEEE-CIS is partly anonymised, so the mutable set is
restricted accordingly."  The claim-by-evidence table lists H3 fraud durability on "IEEE-CIS
(restricted), BAF, vehicle".  This script runs the e4_durability.py protocol unchanged on the
IEEE-CIS temporal split cached by e20, with the mutable set restricted to the ten fields that are
(i) not anonymised and (ii) under the transacting party's control:

    TransactionAmt, dist1, card4, card6, P_emaildomain, R_emaildomain,
    DeviceType, DeviceInfo, id_30 (operating system), id_31 (browser)

Everything else is held fixed: V1-V339 and id_01-id_29 are anonymised engineered features whose
semantics Vesta does not release, C1-C14 and D1-D15 are counts and time deltas the acquirer
computes rather than the payer supplies, addr1/addr2 and card1/card2/card3/card5 identify the
instrument and the billing address, and M1-M9 are match flags computed by the processor.

Grid, rounds and primary cell are those of S4 and of e4_durability.py: q in {0.01,0.05,0.10,0.25},
kappa in {0,0.1,0.5}, Delta in {30,90,180} days, T = 10 rounds, primary (q=0.05, kappa=0.1,
Delta=90); the reduced cross (every q at the primary kappa, every kappa at the primary q), three
seeds at the primary cell and one elsewhere, frozen and retrained detectors.

TWO DEVIATIONS, both logged in DEVIATIONS.md 8.5:
 (1) The plausibility copula is fitted to the TEN mutable coordinates only.  On the vehicle (31
     coordinates) and Medicare (13) arms it spans every coordinate; over IEEE-CIS's 432 coordinates
     a Gaussian copula estimated from 8,000 legitimate records is dominated by estimation error, and
     the constraint is in any case only ever evaluated on coordinates the adversary moves.
 (2) The six coded-categorical mutable coordinates are snapped back to the coded lattice (rounded
     and clipped to the observed code range) after each move, so the adversary cannot occupy a
     value the encoding does not admit.  The vehicle arm moved its coded fields continuously.
"""
from __future__ import annotations
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import common as C

CACHE = os.path.join(C.RES, "cache")
QS = [0.01, 0.05, 0.10, 0.25]
KAPPAS = [0.0, 0.1, 0.5]
DELTAS = [30, 90, 180]
T = 10
PRIMARY = {"q": 0.05, "kappa": 0.1, "delta": 90}

IEEE_MUTABLE = ["TransactionAmt", "dist1", "card4", "card6", "P_emaildomain",
                "R_emaildomain", "DeviceType", "DeviceInfo", "id_30", "id_31"]
IEEE_CODED = {"card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType",
              "DeviceInfo", "id_30", "id_31"}


def best_response(Xf, mut_idx, cop, thresh, score_fn, kappa, scale, code_mask, code_lo, code_hi,
                  n_iter=1, n_grid=7):
    """Query-only coordinate descent on the mutable coordinates (Eq. 2), restricted set."""
    x0 = Xf.copy()
    x = Xf.copy()
    grid = np.linspace(-1.5, 1.5, n_grid)
    for _ in range(n_iter):
        for k, j in enumerate(mut_idx):
            best = x.copy()
            best_obj = score_fn(x) + kappa * (np.abs(x - x0) / scale).mean(1)
            for g in grid:
                if g == 0:
                    continue
                cand = x.copy()
                nv = x[:, j] + g * scale[j]
                if code_mask[k]:
                    nv = np.clip(np.round(nv), code_lo[k], code_hi[k])
                cand[:, j] = nv
                ok = cop.loglik(cand[:, mut_idx]) >= thresh
                o = score_fn(cand) + kappa * (np.abs(cand - x0) / scale).mean(1)
                take = ok & (o < best_obj)
                best[take] = cand[take]
                best_obj = np.where(take, o, best_obj)
            x = best
    return x


def main():
    t0 = time.time()
    Xtr = np.load(os.path.join(CACHE, "ieee_Xtr.npy"))
    ytr = np.load(os.path.join(CACHE, "ieee_ytr.npy")).astype(int)
    Xte = np.load(os.path.join(CACHE, "ieee_Xte.npy"))
    yte = np.load(os.path.join(CACHE, "ieee_yte.npy")).astype(int)
    vte = np.load(os.path.join(CACHE, "ieee_vte.npy"))
    cols = json.load(open(os.path.join(CACHE, "ieee_feats.json")))
    mut_cols = [c for c in IEEE_MUTABLE if c in cols]
    mut_idx = [cols.index(c) for c in mut_cols]
    code_mask = np.array([c in IEEE_CODED for c in mut_cols])
    print(f"mutable {len(mut_cols)}/{len(IEEE_MUTABLE)}: {mut_cols}", flush=True)

    scale = Xtr.std(0) + 1e-9
    legit = Xtr[ytr == 0]
    rng0 = np.random.default_rng(0)
    samp = legit[rng0.choice(len(legit), min(len(legit), 8000), replace=False)]
    cop = C.GaussianCopula(samp[:, mut_idx].astype(float))
    code_lo = np.array([Xtr[:, j].min() for j in mut_idx])
    code_hi = np.array([Xtr[:, j].max() for j in mut_idx])
    ll_legit = cop.loglik(Xte[yte == 0][:, mut_idx].astype(float))

    name = "IEEE-CIS"
    rows = []
    cells = [(q, PRIMARY["kappa"]) for q in QS] + \
            [(PRIMARY["q"], k) for k in KAPPAS if k != PRIMARY["kappa"]]
    seeds = (0, 1, 2)
    Xn_all = Xte[yte == 0]
    for q, kappa in cells:
        thresh = float(np.quantile(ll_legit, q))
        for retrain in (False, True):
            use = seeds if (q == PRIMARY["q"] and kappa == PRIMARY["kappa"]) else seeds[:1]
            for seed in use:
                _, model = C.fit_gbdt(Xtr, ytr, Xtr[:1], seed=seed)
                score_fn = lambda z: model.predict_proba(z)[:, 1]
                Xf, vf, Xn = Xte[yte == 1].copy(), vte[yte == 1].copy(), Xn_all
                tr_X, tr_y = Xtr.copy(), ytr.copy()
                for t in range(T + 1):
                    sn, sf = score_fn(Xn), score_fn(Xf)
                    s_all = np.r_[sn, sf]
                    y_all = np.r_[np.zeros(len(Xn), int), np.ones(len(Xf), int)]
                    v_all = np.r_[np.zeros(len(Xn)), vf]
                    thr = np.quantile(sn, 0.99)
                    rows.append({"dataset": name, "q": q, "kappa": kappa, "seed": seed,
                                 "retrain": retrain, "round": t,
                                 "vw_recall": C.value_weighted_recall_at_fpr(y_all, s_all, v_all, 0.01),
                                 "count_recall": C.count_recall_at_fpr(y_all, s_all, 0.01),
                                 "value_stolen": float(vf[sf < thr].sum())})
                    if t == T:
                        break
                    Xf = best_response(Xf, mut_idx, cop, thresh, score_fn, kappa, scale,
                                       code_mask, code_lo, code_hi)
                    if retrain:
                        tr_X = np.vstack([tr_X, Xf])
                        tr_y = np.r_[tr_y, np.ones(len(Xf), int)]
                        _, model = C.fit_gbdt(tr_X, tr_y, tr_X[:1], seed=seed)
                        score_fn = lambda z, m=model: m.predict_proba(z)[:, 1]
                print(f"  {name} q={q} k={kappa} retrain={retrain} seed={seed} "
                      f"done {time.time()-t0:.0f}s", flush=True)
                C.dump("fig5_durability_partial_IEEE.csv", pd.DataFrame(rows))

    df = pd.DataFrame(rows)
    C.dump("fig5_durability_raw_IEEE.csv", df)

    prim = []
    sel = df[(df["q"] == PRIMARY["q"]) & (df["kappa"] == PRIMARY["kappa"])]
    for retrain in (False, True):
        s = sel[sel.retrain == retrain]
        if s.empty:
            continue
        p = s.pivot_table(index="seed", columns="round", values="value_stolen")
        r = s.pivot_table(index="seed", columns="round", values="vw_recall")
        rc = s.pivot_table(index="seed", columns="round", values="count_recall")
        tmax = int(max(p.columns))
        ratio = p[tmax] / p[0].replace(0, np.nan)
        prim.append({"dataset": name, "retrain": retrain, "T": tmax,
                     "V_T_over_V_0_mean": float(ratio.mean()),
                     "V_T_over_V_0_sd": float(ratio.std(ddof=1)) if len(ratio) > 1 else float("nan"),
                     "R_0_mean": float(r[0].mean()), "R_T_mean": float(r[tmax].mean()),
                     "R_drop": float(r[0].mean() - r[tmax].mean()),
                     "count_R_0_mean": float(rc[0].mean()), "count_R_T_mean": float(rc[tmax].mean()),
                     "monotone_nonincreasing": bool(np.all(np.diff(r.mean(0).values) <= 1e-12)),
                     "n_seeds": int(len(p)),
                     "calendar_years_at_delta90": tmax * PRIMARY["delta"] / 365.0,
                     "mutable_fields": ";".join(mut_cols),
                     "value_field": "TransactionAmt (real monetary amount)"})
    C.dump("h3_primary_endpoint_IEEE.csv", pd.DataFrame(prim))

    last = df[df["round"] == df["round"].max()]
    base = last.groupby(["q", "kappa", "retrain"])["vw_recall"].mean().reset_index()
    C.dump("h3_spread_by_free_quantity_IEEE.csv", pd.DataFrame([{
        "dataset": name,
        "sd_across_q": float(base[base.kappa == PRIMARY["kappa"]].groupby("q")["vw_recall"].mean().std(ddof=1)),
        "sd_across_kappa": float(base[base.q == PRIMARY["q"]].groupby("kappa")["vw_recall"].mean().std(ddof=1)),
        "sd_across_retrain": float(base.groupby("retrain")["vw_recall"].mean().std(ddof=1)),
        "note": "Delta enters only as the round-to-calendar mapping, so it produces no spread in "
                "R_t by round index."}]))
    print(json.dumps(prim, indent=1, default=float), flush=True)
    print(f"IEEE-CIS adversary complete {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

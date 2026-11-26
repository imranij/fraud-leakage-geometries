"""e19_paysim_h4_fix.py -- H4 value-at-budget on the PaySim host with the budget-period defect of
DEVIATIONS.md 7.5 fixed (injected records floored into the host's own periods).
The E7 gate on the PaySim host is computed by e16_paysim_host.py and is unaffected by the defect.
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import common as C
import e16_paysim_host as P
import e18_h4_period_fix as F

FEATS = ["type_code", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
         "newbalanceDest", "errBalOrig", "errBalDest", "hour"]


def main():
    t0 = time.time()
    d = P.load_paysim()
    d["errBalOrig"] = (d.oldbalanceOrg - d.amount - d.newbalanceOrig).astype(np.float32)
    d["errBalDest"] = (d.oldbalanceDest + d.amount - d.newbalanceDest).astype(np.float32)
    d["hour"] = (d.step % 24).astype(np.int8)
    q80 = int(np.quantile(d.step.to_numpy(), 0.8))
    tr_m = (d.step <= q80).to_numpy()
    rng = np.random.default_rng(0)
    ytr_all = d.isFraud.to_numpy()[tr_m]
    sub = np.sort(np.concatenate([np.flatnonzero(ytr_all == 1),
        rng.choice(np.flatnonzero(ytr_all == 0), min(200000, int((ytr_all == 0).sum())), replace=False)]))
    Xtr = d.loc[tr_m, FEATS].to_numpy(np.float32)[sub]
    ytr = ytr_all[sub]
    te = d.loc[~tr_m].reset_index(drop=True)
    n_full = len(te)
    del d
    keep = np.sort(np.random.default_rng(0).choice(len(te), P.HOST_CAP, replace=False))
    te = te.iloc[keep].reset_index(drop=True)
    s_f, _ = C.fit_gbdt(Xtr, ytr, te[FEATS].to_numpy(np.float32), seed=0)
    pra = C.pr_auc(te.isFraud.to_numpy(), s_f)
    print("PaySim host %d of %d test records, fraud %d, fraud-model test PR-AUC %.4f (%.0fs)"
          % (len(te), n_full, int(te.isFraud.sum()), pra, time.time() - t0), flush=True)
    harm = pd.DataFrame({"record_id": np.arange(len(te)), "day": te["step"].to_numpy().astype(float),
                         "entity_id": te["orig_id"].to_numpy(),
                         "counterparty_id": te["dest_id"].to_numpy(),
                         "amount": te["amount"].to_numpy().astype(float), "currency": "EUR",
                         "y": te["isFraud"].to_numpy().astype(int)})
    F.run(harm, s_f, "paysim_periodfix",
          "PaySim test split (real fraud labels and amounts, seeded uniform subsample) with "
          "LeakBench-injected leakage, injected records floored into the host's step periods -- "
          "the host configs/default.yaml names; fraud-model test PR-AUC %.4f" % pra)
    print("done %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

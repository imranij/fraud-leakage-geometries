"""e22_ftt.py -- a GENUINE feature-tokenised transformer (Gorishniy et al. 2021) for the FTT row
of Table 3 (tab:dissociation) and its Supplementary S2 companions.

WHY THIS FILE EXISTS.  Every earlier run of the FTT row substituted scikit-learn's MLPClassifier
(`common.FTTLite`, substitution 2.1 in DEVIATIONS.md) because PyTorch was believed unavailable.
PyTorch 2.14.0 is in fact importable in this container and runs on CPU (torch.cuda.is_available()
is False), so the substitution is no longer necessary and the real architecture is implemented
here.  Nothing in this file touches the MLP results: they stay where they are, and the FTT numbers
are written to their own files (results/ftt_*).

THE ARCHITECTURE (Gorishniy, Rubachev, Khrulkov & Babenko, "Revisiting Deep Learning Models for
Tabular Data", NeurIPS 2021, Section 3.1-3.2):
  * Feature tokeniser.  Every one of the p features becomes its own token in R^d.
      - numerical feature j:    T_j = x_j * W_j + b_j,       W_j, b_j in R^d  (learned linear
                                embedding, one weight vector per feature)
      - categorical feature j:  T_j = E_j[c_j] + b_j,        E_j an embedding table of size
                                (cardinality_j, d), one table per feature
  * A learned [CLS] token is appended, giving a sequence of p+1 tokens.
  * L pre-norm transformer blocks: LayerNorm -> multi-head self-attention -> residual;
    LayerNorm -> feed-forward (d -> 2d -> d, ReLU) -> residual.
  * Prediction head on the [CLS] token only: LayerNorm -> ReLU -> Linear(d, 1).
This is an FTT and not an MLP: there is per-feature tokenisation, a [CLS] token, and self-attention
between feature tokens.  It is deliberately small (see SIZING below); it is not tuned to be
state of the art, and no claim is made that it is.

PROTOCOL.  Identical to the other detectors in e1_dissociation.py, e14_baf_dissociation.py and
e20_ieee_dissociation.py: the SAME splits (read from those scripts, not re-invented), the SAME five
seeds [0,1,2,3,4], value-weighted recall at a 1% false-positive rate with count-based recall beside
it, and PR-AUC.  Class imbalance is handled by pos_weight = n_neg/n_pos in the BCE loss, which is
the loss-side equivalent of the scale_pos_weight LightGBM is given in common.fit_gbdt.

SIZING (declared; written into results/ftt_config.json and DEVIATIONS_ADDENDUM_A.md).  Two cores,
no GPU.  Self-attention is quadratic in the number of feature tokens, so the cost is dominated by
IEEE-CIS with its 432 features.  Measured throughput at d=16, 2 heads, 2 blocks, batch 256 is
~320 training rows/s on this machine.  The model is therefore small (d_token=16, 2 heads, 2 blocks
for wide inputs; d_token=32, 4 heads, 3 blocks for inputs with <= 64 features) and the IEEE-CIS arm
trains on a further subsample of the training window (all training positives plus a seeded random
sample of negatives).  Both departures are computational and are recorded; neither touches the
test set, the split, the metric or the seeds.
"""
from __future__ import annotations
import sys, os, json, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import common as C

N_THREADS = int(os.environ.get("FTT_THREADS", "2"))
torch.set_num_threads(N_THREADS)

FPR = 0.01
SEEDS = C.SEEDS
RES = C.RES
CACHE = os.path.join(RES, "cache")


# ----------------------------------------------------------------- the model
class FeatureTokenizer(nn.Module):
    """Per-feature tokenisation: numerical -> learned linear embedding, categorical -> table."""

    def __init__(self, n_num: int, cat_cardinalities: list[int], d: int):
        super().__init__()
        self.n_num, self.d = n_num, d
        self.n_cat = len(cat_cardinalities)
        if n_num:
            self.num_w = nn.Parameter(torch.empty(n_num, d))
            self.num_b = nn.Parameter(torch.empty(n_num, d))
            nn.init.uniform_(self.num_w, -d ** -0.5, d ** -0.5)
            nn.init.uniform_(self.num_b, -d ** -0.5, d ** -0.5)
        if self.n_cat:
            # one embedding table PER categorical feature, laid out contiguously in a single
            # nn.Embedding with per-feature offsets (mathematically identical to a ModuleList of
            # tables, one kernel launch instead of n_cat)
            off = np.concatenate([[0], np.cumsum(cat_cardinalities)[:-1]]).astype(np.int64)
            self.register_buffer("cat_off", torch.from_numpy(off))
            self.cat_emb = nn.Embedding(int(sum(cat_cardinalities)), d)
            nn.init.uniform_(self.cat_emb.weight, -d ** -0.5, d ** -0.5)
            self.cat_b = nn.Parameter(torch.empty(self.n_cat, d))
            nn.init.uniform_(self.cat_b, -d ** -0.5, d ** -0.5)
        self.cls = nn.Parameter(torch.empty(1, 1, d))
        nn.init.uniform_(self.cls, -d ** -0.5, d ** -0.5)

    def forward(self, xn, xc):
        toks = []
        if self.n_num:
            toks.append(xn.unsqueeze(-1) * self.num_w + self.num_b)       # (B, n_num, d)
        if self.n_cat:
            toks.append(self.cat_emb(xc + self.cat_off) + self.cat_b)      # (B, n_cat, d)
        T = torch.cat(toks, 1) if len(toks) > 1 else toks[0]
        return torch.cat([self.cls.expand(T.shape[0], -1, -1), T], 1)      # [CLS] first


class Block(nn.Module):
    """Pre-norm transformer block with multi-head self-attention over the feature tokens."""

    def __init__(self, d, heads, p_drop):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Dropout(p_drop),
                                nn.Linear(2 * d, d))
        self.drop = nn.Dropout(p_drop)
        self.p_drop = p_drop

    def forward(self, x):
        B, T, D = x.shape
        z = self.n1(x)
        q, k, v = self.qkv(z).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        a = nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.p_drop if self.training else 0.0)
        a = a.transpose(1, 2).reshape(B, T, D)
        x = x + self.drop(self.proj(a))
        return x + self.drop(self.ff(self.n2(x)))


class FTTransformer(nn.Module):
    def __init__(self, n_num, cat_cardinalities, d=32, heads=4, blocks=3, p_drop=0.1):
        super().__init__()
        self.tok = FeatureTokenizer(n_num, cat_cardinalities, d)
        self.blocks = nn.ModuleList([Block(d, heads, p_drop) for _ in range(blocks)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, xn, xc):
        x = self.tok(xn, xc)
        for b in self.blocks:
            x = b(x)
        return self.head(x[:, 0]).squeeze(-1)                              # [CLS] -> logit


# ----------------------------------------------------------------- training
def prep(Xtr, Xte, cat_idx):
    """Split the matrix into (standardised numerical block, integer categorical block)."""
    p = Xtr.shape[1]
    cat_idx = sorted(set(int(i) for i in cat_idx))
    num_idx = [j for j in range(p) if j not in set(cat_idx)]
    A = np.asarray(Xtr[:, num_idx], np.float32) if num_idx else np.zeros((len(Xtr), 0), np.float32)
    B = np.asarray(Xte[:, num_idx], np.float32) if num_idx else np.zeros((len(Xte), 0), np.float32)
    if num_idx:
        mu, sd = A.mean(0), A.std(0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        A = np.clip((A - mu) / sd, -10, 10).astype(np.float32)
        B = np.clip((B - mu) / sd, -10, 10).astype(np.float32)
        A = np.nan_to_num(A); B = np.nan_to_num(B)
    cards, Ctr, Cte = [], [], []
    for j in cat_idx:
        a = np.nan_to_num(np.asarray(Xtr[:, j], np.float64)).round().astype(np.int64)
        b = np.nan_to_num(np.asarray(Xte[:, j], np.float64)).round().astype(np.int64)
        lo = min(a.min(), 0)
        a = a - lo; b = b - lo
        k = int(a.max()) + 1
        b = np.clip(b, 0, k - 1)          # unseen test codes fold onto the last training code
        cards.append(k); Ctr.append(a); Cte.append(b)
    Ctr = np.column_stack(Ctr) if cards else np.zeros((len(Xtr), 0), np.int64)
    Cte = np.column_stack(Cte) if cards else np.zeros((len(Xte), 0), np.int64)
    return A, Ctr, B, Cte, cards, len(num_idx)


def fit_predict_ftt(Xtr, ytr, Xte, cat_idx, seed, d=32, heads=4, blocks=3, epochs=8,
                    batch=256, lr=1e-3, wd=1e-5, log=print):
    A, Ctr, Bm, Cte, cards, n_num = prep(Xtr, Xte, cat_idx)
    torch.manual_seed(seed); np.random.seed(seed)
    net = FTTransformer(n_num, cards, d=d, heads=heads, blocks=blocks)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    y = np.asarray(ytr, np.float32)
    npos = max(float(y.sum()), 1.0)
    pw = torch.tensor((len(y) - npos) / npos, dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    At, Ct, yt = torch.from_numpy(A), torch.from_numpy(Ctr), torch.from_numpy(y)
    n = len(y)
    g = np.random.default_rng(seed)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(epochs * int(np.ceil(n / batch)), 1), pct_start=0.2)
    net.train()
    for ep in range(epochs):
        order = g.permutation(n)
        tot, t0 = 0.0, time.time()
        for s in range(0, n, batch):
            i = torch.from_numpy(order[s:s + batch])
            opt.zero_grad()
            out = net(At[i], Ct[i])
            L = lossf(out, yt[i])
            L.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            tot += float(L) * len(i)
        log(f"      epoch {ep} loss {tot/n:.4f} ({time.time()-t0:.0f}s)")
    net.eval()
    Bt, Cb = torch.from_numpy(Bm), torch.from_numpy(Cte)
    out = np.empty(len(Bm), np.float64)
    with torch.no_grad():
        for s in range(0, len(Bm), 2048):
            out[s:s + 2048] = torch.sigmoid(net(Bt[s:s + 2048], Cb[s:s + 2048])).numpy()
    n_par = sum(p.numel() for p in net.parameters())
    return out, {"n_params": int(n_par), "n_num_tokens": n_num, "n_cat_tokens": len(cards),
                 "cat_cardinalities": cards, "d_token": d, "n_heads": heads, "n_blocks": blocks,
                 "epochs": epochs, "batch": batch, "lr": lr, "weight_decay": wd,
                 "pos_weight": float(pw)}


# ----------------------------------------------------------------- arms
ROWS_PATH = os.path.join(RES, "ftt_raw.csv")
CFG_PATH = os.path.join(RES, "ftt_config.json")
_rows, _cfg = [], {}


def flush():
    pd.DataFrame(_rows).to_csv(ROWS_PATH, index=False)
    with open(CFG_PATH, "w") as f:
        json.dump(_cfg, f, indent=2, default=float)


def run_arm(name, Xtr, ytr, Xte, yte, vte, cat_idx, **kw):
    t0 = time.time()
    print(f"=== {name}: train {Xtr.shape} pos {int(np.sum(ytr))} | "
          f"test {Xte.shape} pos {int(np.sum(yte))} | cats {len(cat_idx)}", flush=True)
    meta = None
    for seed in SEEDS:
        s, meta = fit_predict_ftt(Xtr, ytr, Xte, cat_idx, seed,
                                  log=lambda m: print(m, flush=True), **kw)
        _rows.append({"dataset": name, "detector": "FTT", "seed": seed,
                      "vw_recall_at_1pct_fpr": C.value_weighted_recall_at_fpr(yte, s, vte, FPR),
                      "count_recall_at_1pct_fpr": C.count_recall_at_fpr(yte, s, FPR),
                      "count_recall_at_0.1pct_fpr": C.count_recall_at_fpr(yte, s, 0.001),
                      "pr_auc": C.pr_auc(yte, s)})
        flush()
        print(f"   {name} seed {seed}: vw={_rows[-1]['vw_recall_at_1pct_fpr']:.4f} "
              f"cnt={_rows[-1]['count_recall_at_1pct_fpr']:.4f} "
              f"pr_auc={_rows[-1]['pr_auc']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    _cfg[name] = dict(meta or {}, n_train=int(len(ytr)), n_train_pos=int(np.sum(ytr)),
                      n_test=int(len(yte)), n_test_pos=int(np.sum(yte)),
                      seconds=round(time.time() - t0, 1), torch=torch.__version__,
                      threads=N_THREADS)
    flush()


VEHICLE_EPOCHS = int(os.environ.get("VEHICLE_EPOCHS", "40"))
BAF_EPOCHS = int(os.environ.get("BAF_EPOCHS", "30"))


def arm_vehicle():
    """Split by year (1994-95 train, 1996 test) -- identical to e1_dissociation.py.

    EPOCH BUDGET.  The first run of this arm (2026-09-20, contended CPU) used 8 epochs and the
    training loss was still falling when the budget ran out (ADDENDUM A.1.3-a).  It is re-run here
    to convergence at VEHICLE_EPOCHS (default 40) on uncontended CPU; the 8-epoch rows are kept for
    provenance in results/ftt_raw_prebudgetfix.csv.  Nothing else about the arm changes: same
    split, same seeds, same metric, same architecture.
    """
    cc = C.load_carclaims()
    X, y, val, year, cols = C.carclaims_matrix(cc)
    raw = cc.drop(columns=["FraudFound_P", "PolicyNumber", "Year"])
    cat_idx = [i for i, c in enumerate(cols) if not pd.api.types.is_numeric_dtype(raw[c])]
    tr = year < 1996
    run_arm("Vehicle", X[tr].values, y[tr], X[~tr].values, y[~tr], val[~tr], cat_idx,
            d=32, heads=4, blocks=3, epochs=VEHICLE_EPOCHS, batch=256)


def arm_medicare():
    """Provider level, split at the 70th percentile of mean claim start -- as e1_dissociation.py."""
    m, F, y, val, feat = C.medicare_provider_features()
    cut = np.quantile(m["start_mean"].values, 0.7)
    tr = m["start_mean"].values <= cut
    run_arm("Medicare", F[tr].values, y[tr], F[~tr].values, y[~tr], val[~tr], [],
            d=32, heads=4, blocks=3, epochs=20, batch=256)


def arm_ulb():
    """Chronological 70/10/20 with the same seeded 50k-negative training subsample as e1."""
    d = C.load_ulb().sort_values("Time").reset_index(drop=True)
    n = len(d)
    i_va = int(0.8 * n)
    feats = [c for c in d.columns if c != "Class"]
    Xtr = d.loc[:i_va - 1, feats].values
    ytr = d.loc[:i_va - 1, "Class"].values
    te = d.loc[i_va:].reset_index(drop=True)
    rng = np.random.default_rng(0)
    pos_i = np.flatnonzero(ytr == 1)
    neg_i = rng.choice(np.flatnonzero(ytr == 0), 50000, replace=False)
    sub = np.sort(np.concatenate([pos_i, neg_i]))
    run_arm("ULB", Xtr[sub], ytr[sub], te[feats].values, te["Class"].values,
            te["Amount"].values, [], d=32, heads=4, blocks=3, epochs=5, batch=512)


def arm_baf():
    """Temporal: train months 0-5 (same seeded 120k-negative subsample as e14), test months 6-7."""
    CATB = ["payment_type", "employment_status", "housing_status", "source", "device_os"]
    d = pd.read_csv(os.path.join(C.RAW, "baf", "Base.csv"), engine="c")
    assert d.shape == (1_000_000, 32) and int(d.fraud_bool.sum()) == 11_029
    y = d["fraud_bool"].to_numpy(np.int8)
    month = d["month"].to_numpy(np.int8)
    value = d["proposed_credit_limit"].to_numpy(np.float64)
    X = d.drop(columns=["fraud_bool", "month"]).copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes
    cols = list(X.columns)
    cat_idx = [cols.index(c) for c in CATB]
    Xa = X.to_numpy(np.float32)
    del X, d
    tr = month <= 5
    rng = np.random.default_rng(0)
    pos_i = np.flatnonzero(tr & (y == 1))
    neg_i = rng.choice(np.flatnonzero(tr & (y == 0)), 120_000, replace=False)
    sub = np.sort(np.concatenate([pos_i, neg_i]))
    # EPOCH BUDGET. The first run used 2 epochs with the training loss still falling
    # (ADDENDUM A). The Vehicle arm's cell quadrupled (1.7 -> 7.2) once trained to convergence,
    # so a still-falling BAF cell is not a trustworthy measurement of what this architecture
    # reaches. Re-run to convergence at BAF_EPOCHS; the 2-epoch rows stay in
    # results/ftt_raw_prebudgetfix.csv. Nothing else about the arm changes.
    run_arm("BAF", Xa[sub], y[sub].astype(int), Xa[~tr], y[~tr].astype(int), value[~tr],
            cat_idx, d=32, heads=4, blocks=3, epochs=BAF_EPOCHS, batch=512)


IEEE_FTT_NEG = int(os.environ.get("IEEE_FTT_NEG", "20000"))


def arm_ieee():
    """Reuses the cached matrices e20_ieee_dissociation.py wrote: same temporal split (75th
    percentile of TransactionDT), same imputation, same features, same test set.

    FURTHER TRAINING SUBSAMPLE (declared).  e20 already subsamples the training window to all
    positives plus 120,000 negatives.  Self-attention over 432 feature tokens costs ~320 training
    rows/s on two cores, so the FTT arm trains on all training positives plus a seeded random
    sample of IEEE_FTT_NEG of those 120,000 negatives.  The test set is untouched.
    """
    feats = json.load(open(os.path.join(CACHE, "ieee_feats.json")))
    TXN_OBJ = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
               "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]
    IDN_OBJ = ["id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29", "id_30", "id_31",
               "id_33", "id_34", "id_35", "id_36", "id_37", "id_38", "DeviceType", "DeviceInfo"]
    cat_idx = [i for i, c in enumerate(feats) if c in set(TXN_OBJ + IDN_OBJ)]
    Xtr = np.load(os.path.join(CACHE, "ieee_Xtr.npy"))
    ytr = np.load(os.path.join(CACHE, "ieee_ytr.npy")).astype(int)
    rng = np.random.default_rng(0)
    pos = np.flatnonzero(ytr == 1)
    negall = np.flatnonzero(ytr == 0)
    neg = rng.choice(negall, min(IEEE_FTT_NEG, len(negall)), replace=False)
    sub = np.sort(np.concatenate([pos, neg]))
    Xtr, ytr = Xtr[sub], ytr[sub]
    Xte = np.load(os.path.join(CACHE, "ieee_Xte.npy"))
    yte = np.load(os.path.join(CACHE, "ieee_yte.npy")).astype(int)
    vte = np.load(os.path.join(CACHE, "ieee_vte.npy"))
    _cfg["IEEE-CIS_subsample_note"] = {
        "n_train_available_in_e20_cache": int(len(negall) + len(pos)),
        "n_train_used_by_ftt": int(len(sub)),
        "n_negatives_sampled": int(len(neg)), "sampling_seed": 0,
        "reason": "self-attention over 432 feature tokens at ~320 training rows/s on two cores"}
    run_arm("IEEE-CIS", Xtr, ytr, Xte, yte, vte, cat_idx,
            d=16, heads=2, blocks=2, epochs=3, batch=256)


ARMS = {"vehicle": arm_vehicle, "medicare": arm_medicare, "ulb": arm_ulb,
        "baf": arm_baf, "ieee": arm_ieee}


def main(which):
    if os.path.exists(ROWS_PATH):
        prev = pd.read_csv(ROWS_PATH)
        _rows.extend(prev.to_dict("records"))
    if os.path.exists(CFG_PATH):
        _cfg.update(json.load(open(CFG_PATH)))
    # re-running an arm REPLACES its rows; it never appends a second copy of them
    names = {"vehicle": "Vehicle", "medicare": "Medicare", "ulb": "ULB",
             "baf": "BAF", "ieee": "IEEE-CIS"}
    drop = {names[w] for w in which}
    keep = [r for r in _rows if r.get("dataset") not in drop]
    _rows.clear(); _rows.extend(keep)
    for w in which:
        ARMS[w]()
    df = pd.DataFrame(_rows)
    agg = (df.groupby(["dataset", "detector"])["vw_recall_at_1pct_fpr"]
           .agg(["mean", "std", "count"]).reset_index())
    agg2 = (df.groupby(["dataset", "detector"])[
        ["count_recall_at_1pct_fpr", "count_recall_at_0.1pct_fpr", "pr_auc"]]
        .mean().reset_index())
    agg.merge(agg2, on=["dataset", "detector"]).to_csv(
        os.path.join(RES, "ftt_summary.csv"), index=False)
    print(pd.read_csv(os.path.join(RES, "ftt_summary.csv")).to_string(index=False), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["vehicle", "medicare", "ulb", "baf", "ieee"])

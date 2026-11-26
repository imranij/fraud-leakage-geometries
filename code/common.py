"""Shared loaders, metrics and detectors for the real-data run, 2026-09-19.

Every number produced by this run is written to ../results/. Nothing here fabricates data:
if a dataset is absent the caller raises and the endpoint is reported "not computed".
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
RAW = os.path.join(PAPER, "data", "raw")
RES = os.path.join(PAPER, "results")
os.makedirs(RES, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(name: str, obj):
    p = os.path.join(RES, name)
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(p, index=False)
    else:
        with open(p, "w") as f:
            json.dump(obj, f, indent=2, default=float)
    print("wrote", p, flush=True)
    return p


# ---------------------------------------------------------------- metrics
def value_weighted_recall_at_fpr(y_true, score, value, fpr: float = 0.01) -> float:
    """Fraction of positive VALUE captured at the threshold giving `fpr` on the negatives."""
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, float)
    value = np.asarray(value, float)
    neg = score[y_true == 0]
    if len(neg) == 0 or (y_true == 1).sum() == 0:
        return float("nan")
    thr = np.quantile(neg, 1.0 - fpr)
    pos = y_true == 1
    tot = value[pos].sum()
    if tot <= 0:
        return float("nan")
    return float(value[pos & (score >= thr)].sum() / tot)


def count_recall_at_fpr(y_true, score, fpr: float = 0.01) -> float:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, float)
    neg = score[y_true == 0]
    if len(neg) == 0 or (y_true == 1).sum() == 0:
        return float("nan")
    thr = np.quantile(neg, 1.0 - fpr)
    return float(((score >= thr) & (y_true == 1)).sum() / (y_true == 1).sum())


def pr_auc(y_true, score) -> float:
    from sklearn.metrics import average_precision_score
    try:
        return float(average_precision_score(y_true, score))
    except Exception:
        return float("nan")


def paired_tests(a, b):
    """Paired comparison of two length-5 seed vectors: mean diff, paired t, Wilcoxon, sign test."""
    from scipy import stats
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    out = {"mean_diff": float(np.mean(d)), "sd_diff": float(np.std(d, ddof=1)) if len(d) > 1 else float("nan")}
    try:
        t = stats.ttest_rel(a, b)
        out["t_stat"], out["t_p"] = float(t.statistic), float(t.pvalue)
    except Exception:
        out["t_stat"] = out["t_p"] = float("nan")
    try:
        w = stats.wilcoxon(a, b)
        out["wilcoxon_p"] = float(w.pvalue)
    except Exception:
        out["wilcoxon_p"] = float("nan")
    n_pos = int(np.sum(d > 0))
    n_eff = int(np.sum(d != 0))
    out["sign_pos"], out["sign_n"] = n_pos, n_eff
    out["sign_p"] = float(stats.binomtest(n_pos, n_eff, 0.5).pvalue) if n_eff > 0 else float("nan")
    return out


# ---------------------------------------------------------------- detectors
def fit_gbdt(Xtr, ytr, Xte, seed=0, weight=True):
    import lightgbm as lgb
    pos = max(int(np.sum(ytr == 1)), 1)
    spw = (len(ytr) - pos) / pos if weight else 1.0
    m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.08, num_leaves=31,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           scale_pos_weight=spw, random_state=seed, n_jobs=2, verbose=-1)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1], m


def fit_isolation_forest(Xtr, Xte, seed=0):
    from sklearn.ensemble import IsolationForest
    m = IsolationForest(n_estimators=200, random_state=seed, n_jobs=2, contamination="auto")
    m.fit(Xtr)
    return -m.score_samples(Xte)


def fit_autoencoder(Xtr, Xte, seed=0, hidden=16, epochs=12):
    """Reconstruction-error detector: an undercomplete autoencoder trained on the training records.

    DEVIATION (logged): PyTorch is not installed in the run environment, so the autoencoder is
    scikit-learn's MLPRegressor fitted to reconstruct its own input through a bottleneck layer
    (64 -> `hidden` -> 64). It is an autoencoder; it is not the torch implementation the scaffold
    assumed. `epochs` maps to max_iter.
    """
    from sklearn.neural_network import MLPRegressor
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = (Xtr - mu) / sd
    B = (Xte - mu) / sd
    net = MLPRegressor(hidden_layer_sizes=(64, hidden, 64), activation="relu", solver="adam",
                       learning_rate_init=1e-3, max_iter=epochs, random_state=seed,
                       batch_size=1024, early_stopping=False, tol=0.0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.fit(A, A)
        rec = net.predict(B)
    return ((rec - B) ** 2).mean(1)


class FTTLite(object):
    """Learner-family control for the manuscript's feature-tokenised transformer.

    DEVIATION (logged): PyTorch is not installed, so the FTT of Gorishniy et al. cannot be run.
    The control is a multi-layer perceptron (scikit-learn MLPClassifier) on the same standardised
    inputs: it still answers the question the FTT row was there to answer -- whether an ordering
    found with a tree ensemble also holds for a neural tabular model -- but it is not an FTT, and
    every FTT row in the results carries this substitution.
    """

    def __init__(self, seed=0, d_model=32, epochs=30):
        self.seed, self.epochs = seed, epochs

    def fit_predict(self, Xtr, ytr, Xte):
        from sklearn.neural_network import MLPClassifier
        from sklearn.utils import resample
        import warnings
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        A = (Xtr - mu) / sd
        B = (Xte - mu) / sd
        # class weighting by oversampling the minority (MLPClassifier takes no sample weights)
        rng = np.random.default_rng(self.seed)
        pos = np.flatnonzero(ytr == 1)
        neg = np.flatnonzero(ytr == 0)
        if len(pos) and len(pos) < len(neg):
            reps = rng.choice(pos, size=min(len(neg), 20 * len(pos)) - len(pos), replace=True)
            idx = np.concatenate([np.arange(len(ytr)), reps])
        else:
            idx = np.arange(len(ytr))
        net = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                            learning_rate_init=1e-3, max_iter=self.epochs, random_state=self.seed,
                            batch_size=1024, early_stopping=False, tol=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            net.fit(A[idx], np.asarray(ytr)[idx])
            return net.predict_proba(B)[:, 1]


def rules_score(df: pd.DataFrame, value_col: str, entity_col: str | None, time_col: str | None):
    """Incumbent stand-in: velocity + exact-duplicate rules. Score = number of rules fired."""
    s = np.zeros(len(df))
    v = df[value_col].values.astype(float)
    s += (v >= np.quantile(v, 0.99)).astype(float)          # unusually large amount
    s += (v == 0).astype(float)                              # zero-value record
    if entity_col is not None and entity_col in df.columns:
        cnt = df.groupby(entity_col)[value_col].transform("size").values
        s += (cnt >= np.quantile(cnt, 0.99)).astype(float)   # velocity
    key = df[[value_col] + ([entity_col] if entity_col in df.columns else [])] if entity_col else df[[value_col]]
    s += key.duplicated(keep=False).values.astype(float)     # exact duplicate
    return s + np.linspace(0, 1e-9, len(df))


# ---------------------------------------------------------------- data loaders
def load_ulb():
    p = os.path.join(RAW, "ulb", "creditcard.csv")
    d = pd.read_csv(p)
    assert d.shape[0] == 284807, d.shape
    assert int(d["Class"].sum()) == 492
    return d


def load_carclaims():
    p = os.path.join(RAW, "carclaims", "fraud_oracle.csv")
    d = pd.read_csv(p)
    assert d.shape[0] == 15420, d.shape
    assert int(d["FraudFound_P"].sum()) == 923
    return d


VEHICLE_PRICE_MID = {
    "less than 20000": 15000.0, "20000 to 29000": 24500.0, "30000 to 39000": 34500.0,
    "40000 to 59000": 49500.0, "60000 to 69000": 64500.0, "more than 69000": 80000.0,
}


def carclaims_matrix(d: pd.DataFrame):
    """Ordinal / one-hot encoding of the 33 coded fields; value proxy from VehiclePrice band midpoints."""
    df = d.copy()
    value = df["VehiclePrice"].map(VEHICLE_PRICE_MID).fillna(24500.0).values
    y = df["FraudFound_P"].values.astype(int)
    year = df["Year"].values
    drop = ["FraudFound_P", "PolicyNumber", "Year"]
    X = df.drop(columns=drop)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes
    return X.astype(float), y, value, year, list(X.columns)


def load_medicare():
    D = os.path.join(RAW, "medicare")
    tr = pd.read_csv(os.path.join(D, "Train-1542865627584.csv"))
    ip = pd.read_csv(os.path.join(D, "Train_Inpatientdata-1542865627584.csv"), low_memory=False)
    op = pd.read_csv(os.path.join(D, "Train_Outpatientdata-1542865627584.csv"), low_memory=False)
    assert tr.shape[0] == 5410 and int((tr.PotentialFraud == "Yes").sum()) == 506
    return tr, ip, op


def medicare_provider_features():
    """Provider-level features aggregated from the claim files; value = total reimbursed."""
    tr, ip, op = load_medicare()
    ip = ip.assign(inpatient=1)
    op = op.assign(inpatient=0)
    cols = ["Provider", "ClaimID", "ClaimStartDt", "ClaimEndDt", "InscClaimAmtReimbursed",
            "AttendingPhysician", "BeneID", "DeductibleAmtPaid", "inpatient"]
    cl = pd.concat([ip[cols], op[cols]], ignore_index=True)
    cl["ClaimStartDt"] = pd.to_datetime(cl.ClaimStartDt, errors="coerce")
    cl["ClaimEndDt"] = pd.to_datetime(cl.ClaimEndDt, errors="coerce")
    cl["dur"] = (cl.ClaimEndDt - cl.ClaimStartDt).dt.days
    g = cl.groupby("Provider")
    f = pd.DataFrame({
        "n_claims": g.ClaimID.size(),
        "n_bene": g.BeneID.nunique(),
        "n_phys": g.AttendingPhysician.nunique(),
        "amt_sum": g.InscClaimAmtReimbursed.sum(),
        "amt_mean": g.InscClaimAmtReimbursed.mean(),
        "amt_max": g.InscClaimAmtReimbursed.max(),
        "amt_std": g.InscClaimAmtReimbursed.std().fillna(0.0),
        "ded_mean": g.DeductibleAmtPaid.mean(),
        "dur_mean": g.dur.mean(),
        "dur_max": g.dur.max(),
        "inpatient_share": g.inpatient.mean(),
        "claims_per_bene": g.ClaimID.size() / g.BeneID.nunique(),
        "start_mean": g.ClaimStartDt.mean().astype("int64") / 86400e9,
    }).reset_index()
    m = tr.merge(f, on="Provider", how="left").fillna(0.0)
    y = (m.PotentialFraud == "Yes").astype(int).values
    value = m["amt_sum"].values.astype(float)
    feat = [c for c in f.columns if c != "Provider"]
    return m, m[feat].astype(float), y, value, feat


# ---------------------------------------------------------------- copula plausibility
class GaussianCopula:
    """Gaussian copula with empirical marginals; log-density used as the plausibility statistic."""

    def __init__(self, X: np.ndarray):
        from scipy import stats
        self.X = np.asarray(X, float)
        self.n, self.d = self.X.shape
        self.sorted = [np.sort(self.X[:, j]) for j in range(self.d)]
        Z = np.column_stack([self._to_z(self.X[:, j], j) for j in range(self.d)])
        self.mu = Z.mean(0)
        C = np.cov(Z.T) + 1e-6 * np.eye(self.d)
        self.C = C
        self.Ci = np.linalg.inv(C)
        sign, logdet = np.linalg.slogdet(C)
        self.logdet = logdet
        self._stats = stats

    def _to_z(self, x, j):
        from scipy import stats
        r = np.searchsorted(self.sorted[j], x, side="right") / (self.n + 1.0)
        r = np.clip(r, 1.0 / (self.n + 1), 1 - 1.0 / (self.n + 1))
        return stats.norm.ppf(r)

    def loglik(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        Z = np.column_stack([self._to_z(X[:, j], j) for j in range(self.d)])
        dz = Z - self.mu
        q = np.einsum("ij,jk,ik->i", dz, self.Ci, dz)
        return -0.5 * (q + self.logdet) + 0.5 * np.sum(Z ** 2, axis=1)


def timeit(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t0

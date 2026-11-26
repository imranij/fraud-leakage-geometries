"""Value-at-Budget (VaB) metrics and budgeted triage policies across two alert queues.

Given per-record P_F (fraud prob), P_L (leak indicator/prob), amount v, leakage value v_L, true labels
y in {0,1,2}, and a daily capacity K, a policy selects which records to investigate.
VaB_F = fraud loss caught / total fraud loss; VaB_L = leakage value recovered / total leakage value.

Changes of 2026-09-16:
  * EV uses a per-record leakage value v_L (loss given deviation type), not the transaction amount.
  * `policy_value_based` implements Algorithm 1 with the termination guard (a queue cannot be forced
    above its candidate count), explicit queue assignment (larger EV term; ties -> L) and tie-breaking
    (smaller index). It always returns exactly k indices when k <= n.
  * New baselines: `policy_oracle` (top-k by realised value; upper bound) and
    `policy_single_queue` (top-k by EV with NO minimum shares) -- the true competitor for H4.
  * `policy_value_per_hour` is the knapsack variant for capacity in investigator hours (greedy by
    EV per hour; (1-1/e)-approximate).
  * `vab_grid` evaluates over a (lgf, recoverability) grid so that the winning region is reported.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def expected_value_terms(p_f, p_l, amount, v_l, lgf: float = 1.0, recoverability: float = 0.5):
    """Return (EV_F, EV_L) per record. p_l is 1[delta>0] or a probability; v_l is loss given deviation."""
    ev_f = np.asarray(p_f, float) * np.asarray(amount, float) * lgf
    ev_l = np.asarray(p_l, float) * np.asarray(v_l, float) * recoverability
    return ev_f, ev_l


def _vl(day_df: pd.DataFrame) -> np.ndarray:
    return day_df.v_l.values if "v_l" in day_df.columns else day_df.amount.values


def policy_single_queue(day_df: pd.DataFrame, k: int, lgf=1.0, recoverability=0.5) -> np.ndarray:
    """Top-k by total EV with no intent split (the competitor that tests whether two queues add anything)."""
    ev_f, ev_l = expected_value_terms(day_df.p_f.values, day_df.p_l.values, day_df.amount.values, _vl(day_df), lgf, recoverability)
    ev = ev_f + ev_l
    order = np.lexsort((np.arange(len(ev)), -ev))   # ties -> smaller index
    return order[:k]


def policy_value_based(day_df: pd.DataFrame, k: int, lgf=1.0, recoverability=0.5,
                       min_share: dict | None = None) -> np.ndarray:
    """Algorithm 1: top-k by EV with minimum share per queue, e.g. {"F":0.1,"L":0.1} (s_F + s_L <= 1).
    Guarded so that |S| == k always (when k <= n)."""
    n = len(day_df)
    k = min(k, n)
    ev_f, ev_l = expected_value_terms(day_df.p_f.values, day_df.p_l.values, day_df.amount.values, _vl(day_df), lgf, recoverability)
    ev = ev_f + ev_l
    queue = np.where(ev_f > ev_l, "F", "L")          # ties -> L (the durable queue)
    order = np.lexsort((np.arange(n), -ev))
    chosen = list(order[:k])
    if min_share:
        assert sum(min_share.values()) <= 1.0 + 1e-9, "minimum shares must sum to <= 1"
        for q, share in min_share.items():
            cand_q = int(np.sum(queue == q))
            m_q = min(int(np.ceil(share * k)), cand_q)            # guard (comment 54)
            in_s = [i for i in chosen if queue[i] == q]
            while len(in_s) < m_q:
                promote = next(i for i in order if queue[i] == q and i not in chosen)
                others = [i for i in chosen if queue[i] != q]
                if not others:
                    break
                demote = min(others, key=lambda i: (ev[i], -i))    # lowest EV; ties -> larger index leaves
                chosen.remove(demote); chosen.append(promote)
                in_s = [i for i in chosen if queue[i] == q]
    assert len(chosen) == k
    return np.array(chosen, int)


def policy_fraud_first(day_df, k, **_):
    return np.lexsort((np.arange(len(day_df)), -day_df.p_f.values))[:k]


def policy_fixed_split(day_df, k, share_f=0.5, **_):
    kf = int(round(share_f * k)); kl = k - kf
    f_idx = list(np.lexsort((np.arange(len(day_df)), -day_df.p_f.values))[:kf])
    l_idx = [i for i in np.lexsort((np.arange(len(day_df)), -day_df.p_l.values)) if i not in f_idx][:kl]
    return np.array(f_idx + l_idx, int)


def policy_oracle(day_df, k, lgf=1.0, recoverability=0.5, **_):
    """Upper bound: top-k by REALISED value (uses labels). Never a deployable policy."""
    realised = np.where(day_df.y.values == 1, day_df.amount.values * lgf,
                        np.where(day_df.y.values == 2, _vl(day_df) * recoverability, 0.0))
    return np.lexsort((np.arange(len(day_df)), -realised))[:k]


def policy_value_per_hour(day_df, hours: float, lgf=1.0, recoverability=0.5, **_):
    """Knapsack variant: capacity in investigator hours; each record has `effort_h`. Greedy by EV/hour."""
    ev_f, ev_l = expected_value_terms(day_df.p_f.values, day_df.p_l.values, day_df.amount.values, _vl(day_df), lgf, recoverability)
    eff = day_df.effort_h.values if "effort_h" in day_df.columns else np.ones(len(day_df))
    order = np.argsort(-(ev_f + ev_l) / np.maximum(eff, 1e-9))
    chosen, used = [], 0.0
    for i in order:
        if used + eff[i] <= hours:
            chosen.append(i); used += eff[i]
    return np.array(chosen, int)


def value_at_budget(df: pd.DataFrame, k_frac: float, policy, **policy_kw) -> dict:
    """df columns: day, p_f, p_l, amount, y, [v_l]. Returns VaB_F, VaB_L, precision per queue."""
    vl_all = df.v_l.values if "v_l" in df.columns else df.amount.values
    tot_f = df.loc[df.y == 1, "amount"].sum()
    tot_l = float(np.sum(vl_all[df.y.values == 2]))
    caught_f = caught_l = 0.0
    n_alert = n_tp = 0
    for _, d in df.groupby("day"):
        d = d.reset_index(drop=True)
        k = max(1, int(round(k_frac * len(d))))
        sel = policy(d, k, **policy_kw)
        s = d.iloc[sel]
        caught_f += s.loc[s.y == 1, "amount"].sum()
        caught_l += float(np.sum(_vl(s)[s.y.values == 2]))
        n_alert += len(s); n_tp += int((s.y > 0).sum())
    return {"k_frac": k_frac, "VaB_F": caught_f / max(tot_f, 1e-9), "VaB_L": caught_l / max(tot_l, 1e-9),
            "alert_precision": n_tp / max(n_alert, 1), "n_alerts": n_alert}


def vab_grid(df: pd.DataFrame, k_fracs=(0.005, 0.01, 0.02, 0.05), lgfs=(0.5, 1.0),
             recs=(0.2, 0.5, 0.8, 1.0), min_share=None) -> pd.DataFrame:
    """Evaluate all policies over the (lgf, recoverability) grid (comment 52) and report the region
    in which two-queue beats the single value-ranked queue."""
    rows = []
    pols = {"oracle": policy_oracle, "single_queue": policy_single_queue,
            "two_queue": lambda d, k, **kw: policy_value_based(d, k, min_share=min_share or {"F": 0.1, "L": 0.1}, **kw),
            "fraud_first": policy_fraud_first, "fixed_50": policy_fixed_split}
    for kf, lgf, rec in itertools.product(k_fracs, lgfs, recs):
        for name, pol in pols.items():
            kw = {} if name in ("fraud_first", "fixed_50") else {"lgf": lgf, "recoverability": rec}
            r = value_at_budget(df, kf, pol, **kw)
            rows.append({"policy": name, "lgf": lgf, "recoverability": rec, **r,
                         "VaB_total": r["VaB_F"] + r["VaB_L"]})
    out = pd.DataFrame(rows)
    piv = out.pivot_table(index=["k_frac", "lgf", "recoverability"], columns="policy", values="VaB_total")
    piv["two_queue_wins"] = piv["two_queue"] > piv["single_queue"]
    return out, piv


if __name__ == "__main__":
    # Smoke test on synthetic inputs. These numbers are NOT results.
    rng = np.random.default_rng(0)
    n = 50000
    y = rng.choice([0, 1, 2], n, p=[0.98, 0.01, 0.01])
    amount = rng.lognormal(5, 1, n)
    p_f = np.clip(rng.beta(1, 30, n) + 0.5 * (y == 1) * rng.random(n), 0, 1)
    p_l = np.clip(rng.beta(1, 30, n) + 0.7 * (y == 2) * rng.random(n), 0, 1)
    v_l = amount * rng.uniform(0.2, 1.0, n)
    df = pd.DataFrame({"day": rng.integers(0, 30, n), "p_f": p_f, "p_l": p_l, "amount": amount, "v_l": v_l, "y": y})
    _, piv = vab_grid(df, k_fracs=(0.01,), lgfs=(1.0,), recs=(0.2, 0.5, 0.8))
    print(piv.round(3).to_string())

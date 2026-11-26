"""LeakBench: seeded, typed leakage injection into a harmonised transaction/claims log.

NOTE: the injected types mirror what rule-based detectors look for. Experiment E7 is a
GATE: injected results count as evidence only if the injected type mix matches the real BPI 2019 deviation mix.

Input schema (pandas DataFrame): record_id, day (float or int), entity_id, counterparty_id,
amount, currency, y (0 legit / 1 fraud). Output adds injected records with y=2, plus
`injected` and `leak_type` columns. Every injection is reproducible from (seed, rates).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LEAK_TYPES = ["DUP_EXACT", "DUP_NEAR", "WRONG_PAYEE", "TRANSPOSE", "DOUBLE_FX", "SPLIT_PAY"]


def _transpose_amount(a: float, rng: np.random.Generator) -> float:
    """Swap two adjacent digits of the integer part (never the cents pair).
    Fixed 2026-09-16: the previous index range rng.integers(0, len-3) raised for
    3-digit strings (empty range) and could swap into the cents pair; now requires >= 2 integer digits
    that differ, else falls back to a x10 keying error."""
    s = f"{a:.2f}"
    int_part, cents = s.split(".")
    if len(int_part) < 2 or len(set(int_part)) == 1:
        return round(a * 10, 2)
    digits = list(int_part)
    candidates = [i for i in range(len(digits) - 1) if digits[i] != digits[i + 1]]
    if not candidates:
        return round(a * 10, 2)
    i = int(rng.choice(candidates))
    digits[i], digits[i + 1] = digits[i + 1], digits[i]
    return float("".join(digits) + "." + cents)


def inject(df: pd.DataFrame, rate: float = 0.01, seed: int = 0,
           type_mix: dict[str, float] | None = None, hard: bool = False) -> pd.DataFrame:
    """Inject leakage records at `rate` of len(df). `hard=True` widens time windows and
    uses near-identical amounts to defeat naive exact-match rules."""
    rng = np.random.default_rng(seed)
    type_mix = type_mix or {t: 1 / len(LEAK_TYPES) for t in LEAK_TYPES}
    n_inj = int(round(rate * len(df)))
    legit = df[df.y == 0]
    src = legit.sample(n=n_inj, random_state=seed, replace=len(legit) < n_inj).reset_index(drop=True)
    types = rng.choice(list(type_mix), n_inj, p=np.array(list(type_mix.values())) / sum(type_mix.values()))
    out = []
    next_id = int(df.record_id.max()) + 1
    counterparties = df.counterparty_id.unique()
    for (_, r), t in zip(src.iterrows(), types):
        rows = []
        dt_short, dt_long = (1.0, 3.0) if not hard else (3.0, 7.0)
        if t == "DUP_EXACT":
            c = r.copy(); c["day"] = r.day + rng.uniform(0, dt_short); rows.append(c)
        elif t == "DUP_NEAR":
            c = r.copy(); c["day"] = r.day + rng.uniform(0, dt_long)
            jitter = rng.uniform(-0.02, 0.02) if not hard else rng.uniform(-0.005, 0.005)   # precedence fixed (comment 94)
            c["amount"] = r.amount * (1 + jitter)
            rows.append(c)
        elif t == "WRONG_PAYEE":
            c = r.copy(); c["day"] = r.day + rng.uniform(0, dt_short)
            c["counterparty_id"] = rng.choice(counterparties); rows.append(c)
        elif t == "TRANSPOSE":
            c = r.copy(); c["day"] = r.day + rng.uniform(0, dt_short)
            c["amount"] = _transpose_amount(float(r.amount), rng); rows.append(c)
        elif t == "DOUBLE_FX":
            c = r.copy(); c["day"] = r.day + rng.uniform(0, dt_short)
            fx = rng.uniform(1.05, 3.75); c["amount"] = r.amount * fx; rows.append(c)
        elif t == "SPLIT_PAY":
            k = int(rng.integers(2, 5)); w = rng.dirichlet(np.ones(k))
            for j in range(k):
                c = r.copy(); c["day"] = r.day + rng.uniform(0, 1 / 24)
                c["amount"] = r.amount * w[j]; rows.append(c)
        for c in rows:
            c["record_id"] = next_id; next_id += 1
            c["y"] = 2; c["injected"] = 1; c["leak_type"] = t; c["source_record_id"] = r.record_id
            out.append(c)
    base = df.copy()
    base["injected"] = 0; base["leak_type"] = "NONE"; base["source_record_id"] = -1
    res = pd.concat([base, pd.DataFrame(out)], ignore_index=True).sort_values("day").reset_index(drop=True)
    return res


def verify(df_injected: pd.DataFrame) -> dict:
    inj = df_injected[df_injected.injected == 1]
    return {"n_injected": int(len(inj)), "rate": float(len(inj) / len(df_injected)),
            "by_type": inj.leak_type.value_counts().to_dict()}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 10000
    toy = pd.DataFrame({"record_id": np.arange(n), "day": rng.uniform(0, 30, n),
                        "entity_id": rng.integers(0, 500, n), "counterparty_id": rng.integers(0, 2000, n),
                        "amount": np.round(rng.lognormal(5, 1, n), 2), "currency": "SAR",
                        "y": (rng.random(n) < 0.01).astype(int)})
    out = inject(toy, rate=0.01, seed=1)
    print(verify(out))

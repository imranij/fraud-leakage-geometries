"""Regression tests for leakbench/inject.py (index-range fix of 2026-09-16)."""
import numpy as np
import pandas as pd
import pytest

from src.leakbench.inject import _transpose_amount, inject, verify


def test_transpose_short_amounts_do_not_raise():
    rng = np.random.default_rng(0)
    for a in [0.5, 1.0, 9.99, 10.0, 12.34, 100.0, 123.45]:
        out = _transpose_amount(a, rng)
        assert out > 0 and out != a


def test_transpose_never_touches_cents():
    rng = np.random.default_rng(1)
    for a in [1234.56, 987.65, 45.67]:
        out = _transpose_amount(a, rng)
        assert f"{out:.2f}".split(".")[1] == f"{a:.2f}".split(".")[1]


def test_dup_near_jitter_range():
    rng = np.random.default_rng(0)
    n = 3000
    toy = pd.DataFrame({"record_id": np.arange(n), "day": rng.uniform(0, 30, n), "entity_id": rng.integers(0, 50, n),
                        "counterparty_id": rng.integers(0, 200, n), "amount": np.round(rng.lognormal(5, 1, n), 2),
                        "currency": "SAR", "y": 0})
    for hard, tol in [(False, 0.02), (True, 0.005)]:
        out = inject(toy, rate=0.05, seed=2, type_mix={"DUP_NEAR": 1.0}, hard=hard)
        inj = out[out.injected == 1].merge(toy[["record_id", "amount"]], left_on="source_record_id", right_on="record_id", suffixes=("", "_src"))
        ratio = inj.amount / inj.amount_src
        assert ((ratio >= 1 - tol - 1e-9) & (ratio <= 1 + tol + 1e-9)).all()


def test_rate_and_reproducibility():
    rng = np.random.default_rng(3)
    n = 5000
    toy = pd.DataFrame({"record_id": np.arange(n), "day": rng.uniform(0, 30, n), "entity_id": rng.integers(0, 50, n),
                        "counterparty_id": rng.integers(0, 200, n), "amount": np.round(rng.lognormal(5, 1, n), 2),
                        "currency": "SAR", "y": (rng.random(n) < 0.01).astype(int)})
    a, b = inject(toy, rate=0.01, seed=7), inject(toy, rate=0.01, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert verify(a)["n_injected"] >= int(0.01 * n)

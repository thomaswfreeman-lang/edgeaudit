"""
Evaluation-survival invariants.

The simulator must be deterministic, its probabilities must be coherent, and
each rule must actually bind when constructed to bind. None of these tests
pin exact probabilities — the module makes no published numeric claims — but
every structural property a consumer relies on is pinned here.
"""

import numpy as np
import pandas as pd
import pytest

from edgeaudit import demo_data, parsers, survival

RULES = survival.PRESETS["50k_trailing"]


def _trades(pnls, per_day=1):
    """Minimal canonical trades frame: given per-trade dollar P&L."""
    n = len(pnls)
    days = pd.bdate_range("2025-01-02", periods=int(np.ceil(n / per_day)))
    entry = [days[i // per_day] + pd.Timedelta(hours=9, minutes=i % per_day)
             for i in range(n)]
    return pd.DataFrame({"entry_time": entry, "net_pnl": pnls})


def _noise_trades():
    df, _ = parsers.normalize(demo_data.pure_noise(400, seed=3))
    return df


def test_deterministic():
    t = _noise_trades()
    a = survival.simulate_evaluation(t, RULES, n_paths=500)
    b = survival.simulate_evaluation(t, RULES, n_paths=500)
    assert a == b


def test_probabilities_coherent():
    ev = survival.simulate_evaluation(_noise_trades(), RULES, n_paths=1000)
    parts = [ev.p_pass, ev.p_breach_drawdown, ev.p_breach_daily, ev.p_expired]
    assert all(0.0 <= p <= 1.0 for p in parts)
    assert sum(parts) == pytest.approx(1.0)
    assert ev.ci[0] <= ev.p_pass <= ev.ci[1]


def test_steady_winner_passes():
    # +$500 every day, never a losing trade: target in 6 days, nothing binds
    ev = survival.simulate_evaluation(_trades([500.0] * 40), RULES, n_paths=200)
    assert ev.p_pass == 1.0
    assert ev.expected_attempts == 1.0
    assert ev.expected_cost == RULES.fee


def test_steady_loser_never_passes():
    ev = survival.simulate_evaluation(_trades([-400.0] * 40), RULES, n_paths=200)
    assert ev.p_pass == 0.0
    assert np.isinf(ev.expected_attempts) and np.isinf(ev.expected_cost)


def test_daily_loss_limit_binds_before_trailing():
    # -$2,000 in one trade breaches the $1,250 daily limit on day one,
    # before cumulative equity ever reaches the $2,500 trailing threshold
    ev = survival.simulate_evaluation(_trades([-2000.0] * 20), RULES, n_paths=200)
    assert ev.p_breach_daily == 1.0
    assert ev.p_breach_drawdown == 0.0


def test_trailing_drawdown_binds_without_daily_limit():
    rules = survival.EvalRules(
        name="test", account=50_000, target=3_000, trailing_dd=2_500,
        daily_loss=None, min_days=1, fee=85)
    ev = survival.simulate_evaluation(_trades([-2000.0] * 20), rules, n_paths=200)
    assert ev.p_breach_drawdown == 1.0
    assert ev.p_breach_daily == 0.0


def test_zero_edge_trader_sometimes_passes():
    # the uncomfortable fact the module exists to quantify: a coin flip
    # passes evaluations at a healthy clip
    ev = survival.simulate_evaluation(_noise_trades(), RULES, n_paths=1000)
    assert 0.0 < ev.p_pass < 1.0


def test_eod_mode_is_kinder_intraday():
    # a V-shaped day (deep dip, full recovery) breaches per-trade trailing
    # but survives end-of-day checking
    pnls = [-2600.0, +2600.0] * 10
    trades = _trades(pnls, per_day=2)
    trade_mode = survival.EvalRules(
        name="t", account=50_000, target=3_000, trailing_dd=2_500,
        daily_loss=None, min_days=1, fee=85, dd_mode="trade")
    eod_mode = survival.EvalRules(
        name="e", account=50_000, target=3_000, trailing_dd=2_500,
        daily_loss=None, min_days=1, fee=85, dd_mode="eod")
    assert survival.simulate_evaluation(trades, trade_mode, n_paths=100).p_breach_drawdown == 1.0
    assert survival.simulate_evaluation(trades, eod_mode, n_paths=100).p_breach_drawdown == 0.0


def test_empty_trades_raises():
    with pytest.raises(ValueError):
        survival.simulate_evaluation(
            pd.DataFrame({"entry_time": [], "net_pnl": []}), RULES)

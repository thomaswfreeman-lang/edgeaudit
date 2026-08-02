"""
Prop-firm evaluation survival: given the trades you actually take, what are
the odds an evaluation account survives to its target?

Method: resample the trader's own *days* (stationary block bootstrap over the
day sequence, geometric block lengths — same stance as stats.py: dependence
is real, destroying it manufactures confidence) and walk the equity path
through the evaluation rules trade by trade.

Model assumptions, stated because they matter:

- Drawdown is computed on end-of-trade equity ("trade" mode) or end-of-day
  equity ("eod" mode). Intraday excursion inside a single trade is invisible
  in a round-trip export, so a real trailing-threshold account can breach in
  places this model cannot see. Treat p_pass as an upper bound.
- Sizing is whatever the trader historically did. No scaling rules modelled.
- Once the profit target is hit the trader is assumed to coast to any
  minimum-day requirement at negligible risk, so target-hit counts as a pass.
- Attempts that neither pass nor breach within `max_days` count as failures
  (in practice: another month of fees, which the reset-fee model absorbs).
- expected_attempts is geometric (1 / p_pass); expected_cost = attempts * fee.

The presets are honest approximations of common 2025/26 futures-eval rule
sets, not any single firm's contract. Check your firm's numbers and build an
EvalRules with them — everything is a parameter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvalRules:
    name: str
    account: float          # nominal account size, cosmetic
    target: float           # profit target, dollars above start
    trailing_dd: float      # trailing max drawdown, dollars off the high-water
    daily_loss: float | None  # daily loss limit, dollars; None if the firm has none
    min_days: int           # minimum trading days (assumed coastable, see module doc)
    fee: float              # cost of a reset / another attempt
    dd_mode: str = "trade"  # "trade": drawdown checked after every trade; "eod": at day close

    def describe(self) -> str:
        dll = f"${self.daily_loss:,.0f} daily loss limit" if self.daily_loss else "no daily loss limit"
        when = "checked per trade" if self.dd_mode == "trade" else "checked end of day"
        return (f"{self.name}: +${self.target:,.0f} target, "
                f"${self.trailing_dd:,.0f} trailing drawdown ({when}), "
                f"{dll}, min {self.min_days} trading days, ${self.fee:,.0f} a reset")


PRESETS: dict[str, EvalRules] = {
    "50k_trailing": EvalRules(
        name="50K trailing eval", account=50_000, target=3_000,
        trailing_dd=2_500, daily_loss=1_250, min_days=7, fee=85, dd_mode="trade"),
    "50k_eod": EvalRules(
        name="50K end-of-day drawdown eval", account=50_000, target=3_000,
        trailing_dd=2_000, daily_loss=1_250, min_days=7, fee=85, dd_mode="eod"),
    "100k_trailing": EvalRules(
        name="100K trailing eval", account=100_000, target=6_000,
        trailing_dd=3_000, daily_loss=2_000, min_days=7, fee=100, dd_mode="trade"),
}


@dataclass
class EvalOutcome:
    p_pass: float
    ci: tuple                # Wilson 95% interval on p_pass
    p_breach_drawdown: float
    p_breach_daily: float
    p_expired: float         # neither passed nor breached within max_days
    expected_attempts: float
    expected_cost: float
    n_paths: int


def _wilson(k: int, n: int, z: float = 1.959964) -> tuple:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _days_from_trades(trades: pd.DataFrame) -> list[np.ndarray]:
    """Per-day arrays of net P&L, chronological within and across days."""
    df = trades.sort_values("entry_time")
    dates = pd.to_datetime(df["entry_time"]).dt.normalize()
    return [g.to_numpy(dtype=float) for _, g in df.groupby(dates)["net_pnl"]]


def simulate_evaluation(
    trades: pd.DataFrame,
    rules: EvalRules,
    n_paths: int = 4000,
    seed: int = 7,
    max_days: int = 60,
) -> EvalOutcome:
    """
    Monte Carlo of evaluation attempts built from the trader's own days.

    Day sequence per path: stationary bootstrap (random start, continue to the
    next historical day with high probability, jump to a random day otherwise;
    geometric blocks, mean length n_days^(1/3)) — preserves multi-day streaks.
    """
    days = _days_from_trades(trades)
    n_days = len(days)
    if n_days == 0:
        raise ValueError("no trades to simulate from")

    rng = np.random.default_rng(seed)
    p_new = 1.0 / max(1.0, float(n_days) ** (1.0 / 3.0))

    PASS, BREACH_DD, BREACH_DAILY, EXPIRED = 0, 1, 2, 3
    outcomes = np.empty(n_paths, dtype=np.int64)

    for path in range(n_paths):
        equity, hw = 0.0, 0.0
        day_idx = int(rng.integers(0, n_days))
        result = EXPIRED
        for _ in range(max_days):
            day_pnl = 0.0
            for pnl in days[day_idx]:
                equity += pnl
                day_pnl += pnl
                if rules.daily_loss is not None and day_pnl <= -rules.daily_loss:
                    result = BREACH_DAILY
                    break
                if rules.dd_mode == "trade":
                    hw = max(hw, equity)
                    if equity <= hw - rules.trailing_dd:
                        result = BREACH_DD
                        break
                if equity >= rules.target:
                    result = PASS
                    break
            if result != EXPIRED:
                break
            if rules.dd_mode == "eod":
                hw = max(hw, equity)
                if equity <= hw - rules.trailing_dd:
                    result = BREACH_DD
                    break
            day_idx = int(rng.integers(0, n_days)) if rng.random() < p_new \
                else (day_idx + 1) % n_days
        outcomes[path] = result

    k_pass = int((outcomes == PASS).sum())
    p_pass = k_pass / n_paths
    attempts = 1.0 / p_pass if p_pass > 0 else np.inf
    return EvalOutcome(
        p_pass=p_pass,
        ci=_wilson(k_pass, n_paths),
        p_breach_drawdown=float((outcomes == BREACH_DD).mean()),
        p_breach_daily=float((outcomes == BREACH_DAILY).mean()),
        p_expired=float((outcomes == EXPIRED).mean()),
        expected_attempts=attempts,
        expected_cost=attempts * rules.fee,
        n_paths=n_paths,
    )

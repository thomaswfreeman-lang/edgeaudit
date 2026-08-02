"""
Schwab/thinkorswim futures-section regressions against a small anonymised
fixture with hand-computed P&L. Every quirk found in a real 83k-line Schwab
statement is pinned here:

- overnight ADJ mark-to-market rows belong to the position open at settlement
- positions opened before the statement window are inferred from the
  open/close pattern and their round trips dropped, never guessed at
- exercises close the option and synthesise the implied futures at the strike
  (skipping them mispairs the underlying AND deletes the premium loss)
- worthless expirations close the whole position at zero
- CBOT 32nds tick prices (/ZN 111'240) parse
- net is ALWAYS gross minus both fills' fees, broker realised is gross
- the reconciliation accounts for every ledger dollar: captured + dropped +
  unmatched = section total
"""

from pathlib import Path

import pytest

from edgeaudit import audit, parsers

FIXTURE = str(Path(__file__).parent / "fixtures" / "tos_statement_small.csv")


@pytest.fixture(scope="module")
def parsed():
    df, fmt = parsers.normalize(FIXTURE)
    assert fmt.startswith("Schwab / thinkorswim")
    return df, parsers.diagnose(FIXTURE)


def test_trade_count(parsed):
    df, _ = parsed
    assert len(df) == 7


def test_simple_round_trip_fees_in_and_out(parsed):
    df, _ = parsed
    t = df[df["symbol"] == "MNQZ25"].iloc[0]
    assert t["direction"] == "long"
    assert t["qty"] == 2
    assert t["gross_pnl"] == pytest.approx(40.0)          # broker's dollars
    assert t["commission"] == pytest.approx(9.60)         # open 4.80 + close 4.80
    assert t["net_pnl"] == pytest.approx(40.0 - 9.60)


def test_overnight_adj_folded_into_trade(parsed):
    df, _ = parsed
    t = df[df["symbol"] == "MESZ25"].iloc[0]
    assert t["direction"] == "short"
    # -50 settlement mark + 25 on the close = -25 gross (6900 -> 6905 short)
    assert t["gross_pnl"] == pytest.approx(-25.0)
    assert t["net_pnl"] == pytest.approx(-25.0 - 4.80)


def test_pre_window_position_inferred_and_dropped(parsed):
    df, diag = parsed
    assert diag["inferred_start_positions"] == {"ESZ25": 1.0}
    assert diag["dropped_prewindow_trades"] == 1
    t = df[df["symbol"] == "ESZ25"].iloc[0]            # the in-window trade
    assert t["gross_pnl"] == pytest.approx(300.0)
    assert t["entry_price"] == pytest.approx(6795.0)


def test_option_round_trip_priced_by_multiplier(parsed):
    df, _ = parsed
    t = df[df["symbol"].str.startswith("GCJ26_")].iloc[0]
    assert t["root"] == "GC"
    assert t["gross_pnl"] == pytest.approx(250.0)         # (12.50-10.00) x 100
    assert t["net_pnl"] == pytest.approx(250.0 - 4.84)


def test_expired_option_counted_as_loss(parsed):
    df, _ = parsed
    t = df[df["symbol"].str.startswith("ESM26_")].iloc[0]
    assert t["direction"] == "long"
    assert t["gross_pnl"] == pytest.approx(-500.0)        # 2 x 5.00 x 50 premium
    assert t["net_pnl"] == pytest.approx(-500.0 - 2.42)


def test_exercise_closes_option_and_creates_futures_trade(parsed):
    df, diag = parsed
    opt = df[df["symbol"].str.startswith("NQH26_")].iloc[0]
    assert opt["gross_pnl"] == pytest.approx(-200.0)      # premium lost at exercise
    fut = df[df["symbol"] == "NQH26"].iloc[0]
    assert fut["direction"] == "long"
    assert fut["entry_price"] == pytest.approx(24000.0)   # the strike
    assert fut["gross_pnl"] == pytest.approx(200.0)       # broker realised on close
    # and the synthesised entry must not trip the pre-window inference
    assert "NQH26" not in diag["inferred_start_positions"]


def test_open_at_eof_dropped_not_counted(parsed):
    df, _ = parsed
    assert not df["symbol"].eq("MGCG26").any()


def test_reconciliation_accounts_for_every_dollar(parsed):
    df, diag = parsed
    r = diag["reconciliation"]
    captured = float(df.loc[df["venue"] == "futures", "gross_pnl"].sum())
    assert r["captured_futures_gross"] == pytest.approx(captured)
    # unaccounted = the pre-window close (+100) + the unparsed EFP row (+12.34)
    assert r["unaccounted"] == pytest.approx(100.0 + 12.34)


def test_unmatched_rows_counted(parsed):
    _, diag = parsed
    # the EFP row and the informational "Opening futures position" notice
    assert diag["unmatched_futures_rows"] == 2


def test_roots_filter_scopes_audit_and_is_disclosed():
    res = audit.run(FIXTURE, roots=["ES", "MES", "NQ", "MNQ"], resamples=200)
    assert res.n_trades == 6                              # the GC option is excluded
    assert set(res.trades["root"]) <= {"ES", "MES", "NQ", "MNQ"}
    assert "filtered to ES, MES, MNQ, NQ" in res.format_name


def test_treasury_tick_prices():
    assert parsers._price_num("111'240") == pytest.approx(111 + 24.0 / 32)
    assert parsers._price_num("108'255") == pytest.approx(108 + 25.5 / 32)
    assert parsers._price_num("6,878.75") == pytest.approx(6878.75)


def test_futures_root_extraction():
    for sym, root in [("/MNQZ25:XCME", "MNQ"), ("SILH26", "SIL"),
                      ("1OZJ26", "1OZ"), ("/6JH26:XCME", "6J"),
                      ("QGG26", "QG"), ("MHGH26", "MHG")]:
        assert parsers._futures_root(sym) == root, sym


def test_scale_out_is_one_trade_not_a_run():
    """A single closing fill over several FIFO lots is one decision; counting
    it as a run of same-sign trades manufactures state-dependence in the
    after-a-loss slices and inflates every sequence q-value."""
    import pandas as pd
    fills = pd.DataFrame({
        "Symbol": ["ESH5"] * 3,
        "Buy/Sell": ["Buy", "Buy", "Sell"],
        "Qty": [1, 1, 2],
        "Price": [5000, 5010, 5030],
        "Fill Time": pd.date_range("2025-01-02 09:30", periods=3, freq="5min"),
    })
    out, _ = parsers.normalize(fills)
    assert len(out) == 1
    assert out["qty"].iloc[0] == 2
    assert out["entry_price"].iloc[0] == pytest.approx(5005.0)   # qty-weighted
    assert out["net_pnl"].iloc[0] == pytest.approx(50.0)         # 30 + 20 points


def test_r_unit_is_per_class_when_scales_differ():
    """$5 SLV losses and $500 SIL losses must not share one 1R unit."""
    import numpy as np, pandas as pd
    rng = np.random.default_rng(0)
    small = rng.normal(0, 6, 60)
    big = rng.normal(0, 600, 60)
    df = pd.DataFrame({
        "symbol": ["SLV"] * 60 + ["SILH26"] * 60,
        "venue": ["equity"] * 60 + ["futures"] * 60,
        "net_pnl": np.r_[small, big],
    })
    r, basis = parsers.to_r_multiples(df)
    assert "per instrument class" in basis
    med_small = np.median(np.abs(r[:60][r[:60] < 0]))
    med_big = np.median(np.abs(r[60:][r[60:] < 0]))
    assert med_small == pytest.approx(1.0, abs=0.01)
    assert med_big == pytest.approx(1.0, abs=0.01)


def test_r_unit_stays_global_on_single_scale_records():
    """The per-class unit must NOT engage on homogeneous records — the
    published post's figures are generated from one, and its reproduction
    is load-bearing."""
    from edgeaudit import demo_data
    df, _ = parsers.normalize(demo_data.pure_noise(400, seed=3))
    _, basis = parsers.to_r_multiples(df)
    assert basis == "median absolute loss (estimated 1R)"

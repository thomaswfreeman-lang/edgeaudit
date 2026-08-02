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

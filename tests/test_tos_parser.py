"""
thinkorswim Account Statement parser regressions.

Every quirk found in a real Schwab/TOS export is pinned here against a small
anonymised fixture with hand-computed P&L:

- Amount books realised P&L on the closing fill for outrights, premium flow
  on every fill for options on futures
- overnight ADJ mark-to-market rows belong to the episode open at settlement
- positions opened before the statement window are inferred and their
  (unattributable) episodes dropped, not guessed
- positions still open at end of file are dropped, not counted as winners
- FUT_ADJ account-level adjustments are excluded and reported
- unparseable TRD descriptions are reported, never silently skipped
"""

from pathlib import Path

import pytest

from edgeaudit import parsers

FIXTURE = str(Path(__file__).parent / "fixtures" / "tos_statement_small.csv")


@pytest.fixture(scope="module")
def parsed():
    text = parsers._tos_text(FIXTURE)
    assert text is not None, "fixture not recognised as a TOS statement"
    return parsers._parse_tos_statement(text)


def test_detected_via_normalize():
    df, fmt = parsers.normalize(FIXTURE)
    assert fmt.startswith("thinkorswim")
    assert len(df) == 7


def test_simple_round_trip(parsed):
    df, _ = parsed
    t = df[df["symbol"] == "/MNQZ25"].iloc[0]
    assert t["direction"] == "long"
    assert t["qty"] == 2
    assert t["gross_pnl"] == pytest.approx(40.0)
    assert t["net_pnl"] == pytest.approx(40.0 - 9.60)  # both fills' fees


def test_overnight_adj_folded_into_episode(parsed):
    df, _ = parsed
    t = df[df["symbol"] == "/MESZ25"].iloc[0]
    assert t["direction"] == "short"
    # -50 settlement mark + 25 on the close = -25 gross (6900 -> 6905 short)
    assert t["gross_pnl"] == pytest.approx(-25.0)
    assert t["net_pnl"] == pytest.approx(-25.0 - 4.80)


def test_pre_window_position_inferred_and_dropped(parsed):
    df, diag = parsed
    # first /ES fill closes a long opened before the statement window
    assert diag["inferred_start_positions"] == {"/ESZ25": 1}
    assert diag["dropped_partial_episodes"] == {"/ESZ25": 1}
    # the in-window /ES round trip is kept and correct
    t = df[df["symbol"] == "/ESZ25"].iloc[0]
    assert t["gross_pnl"] == pytest.approx(300.0)
    assert t["entry_price"] == pytest.approx(6795.0)


def test_option_round_trip_premium_flow(parsed):
    df, _ = parsed
    t = df[df["symbol"].str.startswith("GC ")].iloc[0]
    assert parsers._root(t["symbol"]) == "GC"
    assert t["gross_pnl"] == pytest.approx(250.0)     # -1000 + 1250
    assert t["net_pnl"] == pytest.approx(250.0 - 4.84)


def test_expired_option_counted_as_loss(parsed):
    # bought 2 calls for $500 premium, expired worthless: the loss must be a
    # trade, not a dropped open position
    df, diag = parsed
    assert diag["expiration_rows"] == 1
    t = df[df["symbol"] == "ES 7000 CALL 07JAN26"].iloc[0]
    assert t["direction"] == "long"
    assert t["gross_pnl"] == pytest.approx(-500.0)
    assert t["net_pnl"] == pytest.approx(-500.0 - 2.42)
    assert "ES 7000 CALL 07JAN26" not in diag["dropped_open_at_eof"]


def test_exercise_creates_futures_trade_not_dropped_episode(parsed):
    # long call exercised: option episode closes at -premium, and the
    # resulting futures position must become a real trade at the strike --
    # dropping it deletes the winning half of every exercised call
    df, diag = parsed
    assert diag["exercise_rows"] == 1
    opt = df[df["symbol"] == "NQ 24000 CALL 07JAN26"].iloc[0]
    assert opt["gross_pnl"] == pytest.approx(-200.0)          # premium lost
    fut = df[df["symbol"] == "/NQH26"].iloc[0]
    assert fut["direction"] == "long"
    assert fut["entry_price"] == pytest.approx(24000.0)       # the strike
    assert fut["gross_pnl"] == pytest.approx(200.0)           # 24000 -> 24010
    assert "/NQH26" not in diag["dropped_partial_episodes"]


def test_treasury_tick_prices():
    assert parsers._tos_price("111'240") == pytest.approx(111 + 24.0 / 32)
    assert parsers._tos_price("108'255") == pytest.approx(108 + 25.5 / 32)
    assert parsers._tos_price("6,878.75") == pytest.approx(6878.75)


def test_open_at_eof_dropped_not_counted(parsed):
    df, diag = parsed
    assert "/MGCG26" not in set(df["symbol"])
    assert diag["dropped_open_at_eof"] == {"/MGCG26": 1}


def test_fut_adj_excluded_and_reported(parsed):
    _, diag = parsed
    assert diag["fut_adj_excluded"] == [500.0]


def test_unparsed_descriptions_reported(parsed):
    _, diag = parsed
    # the EFP row and the informational "Opening futures position" notice
    # (the position itself is synthesised from the Exercise row)
    assert len(diag["unparsed_trd"]) == 2
    assert any("EXCHANGE FOR PHYSICAL" in d for d in diag["unparsed_trd"])
    assert any("Opening futures position" in d for d in diag["unparsed_trd"])


def test_reconciliation_accounts_for_every_dollar(parsed):
    df, diag = parsed
    r = diag["reconciliation"]
    assert r["section_amount_total"] == pytest.approx(
        r["captured_in_trades"] + r["in_dropped_episodes"] + r["in_unparsed_rows"])
    assert r["in_dropped_episodes"] == pytest.approx(100.0)   # pre-window close
    assert r["in_unparsed_rows"] == pytest.approx(12.34)      # the EFP row


def test_futures_root_extraction():
    for sym, root in [("/MNQZ25:XCME", "MNQ"), ("/SILH26", "SIL"),
                      ("1OZJ26", "1OZ"), ("/6JH26:XCME", "6J"),
                      ("/QGG26", "QG"), ("ES 03-25", "ES"), ("MHGH26", "MHG")]:
        assert parsers._root(sym) == root, sym

"""
These tests assert the product's claim, not just that the code runs.

If test_noise_yields_no_verified_slices ever fails, the tool has started
manufacturing edges and must not ship.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgeaudit import audit, demo_data, parsers, report, stats  # noqa: E402

RES = 1200


# --- statistical guarantees -------------------------------------------------

def test_noise_yields_no_verified_slices():
    for seed in (3, 17, 42):
        res = audit.run(demo_data.pure_noise(400, seed=seed), resamples=RES)
        assert res.n_survived == 0, f"seed {seed} manufactured {res.n_survived} edges from noise"
        assert not res.verdict.startswith("Positive")


def test_naive_analysis_would_have_found_edges_in_noise():
    """The marketing claim, asserted: uncorrected testing finds edges in nothing."""
    hits = [audit.run(demo_data.pure_noise(500, seed=s), resamples=RES).n_naive_significant
            for s in (3, 17, 42, 88)]
    assert sum(hits) > 0


def test_uniform_edge_is_detected_globally_but_not_localised():
    res = audit.run(demo_data.real_edge(500, seed=11), resamples=RES)
    assert res.verdict.startswith("Positive")
    assert res.n_survived == 0, "a uniform edge must not be attributed to any one slice"


def test_concentrated_edge_is_found():
    res = audit.run(demo_data.concentrated_edge(600, seed=5), resamples=RES)
    verified = res.buckets[res.buckets["verified"]]
    assert len(verified) >= 1
    assert any("RTH open" in lbl for lbl in verified["label"])


def test_small_slices_are_excluded_not_reported():
    res = audit.run(demo_data.pure_noise(120, seed=9), resamples=RES, min_bucket_n=40)
    untested = res.buckets[~res.buckets["testable"]]
    assert untested["p_vs_zero"].isna().all()
    assert untested["verified"].fillna(False).eq(False).all()


# --- statistical primitives -------------------------------------------------

def test_bh_controls_family_and_ignores_nans():
    p = np.array([0.001, 0.02, 0.3, np.nan, 0.9])
    rej, q = stats.benjamini_hochberg(p, q=0.10)
    assert rej[0] and not rej[2] and not rej[3]
    assert np.isnan(q[3])


def test_shrinkage_collapses_when_spread_is_pure_noise():
    rng = np.random.default_rng(0)
    means = rng.normal(0.0, 0.3, 20)      # spread entirely from sampling noise
    ses = np.full(20, 0.3)
    shrunk, w = stats.shrink_toward_global(means, ses, 0.0)
    assert np.nanmax(np.abs(shrunk)) < np.max(np.abs(means))
    assert np.nanmean(w) < 0.5


def test_block_bootstrap_widens_intervals_under_dependence():
    rng = np.random.default_rng(1)
    e = rng.normal(0, 1, 600)
    x = np.zeros(600)
    for i in range(1, 600):          # strongly autocorrelated series
        x[i] = 0.75 * x[i - 1] + e[i]
    iid = stats.bootstrap_mean(x, resamples=RES, seed=1, block=False)
    blk = stats.bootstrap_mean(x, resamples=RES, seed=1, block=True)
    assert (blk["hi"] - blk["lo"]) > (iid["hi"] - iid["lo"])


def test_min_track_record_grows_as_sharpe_shrinks():
    rng = np.random.default_rng(2)
    strong = rng.normal(0.30, 1.0, 500)
    weak = rng.normal(0.05, 1.0, 500)
    assert stats.min_track_record_length(weak) > stats.min_track_record_length(strong)


def test_deflated_sharpe_penalises_searching():
    rng = np.random.default_rng(4)
    x = rng.normal(0.12, 1.0, 400)
    few = stats.deflated_sharpe(x, rng.normal(0.1, 0.2, 3))
    many = stats.deflated_sharpe(x, rng.normal(0.1, 0.2, 60))
    assert many < few


def test_independence_check_flags_autocorrelation():
    rng = np.random.default_rng(5)
    x = np.repeat(rng.normal(0, 1, 100), 4)   # heavy serial dependence
    assert stats.independence_check(x)["dependent"]


# --- intake -----------------------------------------------------------------

def test_ninjatrader_format_detected():
    df, name = parsers.normalize(demo_data.pure_noise(80))
    assert "NinjaTrader" in name
    assert df["direction"].isin(["long", "short"]).all()
    assert df["net_pnl"].notna().all()


def test_fill_level_fifo_pairing():
    fills = pd.DataFrame({
        "Symbol": ["ESH5"] * 4,
        "Buy/Sell": ["Buy", "Buy", "Sell", "Sell"],
        "Qty": [1, 1, 1, 1],
        "Price": [5000, 5010, 5020, 5030],
        "Fill Time": pd.date_range("2025-01-02 09:30", periods=4, freq="5min"),
        "Account": ["APEX-1"] * 4,
    })
    out, name = parsers.normalize(fills)
    assert "Rithmic" in name
    assert len(out) == 2
    assert out["direction"].eq("long").all()
    assert out["net_pnl"].tolist() == [20.0, 20.0]   # FIFO: 5020-5000, 5030-5010


def test_open_position_is_dropped_not_counted():
    fills = pd.DataFrame({
        "Symbol": ["NQH5"] * 3, "Buy/Sell": ["Buy", "Sell", "Buy"],
        "Qty": [2, 2, 1], "Price": [20000, 20050, 20100],
        "Fill Time": pd.date_range("2025-01-02 10:00", periods=3, freq="1min"),
    })
    out, _ = parsers.normalize(fills)
    assert len(out) == 1


def test_unrecognised_file_fails_loudly():
    with pytest.raises(parsers.ParseError):
        parsers.normalize(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))


def test_r_basis_is_disclosed():
    _, basis = parsers.to_r_multiples(
        pd.DataFrame({"net_pnl": [100, -50, 200, -50, -60, -40, 90]}))
    assert "1R" in basis or "P&L" in basis


# --------------------------------------------------------------------------
# Schwab / thinkorswim statement intake, incl. the futures section.
#
# The regression these guard is specific and was live: in the futures ledger
# the Amount column means realised P&L for outright futures but *premium cash
# flow* for futures options. Treating both as P&L turned a -$315k record into
# a +$7.4mm one. If test_futures_option_amount_is_not_pnl ever fails, the
# tool is reporting cash flows as profit and must not ship.
# --------------------------------------------------------------------------

SCHWAB_HEAD = "Account Statement for 12345678SCHW (Individual) since 11/3/25 through 8/1/26\n\n"
CASH_HDR = "Cash Balance\nDATE,TIME,TYPE,REF #,DESCRIPTION,Misc Fees,Commissions & Fees,AMOUNT,BALANCE\n"
FUT_HDR = ("Futures Statements\nTrade Date,Exec Date,Exec Time,Type,Ref #,Description,"
           "Misc Fees,Commissions & Fees,Amount,Balance\n")


def _stmt(cash_rows="", fut_rows=""):
    s = SCHWAB_HEAD + CASH_HDR
    s += cash_rows or "11/5/25,00:00:00,BAL,,Cash balance at the start of business day,,,,0.00\n"
    s += "\n"
    if fut_rows:
        s += FUT_HDR + fut_rows + "\n"
    return io.StringIO(s)


def test_schwab_equity_roundtrip():
    cash = ("11/5/25,09:30:00,TRD,=\"1\",BOT +100 AAPL @100.00,,,\"-10,000.00\",0.00\n"
            "11/5/25,10:30:00,TRD,=\"2\",SOLD -100 AAPL @101.00,,,\"10,100.00\",0.00\n")
    df, fmt = parsers.normalize(_stmt(cash))
    assert "Schwab" in fmt
    assert len(df) == 1
    assert df.net_pnl.iloc[0] == pytest.approx(100.0)
    assert df.venue.iloc[0] == "equity"


def test_outright_futures_uses_broker_realised_pnl():
    """
    Amount on an outright futures close IS realised P&L -- it becomes the
    trade's GROSS verbatim (dollars from the broker, never points*1). Net is
    gross minus BOTH fills' fees: a "net" that quietly excludes commissions
    would flatter exactly the overtrading this tool exists to expose.
    """
    fut = ("11/5/25,11/4/25,17:00:00,TRD,=\"1\",BOT +1 /ESZ25:XCME @6800.00,-1.40,-1.00,--,0.00\n"
           "11/5/25,11/4/25,17:05:00,TRD,=\"2\",SOLD -1 /ESZ25:XCME @6810.00,-1.40,-1.00,500.00,0.00\n")
    df, _ = parsers.normalize(_stmt(fut_rows=fut))
    f = df[df.venue == "futures"]
    assert len(f) == 1
    assert f.gross_pnl.iloc[0] == pytest.approx(500.0)      # broker's dollars
    assert f.commission.iloc[0] == pytest.approx(4.80)      # in + out
    assert f.net_pnl.iloc[0] == pytest.approx(500.0 - 4.80)
    assert f.root.iloc[0] == "ES"


def test_futures_option_amount_is_not_pnl():
    """
    Amount on a futures OPTION is premium cash flow, not P&L. Buying at 2.00
    and selling at 3.00 on a 1/5000 contract is +$5,000 -- NOT the +$15,000
    you get by summing the two cash flows as though they were profits.
    """
    fut = ("2/23/26,2/23/26,10:00:00,TRD,=\"1\",BOT +1 /SIH26:XCEC 1/5000 24 FEB 26 (MAR26) "
           "/SOH26:XCEC 86 CALL @2.00,,,\"-10,000.00\",0.00\n"
           "2/23/26,2/23/26,11:00:00,TRD,=\"2\",SOLD -1 /SIH26:XCEC 1/5000 24 FEB 26 (MAR26) "
           "/SOH26:XCEC 86 CALL @3.00,,,\"15,000.00\",0.00\n")
    df, _ = parsers.normalize(_stmt(fut_rows=fut))
    f = df[df.venue == "futures"]
    assert len(f) == 1
    assert f.net_pnl.iloc[0] == pytest.approx(5000.0)   # 1.00 x 5000
    assert f.net_pnl.iloc[0] != pytest.approx(5000.0 + 10000.0)
    assert f.root.iloc[0] == "SI"


def test_expired_option_is_closed_not_dropped():
    """An option that expires worthless is a 100% loss and must be recorded."""
    fut = ("4/1/26,4/1/26,10:00:00,TRD,=\"1\",BOT +1 /NQM26:XCME 1/20 2 APR 26 (Wk1) "
           "/Q1DJ26:XCME 24250 CALL @10.00,,,\"-200.00\",0.00\n"
           "4/2/26,4/2/26,16:00:00,TRD,=\"2\",Removal of option due to expiration of /NQM26 "
           "XCME 20 (WEEKLY) 2 Apr 2026 24250.0 CALL,,,--,0.00\n")
    df, _ = parsers.normalize(_stmt(fut_rows=fut))
    f = df[df.venue == "futures"]
    assert len(f) == 1, "expiry must close the position, not leave it open"
    assert f.net_pnl.iloc[0] == pytest.approx(-200.0)   # premium fully lost


def test_venue_is_a_tested_slice():
    """Pooling a futures book with an equity book is the thing to avoid."""
    cash = ("11/5/25,09:30:00,TRD,=\"1\",BOT +100 AAPL @100.00,,,\"-10,000.00\",0.00\n"
            "11/5/25,10:30:00,TRD,=\"2\",SOLD -100 AAPL @101.00,,,\"10,100.00\",0.00\n")
    fut = ("11/5/25,11/4/25,17:00:00,TRD,=\"1\",BOT +1 /ESZ25:XCME @6800.00,,,--,0.00\n"
           "11/5/25,11/4/25,17:05:00,TRD,=\"2\",SOLD -1 /ESZ25:XCME @6810.00,,,500.00,0.00\n")
    df, _ = parsers.normalize(_stmt(cash, fut))
    dims = {c["dimension"] for c in bkt_build(df)}
    assert "Venue" in dims


def bkt_build(df, min_n=1):
    from edgeaudit import buckets
    return buckets.build(df, min_n=min_n)


def test_variance_concentration_flags_a_few_trades_carrying_a_book():
    rng = np.random.default_rng(3)
    x = np.r_[rng.normal(0, 1, 990), np.full(10, 400.0)]   # 1% carries everything
    c = stats.variance_concentration(x)
    assert c["effective_n"] < 100          # nominal n is 1000
    assert c["top1pct_variance_share"] > 0.9
    even = stats.variance_concentration(rng.normal(0, 1, 1000))
    assert even["effective_n"] > 400       # no concentration -> near n


def test_nan_fees_do_not_poison_commission():
    """
    `float(np.nan or 0.0)` is NaN, because NaN is truthy. That idiom put NaN
    commissions on real round trips. Blank fee cells must read as zero.
    """
    assert parsers._num0(np.nan) == 0.0
    assert parsers._num0(None) == 0.0
    assert parsers._num0("") == 0.0
    assert parsers._num0(np.nan, 1.0) == 1.0
    assert parsers._num0(2.5) == 2.5
    cash = ("11/5/25,09:30:00,TRD,=\"1\",BOT +100 AAPL @100.00,,,\"-10,000.00\",0.00\n"
            "11/5/25,10:30:00,TRD,=\"2\",SOLD -100 AAPL @101.00,,,\"10,100.00\",0.00\n")
    df, _ = parsers.normalize(_stmt(cash))          # both fee cells blank
    assert df.commission.notna().all(), "blank fee cells must not produce NaN commission"
    assert df.net_pnl.notna().all()


def test_drawdown_profile_finds_the_giveback():
    pnl = np.r_[np.full(50, 100.0), np.full(30, -200.0)]   # up 5000, down 6000
    c = stats.drawdown_profile(pnl)
    assert c["peak"] == pytest.approx(5000.0)
    assert c["final"] == pytest.approx(-1000.0)
    assert c["max_drawdown"] == pytest.approx(-6000.0)
    assert c["longest_underwater_trades"] == 30


def test_trades_to_detect_scales_with_variance_squared():
    """Halving dispersion must quarter the sample -- the learning-rate claim."""
    a = stats.trades_to_detect(100.0, [10.0])[10.0]
    b = stats.trades_to_detect(50.0, [10.0])[10.0]
    assert a / b == pytest.approx(4.0, rel=0.02)


def test_regime_trend_detects_a_real_improvement_and_not_a_fake_one():
    rng = np.random.default_rng(11)
    improving = np.r_[rng.normal(-10, 20, 200), rng.normal(0, 20, 200), rng.normal(+10, 20, 200)]
    t = stats.regime_trend(improving, resamples=1500, seed=3)
    assert t["improved"] is True, "a genuine +20/trade shift must be detected"

    flat = rng.normal(0, 20, 600)
    t2 = stats.regime_trend(flat, resamples=1500, seed=3)
    assert t2["improved"] is False and t2["worsened"] is False, "noise must not read as a trend"


def test_money_and_trend_reach_the_result():
    res = audit.run(demo_data.concentrated_edge(n=400, seed=5), resamples=400)
    assert res.money["per_trade"] == pytest.approx(res.money["net"] / res.n_trades)
    assert np.isfinite(res.money["breakeven_win_rate"])
    assert res.equity_curve["final"] == pytest.approx(res.money["net"], rel=1e-6)
    assert res.forward["table"]


def test_html_renders_for_a_multi_venue_statement():
    """
    The by-account-type table only appears when a statement carries more than
    one venue, so the demo files never exercise it. It shipped broken once.
    """
    cash = ("11/5/25,09:30:00,TRD,=\"1\",BOT +100 AAPL @100.00,,,\"-10,000.00\",0.00\n"
            "11/5/25,10:30:00,TRD,=\"2\",SOLD -100 AAPL @101.00,,,\"10,100.00\",0.00\n")
    fut = ("11/5/25,11/4/25,17:00:00,TRD,=\"1\",BOT +1 /ESZ25:XCME @6800.00,,,--,0.00\n"
           "11/5/25,11/4/25,17:05:00,TRD,=\"2\",SOLD -1 /ESZ25:XCME @6810.00,,,500.00,0.00\n")
    res = audit.run(_stmt(cash, fut), min_bucket_n=1, resamples=200)
    assert len(res.by_class) == 2
    out = report.to_html(res, subject="multi-venue")
    assert "By account type" in out and "<html" in out
    report.to_markdown(res)


def test_drawdown_never_reports_an_absurd_giveback_percentage():
    """A tiny early peak followed by a huge loss must not print 17543%."""
    c = stats.drawdown_profile(np.r_[100.0, -50_000.0])
    assert not np.isfinite(c["pct_of_peak_given_back"])
    assert c["drawdown_vs_peak_multiple"] > 1
    c2 = stats.drawdown_profile(np.r_[np.full(10, 100.0), np.full(3, -100.0)])
    assert 0 < c2["pct_of_peak_given_back"] <= 1


def test_redacted_upload_shape_still_parses():
    """
    Contract between the upload page's browser-side redaction and the parser.
    The page strips the account number, the REF # column, every running
    BALANCE and the equity ledger's AMOUNT column before anything is sent.
    If the parser ever starts depending on those, the free check silently
    breaks for every visitor while the CLI keeps working.
    """
    redacted = io.StringIO(
        "Account Statement for [REDACTED] (Individual) since 11/3/25 through 8/1/26\n\n"
        "Cash Balance\n"
        "DATE,TIME,TYPE,,DESCRIPTION,Misc Fees,Commissions & Fees,,\n"
        "11/5/25,09:30:00,TRD,,BOT +100 AAPL @100.00,,,,\n"
        "11/5/25,10:30:00,TRD,,SOLD -100 AAPL @101.00,,,,\n\n"
        "Futures Statements\n"
        "Trade Date,Exec Date,Exec Time,Type,,Description,Misc Fees,Commissions & Fees,Amount,\n"
        "11/5/25,11/4/25,17:00:00,TRD,,BOT +1 /ESZ25:XCME @6800.00,,,--,\n"
        "11/5/25,11/4/25,17:05:00,TRD,,SOLD -1 /ESZ25:XCME @6810.00,,,500.00,\n"
    )
    df, fmt = parsers.normalize(redacted)
    assert "Schwab" in fmt
    assert set(df.venue) == {"equity", "futures"}
    assert df.net_pnl.sum() == pytest.approx(600.0)   # 100 equity + 500 futures


def test_report_teaches_its_own_vocabulary():
    """Hover glossary, edge-finder contrast and the re-audit offer are load-
    bearing product features of the HTML report, not decoration."""
    res = audit.run(demo_data.pure_noise(200, seed=3), resamples=RES)
    h = report.to_html(res)
    assert h.count('class="term"') >= 15          # tooltips on terms of art
    assert 'data-tip=' in h
    assert "edge finder would have told you" in h.lower()
    assert "Re-audit in 30" in h

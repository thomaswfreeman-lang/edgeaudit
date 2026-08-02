"""
These tests assert the product's claim, not just that the code runs.

If test_noise_yields_no_verified_slices ever fails, the tool has started
manufacturing edges and must not ship.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgeaudit import audit, demo_data, parsers, stats  # noqa: E402

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

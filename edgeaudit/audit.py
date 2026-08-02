"""Orchestration: raw export in, verdict out."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import buckets as bkt
from . import parsers, stats as st


@dataclass
class AuditResult:
    format_name: str
    r_basis: str
    n_trades: int
    date_range: tuple
    global_stats: dict = field(default_factory=dict)
    adequacy: dict = field(default_factory=dict)
    independence: dict = field(default_factory=dict)
    buckets: pd.DataFrame = field(default_factory=pd.DataFrame)
    family_size: int = 0
    n_naive_significant: int = 0
    n_survived: int = 0
    verdict: str = ""
    verdict_detail: str = ""
    coverage: dict = field(default_factory=dict)
    money: dict = field(default_factory=dict)
    equity_curve: dict = field(default_factory=dict)
    trend: dict = field(default_factory=dict)
    forward: dict = field(default_factory=dict)
    by_class: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)


def run(
    source,
    min_bucket_n: int = st.MIN_BUCKET_N,
    fdr_q: float = st.DEFAULT_FDR_Q,
    resamples: int = 4000,
    point_values: dict | None = None,
    seed: int = 7,
    roots: list[str] | None = None,
) -> AuditResult:
    df, fmt = parsers.normalize(source, point_values=point_values)
    _at = dict(getattr(df, "attrs", {}) or {})
    if roots:
        want = {x.strip().upper() for x in roots}
        df = df[df["root"].isin(want)].reset_index(drop=True)
        if df.empty:
            raise parsers.ParseError(
                f"No trades left after filtering to roots {sorted(want)}.")
        # the filter is part of the claim being audited -- it must be on the face
        fmt += f" | filtered to {', '.join(sorted(want))}"
    _ledger = _at.get("futures_ledger_total", np.nan)
    # like-for-like: the ledger total is the futures section's own realised
    # dollars gross of fees, so it is compared against captured futures GROSS
    _fut_gross = float(df.loc[df["venue"] == "futures", "gross_pnl"].sum()) \
        if ("venue" in df and len(df)) else (float(df["gross_pnl"].sum()) if len(df) else np.nan)
    coverage = {
        "ledger_total": _ledger,
        "accounted": _fut_gross,
        "excluded_contracts": _at.get("excluded_contracts", 0),
        "skipped_spread_orders": _at.get("skipped_spread_orders", 0),
        "unmatched_futures_rows": _at.get("unmatched_futures_rows", 0),
        "roots_filter": sorted({x.strip().upper() for x in roots}) if roots else None,
    }
    r, basis = parsers.to_r_multiples(df)
    df = df.assign(r=r)
    n = len(df)

    g = st.bootstrap_mean(r, resamples=resamples, seed=seed)
    sr = st.sharpe_per_trade(r)
    global_stats = {
        "mean_r": g["mean"], "ci_lo": g["lo"], "ci_hi": g["hi"],
        "p_vs_zero": g["p_vs_zero"],
        "win_rate": float((r > 0).mean()),
        "profit_factor": float(r[r > 0].sum() / abs(r[r < 0].sum())) if (r < 0).any() else np.inf,
        "sharpe_per_trade": sr,
        "psr": st.probabilistic_sharpe(r),
        "total_net_pnl": float(df["net_pnl"].sum()),
        "skew": float(sps.skew(r, bias=False)) if n > 2 else np.nan,
        "kurtosis": float(sps.kurtosis(r, fisher=False, bias=False)) if n > 3 else np.nan,
    }

    adequacy = {
        "min_track_record": st.min_track_record_length(r),
        "trades_needed_80pct": st.trades_needed(r),
        "detectable_effect_r": st.detectable_effect(n, float(np.std(r, ddof=1)) if n > 1 else np.nan),
        "shortfall": None,
        "concentration": st.variance_concentration(r),
    }
    # When the required sample is an order of magnitude beyond what the trader
    # has, quoting it is worse than useless -- it reads as "this tool is
    # broken" rather than "your outcome rests on a handful of trades". Flag it
    # so the report substitutes the concentration diagnosis instead.
    _mtrl = adequacy["min_track_record"]
    adequacy["mtrl_uninformative"] = bool(np.isfinite(_mtrl) and _mtrl > 10 * n)
    for k in ("min_track_record", "trades_needed_80pct"):
        v = adequacy[k]
        if np.isfinite(v) and v > n:
            adequacy["shortfall"] = int(np.ceil(v - n))
            break

    # ---- reporting-grade descriptives -------------------------------------
    d = df["net_pnl"].to_numpy(dtype=float)
    fees = float(df["commission"].fillna(0).abs().sum())
    money = {
        "net": float(d.sum()), "per_trade": float(d.mean()),
        "sd_per_trade": float(d.std(ddof=1)) if n > 1 else np.nan,
        "gross_win": float(d[d > 0].sum()), "gross_loss": float(d[d < 0].sum()),
        "avg_win": float(d[d > 0].mean()) if (d > 0).any() else np.nan,
        "avg_loss": float(d[d < 0].mean()) if (d < 0).any() else np.nan,
        "fees": fees,
        "fees_pct_of_result": float(fees / abs(d.sum())) if abs(d.sum()) > 0 else np.nan,
        "biggest_win": float(d.max()), "biggest_loss": float(d.min()),
    }
    if np.isfinite(money["avg_win"]) and np.isfinite(money["avg_loss"]):
        w, l = money["avg_win"], abs(money["avg_loss"])
        money["payoff_ratio"] = float(w / l) if l > 0 else np.nan
        money["breakeven_win_rate"] = float(l / (w + l)) if (w + l) > 0 else np.nan
        money["win_rate_gap"] = float(global_stats["win_rate"] - money["breakeven_win_rate"])

    order = df["exit_time"].fillna(df["entry_time"])
    seq = d[np.argsort(order.to_numpy(), kind="stable")] if order.notna().any() else d
    equity_curve = st.drawdown_profile(seq)
    trend = st.regime_trend(d, order=order.to_numpy() if order.notna().any() else None,
                            resamples=min(resamples, 2000), seed=seed)
    # Trades needed to establish an edge, at THIS trader's own dispersion.
    sd_d = money["sd_per_trade"]
    forward = {"sd": sd_d,
               "table": st.trades_to_detect(sd_d, [sd_d / 20, sd_d / 10, sd_d / 5, sd_d / 2])}

    by_class = pd.DataFrame()
    key = "venue" if ("venue" in df and df["venue"].notna().any()) else None
    if key:
        by_class = (df.groupby(key)["net_pnl"]
                      .agg(trades="size", total="sum", per_trade="mean", sd="std")
                      .reset_index().rename(columns={key: "class"}))
        by_class["win_rate"] = df.groupby(key)["net_pnl"].apply(lambda s: (s > 0).mean()).values
        lo_hi = [st.bootstrap_mean(g["net_pnl"].to_numpy(float), resamples=min(resamples, 2000), seed=seed)
                 for _, g in df.groupby(key)]
        by_class["ci_lo"] = [b["lo"] for b in lo_hi]
        by_class["ci_hi"] = [b["hi"] for b in lo_hi]
        by_class = by_class.sort_values("total")

    cands = bkt.build(df, min_n=min_bucket_n)
    rows = []
    for c in cands:
        row = {"dimension": c["dimension"], "label": c["label"], "n": c["n"],
               "testable": c["testable"]}
        vals = r[c["mask"]]
        row["mean_r"] = float(vals.mean()) if c["n"] else np.nan
        row["se"] = float(vals.std(ddof=1) / np.sqrt(c["n"])) if c["n"] > 1 else np.nan
        row["net_pnl"] = float(df.loc[c["mask"], "net_pnl"].sum())
        row["win_rate"] = float((vals > 0).mean()) if c["n"] else np.nan
        row["sharpe"] = st.sharpe_per_trade(vals)
        if c["testable"]:
            b = st.bootstrap_mean(vals, resamples=resamples, seed=seed)
            row["ci_lo"], row["ci_hi"] = b["lo"], b["hi"]
            row["p_vs_zero"] = b["p_vs_zero"]
            row["p_vs_self"] = st.permutation_vs_global(c["mask"], r, resamples=resamples, seed=seed)
        else:
            row["ci_lo"] = row["ci_hi"] = row["p_vs_zero"] = row["p_vs_self"] = np.nan
        rows.append(row)

    b = pd.DataFrame(rows)
    if not b.empty:
        rej_self, q_self = st.benjamini_hochberg(b["p_vs_self"].to_numpy(), q=fdr_q)
        rej_zero, q_zero = st.benjamini_hochberg(b["p_vs_zero"].to_numpy(), q=fdr_q)
        b["q_vs_self"], b["survives_self"] = q_self, rej_self
        b["q_vs_zero"], b["survives_zero"] = q_zero, rej_zero
        shrunk, weight = st.shrink_toward_global(
            b["mean_r"].to_numpy(), b["se"].to_numpy(), global_stats["mean_r"])
        b["shrunk_mean_r"], b["shrink_weight"] = shrunk, weight
        # a real finding must clear both bars: not zero, and not just "me"
        b["verified"] = b["survives_self"] & b["survives_zero"]
        b["naive_significant"] = b["p_vs_zero"] < 0.05
        b = b.sort_values(["verified", "mean_r"], ascending=[False, False]).reset_index(drop=True)

    family = int(b["testable"].sum()) if not b.empty else 0
    n_naive = int(b["naive_significant"].fillna(False).sum()) if not b.empty else 0
    n_surv = int(b["verified"].fillna(False).sum()) if not b.empty else 0

    trial_sharpes = b.loc[b["testable"], "sharpe"].to_numpy() if not b.empty else np.array([])
    global_stats["dsr"] = st.deflated_sharpe(r, trial_sharpes)

    verdict, detail = _verdict(n, global_stats, adequacy, n_surv, family)

    return AuditResult(
        format_name=fmt, r_basis=basis, n_trades=n,
        date_range=(df["entry_time"].min(), df["entry_time"].max()),
        global_stats=global_stats, adequacy=adequacy,
        independence=st.independence_check(r),
        buckets=b, family_size=family,
        n_naive_significant=n_naive, n_survived=n_surv,
        verdict=verdict, verdict_detail=detail, coverage=coverage,
        money=money, equity_curve=equity_curve, trend=trend, forward=forward,
        by_class=by_class, trades=df,
    )


def _verdict(n, gs, ad, n_surv, family) -> tuple[str, str]:
    lo, hi, mean = gs["ci_lo"], gs["ci_hi"], gs["mean_r"]
    short = ad.get("shortfall")
    if hi < 0:
        return ("Negative expectancy, established",
                f"The 95% interval for your average trade sits entirely below zero "
                f"({lo:+.3f}R to {hi:+.3f}R). This is not variance. Sizing changes will "
                f"not fix a negative mean.")
    if lo > 0 and gs["psr"] >= 0.95:
        s = (f" {n_surv} of {family} slices survived correction." if family else "")
        return ("Positive expectancy, established",
                f"The 95% interval is entirely above zero ({lo:+.3f}R to {hi:+.3f}R) and the "
                f"Sharpe claim survives skew, kurtosis and sample length.{s}")
    if mean > 0:
        need = f" You need roughly {short} more trades before this is answerable." if short else ""
        return ("Not established - insufficient evidence",
                f"Your average trade is {mean:+.3f}R, but the interval spans zero "
                f"({lo:+.3f}R to {hi:+.3f}R). A trader with no edge would produce this "
                f"record often enough that it cannot be called an edge.{need}")
    return ("Not established - insufficient evidence",
            f"Average trade is {mean:+.3f}R with an interval of {lo:+.3f}R to {hi:+.3f}R. "
            f"Nothing here separates from chance in either direction.")

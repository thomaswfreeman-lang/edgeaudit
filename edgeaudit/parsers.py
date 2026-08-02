"""
Intake layer.

The friction that kills conversion is not "CSV vs API" -- it is being asked
to reshape your broker export into someone's template. So: take the raw
export, fingerprint it, map it. No template, no instructions.

Adding a broker = appending one FormatSpec below. Nothing else changes.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CANONICAL = [
    "trade_id", "account", "symbol", "root", "direction", "qty",
    "entry_time", "exit_time", "entry_price", "exit_price",
    "gross_pnl", "commission", "net_pnl", "setup",
]


@dataclass
class FormatSpec:
    name: str
    required: tuple[str, ...]          # headers that must all be present
    level: str                          # "roundtrip" or "fill"
    mapping: dict[str, str] = field(default_factory=dict)
    direction_from: str | None = None
    tz: str | None = None

    def score(self, cols: set[str]) -> int:
        return len(set(self.required) & cols) if set(self.required) <= cols else 0


SPECS: list[FormatSpec] = [
    FormatSpec(
        name="Tradovate (Performance export)",
        required=("symbol", "pnl", "boughtTimestamp", "soldTimestamp", "qty"),
        level="roundtrip",
        mapping={
            "symbol": "symbol", "qty": "qty", "pnl": "net_pnl",
            "buyPrice": "buy_price", "sellPrice": "sell_price",
            "boughtTimestamp": "bought_time", "soldTimestamp": "sold_time",
        },
    ),
    FormatSpec(
        name="NinjaTrader (Trade Performance)",
        required=("Instrument", "Market pos.", "Entry price", "Exit price", "Entry time"),
        level="roundtrip",
        mapping={
            "Instrument": "symbol", "Account": "account", "Quantity": "qty",
            "Entry price": "entry_price", "Exit price": "exit_price",
            "Entry time": "entry_time", "Exit time": "exit_time",
            "Profit": "gross_pnl", "Commission": "commission",
            "Strategy": "setup", "Entry name": "setup",
        },
        direction_from="Market pos.",
    ),
    FormatSpec(
        name="TopstepX / ProjectX",
        required=("ContractName", "EnteredAt", "ExitedAt", "PnL"),
        level="roundtrip",
        mapping={
            "ContractName": "symbol", "Size": "qty", "PnL": "gross_pnl",
            "Fees": "commission", "EntryPrice": "entry_price",
            "ExitPrice": "exit_price", "EnteredAt": "entry_time",
            "ExitedAt": "exit_time", "Id": "trade_id", "Type": "direction_raw",
        },
    ),
    FormatSpec(
        name="Rithmic (R|Trader fills)",
        required=("Symbol", "Buy/Sell", "Qty", "Price"),
        level="fill",
        mapping={
            "Account": "account", "Symbol": "symbol", "Buy/Sell": "side",
            "Qty": "qty", "Price": "price", "Fill Time": "time",
            "Update Time": "time", "Commission": "commission",
        },
    ),
    FormatSpec(
        name="Interactive Brokers (Trades)",
        required=("Symbol", "Date/Time", "Quantity", "T. Price"),
        level="fill",
        mapping={
            "Account": "account", "Symbol": "symbol", "Date/Time": "time",
            "Quantity": "signed_qty", "T. Price": "price",
            "Comm/Fee": "commission", "Realized P/L": "gross_pnl",
        },
    ),
    FormatSpec(
        name="TradeStation (Trades)",
        required=("Symbol", "Side", "Quantity", "Price", "Time"),
        level="fill",
        mapping={
            "Account": "account", "Symbol": "symbol", "Side": "side",
            "Quantity": "qty", "Price": "price", "Time": "time",
            "Commission": "commission",
        },
    ),
]

# fuzzy fallback: header token -> canonical field
FUZZY = {
    "net_pnl": [r"^net.?p.?[l&]", r"^net.?profit", r"realized.?p"],
    "gross_pnl": [r"^p.?[/&]?l$", r"^pnl$", r"^profit$", r"gross.?p"],
    "commission": [r"comm", r"^fees?$"],
    "symbol": [r"^symbol$", r"instrument", r"contract", r"ticker", r"market"],
    "qty": [r"^qty$", r"quantity", r"^size$", r"contracts", r"^lots?$"],
    "entry_time": [r"entry.?(time|date)", r"open.?(time|date)", r"entered"],
    "exit_time": [r"exit.?(time|date)", r"close.?(time|date)", r"exited"],
    "entry_price": [r"entry.?price", r"open.?price", r"^avg.?entry"],
    "exit_price": [r"exit.?price", r"close.?price", r"^avg.?exit"],
    "direction_raw": [r"^side$", r"direction", r"^type$", r"long.?short", r"market.?pos"],
    "account": [r"^account"],
    "setup": [r"setup", r"strategy", r"^tag", r"playbook"],
}

LONG_WORDS = {"long", "buy", "b", "bot", "bought", "l"}
SHORT_WORDS = {"short", "sell", "s", "sld", "sold", "sht"}
ROOT_RE = re.compile(r"^([A-Za-z]{1,4})")


class ParseError(RuntimeError):
    pass


def _read(source) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    for kwargs in ({}, {"sep": ";"}, {"sep": "\t"}, {"skiprows": 1}):
        try:
            df = pd.read_csv(source, **kwargs)
            if df.shape[1] > 2:
                return df
        except Exception:
            if hasattr(source, "seek"):
                source.seek(0)
            continue
        if hasattr(source, "seek"):
            source.seek(0)
    raise ParseError("Could not read this file as delimited text.")


def detect_format(df: pd.DataFrame) -> FormatSpec | None:
    cols = set(df.columns.astype(str).str.strip())
    best, best_score = None, 0
    for spec in SPECS:
        s = spec.score(cols)
        if s > best_score:
            best, best_score = spec, s
    return best


def _norm_direction(series: pd.Series) -> pd.Series:
    def one(v):
        s = str(v).strip().lower()
        if s in LONG_WORDS or s.startswith("long") or s.startswith("buy"):
            return "long"
        if s in SHORT_WORDS or s.startswith("short") or s.startswith("sell"):
            return "short"
        return np.nan
    return series.map(one)


def _to_dt(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce", format="mixed", utc=False)
    if getattr(out.dtype, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out


def _fuzzy_map(df: pd.DataFrame) -> dict[str, str]:
    found: dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        for canon, pats in FUZZY.items():
            if canon in found:
                continue
            if any(re.search(p, key) for p in pats):
                found[canon] = col
                break
    return found


def _pair_fills_fifo(df: pd.DataFrame) -> pd.DataFrame:
    """
    FIFO round-trip construction from fill-level data.

    Futures fills arrive as signed adds/reduces; a 'trade' only exists once
    a position opens and returns to flat. Anything still open at the end of
    the file is dropped -- an unclosed position has no realised outcome and
    including it would bias the sample toward winners.
    """
    rows = []
    df = df.sort_values("time", kind="stable")
    for sym, g in df.groupby("symbol", sort=False):
        book: list[dict] = []  # open lots, all same sign
        for _, r in g.iterrows():
            qty = float(r["signed_qty"])
            price, t = float(r["price"]), r["time"]
            comm = float(r.get("commission", 0.0) or 0.0)
            while qty != 0:
                if book and np.sign(book[0]["qty"]) != np.sign(qty):
                    lot = book[0]
                    matched = min(abs(lot["qty"]), abs(qty))
                    side = 1 if lot["qty"] > 0 else -1
                    rows.append({
                        "symbol": sym,
                        "account": r.get("account", ""),
                        "direction": "long" if side > 0 else "short",
                        "qty": matched,
                        "entry_time": lot["time"], "exit_time": t,
                        "entry_price": lot["price"], "exit_price": price,
                        "gross_pnl": side * (price - lot["price"]) * matched,
                        "commission": abs(comm) * matched / max(abs(float(r["signed_qty"])), 1e-9),
                    })
                    lot["qty"] -= side * matched
                    qty += side * matched
                    if abs(lot["qty"]) < 1e-9:
                        book.pop(0)
                else:
                    book.append({"qty": qty, "price": price, "time": t})
                    qty = 0
    return pd.DataFrame(rows)


def normalize(source, point_values: dict[str, float] | None = None) -> tuple[pd.DataFrame, str]:
    """
    Returns (canonical dataframe, detected format name).

    gross_pnl from fill pairing is in *price points*; supply `point_values`
    ({"ES": 50, "CL": 1000, ...}) to convert to dollars. Without it the audit
    still runs -- R-multiples are scale-free -- but dollar figures are omitted
    rather than guessed, because a guessed dollar figure in a verification
    report is worse than no figure.
    """
    raw = _read(source)
    raw.columns = [str(c).strip() for c in raw.columns]
    spec = detect_format(raw)
    out = pd.DataFrame()

    if spec is not None and spec.level == "roundtrip":
        m = {k: v for k, v in spec.mapping.items() if k in raw.columns}
        work = raw.rename(columns=m)
        if spec.name.startswith("Tradovate"):
            bought, sold = _to_dt(work["bought_time"]), _to_dt(work["sold_time"])
            out["direction"] = np.where(bought <= sold, "long", "short")
            out["entry_time"] = np.minimum(bought, sold)
            out["exit_time"] = np.maximum(bought, sold)
            out["entry_price"] = np.where(out["direction"] == "long",
                                          work.get("buy_price"), work.get("sell_price"))
            out["exit_price"] = np.where(out["direction"] == "long",
                                         work.get("sell_price"), work.get("buy_price"))
        else:
            out["entry_time"] = _to_dt(work.get("entry_time"))
            out["exit_time"] = _to_dt(work.get("exit_time"))
            for c in ("entry_price", "exit_price"):
                if c in work:
                    out[c] = pd.to_numeric(work[c], errors="coerce")
            src = spec.direction_from if spec.direction_from in raw.columns else None
            if src:
                out["direction"] = _norm_direction(raw[src])
            elif "direction_raw" in work:
                out["direction"] = _norm_direction(work["direction_raw"])
        for c in ("symbol", "account", "setup", "trade_id"):
            if c in work:
                out[c] = work[c].astype(str)
        for c in ("qty", "gross_pnl", "net_pnl", "commission"):
            if c in work:
                out[c] = pd.to_numeric(
                    work[c].astype(str).str.replace(r"[$,()]", "", regex=True), errors="coerce"
                )
        fmt_name = spec.name

    elif spec is not None and spec.level == "fill":
        m = {k: v for k, v in spec.mapping.items() if k in raw.columns}
        work = raw.rename(columns=m)
        work["time"] = _to_dt(work["time"])
        work["price"] = pd.to_numeric(work["price"], errors="coerce")
        if "signed_qty" in work:
            work["signed_qty"] = pd.to_numeric(
                work["signed_qty"].astype(str).str.replace(",", ""), errors="coerce")
        else:
            side = _norm_direction(work["side"])
            q = pd.to_numeric(work["qty"], errors="coerce").abs()
            work["signed_qty"] = np.where(side == "long", q, -q)
        work["symbol"] = work["symbol"].astype(str)
        if "commission" not in work:
            work["commission"] = 0.0
        out = _pair_fills_fifo(work.dropna(subset=["time", "price", "signed_qty"]))
        fmt_name = spec.name

    else:  # fuzzy fallback
        found = _fuzzy_map(raw)
        if not {"symbol"} <= set(found) or not ({"net_pnl", "gross_pnl"} & set(found)):
            raise ParseError(
                "Unrecognised export. Needs at least a symbol column and a P&L column. "
                f"Columns seen: {list(raw.columns)[:12]}"
            )
        for canon, col in found.items():
            if canon in ("entry_time", "exit_time"):
                out[canon] = _to_dt(raw[col])
            elif canon == "direction_raw":
                out["direction"] = _norm_direction(raw[col])
            elif canon in ("symbol", "account", "setup"):
                out[canon] = raw[col].astype(str)
            else:
                out[canon] = pd.to_numeric(
                    raw[col].astype(str).str.replace(r"[$,()]", "", regex=True), errors="coerce")
        fmt_name = "Generic (column-matched)"

    for c in CANONICAL:
        if c not in out:
            out[c] = np.nan
    out = out[CANONICAL].copy()

    if out["net_pnl"].isna().all():
        out["net_pnl"] = out["gross_pnl"].fillna(0) - out["commission"].fillna(0).abs()
    if point_values:
        mult = out["symbol"].map(lambda s: point_values.get(_root(s), np.nan))
        conv = mult.notna() & out["net_pnl"].notna()
        out.loc[conv, "net_pnl"] = out.loc[conv, "net_pnl"] * mult[conv] * out.loc[conv, "qty"].fillna(1)

    out["root"] = out["symbol"].map(_root)
    out["trade_id"] = out["trade_id"].fillna(pd.Series(range(len(out))).astype(str))
    out = out.dropna(subset=["net_pnl"])
    if out.empty:
        raise ParseError("File parsed but no trades with a usable P&L were found.")
    if out["entry_time"].notna().any():
        out = out.sort_values("entry_time", kind="stable").reset_index(drop=True)
    return out.reset_index(drop=True), fmt_name


def _root(sym) -> str:
    m = ROOT_RE.match(str(sym).strip().upper())
    return m.group(1) if m else str(sym)


def to_r_multiples(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """
    R-multiples. If the export carries no per-trade risk (almost none do),
    normalise by the median absolute loss -- a robust stand-in for '1R' that
    is unaffected by the few catastrophic trades that would wreck a mean-based
    scale. The report must say which basis was used; a silent proxy is how
    other tools quietly become uncomparable across accounts.
    """
    pnl = df["net_pnl"].to_numpy(dtype=float)
    losses = np.abs(pnl[pnl < 0])
    if losses.size >= 5:
        unit = float(np.median(losses))
        basis = "median absolute loss (estimated 1R)"
    else:
        unit = float(np.std(pnl, ddof=1)) if pnl.size > 1 else 1.0
        basis = "P&L standard deviation (too few losses for a loss-based 1R)"
    if not np.isfinite(unit) or unit <= 0:
        unit, basis = 1.0, "raw P&L (no usable risk scale)"
    return pnl / unit, basis

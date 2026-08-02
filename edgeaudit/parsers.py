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
from pathlib import Path

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
# futures contract code: optional /, root, month letter, 1-2 digit year, optional :EXCH
FUT_SYM_RE = re.compile(r"^/?([A-Z0-9]{1,4}?)([FGHJKMNQUVXZ]\d{1,2})(?::[A-Z]+)?$")


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


# ---------------------------------------------------------------------------
# thinkorswim / Schwab "Account Statement"
#
# Not a table -- a multi-section statement (Cash Balance, Futures Statements,
# Account Trade History, ...) with different columns per section. The one
# FormatSpec-per-broker model doesn't fit, so it gets a dedicated pre-parser.
#
# We audit the *Futures Statements* section: its fills carry the broker's own
# realised dollars ("Amount" books realised P&L on the closing fill for
# outrights, premium flow on every fill for options on futures) plus per-fill
# fees, so no point-value guessing is needed. Overnight marks arrive as ADJ
# rows and are folded into the episode that was open at settlement. Equities
# and equity-option activity (Cash Balance section) is counted and reported,
# not audited -- mixing investment legs into a futures day-trading record
# would corrupt the 1R basis silently.
# ---------------------------------------------------------------------------

_TOS_OUTRIGHT_RE = re.compile(
    r"^(BOT|SOLD)\s+([+-]?[\d,]+)\s+(/?[A-Z0-9]+(?::[A-Z]+)?)\s+@([\d,.']+)$")
_TOS_OPTION_RE = re.compile(
    r"^(BOT|SOLD)\s+([+-]?[\d,]+)\s+(/?[A-Z0-9]+)(?::[A-Z]+)?\s+1/[\d,]+\s+(.*?)"
    r"(/?[A-Z0-9]+)(?::[A-Z]+)?\s+([\d.]+)\s+(CALL|PUT)\s+@([\d,.]+)$")
_TOS_ADJ_SYM_RE = re.compile(r"^(/?[A-Z0-9]+)(?::[A-Z]+)?\s+mark to market")
# expiry date as it appears in fill descriptions ("24 FEB 26") and in
# removal/exercise rows ("24 Feb 2026")
_TOS_EXPIRY_RE = re.compile(r"\b(\d{1,2}) ([A-Za-z]{3}) (\d{2,4})\b")
_TOS_REMOVAL_RE = re.compile(
    r"^Removal of option due to expiration of (/?[A-Z0-9]+)\s+.*?"
    r"([\d.]+)\s+(CALL|PUT)$")
_TOS_EXERCISE_RE = re.compile(
    r"^Exercise, future settle (-?[\d.]+) of (/?[A-Z0-9]+)\s+.*?"
    r"([\d.]+)\s+(CALL|PUT)$")


def _tos_price(s: str) -> float:
    """Prices, including CBOT 32nds tick notation: 111'245 = 111 + 24.5/32."""
    s = str(s).replace(",", "")
    if "'" in s:
        whole, frac = s.split("'")
        ticks = float(frac) / 10.0 if len(frac) >= 3 else float(frac)
        return float(whole) + ticks / 32.0
    return float(s)


def _tos_expiry_key(text: str) -> str:
    m = _TOS_EXPIRY_RE.search(text)
    if not m:
        return "?"
    return f"{int(m.group(1)):02d}{m.group(2).upper()}{int(m.group(3)) % 100:02d}"


def _tos_option_key(underlying: str, expiry_src: str, strike: str, cp: str) -> str:
    """Contract identity shared by fills and removal/exercise rows."""
    return f"{_root(underlying)} {float(strike):g} {cp} {_tos_expiry_key(expiry_src)}"
_TOS_SECTION_NAMES = (
    "Cash Balance", "Futures Statements", "Forex Statements",
    "Account Order History", "Account Trade History", "Equities", "Options",
    "Futures", "Futures Options", "Profits and Losses", "Account Summary",
)


def _tos_text(source) -> str | None:
    """Full text if `source` is a thinkorswim Account Statement, else None."""
    try:
        if hasattr(source, "read"):
            text = source.read()
            if hasattr(source, "seek"):
                source.seek(0)
            if isinstance(text, bytes):
                text = text.decode("utf-8-sig", "replace")
        elif isinstance(source, (str, Path)) and "\n" not in str(source):
            p = Path(source)
            if not p.exists():
                return None
            text = p.read_text(encoding="utf-8-sig", errors="replace")
        else:
            return None
    except OSError:
        return None
    return text if text.lstrip("﻿ \n").startswith("Account Statement for") else None


def _tos_num(s: str) -> float:
    s = str(s).strip().strip('"').replace(",", "")
    if s in ("", "--", "N/A"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _tos_sections(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    cur = "_preamble"
    out[cur] = []
    for line in text.splitlines():
        if line.strip().strip('"') in _TOS_SECTION_NAMES:
            cur = line.strip().strip('"')
            out[cur] = []
        else:
            out[cur].append(line)
    return out


def _infer_start_position(fills: list[tuple[float, bool]], span: int = 30) -> int:
    """
    Starting position for a symbol whose entry may predate the statement.

    Ground truth: an outright fill carries an Amount iff it reduces an open
    position. Search small integer starting positions and keep the one most
    consistent with the observed open/close pattern; ties go to the smallest
    absolute position (0 when nothing contradicts it).
    """
    best, best_score = 0, -1
    for cand in range(-span, span + 1):
        pos, score = cand, 0
        for qty, has_amt in fills:
            closing = pos != 0 and qty * pos < 0
            if closing == has_amt:
                score += 1
            pos += qty
        if score > best_score or (score == best_score and abs(cand) < abs(best)):
            best, best_score = cand, score
    return best


def _parse_tos_statement(text: str) -> tuple[pd.DataFrame, dict]:
    """Round trips from the Futures Statements section, plus a diagnosis dict."""
    import csv as _csv

    sections = _tos_sections(text)
    header = text.lstrip("﻿ \n").splitlines()[0]
    diag: dict = {
        "format": "thinkorswim (Account Statement, futures section)",
        "statement": header,
        "sections": {k: len(v) for k, v in sections.items() if k != "_preamble"},
        "unparsed_trd": [], "row_types": {}, "fut_adj_excluded": [],
        "inferred_start_positions": {}, "orphan_adj": 0,
        "dropped_partial_episodes": {}, "dropped_open_at_eof": {},
        "outright_fills": 0, "option_fills": 0,
        "expiration_rows": 0, "exercise_rows": 0, "orphan_removals": 0,
    }
    if "Futures Statements" not in sections:
        raise ParseError(
            "thinkorswim statement detected but it has no Futures Statements "
            "section. Only futures activity is audited for now.")

    lines = [l for l in sections["Futures Statements"] if l.strip()]
    rows = list(_csv.reader(lines))
    fills: list[dict] = []          # parsed TRD fills, file order
    adjs: list[dict] = []           # overnight mark-to-market rows
    for r in rows[1:]:
        if len(r) < 10:
            continue
        typ = r[3].strip()
        diag["row_types"][typ] = diag["row_types"].get(typ, 0) + 1
        desc = r[5].strip()
        if typ == "ADJ":
            m = _TOS_ADJ_SYM_RE.match(desc)
            amt = _tos_num(r[8])
            if m and np.isfinite(amt):
                adjs.append({"dt": f"{r[1]} {r[2]}", "symbol": m.group(1).split(":")[0],
                             "amount": amt})
            continue
        if typ == "FUT_ADJ":
            diag["fut_adj_excluded"].append(_tos_num(r[8]))
            continue
        if typ != "TRD":
            continue
        fees = np.nansum([_tos_num(r[6]), _tos_num(r[7])])
        amt = _tos_num(r[8])
        dt = f"{r[1]} {r[2]}"
        m = _TOS_OUTRIGHT_RE.match(desc)
        if m:
            qty = float(m.group(2).replace(",", "").replace("+", ""))
            fills.append({"dt": dt, "symbol": m.group(3).split(":")[0], "qty": qty,
                          "price": _tos_price(m.group(4)),
                          "amount": amt, "fees": fees, "option": False,
                          "close_all": False})
            diag["outright_fills"] += 1
            continue
        m = _TOS_OPTION_RE.match(desc)
        if m:
            qty = float(m.group(2).replace(",", "").replace("+", ""))
            sym = _tos_option_key(m.group(3), m.group(4), m.group(6), m.group(7))
            fills.append({"dt": dt, "symbol": sym, "qty": qty,
                          "price": float(m.group(8).replace(",", "")),
                          "amount": amt, "fees": fees, "option": True,
                          "close_all": False})
            diag["option_fills"] += 1
            continue
        m = _TOS_REMOVAL_RE.match(desc)
        if m:
            # worthless expiry: closes the whole position at zero. Skipping
            # these silently deletes losses -- the one bias this tool must
            # never have.
            fills.append({"dt": dt,
                          "symbol": _tos_option_key(m.group(1), desc, m.group(2), m.group(3)),
                          "qty": 0.0, "price": 0.0, "amount": amt, "fees": fees,
                          "option": True, "close_all": True})
            diag["expiration_rows"] += 1
            continue
        m = _TOS_EXERCISE_RE.match(desc)
        if m:
            opt_delta = float(m.group(1))
            fills.append({"dt": dt,
                          "symbol": _tos_option_key(m.group(2), desc, m.group(3), m.group(4)),
                          "qty": opt_delta, "price": 0.0, "amount": amt,
                          "fees": fees, "option": True, "close_all": False})
            # exercising converts the option into the underlying at the
            # strike: emit the futures opening the broker implies, so the
            # resulting position closes as a real trade instead of being
            # dropped as unattributable. Long calls removed -> long futures;
            # long puts removed -> short futures.
            fut_qty = -opt_delta if m.group(4) == "CALL" else opt_delta
            fills.append({"dt": dt, "symbol": m.group(2).split(":")[0],
                          "qty": fut_qty, "price": float(m.group(3)),
                          "amount": np.nan, "fees": 0.0, "option": False,
                          "close_all": False})
            diag["exercise_rows"] += 1
            continue
        diag["unparsed_trd"].append(desc)
        if np.isfinite(amt):
            diag["unparsed_amount"] = diag.get("unparsed_amount", 0.0) + amt

    events = pd.DataFrame(fills)
    if events.empty:
        raise ParseError("Futures Statements section contained no parseable fills.")
    events["dt"] = pd.to_datetime(events["dt"], format="%m/%d/%y %H:%M:%S")
    adj_df = pd.DataFrame(adjs)
    if not adj_df.empty:
        adj_df["dt"] = pd.to_datetime(adj_df["dt"], format="%m/%d/%y %H:%M:%S")

    trades: list[dict] = []
    for sym, g in events.groupby("symbol", sort=False):
        g = g.sort_values("dt", kind="stable")
        is_opt = bool(g["option"].iat[0])
        pos0 = 0
        if not is_opt:
            pos0 = _infer_start_position(
                [(q, np.isfinite(a)) for q, a in zip(g["qty"], g["amount"])])
            if pos0 != 0:
                diag["inferred_start_positions"][sym] = pos0
        sym_adjs = (adj_df[adj_df["symbol"] == sym].sort_values("dt")
                    if not adj_df.empty else pd.DataFrame())
        stream = [("fill", r) for r in g.itertuples()]
        stream += [("adj", r) for r in sym_adjs.itertuples()]
        stream.sort(key=lambda e: e[1].dt)

        pos, cur = pos0, None
        for kind, ev in stream:
            if kind == "adj":
                if cur is not None:
                    cur["pnl"] += ev.amount
                else:
                    diag["orphan_adj"] += 1
                continue
            if ev.close_all:
                if cur is None or pos == 0:
                    diag["orphan_removals"] += 1
                    continue
                q = -pos
            else:
                q = ev.qty
            first_portion = True
            while q != 0:
                if cur is None:
                    cur = {"partial": pos != 0, "dir": np.sign(q if pos == 0 else pos),
                           "entry_dt": ev.dt, "exit_dt": ev.dt, "pnl": 0.0,
                           "fees": 0.0, "peak": abs(pos),
                           "e_qty": 0.0, "e_px": 0.0, "x_qty": 0.0, "x_px": 0.0}
                if first_portion:
                    cur["fees"] += ev.fees if np.isfinite(ev.fees) else 0.0
                    if np.isfinite(ev.amount):
                        cur["pnl"] += ev.amount
                    first_portion = False
                cur["exit_dt"] = ev.dt
                if pos == 0 or np.sign(q) == np.sign(pos):     # opening / adding
                    pos += q
                    cur["e_qty"] += abs(q)
                    cur["e_px"] += abs(q) * ev.price
                    q = 0.0
                else:                                          # reducing
                    take = np.sign(q) * min(abs(q), abs(pos))
                    pos += take
                    q -= take
                    cur["x_qty"] += abs(take)
                    cur["x_px"] += abs(take) * ev.price
                cur["peak"] = max(cur["peak"], abs(pos))
                if pos == 0 and cur is not None:
                    if cur["partial"]:
                        diag["dropped_partial_episodes"][sym] = \
                            diag["dropped_partial_episodes"].get(sym, 0) + 1
                    else:
                        trades.append({
                            "symbol": sym, "account": "",
                            "direction": "long" if cur["dir"] > 0 else "short",
                            "qty": cur["peak"],
                            "entry_time": cur["entry_dt"], "exit_time": cur["exit_dt"],
                            "entry_price": cur["e_px"] / cur["e_qty"] if cur["e_qty"] else np.nan,
                            "exit_price": cur["x_px"] / cur["x_qty"] if cur["x_qty"] else np.nan,
                            "gross_pnl": cur["pnl"],
                            "commission": abs(cur["fees"]),
                            "net_pnl": cur["pnl"] - abs(cur["fees"]),
                            "setup": "option" if is_opt else "outright",
                        })
                    cur = None
        if cur is not None:
            diag["dropped_open_at_eof"][sym] = 1

    out = pd.DataFrame(trades)
    if out.empty:
        raise ParseError("No completed round trips found in the futures section.")
    diag["trades"] = len(out)
    diag["date_range"] = (str(out["entry_time"].min()), str(out["exit_time"].max()))
    diag["gross_pnl_sum"] = float(out["gross_pnl"].sum())
    diag["fees_sum"] = float(out["commission"].sum())
    diag["net_pnl_sum"] = float(out["net_pnl"].sum())
    # reconciliation: every dollar in the section is either in a trade,
    # attributed to a dropped episode, or listed here -- never silently lost
    parsed_amt = float(np.nansum(events["amount"].to_numpy())) + \
        (float(adj_df["amount"].sum()) if not adj_df.empty else 0.0)
    unparsed_amt = float(diag.get("unparsed_amount", 0.0))
    diag["reconciliation"] = {
        "section_amount_total": parsed_amt + unparsed_amt,
        "captured_in_trades": diag["gross_pnl_sum"],
        "in_dropped_episodes": parsed_amt - diag["gross_pnl_sum"],
        "in_unparsed_rows": unparsed_amt,
    }
    return out, diag


def diagnose(source) -> dict:
    """Parse without auditing; report exactly what was read, used and dropped."""
    text = _tos_text(source)
    if text is not None:
        df, diag = _parse_tos_statement(text)
        return diag
    raw = _read(source)
    raw.columns = [str(c).strip() for c in raw.columns]
    spec = detect_format(raw)
    df, fmt = normalize(source)
    return {
        "format": fmt, "spec_matched": spec.name if spec else None,
        "columns_seen": list(raw.columns), "rows_in_file": len(raw),
        "trades": len(df),
        "date_range": (str(df["entry_time"].min()), str(df["entry_time"].max())),
        "net_pnl_sum": float(df["net_pnl"].sum()),
    }


def normalize(source, point_values: dict[str, float] | None = None) -> tuple[pd.DataFrame, str]:
    """
    Returns (canonical dataframe, detected format name).

    gross_pnl from fill pairing is in *price points*; supply `point_values`
    ({"ES": 50, "CL": 1000, ...}) to convert to dollars. Without it the audit
    still runs -- R-multiples are scale-free -- but dollar figures are omitted
    rather than guessed, because a guessed dollar figure in a verification
    report is worse than no figure.
    """
    text = _tos_text(source)
    if text is not None:
        out, _ = _parse_tos_statement(text)
        return _finish(out, "thinkorswim (Account Statement, futures section)", point_values)

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

    return _finish(out, fmt_name, point_values)


def _finish(out: pd.DataFrame, fmt_name: str,
            point_values: dict[str, float] | None = None) -> tuple[pd.DataFrame, str]:
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
    s = str(sym).strip().upper()
    tok = s.split()[0] if s else s
    m = FUT_SYM_RE.match(tok)          # "/MNQZ25:XCME" -> MNQ, "SILH26" -> SIL
    if m:
        return m.group(1)
    m = ROOT_RE.match(s)
    return m.group(1) if m else s


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

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
    "trade_id", "account", "venue", "symbol", "root", "direction", "qty",
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


# ---------------------------------------------------------------------------
# Schwab / thinkorswim combined "Account Statement" export.
#
# This is not a trades file -- it's the full statement: Cash Balance ledger,
# Account Order History, Account Trade History, Options/Futures Options
# positions, a Profits-and-Losses Greeks watchlist, Account Summary, all
# concatenated in one CSV with different column shapes per section. The
# executed trades live inside the Cash Balance ledger as free-text
# DESCRIPTION strings ("SOLD -1 SLV @64.18", "BOT +1 QQQ 100 (Weeklys) 8 JUN
# 26 721 CALL @.48 CBOE"), not as a clean symbol/qty/price table -- so this
# needs real text parsing before it can join the fill-pairing pipeline the
# same way Rithmic/IBKR/TradeStation fills do.
#
# What this covers: single-leg equity and single-leg option fills, and
# option expirations (the "Removed due to Expiration ... .OCC" rows).
#
# What this deliberately does NOT cover: multi-leg spread orders (vertical,
# iron condor, calendar, custom combo, backratio -- anything with a slash-
# separated strike/expiry/right list and one net price for the whole
# package). Splitting a combo's net price across legs isn't a parsing
# problem, it's a pricing-model problem, and guessing would put a fabricated
# per-leg P&L into a verification report. So instead: every option contract
# that ever appears inside a spread description is identified and excluded
# from the *entire* file, single-leg fills included, not just the spread
# rows themselves. Partial coverage of a contract's fill history is worse
# than no coverage -- it can silently misrepresent a position as flat or
# open when it wasn't -- so the tool drops the whole contract rather than
# guess. This does mean an active spread trader's true trade count is
# understated here; report.py should surface how many contracts were
# excluded so that's visible, not silent.
# ---------------------------------------------------------------------------

_SCHWAB_TITLE_RE = re.compile(r"Account Statement for (\S+)")
# price forms: plain decimal, bare integer, or CBOT 32nds tick notation
# (111'240 = 111 + 24.0/32 -- treasuries /ZN, /ZF, /ZB)
_SCHWAB_PRICE = r"(?:\d[\d,]*'\d{2,3}|\d[\d,]*\.\d+|\.\d+|\d[\d,]*)"
_SCHWAB_TRD_RE = re.compile(
    r"^(?P<side>BOT|SOLD)\s+(?P<qty>[+-]?[\d,]+(?:\.\d+)?)\s+"
    r"(?P<underlying>[A-Za-z0-9./]+)\s+"
    r"(?:(?P<mult>\d+)\s+(?:\((?:Weeklys|Quarterlys|Monthlys|EOM)\)\s+)?"
    r"(?P<day>\d{1,2})\s+(?P<mon>[A-Za-z]{3})\s+(?P<yr>\d{2})\s+(?:\[AM\]\s+)?"
    r"(?P<strike>[\d.]+)\s+(?P<right>CALL|PUT)\s+)?"
    r"@\s*(?P<price>" + _SCHWAB_PRICE + r")"
)
_SCHWAB_RAD_RE = re.compile(
    r"Removed due to Expiration.*?(?P<expqty>[+-]?[\d.]+)\s+\.(?P<occ>[A-Za-z0-9.]+)\s*$"
)
_SCHWAB_SPREAD_KW_RE = re.compile(
    r"(?:CUSTOM|VERTICAL|IRON CONDOR|CALENDAR|DIAGONAL|BACKRATIO)\s+"
    r"(?P<underlying>[A-Za-z0-9./]+)\s+\d+\s+"
)
_SCHWAB_DATE_RE = re.compile(r"(\d{1,2})\s+([A-Za-z]{3})\s+(\d{2})")
_SCHWAB_TAIL_RE = re.compile(
    r"(?P<strikes>[\d.]+(?:/[\d.]+)*)\s+(?P<rights>(?:CALL|PUT)(?:/(?:CALL|PUT))*)\s*@"
)
_SCHWAB_MONTHS = {m.upper(): i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}

# Futures Statements section. Same statement file, different section, different
# column names ("Trade Date/Exec Time/Amount/Balance"), and crucially the
# Amount column already carries the broker's own realised P&L in DOLLARS on
# each closing fill. That is strictly better than pairing prices ourselves and
# multiplying by a point-value table we'd have to maintain and could get wrong,
# so where the broker states realised P&L we use it and only use FIFO pairing
# to recover the round-trip *structure* (entry time, hold time, direction).
_SCHWAB_FUT_RE = re.compile(
    r"^(?P<side>BOT|SOLD)\s+(?P<qty>[+-]?[\d,]+(?:\.\d+)?)\s+"
    r"(?P<symbol>/[A-Z0-9]+(?::[A-Z]+)?)\s+"
    r"@\s*(?P<price>" + _SCHWAB_PRICE + r")"
)
# Options ON futures, e.g.
#   BOT +1 /SIH26:XCEC 1/5000 24 FEB 26 (MAR26) (MAR 26) /SOH26:XCEC 75.75 CALL @2.619
# The tradeable instrument is the option (/SOH26 75.75 CALL, exp 24 Feb 26);
# the leading /SIH26 is the underlying future and is what the trader thinks of
# as the instrument, so the canonical symbol keeps both: underlying first (so
# instrument bucketing rolls up to "SI"), then the specific contract.
_SCHWAB_FUTOPT_RE = re.compile(
    r"^(?P<side>BOT|SOLD)\s+(?P<qty>[+-]?[\d,]+(?:\.\d+)?)\s+"
    r"(?P<under>/[A-Z0-9]+)(?::[A-Z]+)?\s+"
    r"\d+/(?P<mult>\d+)\s+"
    r"(?P<day>\d{1,2})\s+(?P<mon>[A-Z]{3})\s+(?P<yr>\d{2})\s+"
    r"(?:\([^)]*\)\s*)*"
    r"(?P<opt>/[A-Z0-9]+)(?::[A-Z]+)?\s+"
    r"(?P<strike>[\d.]+)\s+(?P<right>CALL|PUT)\s+"
    r"@\s*(?P<price>" + _SCHWAB_PRICE + r")"
)
# /ESZ25:XCME -> ES ; /MNQZ25 -> MNQ ; /M2KH6 -> M2K ; /6EM6 -> 6E
_FUT_ROOT_RE = re.compile(r"^/?([A-Z0-9]+?)([FGHJKMNQUVXZ])(\d{1,2})(?::[A-Z]+)?$")
# multi-leg futures option orders: keyword, then the underlying is the next
# slash-token AFTER the keyword (a leading ratio like "7/10 CUSTOM" would
# otherwise be misread as the underlying "/10")
_SCHWAB_FUT_SPREAD_RE = re.compile(
    r"(?:VERTICAL|CUSTOM|BACKRATIO|CONDOR|BUTTERFLY|CALENDAR|DIAGONAL)\s+"
    r"(?P<under>/[A-Z0-9]+)")
_SCHWAB_FUT_SPREAD_DATE_RE = re.compile(r"\b(\d{1,2})\s+([A-Z]{3})\s+(\d{2})\b")


def _schwab_excluded_fut_contracts(descs: list[str]) -> set[str]:
    """
    Every futures-option contract key touched by ANY multi-leg description,
    via the cross-product of dates x strikes x rights found in it --
    deliberately over-broad within the order, never broader than it.
    """
    excluded: set[str] = set()
    for desc in descs:
        mu = _SCHWAB_FUT_SPREAD_RE.search(desc)
        tail = _SCHWAB_TAIL_RE.search(desc)
        if not mu or not tail:
            continue
        under = mu.group("under")
        dates = _SCHWAB_FUT_SPREAD_DATE_RE.findall(desc.upper())
        strikes = tail.group("strikes").split("/")
        rights = tail.group("rights").split("/")
        for day, mon, yr in dates:
            if mon not in _SCHWAB_MONTHS:
                continue
            for strike in strikes:
                for right in set(rights):
                    excluded.add(_futopt_key(under, day, mon, yr, strike, right))
    return excluded


# "Removal of option due to expiration of /NQH26 XCME 20 (WEEKLY) 11 Mar 2026 25150.0 CALL"
_SCHWAB_FUT_EXP_RE = re.compile(
    r"Removal of option due to expiration of\s+(?P<under>/[A-Z0-9]+)\s+\S+\s+(?P<mult>\d+)\s+"
    r"(?:\([^)]*\)\s*)*"
    r"(?P<day>\d{1,2})\s+(?P<mon>[A-Za-z]{3})\s+(?P<yr>\d{4})\s+"
    r"(?P<strike>[\d.]+)\s+(?P<right>CALL|PUT)"
)


def _futopt_key(under: str, day, mon: str, yr, strike: str, right: str) -> str:
    """
    Canonical futures-option contract id: underlying, expiry date, right,
    strike. Deliberately NOT keyed on the option's own root symbol (/QNEH26,
    /Q1DJ26, ...) because the expiration rows in the ledger don't carry it --
    and an expiry we can't match to its opening fill is a position that never
    closes, which silently drops the trade from the record.
    """
    yy, mm = int(yr) % 100, _SCHWAB_MONTHS[mon.upper()[:3]]
    s = strike.rstrip("0").rstrip(".") if "." in strike else strike
    return f"{str(under).lstrip('/')}_{yy:02d}{mm:02d}{int(day):02d}{right[0]}{s}"


def _futures_root(sym: str) -> str:
    """Contract root for bucketing. Futures-option symbols carry their
    underlying ahead of an underscore so they roll up to the same root."""
    s = str(sym).strip().upper().lstrip("/")
    for cand in (s, s.split("_")[0]):
        m = _FUT_ROOT_RE.match(cand)
        if m:
            return m.group(1)
    return s.split("_")[0].split(":")[0]


def _price_num(s) -> float:
    """Price string to float, including CBOT 32nds: 111'245 = 111 + 24.5/32."""
    s = str(s).replace(",", "")
    if "'" in s:
        whole, frac = s.split("'")
        ticks = float(frac) / 10.0 if len(frac) >= 3 else float(frac)
        return float(whole) + ticks / 32.0
    return float(s)


def _infer_start_position(fills: list[tuple[float, bool]], span: int = 30) -> int:
    """
    Starting position for an outright contract whose entry may predate the
    statement window.

    Ground truth: an outright fill carries a realised Amount iff it reduces an
    open position. Search small integer starting positions and keep the one
    most consistent with the observed open/close pattern; ties go to the
    smallest absolute position (0 when nothing contradicts it).
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


# "Exercise, future settle -10.0 of /NQM26 XCME 20 (QUARTERLY) 31 Mar 2026 23900.0 CALL"
_SCHWAB_FUT_EXER_RE = re.compile(
    r"Exercise, future settle\s+(?P<dqty>-?[\d.]+)\s+of\s+(?P<under>/[A-Z0-9]+)\s+\S+\s+(?P<mult>\d+)\s+"
    r"(?:\([^)]*\)\s*)*"
    r"(?P<day>\d{1,2})\s+(?P<mon>[A-Za-z]{3})\s+(?P<yr>\d{4})\s+"
    r"(?P<strike>[\d.]+)\s+(?P<right>CALL|PUT)"
)

# sentinel: synthetic opening lots for positions inferred to predate the
# statement window are stamped before this date, so the round trips that
# close them can be identified and dropped (their true entry is unknowable)
_PREWINDOW = pd.Timestamp("1990-01-01")


def _num0(v, default: float = 0.0) -> float:
    """
    Coerce to float, mapping NaN/None/blank to `default`.

    Written out rather than using `float(x or default)`: NaN is *truthy* in
    Python, so `np.nan or 0.0` evaluates to NaN and silently poisons any
    column it lands in. That put NaN commissions on 49 real round trips
    before it was caught.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float(default)
    return float(default) if not np.isfinite(f) else f


def _load_text(source) -> str | None:
    """Best-effort raw text of `source`, without disturbing it for `_read`."""
    if isinstance(source, pd.DataFrame):
        return None
    try:
        if hasattr(source, "read"):
            raw = source.read()
            if hasattr(source, "seek"):
                source.seek(0)
        else:
            with open(source, "rb") as f:
                raw = f.read()
    except Exception:
        return None
    if isinstance(raw, bytes):
        try:
            return raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            return raw.decode("latin-1", errors="replace")
    return raw


def _sniff_schwab(text: str) -> bool:
    head = text[:400]
    return bool(_SCHWAB_TITLE_RE.search(head)) and "Cash Balance" in text[:2000]


def _schwab_opt_symbol(underlying: str, day, mon: str, yr, strike: str, right: str) -> str:
    yr = int(yr) % 100
    m = _SCHWAB_MONTHS[mon.upper()[:3]]
    strike_s = strike.rstrip("0").rstrip(".") if "." in strike else strike
    return f"{underlying.upper()}{yr:02d}{m:02d}{int(day):02d}{right[0]}{strike_s}"


def _extract_section(text: str, title: str, header_prefix: str) -> str | None:
    """
    Pull one named section out of a multi-section statement. Sections are
    terminated by a blank line, which is what separates them in the export.
    """
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.strip() == title:
            for j in range(i + 1, min(i + 5, len(lines))):
                if lines[j].startswith(header_prefix):
                    start = j
                    break
            break
    if start is None:
        return None
    end = len(lines)
    for k in range(start + 1, len(lines)):
        if not lines[k].strip():
            end = k
            break
    return "\n".join(lines[start:end])


def _extract_cash_balance_block(text: str) -> str | None:
    return _extract_section(text, "Cash Balance", "DATE,TIME,TYPE")


def _extract_futures_block(text: str) -> str | None:
    return _extract_section(text, "Futures Statements", "Trade Date,")


def _parse_schwab_futures(text: str, account: str) -> pd.DataFrame:
    """Fill-level rows from the Futures Statements section (may be absent)."""
    block = _extract_futures_block(text)
    if block is None:
        return pd.DataFrame()
    raw = pd.read_csv(io.StringIO(block))
    raw.columns = [str(c).strip() for c in raw.columns]
    if "Type" not in raw.columns or "Description" not in raw.columns:
        return pd.DataFrame()
    # TRD fills plus ADJ overnight mark-to-market rows: for a position held
    # across the 16:00 settle, part of the realised P&L arrives as an ADJ,
    # and the eventual closing fill's Amount only covers settle-to-exit.
    # Ignoring ADJ rows misstates every overnight trade.
    raw = raw[raw["Type"].isin(["TRD", "ADJ"])]

    def num(s):
        return pd.to_numeric(
            s.astype(str).str.replace(r"[$,()]", "", regex=True).replace("--", np.nan),
            errors="coerce")

    fills, unmatched, exp_fills = [], [], []
    spread_descs_fut: list[str] = []
    for _, r in raw.iterrows():
        desc = str(r["Description"]).strip()
        if str(r["Type"]).strip() == "ADJ":
            ma = re.match(r"^(?P<symbol>/[A-Z0-9]+)(?::[A-Z]+)?\s+mark to market", desc)
            amt = num(pd.Series([r.get("Amount")])).iloc[0]
            if ma is not None and pd.notna(amt):
                fills.append({
                    "symbol": ma.group("symbol").split(":")[0].lstrip("/"),
                    "account": account, "venue": "futures",
                    "time": pd.Timestamp(f"{r['Exec Date']} {r['Exec Time']}"),
                    "price": 0.0, "signed_qty": 0.0,   # zero-qty: adjustment, not a fill
                    "commission": 0.0, "realized": float(amt),
                    "multiplier": 1.0, "cashflow": float(amt),
                })
            continue
        mo = _SCHWAB_FUTOPT_RE.match(desc)
        m = None if mo else _SCHWAB_FUT_RE.match(desc)
        if mo is None and m is None:
            # An option that expired: close the position at zero. Without
            # this the contract never returns to flat, FIFO drops the whole
            # round trip, and a total loss quietly leaves the record.
            me = _SCHWAB_FUT_EXP_RE.search(desc)
            if me is not None:
                g = me.groupdict()
                exp_fills.append({
                    "symbol": _futopt_key(g["under"], g["day"], g["mon"], g["yr"],
                                          g["strike"], g["right"]),
                    "account": account, "venue": "futures",
                    "time": pd.Timestamp(f"{r['Exec Date']} {r['Exec Time']}"),
                    "price": 0.0, "signed_qty": np.nan,  # filled in below
                    "commission": 0.0, "realized": np.nan,
                    "multiplier": float(g["mult"]), "cashflow": 0.0,
                })
                continue
            # An exercise converts the option into the underlying at the
            # strike. Close the option (its premium is already in the book)
            # and synthesise the futures opening the broker implies -- long
            # calls removed -> long futures, long puts removed -> short.
            # Skipping this mispairs every subsequent fill on the underlying
            # AND silently deletes the option's premium from the record.
            mx = _SCHWAB_FUT_EXER_RE.search(desc)
            if mx is not None:
                g = mx.groupdict()
                opt_delta = float(g["dqty"])
                t = pd.Timestamp(f"{r['Exec Date']} {r['Exec Time']}")
                fee = abs(_num0(num(pd.Series([r.get("Misc Fees")])).iloc[0])) \
                    + abs(_num0(num(pd.Series([r.get("Commissions & Fees")])).iloc[0]))
                fills.append({
                    "symbol": _futopt_key(g["under"], g["day"], g["mon"], g["yr"],
                                          g["strike"], g["right"]),
                    "account": account, "venue": "futures", "time": t,
                    "price": 0.0, "signed_qty": opt_delta,
                    "commission": fee, "realized": np.nan,
                    "multiplier": float(g["mult"]), "cashflow": 0.0,
                })
                fut_qty = -opt_delta if g["right"] == "CALL" else opt_delta
                fills.append({
                    "symbol": g["under"].lstrip("/").upper(),
                    "account": account, "venue": "futures", "time": t,
                    "price": float(g["strike"]), "signed_qty": fut_qty,
                    "commission": 0.0, "realized": np.nan,
                    "multiplier": 1.0, "cashflow": np.nan,
                })
                continue
            # Multi-leg futures orders (vertical, backratio, custom): one net
            # premium for the whole package, so the specific CONTRACTS they
            # touch are excluded rather than allocated -- per contract, like
            # the equity side, never per underlying: nuking every NQ option
            # because of one NQ vertical would silently delete a book.
            if _SCHWAB_FUT_SPREAD_RE.search(desc):
                spread_descs_fut.append(desc)
            unmatched.append(desc)
            continue
        if mo is not None:
            g = mo.groupdict()
            yy = int(g["yr"]) % 100
            mm = _SCHWAB_MONTHS[g["mon"].upper()[:3]]
            sym = _futopt_key(g["under"], g["day"], g["mon"], 2000 + yy,
                              g["strike"], g["right"])
            price, qty = g["price"], g["qty"]
            # CRITICAL: for futures OPTIONS the Amount column is the premium
            # cash flow (price x contract multiplier x qty), NOT realised P&L
            # the way it is for outright futures. Booking it as P&L turns a
            # -$315k record into a +$7.4mm one. So options are priced by FIFO
            # pairing with the multiplier taken from the description ("1/5000"),
            # and Amount is used only to verify that multiplier is right.
            mult, realized = float(g["mult"]), np.nan
        else:
            sym = m.group("symbol").split(":")[0].lstrip("/")
            price, qty = m.group("price"), m.group("qty")
            mult, realized = 1.0, num(pd.Series([r.get("Amount")])).iloc[0]
        fee = abs(_num0(num(pd.Series([r.get("Misc Fees")])).iloc[0])) \
            + abs(_num0(num(pd.Series([r.get("Commissions & Fees")])).iloc[0]))
        fills.append({
            "symbol": sym, "account": account, "venue": "futures",
            "time": pd.Timestamp(f"{r['Exec Date']} {r['Exec Time']}"),
            "price": _price_num(price),
            "signed_qty": float(str(qty).replace(",", "")),
            "commission": fee,
            "realized": realized, "multiplier": mult,
            "cashflow": num(pd.Series([r.get("Amount")])).iloc[0],
        })
    out = pd.DataFrame(fills)
    # Size each expiration close to whatever remains open on that contract.
    if len(out) and exp_fills:
        net = out.groupby("symbol")["signed_qty"].sum()
        sized = []
        for e in exp_fills:
            open_qty = float(net.get(e["symbol"], 0.0))
            if abs(open_qty) < 1e-9:
                continue
            sized.append({**e, "signed_qty": -open_qty})
        if sized:
            out = pd.concat([out, pd.DataFrame(sized)], ignore_index=True)
    excluded_fut = _schwab_excluded_fut_contracts(spread_descs_fut)
    if len(out) and excluded_fut:
        # Drop every specific contract a multi-leg order touched: partial
        # coverage of a contract's fill history is worse than none.
        out = out[~out["symbol"].isin(excluded_fut)].reset_index(drop=True)
    # Positions opened before the statement window: infer the starting
    # position from the open/close pattern (a fill has a realised Amount iff
    # it reduces the position) and prepend a synthetic lot stamped before
    # _PREWINDOW so the trades that close it can be dropped, not guessed at.
    inferred: dict[str, float] = {}
    if len(out):
        pre_rows = []
        outright = out[out["multiplier"].eq(1.0) & out["signed_qty"].ne(0.0)]
        for sym, g in outright.groupby("symbol", sort=False):
            g = g.sort_values("time", kind="stable")
            pos0 = _infer_start_position(
                [(q, bool(pd.notna(a))) for q, a in zip(g["signed_qty"], g["realized"])])
            if pos0:
                inferred[sym] = float(pos0)
                pre_rows.append({
                    "symbol": sym, "account": account, "venue": "futures",
                    "time": _PREWINDOW - pd.Timedelta(days=1),
                    "price": float(g["price"].iloc[0]), "signed_qty": float(pos0),
                    "commission": 0.0, "realized": np.nan,
                    "multiplier": 1.0, "cashflow": np.nan,
                })
        if pre_rows:
            out = pd.concat([pd.DataFrame(pre_rows), out], ignore_index=True)
    out.attrs["inferred_start_positions"] = inferred
    out.attrs["unmatched_futures_rows"] = len(unmatched)
    out.attrs["unmatched_futures_samples"] = unmatched[:5]
    out.attrs["excluded_fut_contracts"] = sorted(excluded_fut)
    out.attrs["skipped_fut_spread_orders"] = len(spread_descs_fut)
    # The broker's own bottom line for this section, so the report can state
    # what share of it this audit actually accounts for. A tool that silently
    # covers 86% of a book has no business calling anything "verified".
    out.attrs["ledger_realised_total"] = float(
        num(raw["Amount"]).sum()) if "Amount" in raw.columns else np.nan
    return out


def _schwab_excluded_contracts(rows: list[str]) -> set[str]:
    """Every option contract touched by ANY multi-leg spread description."""
    excluded: set[str] = set()
    for desc in rows:
        mu = _SCHWAB_SPREAD_KW_RE.search(desc)
        tail = _SCHWAB_TAIL_RE.search(desc)
        if not mu or not tail:
            continue
        underlying = mu.group("underlying").upper()
        dates = _SCHWAB_DATE_RE.findall(desc)
        strikes = tail.group("strikes").split("/")
        rights = tail.group("rights").split("/")
        for day, mon, yr in dates:
            for strike in strikes:
                for right in rights:
                    excluded.add(_schwab_opt_symbol(underlying, day, mon, yr, strike, right))
    return excluded


def _parse_schwab_statement(source) -> pd.DataFrame:
    """
    Fill-level dataframe (symbol, account, time, price, signed_qty,
    commission) from a Schwab/thinkorswim combined Account Statement export.
    Raises ParseError if this doesn't look like that format after all.
    """
    text = _load_text(source)
    if text is None:
        raise ParseError("Could not read this file as text.")
    title_m0 = _SCHWAB_TITLE_RE.search(text[:400])
    acct0 = title_m0.group(1) if title_m0 else ""
    fut = _parse_schwab_futures(text, acct0)

    block = _extract_cash_balance_block(text)
    if block is None:
        if len(fut):
            fut.attrs["excluded_contracts"] = 0
            fut.attrs["skipped_spread_orders"] = 0
            return fut
        raise ParseError("Found a Schwab/thinkorswim statement header but no Cash Balance ledger.")
    raw = pd.read_csv(io.StringIO(block))
    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw[raw["DATE"].notna()]  # drop the trailing TOTAL row (blank DATE)

    title_m = _SCHWAB_TITLE_RE.search(text[:400])
    account = title_m.group(1) if title_m else ""

    trd = raw[raw["TYPE"] == "TRD"]
    spread_descs = [str(d).strip() for d in trd["DESCRIPTION"]
                     if not _SCHWAB_TRD_RE.match(str(d).strip())]
    excluded = _schwab_excluded_contracts(spread_descs)

    fills = []
    for _, r in trd.iterrows():
        desc = str(r["DESCRIPTION"]).strip()
        m = _SCHWAB_TRD_RE.match(desc)
        if not m:
            continue
        g = m.groupdict()
        if g["right"]:
            sym = _schwab_opt_symbol(g["underlying"], g["day"], g["mon"], g["yr"], g["strike"], g["right"])
            if sym in excluded:
                continue
        else:
            sym = g["underlying"].upper()
        fee = abs(_num0(pd.to_numeric(r.get("Misc Fees"), errors="coerce"))) \
            + abs(_num0(pd.to_numeric(r.get("Commissions & Fees"), errors="coerce")))
        fills.append({
            "symbol": sym, "account": account, "venue": "equity",
            "time": pd.Timestamp(f"{r['DATE']} {r['TIME']}"),
            "price": _price_num(g["price"]),
            "signed_qty": float(g["qty"].replace(",", "")),
            "commission": fee, "realized": np.nan,
        })

    rad = raw[raw["TYPE"] == "RAD"]
    for _, r in rad.iterrows():
        desc = str(r["DESCRIPTION"]).strip()
        m = _SCHWAB_RAD_RE.search(desc)
        if not m:
            continue  # non-expiration removal (e.g. an account transfer) -- see module docstring
        occ = m.group("occ").upper()
        if occ in excluded:
            continue
        fills.append({
            "symbol": occ, "account": account, "venue": "equity",
            "time": pd.Timestamp(f"{r['DATE']} {r['TIME']}"),
            "price": 0.0,
            "signed_qty": float(m.group("expqty")),
            "commission": 0.0, "realized": np.nan,
        })

    if not fills and not len(fut):
        raise ParseError("Schwab/thinkorswim statement recognised but no parseable fills were found.")
    df = pd.concat([pd.DataFrame(fills), fut], ignore_index=True) if len(fut) \
        else pd.DataFrame(fills)
    df.attrs["excluded_contracts"] = len(excluded)
    df.attrs["skipped_spread_orders"] = len(spread_descs)
    df.attrs["n_futures_fills"] = int(len(fut))
    df.attrs["unmatched_futures_rows"] = fut.attrs.get("unmatched_futures_rows", 0) if len(fut) else 0
    df.attrs["futures_ledger_total"] = fut.attrs.get("ledger_realised_total", np.nan) if len(fut) else np.nan
    df.attrs["inferred_start_positions"] = fut.attrs.get("inferred_start_positions", {}) if len(fut) else {}
    df.attrs["excluded_fut_contracts"] = fut.attrs.get("excluded_fut_contracts", []) if len(fut) else []
    df.attrs["skipped_fut_spread_orders"] = fut.attrs.get("skipped_fut_spread_orders", 0) if len(fut) else 0
    return df


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
    has_realized = "realized" in df.columns
    for sym, g in df.groupby("symbol", sort=False):
        book: list[dict] = []  # open lots, all same sign
        for _, r in g.iterrows():
            qty = float(r["signed_qty"])
            price, t = float(r["price"]), r["time"]
            comm = _num0(r.get("commission", 0.0))
            # Contract multiplier: 1 for cash equities and for outright
            # futures (whose P&L the broker states directly); the real point
            # value for futures options, whose P&L we must compute ourselves.
            mult = _num0(r.get("multiplier", 1.0), 1.0)
            comm_pc = abs(comm) / max(abs(float(r["signed_qty"])), 1e-9)  # per contract
            # zero-qty row with a realised amount = an overnight
            # mark-to-market adjustment: it belongs to whatever is open,
            # spread per contract across the open lots
            if qty == 0:
                rz = r.get("realized", np.nan) if has_realized else np.nan
                if pd.notna(rz) and book:
                    tot_open = sum(abs(l["qty"]) for l in book)
                    for l in book:
                        l["adj_pc"] = l.get("adj_pc", 0.0) + float(rz) / tot_open
                continue
            fill_rows: list[dict] = []
            while qty != 0:
                if book and np.sign(book[0]["qty"]) != np.sign(qty):
                    lot = book[0]
                    matched = min(abs(lot["qty"]), abs(qty))
                    side = 1 if lot["qty"] > 0 else -1
                    fill_rows.append({
                        "symbol": sym,
                        "account": r.get("account", ""),
                        "venue": r.get("venue", np.nan),
                        "direction": "long" if side > 0 else "short",
                        "qty": matched,
                        "entry_time": lot["time"], "exit_time": t,
                        "entry_price": lot["price"], "exit_price": price,
                        "gross_pnl": side * (price - lot["price"]) * matched * mult,
                        # a round trip pays to get in AND out: the opening
                        # lot's fee share plus this closing fill's share
                        "commission": matched * (lot.get("comm_pc", 0.0) + comm_pc),
                        "_adj": matched * lot.get("adj_pc", 0.0),
                    })
                    lot["qty"] -= side * matched
                    qty += side * matched
                    if abs(lot["qty"]) < 1e-9:
                        book.pop(0)
                else:
                    book.append({"qty": qty, "price": price, "time": t,
                                 "comm_pc": comm_pc})
                    qty = 0
            # Where the broker states realised P&L for this closing fill, that
            # figure is authoritative (it already carries the contract point
            # value); split it across the lots this fill closed, pro rata by
            # matched quantity, as the trade's GROSS. Net is then uniformly
            # gross minus fees for every venue -- a "net" that quietly
            # excludes commissions would flatter exactly the overtrading this
            # tool exists to expose.
            if fill_rows:
                rz = r.get("realized", np.nan) if has_realized else np.nan
                tot = sum(fr["qty"] for fr in fill_rows) or 1.0
                for fr in fill_rows:
                    adj = fr.pop("_adj", 0.0)
                    if pd.notna(rz):
                        # broker realised covers settle-to-exit; the marks
                        # that accrued while the lot was open cover the rest
                        fr["gross_pnl"] = float(rz) * fr["qty"] / tot + adj
                    else:
                        fr["gross_pnl"] = fr["gross_pnl"] + adj
            rows.extend(fill_rows)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # One closing decision = one trade. A single closing fill that empties
    # several FIFO lots would otherwise appear as a run of same-sign "trades"
    # -- and every sequence-based slice (after a loss, trade order in day)
    # would then be fed each scale-out's own siblings, manufacturing
    # state-dependence and overstating the independent sample size behind
    # every q-value.
    key = ["symbol", "direction", "exit_time"]
    w = (out["entry_price"] * out["qty"]).groupby(
        [out[k] for k in key], sort=False, dropna=False).sum()
    g = out.groupby(key, sort=False, dropna=False)
    merged = g.agg(
        account=("account", "first"), venue=("venue", "first"),
        qty=("qty", "sum"), entry_time=("entry_time", "min"),
        exit_price=("exit_price", "first"),
        gross_pnl=("gross_pnl", "sum"), commission=("commission", "sum"),
    )
    merged["entry_price"] = (w / merged["qty"]).astype(float)
    return merged.reset_index()


def normalize(source, point_values: dict[str, float] | None = None) -> tuple[pd.DataFrame, str]:
    """
    Returns (canonical dataframe, detected format name).

    gross_pnl from fill pairing is in *price points*; supply `point_values`
    ({"ES": 50, "CL": 1000, ...}) to convert to dollars. Without it the audit
    still runs -- R-multiples are scale-free -- but dollar figures are omitted
    rather than guessed, because a guessed dollar figure in a verification
    report is worse than no figure.
    """
    schwab_text = _load_text(source)
    schwab_meta = {}
    if schwab_text is not None and _sniff_schwab(schwab_text):
        fills = _parse_schwab_statement(source)
        out = _pair_fills_fifo(fills.dropna(subset=["time", "price", "signed_qty"]))
        fmt_name = "Schwab / thinkorswim (Account Statement)"
        # round trips that close a synthetic pre-window lot have no knowable
        # entry; they are dropped and counted, never guessed at
        pre_mask = out["entry_time"] < _PREWINDOW if len(out) else pd.Series(dtype=bool)
        n_pre = int(pre_mask.sum()) if len(out) else 0
        if n_pre:
            out = out[~pre_mask].reset_index(drop=True)
        schwab_meta = {
            "excluded_contracts": fills.attrs.get("excluded_contracts", 0),
            "skipped_spread_orders": fills.attrs.get("skipped_spread_orders", 0),
            "unmatched_futures_rows": fills.attrs.get("unmatched_futures_rows", 0),
            "futures_ledger_total": fills.attrs.get("futures_ledger_total", np.nan),
            "inferred_start_positions": fills.attrs.get("inferred_start_positions", {}),
            "excluded_fut_contracts": fills.attrs.get("excluded_fut_contracts", []),
            "skipped_fut_spread_orders": fills.attrs.get("skipped_fut_spread_orders", 0),
            "dropped_prewindow_trades": n_pre,
        }
    else:
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

    # Fill only the gaps: a source may state realised P&L on some rows
    # (e.g. futures fills) and not others, so an all-or-nothing check would
    # silently drop every row the broker didn't price for us.
    _fallback = out["gross_pnl"].fillna(0) - out["commission"].fillna(0).abs()
    out["net_pnl"] = out["net_pnl"].fillna(_fallback)
    if point_values:
        mult = out["symbol"].map(lambda s: point_values.get(_root(s), np.nan))
        conv = mult.notna() & out["net_pnl"].notna()
        out.loc[conv, "net_pnl"] = out.loc[conv, "net_pnl"] * mult[conv] * out.loc[conv, "qty"].fillna(1)

    is_fut = (out["venue"] == "futures") if "venue" in out else pd.Series(False, index=out.index)
    out["root"] = np.where(is_fut, out["symbol"].map(_futures_root), out["symbol"].map(_root))
    out["trade_id"] = out["trade_id"].fillna(pd.Series(range(len(out))).astype(str))
    out = out.dropna(subset=["net_pnl"])
    if out.empty:
        raise ParseError("File parsed but no trades with a usable P&L were found.")
    if out["entry_time"].notna().any():
        out = out.sort_values("entry_time", kind="stable").reset_index(drop=True)
    out = out.reset_index(drop=True)
    if schwab_meta:
        out.attrs.update(schwab_meta)
    return out, fmt_name


def _root(sym) -> str:
    m = ROOT_RE.match(str(sym).strip().upper())
    return m.group(1) if m else str(sym)


def diagnose(source) -> dict:
    """Parse without auditing; report exactly what was read, used and dropped."""
    df, fmt = normalize(source)
    meta = dict(getattr(df, "attrs", {}) or {})
    out = {
        "format": fmt,
        "trades": int(len(df)),
        "date_range": (str(df["entry_time"].min()), str(df["exit_time"].max())),
        "net_pnl_sum": float(df["net_pnl"].sum()),
        "gross_pnl_sum": float(df["gross_pnl"].sum()) if df["gross_pnl"].notna().any() else None,
        "fees_sum": float(df["commission"].fillna(0).abs().sum()),
        "by_venue": {str(k): int(v) for k, v in df.groupby("venue").size().items()}
        if "venue" in df and df["venue"].notna().any() else {},
        "by_root": {str(k): int(v) for k, v in
                    df.groupby("root").size().sort_values(ascending=False).head(20).items()},
    }
    out.update(meta)
    ledger = meta.get("futures_ledger_total")
    if ledger is not None and np.isfinite(ledger):
        fut_gross = float(df.loc[df["venue"] == "futures", "gross_pnl"].sum()) \
            if "venue" in df else float(df["gross_pnl"].sum())
        out["reconciliation"] = {
            "futures_ledger_total": float(ledger),
            "captured_futures_gross": fut_gross,
            "unaccounted": float(ledger) - fut_gross,
        }
    return out


_EQ_OPT_RE = re.compile(r"^[A-Z.]+\d{6}[CP][\d.]+$")


def _scale_class(sym, venue) -> str:
    """
    Risk-scale class for the 1R unit: instrument root plus option-ness.
    A $5 SLV share loss and a $500 SIL futures loss are not the same risk
    unit, and an option on NQ is not the same unit as an NQ future.
    """
    s = str(sym)
    kind = "opt" if ("_" in s or _EQ_OPT_RE.match(s)) else "out"
    root = _futures_root(s) if venue == "futures" else _root(s)
    return f"{root}|{kind}"


def to_r_multiples(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """
    R-multiples. If the export carries no per-trade risk (almost none do),
    normalise by the median absolute loss -- a robust stand-in for '1R' that
    is unaffected by the few catastrophic trades that would wreck a mean-based
    scale.

    When one record spans materially different risk scales (shares next to
    full-size futures next to options), a single global unit makes the big
    book's trades look like +/-40R monsters and the small book's like
    rounding errors -- so each instrument class with enough losses gets its
    own unit, and the gate (largest class unit >= 3x the smallest) keeps
    single-scale records on the one global unit. The report must say which
    basis was used; a silent proxy is how other tools quietly become
    uncomparable across accounts.
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
        return pnl / 1.0, "raw P&L (no usable risk scale)"

    if "symbol" not in df:
        return pnl / unit, basis
    venues = df["venue"] if "venue" in df else pd.Series(np.nan, index=df.index)
    keys = np.array([_scale_class(s, v) for s, v in zip(df["symbol"], venues)])
    class_units: dict[str, float] = {}
    for k in pd.unique(keys):
        kl = np.abs(pnl[(keys == k) & (pnl < 0)])
        if kl.size >= 8:                       # enough losses to estimate a unit
            u = float(np.median(kl))
            if np.isfinite(u) and u > 0:
                class_units[k] = u
    if len(class_units) >= 2 and max(class_units.values()) >= 3.0 * min(class_units.values()):
        units = np.array([class_units.get(k, unit) for k in keys])
        return pnl / units, "median absolute loss per instrument class (estimated 1R)"
    return pnl / unit, basis

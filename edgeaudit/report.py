"""
Report rendering.

The signature device is the interval rail: every claim is drawn as a
confidence interval on one shared, zero-anchored axis. Verified intervals are
solid; unproven ones are hollow outlines. The reader sees which bars touch
zero before reading a single number -- which is the entire product in one
glance. No JavaScript: this has to survive email clients, print, and being
forwarded to a risk desk.
"""

from __future__ import annotations

import html
import math
from datetime import datetime

import numpy as np

CSS = """
:root{
  --paper:#E9EDEF; --card:#F6F8F9; --ink:#101619; --muted:#667279;
  --verified:#0F5F52; --unproven:#A9B3B8; --adverse:#8C2F22; --rule:#C6D0D5;
  --display:"Charter","Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
  --body:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--body);
  font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased}
.sheet{max-width:940px;margin:0 auto;padding:48px 28px 96px}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--muted)}
h1{font-family:var(--display);font-size:40px;line-height:1.1;margin:6px 0 4px;font-weight:600}
h2{font-family:var(--display);font-size:21px;font-weight:600;margin:44px 0 6px}
p.lede{font-size:17px;max-width:64ch;margin:10px 0 0}
.masthead{border-bottom:2px solid var(--ink);padding-bottom:22px}
.meta{font-family:var(--mono);font-size:12px;color:var(--muted);margin-top:14px}
.meta span{margin-right:18px;white-space:nowrap}

.verdict{margin:30px 0 6px;padding:22px 24px;background:var(--card);
  border-left:5px solid var(--unproven)}
.verdict.pos{border-left-color:var(--verified)}
.verdict.neg{border-left-color:var(--adverse)}
.verdict h3{font-family:var(--display);font-size:26px;margin:0 0 8px;font-weight:600}
.verdict p{margin:0;max-width:70ch}

.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));
  gap:1px;background:var(--rule);border:1px solid var(--rule);margin-top:24px}
.cell{background:var(--card);padding:14px 16px}
.cell .k{font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--muted)}
.cell .v{font-family:var(--mono);font-size:22px;margin-top:3px}
.cell .n{font-size:12px;color:var(--muted);margin-top:2px}

table{width:100%;border-collapse:collapse;margin-top:14px;font-size:13.5px}
th{font-family:var(--mono);font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;
  color:var(--muted);text-align:left;padding:8px 10px;border-bottom:1px solid var(--ink);
  font-weight:400}
td{padding:9px 10px;border-bottom:1px solid var(--rule);vertical-align:middle}
td.num{font-family:var(--mono);text-align:right;white-space:nowrap}
tr.ghost td{color:var(--muted)}
.dim{font-family:var(--mono);font-size:10.5px;letter-spacing:.08em;
  text-transform:uppercase;color:var(--muted);display:block}

/* interval rail */
.rail{position:relative;height:20px;min-width:190px}
.rail .axis{position:absolute;inset:9px 0 auto 0;height:1px;background:var(--rule)}
.rail .zero{position:absolute;top:0;bottom:0;width:1px;background:var(--ink);opacity:.55}
.rail .bar{position:absolute;top:6px;height:7px;border-radius:1px}
.rail .bar.solid{background:var(--verified)}
.rail .bar.solid.neg{background:var(--adverse)}
.rail .bar.hollow{background:transparent;border:1px solid var(--unproven)}
.rail .tick{position:absolute;top:2px;height:15px;width:2px;background:var(--ink)}
.rail .tick.faint{background:var(--unproven)}

.badge{font-family:var(--mono);font-size:10px;letter-spacing:.08em;text-transform:uppercase;
  padding:2px 6px;border:1px solid currentColor;border-radius:2px;white-space:nowrap}

/* hover glossary: CSS-only so it survives email clients and print; the
   dotted underline degrades to a plain hint where :hover doesn't exist */
.term{border-bottom:1px dotted var(--muted);cursor:help;position:relative}
.term:hover::after{content:attr(data-tip);position:absolute;left:0;bottom:calc(100% + 7px);
  z-index:20;width:min(340px,82vw);background:var(--ink);color:#F6F8F9;padding:10px 13px;
  font-family:var(--body);font-size:12.5px;line-height:1.5;font-weight:400;
  letter-spacing:0;text-transform:none;white-space:normal;text-align:left;
  border-radius:3px;box-shadow:0 5px 16px rgba(16,22,25,.3)}
.term:hover::before{content:"";position:absolute;left:12px;bottom:calc(100% + 1px);
  border:6px solid transparent;border-top-color:var(--ink);z-index:21}
@media print{.term{border-bottom:none}}
.badge.ok{color:var(--verified)}
.badge.no{color:var(--muted)}
.badge.bad{color:var(--adverse)}

.callout{margin-top:18px;padding:18px 20px;border:1px solid var(--ink);background:transparent}
.callout .big{font-family:var(--display);font-size:30px;line-height:1.15;margin:0 0 6px}
.note{font-size:13px;color:var(--muted);max-width:72ch;margin-top:10px}
ul.method{font-size:13px;color:var(--muted);max-width:74ch;padding-left:18px}
ul.method li{margin:5px 0}
footer{margin-top:56px;padding-top:16px;border-top:1px solid var(--rule);
  font-family:var(--mono);font-size:11px;color:var(--muted)}
@media (max-width:640px){
  .sheet{padding:28px 16px 64px} h1{font-size:30px}
  table{font-size:12.5px} .rail{min-width:120px}
}
@media print{body{background:#fff} .verdict,.cell{background:#fff}}
"""


def _fmt(v, nd=3, pct=False, dollar=False, dash="—"):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return dash if not (isinstance(v, float) and math.isinf(v)) else "∞"
    if pct:
        return f"{v*100:.1f}%"
    if dollar:
        return "$0" if abs(v) < 0.5 else f"${v:,.0f}"
    return f"{v:+.{nd}f}" if nd else f"{v:,.0f}"


# Plain-language glossary, shown on hover. Written for someone who has never
# journaled or taken a statistics class; every term of art on the page should
# route through _t() so the report never assumes vocabulary it didn't teach.
GLOSSARY = {
    "r": ("R measures every trade in units of your own typical loss. We set 1R to the "
          "median losing trade in this record (in dollars), so +2.0R means a win worth "
          "two typical losses and -1.0R is a standard loss. It makes different months, "
          "instruments and account sizes comparable on one scale."),
    "slice": ("A slice is a group of your trades that share one trait: every Monday trade, "
              "every NQ trade, every trade held under an hour, every trade taken right "
              "after a loss. Journals cut your history into ~30 of these and show you the "
              "winners. The catch: cut 30 slices from ANY record - even coin flips - and "
              "one or two will look profitable by pure luck. This report tests which "
              "slices are luck and which are real."),
    "ci": ("The 95% confidence interval: the range the true value plausibly lives in, "
           "given how much your trades vary. If it contains zero, the honest reading is "
           "'could be zero' - a positive average whose interval spans zero is not "
           "evidence of an edge, just a sample that hasn't decided yet."),
    "q": ("A p-value answers 'how surprising is this slice if it were pure luck?'. A "
          "q-value corrects that for how many slices were examined at once: it is the "
          "share of your 'real' findings you should expect to be false if you accept "
          "this one. We require q <= 0.10 - at most one in ten accepted findings false."),
    "psr": ("Probabilistic Sharpe Ratio: the probability that your true risk-adjusted "
            "return is above zero, after accounting for skew (lopsided wins/losses), fat "
            "tails (rare huge trades) and how short the sample is. 95% or higher is the "
            "bar for 'established'."),
    "dsr": ("Deflated Sharpe Ratio: the same probability after ALSO charging you for "
            "every slice that was searched. Searching 40 slices guarantees the best one "
            "looks impressive by luck alone; this number deflates the claim accordingly. "
            "It is the figure that kills most 'I found my edge' stories."),
    "pf": ("Profit factor: gross dollars won divided by gross dollars lost. 1.0 is "
           "break-even before costs; below 1.0 means the losses outweighed the wins."),
    "winrate": ("The share of trades closed at a profit. Meaningless on its own: a 90% "
                "win rate loses money if the 10% of losers are big enough, and a 35% win "
                "rate prints money if wins are triple the size of losses."),
    "adjusted": ("The slice's average after being pulled toward your overall average, by "
                 "an amount set by how much of the slice-to-slice spread is explainable "
                 "as noise. When the spread IS noise, every slice collapses onto your "
                 "overall number - which is the statistically correct answer to 'which "
                 "setup is my best', and the one an edge-finder will never give you."),
    "bootstrap": ("How the intervals are computed: your record is resampled thousands of "
                  "times in short consecutive runs (not one trade at a time), so streaks "
                  "and tilt-clusters count as the dependent episodes they are. Resampling "
                  "single trades would make the intervals artificially narrow - which is "
                  "how fake confidence gets manufactured."),
    "payoff": ("Average winning trade divided by average losing trade. Together with win "
               "rate it fully determines whether you make money."),
    "breakeven": ("The win rate your current payoff ratio requires just to break even. "
                  "Your actual win rate minus this number is your entire economics in "
                  "one figure."),
    "drawdown": ("The deepest fall from any high-water mark of your cumulative P&L - the "
                 "losing stretch you would have had to sit through without quitting."),
    "effective": ("Your trades weighted by how much risk each one actually carried. Four "
                  "thousand trades where forty giants carry most of the variance behave, "
                  "statistically, like a few hundred equal-sized trades - so the record "
                  "knows less than its trade count suggests."),
    "verified": ("Cleared all three bars: distinguishable from zero, distinguishable from "
                 "your own overall average, and still standing after the correction for "
                 "how many slices were examined."),
    "notdist": ("Could be luck. Not 'bad' - unproven. An edge-finder would report many of "
                "these as edges; this report leaves them unclaimed."),
    "expectancy": ("Your average result per trade. 'Negative expectancy, established' "
                   "means the average is below zero AND the uncertainty interval is "
                   "entirely below zero - it is no longer plausibly variance."),
    "mtrl": ("Minimum track record length: how many trades a record with this Sharpe, "
             "skew and tail-fatness needs before the claim 'this is above zero' is "
             "statistically supportable at 95% confidence."),
}


def _t(text: str, key: str) -> str:
    """Wrap a term with its plain-language hover explanation."""
    tip = html.escape(GLOSSARY[key], quote=True)
    return f'<span class="term" data-tip="{tip}">{text}</span>'


def _rail(lo, hi, mean, scale, verified):
    """One confidence interval drawn on a shared zero-anchored axis."""
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in (lo, hi, mean)):
        return '<div class="rail"><div class="axis"></div>'\
               '<div class="zero" style="left:50%"></div></div>'
    def pos(x):
        return max(1.0, min(99.0, 50.0 + 50.0 * x / scale))
    l, h, m = pos(lo), pos(hi), pos(mean)
    cls = "solid" + (" neg" if hi < 0 else "") if verified else "hollow"
    tick = "tick" if verified else "tick faint"
    return (
        '<div class="rail"><div class="axis"></div>'
        f'<div class="bar {cls}" style="left:{l:.2f}%;width:{max(h-l,0.6):.2f}%"></div>'
        f'<div class="{tick}" style="left:{m:.2f}%"></div>'
        '<div class="zero" style="left:50%"></div></div>'
    )


def _coverage_note(res) -> str:
    """
    State what share of the broker's own realised total this audit accounts
    for. Everything else on the page is conditional on this number: a verdict
    drawn from 70% of a book is a verdict about a different book. Silence here
    would be the exact failure this tool exists to call out in other tools.
    """
    c = getattr(res, "coverage", None) or {}
    led, acc = c.get("ledger_total", np.nan), c.get("accounted", np.nan)
    bits = []
    if np.isfinite(led) and np.isfinite(acc) and abs(led) > 1e-9:
        pct = 100.0 * acc / led
        gap = led - acc
        cls = "" if pct >= 99 else ' style="color:#8C2F22"'
        bits.append(
            f'<span{cls}><strong>{pct:.1f}%</strong> of the statement&rsquo;s realised P&amp;L '
            f'is accounted for here ({_fmt(acc, nd=0, dollar=True)} of '
            f'{_fmt(led, nd=0, dollar=True)}; {_fmt(gap, nd=0, dollar=True)} unattributed)</span>')
    ex, sp = c.get("excluded_contracts", 0), c.get("skipped_spread_orders", 0)
    if ex or sp:
        bits.append(f"{sp:,} multi-leg orders across {ex:,} contracts excluded — a combo carries "
                    f"one net price for several legs, and a per-leg P&amp;L would be invented")
    if not bits:
        return ""
    return ('<p class="meta" style="margin-top:10px;max-width:70ch;line-height:1.5">'
            + " &middot; ".join(bits) + "</p>")


def _money_section(res) -> str:
    """
    The money question, in money. R-multiples are scale-free and comparable
    across accounts, which is why the engine works in them -- but no one has
    ever read "-1.106R" and felt informed. A paid report states the result in
    the units the customer's account is denominated in, and shows the two
    numbers that decide everything: what a win pays relative to a loss, and
    the win rate that combination requires.
    """
    m = getattr(res, "money", None) or {}
    if not m:
        return ""
    be = m.get("breakeven_win_rate", np.nan)
    gap = m.get("win_rate_gap", np.nan)
    gapcls = "pos" if (np.isfinite(gap) and gap >= 0) else "neg"
    gapline = ""
    if np.isfinite(be):
        gapline = (f"<p>Your average win is <strong>{_fmt(m['avg_win'], nd=2, dollar=True)}</strong> "
                   f"against an average loss of <strong>{_fmt(abs(m['avg_loss']), nd=2, dollar=True)}</strong> "
                   f"— a {_t('payoff ratio', 'payoff')} of {m.get('payoff_ratio', float('nan')):.2f}. That combination needs a "
                   f"<strong>{be*100:.1f}%</strong> {_t('win rate to break even', 'breakeven')}. "
                   f"You are at <span class=\"{gapcls}\">{100*(be+gap):.1f}%</span>, "
                   f"<strong>{abs(gap)*100:.1f} points {'above' if gap >= 0 else 'below'}</strong> it. "
                   f"Closing that gap is the whole problem, stated as one number.</p>")
    fee = ""
    if np.isfinite(m.get("fees_pct_of_result", np.nan)) and m.get("fees", 0) > 0:
        fee = (f"<p>Costs: <strong>{_fmt(m['fees'], nd=0, dollar=True)}</strong> in commissions and "
               f"fees, {m['fees_pct_of_result']*100:.0f}% of the size of your net result, or "
               f"{_fmt(m['fees']/max(res.n_trades,1), nd=2, dollar=True)} per trade. This is the one "
               f"component that shrinks in exact proportion to trading less.</p>")
    return f"""
<h2>The result, in money</h2>
<div class="grid">
  <div class="cell"><div class="k">Per trade</div><div class="v">{_fmt(m['per_trade'], nd=2, dollar=True)}</div>
    <div class="n">across {res.n_trades:,} trades</div></div>
  <div class="cell"><div class="k">Typical swing</div><div class="v">{_fmt(m['sd_per_trade'], nd=0, dollar=True)}</div>
    <div class="n">std dev of one trade</div></div>
  <div class="cell"><div class="k">Average win</div><div class="v">{_fmt(m['avg_win'], nd=0, dollar=True)}</div>
    <div class="n">biggest {_fmt(m['biggest_win'], nd=0, dollar=True)}</div></div>
  <div class="cell"><div class="k">Average loss</div><div class="v">{_fmt(m['avg_loss'], nd=0, dollar=True)}</div>
    <div class="n">biggest {_fmt(m['biggest_loss'], nd=0, dollar=True)}</div></div>
</div>
<div class="callout">{gapline}{fee}</div>"""


def _curve_section(res) -> str:
    c = getattr(res, "equity_curve", None) or {}
    if not c:
        return ""
    gb = c.get("pct_of_peak_given_back", np.nan)
    mult = c.get("drawdown_vs_peak_multiple", np.nan)
    if c.get("never_profitable"):
        gbtxt = (" This curve was never above water: there was no peak to give back, "
                 "so the drawdown figure is simply the deepest point reached.")
    elif np.isfinite(mult):
        gbtxt = (f" Your worst drawdown was <strong>{mult:,.1f}&times;</strong> the largest "
                 f"cumulative profit you ever held — the downside was never on the same "
                 f"scale as the upside.")
    elif np.isfinite(gb) and gb > 0:
        gbtxt = f" You gave back <strong>{gb*100:.0f}%</strong> of your peak."
    else:
        gbtxt = ""
    return f"""
<h2>The path, not just the destination</h2>
<div class="grid">
  <div class="cell"><div class="k">Peak</div><div class="v">{_fmt(c['peak'], nd=0, dollar=True)}</div>
    <div class="n">best cumulative point</div></div>
  <div class="cell"><div class="k">Finished at</div><div class="v">{_fmt(c['final'], nd=0, dollar=True)}</div>
    <div class="n">realised, cumulative</div></div>
  <div class="cell"><div class="k">{_t("Worst drawdown", "drawdown")}</div><div class="v">{_fmt(c['max_drawdown'], nd=0, dollar=True)}</div>
    <div class="n">peak to trough</div></div>
  <div class="cell"><div class="k">Time under water</div><div class="v">{c['pct_of_trades_underwater']*100:.0f}%</div>
    <div class="n">longest run {c['longest_underwater_trades']:,} trades</div></div>
</div>
<p class="note">A final number says what happened; a drawdown says whether you could have
stayed in the seat while it happened.{gbtxt}</p>"""


def _trend_section(res) -> str:
    """
    Is the recent stretch different from the early one? A single verdict over
    a whole sample answers a question nobody asked. This is reported as a
    difference with an interval -- eyeballing three point estimates is how
    people talk themselves into believing they have turned a corner.
    """
    t = getattr(res, "trend", None) or {}
    if not t or "parts" not in t:
        return ""
    cells = "".join(
        f"<div class=\"cell\"><div class=\"k\">Period {p['i']} of {len(t['parts'])}</div>"
        f"<div class=\"v\">{_fmt(p['mean'], nd=2, dollar=True)}</div>"
        f"<div class=\"n\">{p['n']:,} trades · {_fmt(p['total'], nd=0, dollar=True)}</div></div>"
        for p in t["parts"])
    d, lo, hi = t["diff_last_minus_first"], t["diff_lo"], t["diff_hi"]
    if t.get("improved"):
        line = (f"<strong>Improving.</strong> The most recent period beats the earliest by "
                f"{_fmt(d, nd=2, dollar=True)} per trade, and the interval "
                f"({_fmt(lo, nd=2, dollar=True)} to {_fmt(hi, nd=2, dollar=True)}) stays above zero.")
    elif t.get("worsened"):
        line = (f"<strong>Deteriorating.</strong> The most recent period is "
                f"{_fmt(abs(d), nd=2, dollar=True)} per trade worse than the earliest, "
                f"interval {_fmt(lo, nd=2, dollar=True)} to {_fmt(hi, nd=2, dollar=True)}.")
    else:
        line = (f"<strong>No detectable trend.</strong> The most recent period differs from the "
                f"earliest by {_fmt(d, nd=2, dollar=True)} per trade, but the interval "
                f"({_fmt(lo, nd=2, dollar=True)} to {_fmt(hi, nd=2, dollar=True)}) contains zero — "
                f"consistent with chance. Three periods that look different to the eye usually are not.")
    return f"""
<h2>Are you getting better?</h2>
<div class="grid">{cells}</div>
<div class="callout"><p>{line}</p></div>"""


def _recheck_section(res) -> str:
    """
    The standing offer: a verdict is a snapshot, and the only way to find out
    whether a change worked is new trades tested the same way. This is also
    the business: the re-audit is where a report becomes a relationship, and
    the before/after comparison is statistically answerable (regime_trend),
    not a feeling.
    """
    t = getattr(res, "trend", None) or {}
    if t.get("improved"):
        hook = "The trend section already shows improvement — a re-audit would date-stamp it."
    elif t.get("worsened"):
        hook = "The trend section shows deterioration — a re-audit tells you if a change arrested it."
    else:
        hook = ("Nothing in this report can move until there are new trades to test — "
                "old trades cannot get better.")
    return f"""
<h2>Re-audit in 30&ndash;60 days</h2>
<div class="callout">
  <p>This verdict describes the trades supplied, and nothing else. If you change anything —
  size, instruments, what you do after a loss — the only way to know whether it worked is to
  run the <em>new</em> trades through the same tests. {hook}</p>
  <p class="note">Send your next statement in 30&ndash;60 days and the &ldquo;Are you getting
  better?&rdquo; section becomes a before/after comparison with an interval on it, not a
  feeling. The re-check is part of the audit.</p>
</div>"""


def _forward_section(res) -> str:
    f = getattr(res, "forward", None) or {}
    tbl = f.get("table") or {}
    if not tbl:
        return ""
    rows = "".join(
        f"<tr><td>an edge of {_fmt(e, nd=2, dollar=True)} per trade</td>"
        f"<td class=\"num\">{n:,.0f} trades</td></tr>"
        for e, n in sorted(tbl.items()))
    return f"""
<h2>How long until you know</h2>
<p>The sample needed to establish an edge scales with the <em>square</em> of your dispersion,
so trade size is not only a risk decision — it sets how fast you can learn anything at all.
At your typical swing of {_fmt(f['sd'], nd=0, dollar=True)} per trade:</p>
<table><thead><tr><th>To establish</th><th class="num">You need</th></tr></thead>
<tbody>{rows}</tbody></table>
<p class="note">80% power, 5% significance, two-sided. Halving your position size roughly
quarters the sample you need — the cheapest way to buy statistical power you will ever find.</p>"""


def _class_section(res) -> str:
    bc = getattr(res, "by_class", None)
    if bc is None or len(bc) == 0:
        return ""
    rows = "".join(
        f"<tr><td>{html.escape(str(r['class']).capitalize())}</td><td class=\"num\">{r['trades']:,}</td>"
        f"<td class=\"num\">{_fmt(r['per_trade'], nd=2, dollar=True)}</td>"
        f"<td class=\"num\">{_fmt(r['ci_lo'], nd=2, dollar=True)} to {_fmt(r['ci_hi'], nd=2, dollar=True)}</td>"
        f"<td class=\"num\">{_fmt(r['total'], nd=0, dollar=True)}</td></tr>"
        for _, r in bc.iterrows())
    return f"""
<h2>By account type</h2>
<p>Pooling books that trade at different scales is the fastest way to a wrong answer:
a flat equity book and a bleeding futures book average into a mild-looking loss that
describes neither.</p>
<table><thead><tr><th>Book</th><th class="num">Trades</th><th class="num">Per trade</th>
<th class="num">95% interval</th><th class="num">Total</th></tr></thead>
<tbody>{rows}</tbody></table>"""


def to_html(res, title="Trade Edge Audit", subject="Account") -> str:
    gs, ad = res.global_stats, res.adequacy
    e = html.escape
    pos = res.verdict.startswith("Positive")
    neg = res.verdict.startswith("Negative")
    vclass = "pos" if pos else ("neg" if neg else "")

    b = res.buckets
    tested = b[b["testable"]] if len(b) else b
    finite = []
    for col in ("ci_lo", "ci_hi"):
        if len(tested):
            finite += [v for v in tested[col].tolist() if np.isfinite(v)]
    finite += [gs["ci_lo"], gs["ci_hi"]]
    scale = max(0.05, max(abs(v) for v in finite if np.isfinite(v)))

    d0, d1 = res.date_range
    span = ""
    if d0 is not None and d1 is not None and not (isinstance(d0, float) and math.isnan(d0)):
        try:
            span = f"{d0:%d %b %Y} – {d1:%d %b %Y}"
        except Exception:
            span = ""

    rows = []
    for _, r in (tested.iterrows() if len(tested) else iter(())):
        ver = bool(r["verified"])
        badge = (f'<span class="badge ok">{_t("Verified", "verified")}</span>' if ver and r["mean_r"] > 0 else
                 f'<span class="badge bad">{_t("Verified loss", "verified")}</span>' if ver else
                 f'<span class="badge no">{_t("Not distinguishable", "notdist")}</span>')
        rows.append(f"""<tr class="{'' if ver else 'ghost'}">
  <td><span class="dim">{e(str(r['dimension']))}</span>{e(str(r['label']))}</td>
  <td class="num">{int(r['n'])}</td>
  <td class="num">{_fmt(r['mean_r'])}</td>
  <td class="num">{_fmt(r['shrunk_mean_r'])}</td>
  <td>{_rail(r['ci_lo'], r['ci_hi'], r['shrunk_mean_r'], scale, ver)}</td>
  <td class="num">{_fmt(r['q_vs_self'], nd=3).lstrip('+')}</td>
  <td>{badge}</td></tr>""")

    untested = b[~b["testable"]] if len(b) else b
    untested_note = ""
    if len(untested):
        names = ", ".join(f"{e(str(r['label']))} (n={int(r['n'])})"
                          for _, r in untested.head(8).iterrows())
        untested_note = (f'<p class="note"><strong>Not tested:</strong> {len(untested)} slices held '
                         f'fewer than {int(tested["n"].min()) if len(tested) else 30} trades and were '
                         f'excluded rather than reported with a number that would not mean anything — '
                         f'{names}{" …" if len(untested) > 8 else ""}.</p>')

    shortfall = ad.get("shortfall")
    mtrl = ad["min_track_record"]
    conc = ad.get("concentration") or {}
    if ad.get("mtrl_uninformative") and np.isfinite(conc.get("effective_n", np.nan)):
        # Substitute the diagnosis for the number. "Come back in 88,452
        # trades" is arithmetically true and behaviourally useless; the
        # reason it came out that large is the thing worth reporting.
        adequacy_head = f"{conc['effective_n']:,.0f} effective trades"
        eff_term = _t(f"{conc['effective_n']:,.0f} effective trades", "effective")
        adequacy_line = (
            f"You have {conc['n']:,} trades, but they do not carry equal information. "
            f"Weighted by size, this record behaves like a sample of about "
            f"<strong>{eff_term}</strong>. "
            f"Your largest {conc['top1pct_count']:,} trades "
            f"({100*conc['top1pct_variance_share']:.0f}% of all variance) are doing the work: "
            f"without them the same record totals {_fmt(conc['total_ex_top1pct'], nd=0)}R "
            f"instead of {_fmt(conc['total'], nd=0)}R. "
            f"A minimum-track-record figure is quoted for records like this "
            f"({mtrl:,.0f} trades) but it is not the useful statement — trade count is not "
            f"your constraint, dispersion in position size is.")
    elif not np.isfinite(mtrl):
        adequacy_head = "No sample size would help"
        adequacy_line = ("Minimum track record length is undefined here because the measured edge "
                         "is not above zero to begin with. More trades cannot establish a claim "
                         "there is no evidence for; they can only measure the loss more precisely.")
    else:
        adequacy_head = f"{mtrl:,.0f} trades required"
        adequacy_line = (f"You are about <strong>{shortfall:,} trades short</strong> of the sample "
                         f"this record would need to support its own claim."
                         if shortfall else
                         "Your sample is long enough to support the claim being made.")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(title)}</title><style>{CSS}</style></head><body>
<div class="sheet">

<header class="masthead">
  <div class="eyebrow">Statistical trade audit</div>
  <h1>{e(subject)}</h1>
  <p class="lede">Every figure below is reported with the uncertainty attached to it.
  Slices that cannot be distinguished from chance are drawn hollow and left unclaimed.</p>
  <div class="meta">
    <span>{res.n_trades:,} closed trades</span><span>{e(span)}</span>
    <span>Source: {e(res.format_name)}</span><span>{_t("1R basis", "r")}: {e(res.r_basis)}</span>
  </div>
  {_coverage_note(res)}
</header>

<div class="verdict {vclass}">
  <h3>{e(res.verdict)}</h3>
  <p>{e(res.verdict_detail)}</p>
</div>

<div class="grid">
  <div class="cell"><div class="k">{_t("Average trade", "r")}</div><div class="v">{_fmt(gs['mean_r'])}R</div>
    <div class="n">{_t("95% CI", "ci")} {_fmt(gs['ci_lo'])} to {_fmt(gs['ci_hi'])}</div></div>
  <div class="cell"><div class="k">{_t("Win rate", "winrate")}</div><div class="v">{_fmt(gs['win_rate'], pct=True)}</div>
    <div class="n">{_t("Profit factor", "pf")} {_fmt(gs['profit_factor'], nd=2).lstrip('+')}</div></div>
  <div class="cell"><div class="k">{_t("Prob. Sharpe", "psr")}</div><div class="v">{_fmt(gs['psr'], pct=True)}</div>
    <div class="n">Skew/kurtosis adjusted</div></div>
  <div class="cell"><div class="k">{_t("Deflated Sharpe", "dsr")}</div><div class="v">{_fmt(gs['dsr'], pct=True)}</div>
    <div class="n">After {res.family_size} {_t("slices", "slice")} searched</div></div>
  <div class="cell"><div class="k">Net P&amp;L</div><div class="v">{_fmt(gs['total_net_pnl'], dollar=True)}</div>
    <div class="n">As supplied, net of fees</div></div>
</div>

{_money_section(res)}
{_class_section(res)}
{_curve_section(res)}
{_trend_section(res)}

<h2>Sample adequacy</h2>
<div class="callout">
  <p class="big">{adequacy_head}</p>
  <p>{adequacy_line}</p>
  <p class="note">{_t("Minimum track record length", "mtrl")} is the sample at which a Sharpe of this size,
  with this skew and this fat a tail, becomes distinguishable from zero at 95% confidence.
  At your current sample the smallest average trade you could reliably detect is
  {_fmt(ad['detectable_effect_r'])}R — anything smaller than that is invisible to you no
  matter how carefully you journal it.</p>
</div>

<h2>What an edge finder would have told you</h2>
<div class="callout">
  <p class="big">{res.n_naive_significant} of {res.family_size} {_t("slices", "slice")} look significant.
   {res.n_survived} survive{'' if res.n_survived == 1 else ''}.</p>
  <p class="note">A journal&rsquo;s &ldquo;edge finder&rdquo; stops at the first number: it slices your
  history, runs no correction, and reports every slice past p&lt;0.05 as a setup you own.
  Testing {res.family_size} slices that way produces roughly
  {max(1, round(res.family_size*0.05))} false positives by construction, before any real effect
  exists — it would have handed you <strong>{res.n_naive_significant}</strong> &ldquo;edges&rdquo; from this
  record. Every row below is shown, including the ones that failed; the count that survives the
  {_t("multiple-comparison correction", "q")} — and clears both the vs-zero and
  vs-your-own-average tests — is the honest one.</p>
</div>

<table>
<thead><tr><th>{_t("Slice", "slice")}</th><th class="num">Trades</th><th class="num">{_t("Raw R", "r")}</th>
<th class="num">{_t("Adjusted R", "adjusted")}</th><th>{_t("95% interval", "ci")} (zero-anchored)</th>
<th class="num">{_t("q-value", "q")}</th><th>Status</th></tr></thead>
<tbody>{''.join(rows) if rows else '<tr><td colspan="7">No slice reached the minimum sample size.</td></tr>'}</tbody>
</table>
{untested_note}
<p class="note"><strong>Adjusted R</strong> is the slice average pulled toward your overall average by
an amount set by how much of the spread between slices is explainable as noise. When the spread
between your slices is entirely noise, every adjusted figure collapses onto your overall
average — which is the correct answer to "which setup is my best", and the one most tools
will not give you.</p>

{_forward_section(res)}
{_recheck_section(res)}

<h2>Method</h2>
<ul class="method">
  <li>Intervals from a stationary {_t("block bootstrap", "bootstrap")} ({4000:,} resamples), so streaks and
  tilt-clusters are not mistaken for independent observations.</li>
  <li>Each slice tested twice: against zero, and against your own overall average by label
  permutation. A slice that only beats zero is inheriting your global edge, not adding one.</li>
  <li>Multiplicity controlled with Benjamini-Hochberg across all {res.family_size} tested
  slices at q=0.10. Slices below the minimum sample were excluded from the family, not
  quietly counted.</li>
  <li>Slice averages shrunk toward the global mean by empirical Bayes.</li>
  <li>Sharpe figures reported as Probabilistic and Deflated Sharpe (Bailey &amp; López de Prado),
  adjusting for skew, kurtosis, sample length and the number of slices searched.</li>
  <li>Serial dependence check: lag-1 autocorrelation
  {_fmt(res.independence.get('lag1_autocorr'), nd=3)},
  runs-test z {_fmt(res.independence.get('runs_z'), nd=2)}
  {'— trades are not independent; intervals widened accordingly.'
   if res.independence.get('dependent') else '— no material dependence detected.'}</li>
  <li>Session bands assume exchange local time (US futures). Timezone-free exports are read
  as supplied.</li>
</ul>

<footer>Generated {datetime.now():%d %b %Y %H:%M} · Past results describe the sample supplied and
do not establish future performance · Not investment advice</footer>
</div></body></html>"""


def to_markdown(res) -> str:
    gs, ad = res.global_stats, res.adequacy
    L = [f"# Trade Edge Audit", "",
         f"**{res.verdict}**", "", res.verdict_detail, "",
         f"- Trades: {res.n_trades:,} | Source: {res.format_name} | 1R basis: {res.r_basis}",
         f"- Average trade: {_fmt(gs['mean_r'])}R (95% CI {_fmt(gs['ci_lo'])} to {_fmt(gs['ci_hi'])})",
         f"- Win rate {_fmt(gs['win_rate'], pct=True)} | Profit factor "
         f"{_fmt(gs['profit_factor'], nd=2).lstrip('+')}",
         f"- Probabilistic Sharpe {_fmt(gs['psr'], pct=True)} | Deflated Sharpe "
         f"{_fmt(gs['dsr'], pct=True)} after {res.family_size} slices",
         f"- Minimum track record length: {_fmt(ad['min_track_record'], nd=0)} trades"
         + (f" (short by {ad['shortfall']:,})" if ad.get("shortfall") else ""),
         "", f"## Slices: {res.n_naive_significant} look significant, {res.n_survived} survive", "",
         "| Slice | n | Raw R | Adjusted R | 95% CI | q | Status |",
         "|---|---:|---:|---:|---|---:|---|"]
    b = res.buckets
    for _, r in (b[b["testable"]].iterrows() if len(b) else iter(())):
        status = "VERIFIED" if r["verified"] else "not distinguishable"
        L.append(f"| {r['dimension']}: {r['label']} | {int(r['n'])} | {_fmt(r['mean_r'])} | "
                 f"{_fmt(r['shrunk_mean_r'])} | {_fmt(r['ci_lo'])} to {_fmt(r['ci_hi'])} | "
                 f"{_fmt(r['q_vs_self'], nd=3).lstrip('+')} | {status} |")
    return "\n".join(L)

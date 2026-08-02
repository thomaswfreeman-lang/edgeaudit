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
        badge = ('<span class="badge ok">Verified</span>' if ver and r["mean_r"] > 0 else
                 '<span class="badge bad">Verified loss</span>' if ver else
                 '<span class="badge no">Not distinguishable</span>')
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
    if not np.isfinite(mtrl):
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
    <span>Source: {e(res.format_name)}</span><span>1R basis: {e(res.r_basis)}</span>
  </div>
</header>

<div class="verdict {vclass}">
  <h3>{e(res.verdict)}</h3>
  <p>{e(res.verdict_detail)}</p>
</div>

<div class="grid">
  <div class="cell"><div class="k">Average trade</div><div class="v">{_fmt(gs['mean_r'])}R</div>
    <div class="n">95% CI {_fmt(gs['ci_lo'])} to {_fmt(gs['ci_hi'])}</div></div>
  <div class="cell"><div class="k">Win rate</div><div class="v">{_fmt(gs['win_rate'], pct=True)}</div>
    <div class="n">Profit factor {_fmt(gs['profit_factor'], nd=2).lstrip('+')}</div></div>
  <div class="cell"><div class="k">Prob. Sharpe</div><div class="v">{_fmt(gs['psr'], pct=True)}</div>
    <div class="n">Skew/kurtosis adjusted</div></div>
  <div class="cell"><div class="k">Deflated Sharpe</div><div class="v">{_fmt(gs['dsr'], pct=True)}</div>
    <div class="n">After {res.family_size} slices searched</div></div>
  <div class="cell"><div class="k">Net P&amp;L</div><div class="v">{_fmt(gs['total_net_pnl'], dollar=True)}</div>
    <div class="n">As supplied, net of fees</div></div>
</div>

<h2>Sample adequacy</h2>
<div class="callout">
  <p class="big">{adequacy_head}</p>
  <p>{adequacy_line}</p>
  <p class="note">Minimum track record length is the sample at which a Sharpe of this size,
  with this skew and this fat a tail, becomes distinguishable from zero at 95% confidence.
  At your current sample the smallest average trade you could reliably detect is
  {_fmt(ad['detectable_effect_r'])}R — anything smaller than that is invisible to you no
  matter how carefully you journal it.</p>
</div>

<h2>Where the money supposedly comes from</h2>
<div class="callout">
  <p class="big">{res.n_naive_significant} of {res.family_size} slices look significant.
   {res.n_survived} survive.</p>
  <p class="note">Testing {res.family_size} slices at p&lt;0.05 with no correction produces roughly
  {max(1, round(res.family_size*0.05))} false positives by construction, before any real effect
  exists. The count that survives Benjamini-Hochberg correction — and clears both the
  vs-zero and vs-your-own-average tests — is the honest one.</p>
</div>

<table>
<thead><tr><th>Slice</th><th class="num">Trades</th><th class="num">Raw R</th>
<th class="num">Adjusted R</th><th>95% interval (zero-anchored)</th>
<th class="num">q-value</th><th>Status</th></tr></thead>
<tbody>{''.join(rows) if rows else '<tr><td colspan="7">No slice reached the minimum sample size.</td></tr>'}</tbody>
</table>
{untested_note}
<p class="note"><strong>Adjusted R</strong> is the slice average pulled toward your overall average by
an amount set by how much of the spread between slices is explainable as noise. When the spread
between your slices is entirely noise, every adjusted figure collapses onto your overall
average — which is the correct answer to "which setup is my best", and the one most tools
will not give you.</p>

<h2>Method</h2>
<ul class="method">
  <li>Intervals from a stationary block bootstrap ({4000:,} resamples), so streaks and
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

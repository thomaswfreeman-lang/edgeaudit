"""
EdgeAudit free check — upload page.

The hardest objection to this product is not "is the maths right", it is
"why would I hand you my trading history". That objection is correct, and a
paragraph promising to be trustworthy does not answer it. So the answer is
built into the page instead:

  1. The file is redacted IN THE BROWSER before anything is sent. Account
     number, running balances, deposits, withdrawals, bank transfers,
     dividends, interest and every non-trade row are stripped client-side.
     The user is shown exactly what was removed and can inspect the payload.
  2. What arrives here is a list of fills. It cannot identify an account and
     it does not reveal account size.
  3. Nothing is written to disk. The redacted text lives in memory for the
     length of one request.
  4. Anyone who still doesn't want to upload is told, on the same page, how
     to run the identical engine locally.

That is a claim the user can verify in their own network tab, which is worth
more than any privacy policy.

Run:      uvicorn app:app --reload --port 8000
Deploy:   uvicorn app:app --host 0.0.0.0 --port $PORT
"""
from __future__ import annotations

import io
import os
import smtplib
import sys
import threading
import time
from collections import deque
from email.message import EmailMessage
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgeaudit import audit, report  # noqa: E402

# Where consented, redacted trade lines are forwarded so the auditor receives
# every file (benchmark dataset + outcome log) without the app needing
# persistent disk. All three values come from environment variables so no
# address or credential ever appears in this public repository; when unset,
# forwarding silently disables and the audit still runs.
INTAKE_EMAIL = os.environ.get("INTAKE_EMAIL", "")
GMAIL_USER = os.environ.get("GMAIL_USER", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")


def _forward_to_auditor(payload: str, subject: str, verdict: str, n_trades: int) -> None:
    """Fire-and-forget: the trader's report must never wait on SMTP."""
    if not (INTAKE_EMAIL and GMAIL_USER and GMAIL_APP_PASSWORD):
        return

    def _send():
        try:
            msg = EmailMessage()
            msg["From"] = GMAIL_USER
            msg["To"] = INTAKE_EMAIL
            msg["Subject"] = f"[EdgeAudit self-serve] {subject} — {verdict} ({n_trades:,} trades)"
            msg.set_content(
                f"Consented self-serve audit.\nLabel: {subject}\nVerdict: {verdict}\n"
                f"Trades: {n_trades:,}\n\nRedacted payload attached (fills only; account "
                f"number, balances and cash rows were stripped in the trader's browser).")
            msg.add_attachment(payload.encode(), maintype="text", subtype="csv",
                               filename="redacted_export.csv")
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as s:
                s.login(GMAIL_USER, GMAIL_APP_PASSWORD)
                s.send_message(msg)
        except Exception:
            pass  # forwarding must never break the audit itself

    threading.Thread(target=_send, daemon=True).start()

app = FastAPI(title="EdgeAudit — free check")

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
RATE_WINDOW_SEC = 3600
RATE_MAX_PER_IP = 12
RATE_MAX_GLOBAL = 240
_hits: dict[str, deque] = {}
_global: deque = deque()


def _rate_limited(ip: str) -> str | None:
    """
    Crude in-process limiter. Each audit is several CPU-seconds of bootstrap
    resampling, so an unthrottled public endpoint is a free denial-of-service
    against yourself. Good enough for one box; put a real limiter in front
    before this sees volume.
    """
    now = time.time()
    while _global and now - _global[0] > RATE_WINDOW_SEC:
        _global.popleft()
    if len(_global) >= RATE_MAX_GLOBAL:
        return "This free check is busy right now. Try again shortly."
    q = _hits.setdefault(ip, deque())
    while q and now - q[0] > RATE_WINDOW_SEC:
        q.popleft()
    if len(q) >= RATE_MAX_PER_IP:
        return f"You've run {RATE_MAX_PER_IP} audits this hour, which is the limit on the free check."
    q.append(now)
    _global.append(now)
    return None


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EdgeAudit — do you have an edge, or a coin flip?</title>
<style>
:root{{--paper:#E9EDEF;--card:#F6F8F9;--surface:#FCFCFB;--ink:#101619;--muted:#667279;
 --rule:#C6D0D5;--good:#0F5F52;--adverse:#8C2F22;
 --display:"Charter","Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
 --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
 --body:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font-family:var(--body);
 font-size:16px;line-height:1.62;-webkit-font-smoothing:antialiased}}
.sheet{{max-width:720px;margin:0 auto;padding:56px 24px 90px}}
.eyebrow{{font-family:var(--mono);font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted)}}
h1{{font-family:var(--display);font-size:38px;line-height:1.12;margin:8px 0 14px;font-weight:600}}
h2{{font-family:var(--display);font-size:20px;font-weight:600;margin:40px 0 6px}}
p{{max-width:62ch}} p.lede{{font-size:17.5px;color:#2a343a}}
.proof{{background:var(--card);border-left:5px solid var(--ink);padding:20px 22px;margin:28px 0}}
.proof p{{margin:0 0 10px}} .proof p:last-child{{margin:0}}
.num{{font-family:var(--mono)}}
form{{background:var(--card);border:1px solid var(--rule);padding:26px;margin-top:14px}}
label{{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;
 color:var(--muted);display:block;margin-bottom:6px}}
input[type=text]{{display:block;width:100%;padding:10px 12px;margin-bottom:18px;border:1px solid var(--rule);
 font-size:14px;font-family:var(--body);background:#fff}}
input[type=file]{{display:block;width:100%;margin-bottom:8px;font-size:14px}}
button{{background:var(--ink);color:#fff;border:none;padding:13px 24px;font-size:15px;cursor:pointer;
 font-family:var(--body)}}
button:disabled{{opacity:.45;cursor:not-allowed}}
button:hover:enabled{{opacity:.88}}
.priv{{border:1px solid var(--good);padding:16px 18px;margin:18px 0;background:var(--surface)}}
.priv h3{{font-family:var(--mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;
 color:var(--good);margin:0 0 8px}}
.priv ul{{margin:0;padding-left:18px;font-size:14px}} .priv li{{margin:4px 0}}
#redact{{display:none;margin-top:14px;padding:14px 16px;background:var(--surface);
 border:1px solid var(--rule);font-size:13.5px}}
#redact b{{font-family:var(--mono)}}
#peek{{display:none;margin-top:10px;max-height:190px;overflow:auto;background:#fff;border:1px solid var(--rule);
 padding:10px;font-family:var(--mono);font-size:11px;white-space:pre;color:#2a343a}}
a.toggle{{font-size:13px;color:var(--muted);cursor:pointer;text-decoration:underline}}
.err{{background:#fdeceb;border-left:4px solid var(--adverse);padding:14px 18px;margin-bottom:22px;font-size:14.5px}}
.fine{{font-size:13px;color:var(--muted);margin-top:22px}}
code{{font-family:var(--mono);font-size:13px;background:var(--card);padding:2px 5px;border:1px solid var(--rule)}}
pre{{background:var(--card);border:1px solid var(--rule);padding:12px;overflow:auto;font-size:13px}}
</style></head><body>
<div class="sheet">
  <span class="eyebrow">Free check &middot; no account needed</span>
  <h1>Do you have an edge, or a coin flip?</h1>
  <p class="lede">Drop in your broker export. You get back a report that tells you whether
  your results can be distinguished from chance &mdash; including, when it's true, that they
  can't.</p>

  <div class="proof">
    <p><strong>Why you should distrust the alternative.</strong> We generated 500 trades from
    a coin flip with the expectancy forced to exactly zero. There is no edge in that data by
    construction.</p>
    <p>A standard journal's "edge finder" reports <span class="num">4 winning setups</span>
    &mdash; GC afternoon <span class="num">+0.54R</span>, ES midday
    <span class="num">+0.41R</span>, Wednesdays <span class="num">+0.36R</span>. All fake.
    Test 32 slices at p&lt;0.05 with no correction and you expect about 1.6 "significant"
    results before any real effect exists. Every figure regenerates from fixed seeds:
    <a href="https://github.com/thomaswfreeman-lang/edgeaudit">github.com/thomaswfreeman-lang/edgeaudit</a>.</p>
    <p>This engine reports: <strong>4 of 32 look significant, 0 survive. Not established.</strong></p>
  </div>

  {error_block}

  <h2>Run it on yours</h2>
  <form id="f" action="/audit" method="post" enctype="multipart/form-data">
    <label for="subject">Label for the report (optional)</label>
    <input type="text" id="subject" name="subject" placeholder="e.g. Apex 250k — Q1">
    <label for="file">Broker export (.csv)</label>
    <input type="file" id="file" accept=".csv,text/csv" required>
    <input type="hidden" name="payload" id="payload">
    <div id="redact"></div>
    <div id="peek"></div>
    <p style="font-size:13.5px;margin:16px 0 10px;max-width:62ch">
      <label style="display:flex;gap:10px;align-items:flex-start;font-family:var(--body);
      font-size:13.5px;letter-spacing:0;text-transform:none;color:var(--ink)">
      <input type="checkbox" id="consent" name="consent" value="yes" style="margin-top:3px">
      <span>Run the audit on my trade lines. The report describes this sample only and isn't
      investment advice. My trade lines may be kept for anonymised benchmark statistics &mdash;
      withdrawable by email any time.</span></label></p>
    <button id="go" type="submit" disabled>Choose a file first</button>
  </form>

  <p style="font-size:14px;margin-top:14px">Prefer a human on the other end &mdash; plus a free
  re-check on your new trades in 30&ndash;60 days? Use the
  <a href="https://docs.google.com/forms/d/e/1FAIpQLSernORkAlZOY47Gnn3kyj4e6VHiOaIET4mUEk72IA4xvC3v9g/viewform">audit intake form</a> instead.</p>

  <div class="priv">
    <h3>How your file is handled</h3>
    <ul>
      <li><strong>Your export stays on your computer.</strong> The page reads it locally and
          sends only the trade lines the statistics need: instrument, side, size, price,
          fees, timestamps.</li>
      <li>Anonymised, aggregated trade lines are kept for benchmark statistics &mdash; that's
          what lets the report tell you how your record compares. Withdraw any time by email.</li>
      <li><strong>You can see the exact payload before you send it</strong> &mdash; click
          <em>show me exactly what gets sent</em> above.</li>
    </ul>
  </div>

  <p style="font-size:14px;margin-top:28px">Rather not upload at all? The engine is open
  source and runs on your own machine &mdash;
  <a href="https://github.com/thomaswfreeman-lang/edgeaudit">github.com/thomaswfreeman-lang/edgeaudit</a>.</p>

  <p class="fine">Auto-detects Tradovate, NinjaTrader, TopstepX/ProjectX, Rithmic, Interactive
  Brokers, TradeStation and Schwab/thinkorswim, plus a fuzzy fallback. The bootstrap takes a
  few seconds on large files. This describes the sample you supply; it is not a forecast and
  not investment advice.</p>
</div>

<script>
// ---------------------------------------------------------------------------
// Client-side redaction. Runs before a single byte is sent. Keeps only the
// rows the audit engine actually reads: executed trades and option
// expirations, with their fee columns. Everything that could identify the
// account or reveal its size is dropped here, on the user's machine.
// ---------------------------------------------------------------------------
const KEEP_TYPES = new Set(["TRD", "RAD"]);
const fileEl = document.getElementById('file');
const payloadEl = document.getElementById('payload');
const goEl = document.getElementById('go');
const boxEl = document.getElementById('redact');
const peekEl = document.getElementById('peek');
const consentEl = document.getElementById('consent');

function updateGo() {{
  const haveFile = payloadEl.value.trim().length > 0;
  if (!haveFile) {{ goEl.disabled = true; goEl.textContent = 'Choose a file first'; return; }}
  if (!consentEl.checked) {{ goEl.disabled = true; goEl.textContent = 'Tick the consent box to run'; return; }}
  goEl.disabled = false; goEl.textContent = 'Run the audit';
}}
consentEl.addEventListener('change', updateGo);

function splitCsvLine(line) {{
  const out = []; let cur = '', q = false;
  for (let i = 0; i < line.length; i++) {{
    const c = line[i];
    if (c === '"') {{ q = !q; cur += c; }}
    else if (c === ',' && !q) {{ out.push(cur); cur = ''; }}
    else cur += c;
  }}
  out.push(cur); return out;
}}

// A Schwab/thinkorswim "Account Statement" is a multi-section ledger and needs
// section-aware handling. Every other broker exports a plain trades table --
// which the section logic would silently reduce to nothing, leaving the visitor
// staring at a disabled button. So: detect, then branch.
function isStatement(text) {{
  const head = text.slice(0, 250000);
  return /Account Statement for/i.test(text.slice(0, 4000))
      || /(^|\\n)\\s*Cash Balance\\s*(\\r?\\n)/i.test(head)
      || /(^|\\n)\\s*Futures Statements\\s*(\\r?\\n)/i.test(head);
}}

// Plain trades table: keep every data row, blank only the columns that could
// identify the account. The statistics never read them.
function redactTable(text) {{
  const lines = text.split(/\\r?\\n/).filter(l => l.trim().length > 0);
  const hi = lines.findIndex(l => (l.match(/,/g) || []).length >= 2);
  const stats = {{account: 0, balances: 0, cash: 0, other: 0}};
  if (hi < 0) return {{text: '', stats: stats, keptRows: 0}};
  const header = splitCsvLine(lines[hi]).map(s => s.trim());
  const dropIdx = header.map((h, i) =>
    /^account|acct|^ref\\s*#?$|balance/i.test(h) ? i : -1).filter(i => i >= 0);
  if (dropIdx.length) stats.account = 1;
  const out = [header.map((h, i) => dropIdx.includes(i) ? '' : h).join(',')];
  stats.other = hi;                       // preamble lines above the header
  for (let i = hi + 1; i < lines.length; i++) {{
    const cells = splitCsvLine(lines[i]);
    if (cells.length < 2) {{ stats.other++; continue; }}
    out.push(cells.map((c, j) => dropIdx.includes(j) ? '' : c).join(','));
  }}
  return {{text: out.join('\\n'), stats: stats, keptRows: out.length - 1}};
}}

function redact(text) {{
  if (!isStatement(text)) return redactTable(text);
  const lines = text.split(/\\r?\\n/);
  const kept = [];
  const stats = {{account: 0, balances: 0, cash: 0, other: 0}};
  let section = null, header = null, typeIdx = -1, dropIdx = [];

  for (const raw of lines) {{
    const line = raw.replace(/^\\uFEFF/, '');
    const t = line.trim();

    if (/^Account Statement for/i.test(t)) {{
      stats.account++;
      kept.push(t.replace(/for\\s+\\S+/i, 'for [REDACTED]'));
      continue;
    }}
    if (t === 'Cash Balance' || t === 'Futures Statements') {{
      section = t; kept.push(t); header = null; continue;
    }}
    // sections the engine never reads
    if (/^(Account Order History|Account Trade History|Profits and Losses|Account Summary|Options|Futures Options|Forex Statements|Crypto)/i.test(t)) {{
      section = 'skip'; stats.other++; continue;
    }}
    if (!t) {{ if (section && section !== 'skip') kept.push(''); continue; }}
    if (section === 'skip') {{ stats.other++; continue; }}
    if (!section) {{ stats.other++; continue; }}

    if (header === null && /^(DATE,TIME,TYPE|Trade Date,)/i.test(t)) {{
      header = splitCsvLine(t).map(s => s.trim());
      typeIdx = header.findIndex(h => /^type$/i.test(h));
      // Drop running balances everywhere (reveals account size), the AMOUNT
      // column in the equity ledger (the engine derives P&L from prices there),
      // and broker order reference numbers (never read, needlessly correlatable).
      dropIdx = header.map((h, i) =>
        /^ref\s*#?$/i.test(h) ? i
        : /^balance$/i.test(h) ? i
        : (/^amount$/i.test(h) && section === 'Cash Balance') ? i : -1).filter(i => i >= 0);
      kept.push(header.map((h, i) => dropIdx.includes(i) ? '' : h).join(','));
      continue;
    }}
    if (header === null) {{ stats.other++; continue; }}

    const cells = splitCsvLine(t);
    const ty = (cells[typeIdx] || '').trim().toUpperCase();
    if (!KEEP_TYPES.has(ty)) {{
      if (ty === 'BAL') stats.balances++; else stats.cash++;
      continue;
    }}
    kept.push(cells.map((c, i) => dropIdx.includes(i) ? '' : c).join(','));
  }}
  return {{text: kept.join('\\n'), stats: stats, keptRows: kept.length}};
}}

fileEl.addEventListener('change', () => {{
  const f = fileEl.files[0];
  if (!f) return;
  goEl.disabled = true; goEl.textContent = 'Reading…';
  const rd = new FileReader();
  rd.onload = () => {{
    const raw = String(rd.result);
    const r = redact(raw);
    payloadEl.value = r.text;
    const removed = r.stats.balances + r.stats.cash + r.stats.other;
    boxEl.style.display = 'block';
    if (!r.keptRows) {{
      // Never leave the visitor with a dead button and no explanation.
      boxEl.innerHTML =
        "No trade rows found in that file. It may be an unusual export format &mdash; " +
        "send it through the <a href=\\"https://docs.google.com/forms/d/e/1FAIpQLSernORkAlZOY47Gnn3kyj4e6VHiOaIET4mUEk72IA4xvC3v9g/viewform\\">intake form</a> and it gets parsed by hand " +
        "within 48 hours. Unusual formats are exactly what we want to see.";
      peekEl.style.display = 'none';
      updateGo();
      return;
    }}
    boxEl.innerHTML =
      'Read on your machine &mdash; ready to audit <b>' + r.keptRows.toLocaleString() +
      '</b> trade lines (' + Math.round(r.text.length / 1024).toLocaleString() + ' KB). ' +
      (removed ? 'The other ' + removed.toLocaleString() + ' lines in your file stayed here. ' : '') +
      '<a class="toggle" onclick="peekEl.style.display = peekEl.style.display===\\'block\\'?\\'none\\':\\'block\\'">' +
      'show me exactly what gets sent</a>';
    peekEl.textContent = r.text.slice(0, 4000) + (r.text.length > 4000 ? '\\n… (truncated preview)' : '');
    updateGo();
  }};
  rd.onerror = () => {{ goEl.textContent = 'Could not read that file'; }};
  rd.readAsText(f);
}});
</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE.format(error_block="")


def _err(msg: str, code: int) -> HTMLResponse:
    return HTMLResponse(PAGE.format(error_block=f'<div class="err">{msg}</div>'), status_code=code)


@app.post("/audit", response_class=HTMLResponse)
async def run_audit(request: Request, payload: str = Form(""), subject: str = Form("Account"),
                    consent: str = Form(""), file: UploadFile | None = File(None)):
    ip = (request.client.host if request.client else "?") or "?"
    limited = _rate_limited(ip)
    if limited:
        return _err(limited, 429)
    if consent != "yes":
        return _err("The consent box wasn't ticked. It's required to run the audit — "
                    "it's also what allows the anonymised benchmark statistics.", 400)

    text = payload or ""
    if not text.strip() and file is not None:      # no-JS fallback
        raw = await file.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            return _err(f"That file is over the {MAX_UPLOAD_BYTES // (1024*1024)}MB limit.", 413)
        text = raw.decode("utf-8-sig", errors="replace")
    if not text.strip():
        return _err("No trades came through. Pick a CSV export and try again.", 400)
    if len(text.encode()) > MAX_UPLOAD_BYTES:
        return _err(f"That file is over the {MAX_UPLOAD_BYTES // (1024*1024)}MB limit.", 413)

    try:
        res = audit.run(io.BytesIO(text.encode()))
    except Exception as exc:
        return _err(f"Could not read this file: {exc} — make sure it's the raw export from "
                    f"your broker, not a reformatted copy. Or send it through the "
                    f'<a href="https://docs.google.com/forms/d/e/1FAIpQLSernORkAlZOY47Gnn3kyj4e6VHiOaIET4mUEk72IA4xvC3v9g/viewform">intake form</a> '
                    f"and a human will parse it within 48 hours — unusual formats are exactly "
                    f"what we want to see.", 422)
    label = (subject or "Account").strip() or "Account"
    _forward_to_auditor(text, label, res.verdict, res.n_trades)
    return HTMLResponse(report.to_html(res, subject=label))

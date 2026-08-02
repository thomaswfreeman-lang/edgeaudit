"""
Minimal free-check upload page for EdgeAudit.

Deliberately small: one route serves the upload form, one route runs the
audit and returns the report. No accounts, no storage — the uploaded file
is held in memory for the duration of the request and never written to
disk. That's a real claim worth keeping true, not just a privacy footnote:
don't add persistence here without updating it on the page.

Run locally:
    uvicorn app:app --reload --port 8000

Deploy: this is a stock ASGI app (FastAPI + uvicorn), so it runs unchanged
on Render, Railway, Fly.io, a plain VM, or anywhere else that can run
`uvicorn app:app --host 0.0.0.0 --port $PORT`. No platform-specific code.
Needs numpy/pandas/scipy at runtime (same as the CLI), so it must run as a
server process — this is not deployable as a static site.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgeaudit import audit, report  # noqa: E402

app = FastAPI(title="EdgeAudit — free check")

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15MB — generous for any CSV export we've seen

UPLOAD_FORM = """<!doctype html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>EdgeAudit — is there an edge here?</title>
<style>
  :root{{ --paper:#E9EDEF; --card:#F6F8F9; --ink:#101619; --muted:#667279; --rule:#C6D0D5;
         --display:"Charter","Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
         --body:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--paper);color:var(--ink);font-family:var(--body);
        font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased}}
  .sheet{{max-width:640px;margin:0 auto;padding:64px 28px 96px}}
  .eyebrow{{font-family:ui-monospace,monospace;font-size:11px;letter-spacing:.16em;
            text-transform:uppercase;color:var(--muted)}}
  h1{{font-family:var(--display);font-size:34px;line-height:1.15;margin:6px 0 14px;font-weight:600}}
  p.lede{{font-size:17px;color:var(--muted);max-width:56ch;margin:0 0 36px}}
  form{{background:var(--card);border:1px solid var(--rule);padding:28px;border-radius:2px}}
  input[type=file]{{display:block;width:100%;margin-bottom:16px;font-size:14px}}
  input[type=text]{{display:block;width:100%;padding:10px 12px;margin-bottom:16px;
                     border:1px solid var(--rule);border-radius:2px;font-size:14px;
                     font-family:var(--body);background:#fff}}
  label{{font-family:ui-monospace,monospace;font-size:11px;letter-spacing:.08em;
         text-transform:uppercase;color:var(--muted);display:block;margin-bottom:6px}}
  button{{background:var(--ink);color:#fff;border:none;padding:12px 22px;font-size:14px;
          border-radius:2px;cursor:pointer;font-family:var(--body)}}
  button:hover{{opacity:.88}}
  .fine{{font-size:12.5px;color:var(--muted);margin-top:18px;max-width:56ch}}
  .err{{background:#fdeceb;border-left:4px solid #8C2F22;padding:14px 18px;margin-bottom:24px;
        font-size:14px;color:#5a1c12}}
</style>
</head><body>
<div class="sheet">
  <span class="eyebrow">Free check</span>
  <h1>Is there an edge here, or is this a coin flip?</h1>
  <p class="lede">Drop in your raw broker export. No template, no reformatting. Auto-detects
  Tradovate, NinjaTrader, TopstepX/ProjectX, Rithmic, IBKR, and TradeStation. Nothing is stored —
  the file is read in memory and discarded after the report is built.</p>
  {error_block}
  <form action="/audit" method="post" enctype="multipart/form-data">
    <label for="subject">Label (optional)</label>
    <input type="text" id="subject" name="subject" placeholder="e.g. Apex 250k - Q1">
    <label for="file">Broker export (.csv)</label>
    <input type="file" id="file" name="file" accept=".csv" required>
    <button type="submit">Run the audit</button>
  </form>
  <p class="fine">Runs a stationary block bootstrap and Benjamini–Hochberg correction across
  every slice tested — this can take a few seconds on larger exports. Not investment advice;
  this describes the sample supplied, not a forecast.</p>
</div>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def upload_form():
    return UPLOAD_FORM.format(error_block="")


@app.post("/audit", response_class=HTMLResponse)
async def run_audit(file: UploadFile = File(...), subject: str = "Account"):
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        return HTMLResponse(
            UPLOAD_FORM.format(
                error_block=f'<div class="err">That file is larger than the '
                f'{MAX_UPLOAD_BYTES // (1024*1024)}MB limit for the free check. '
                f"Trim it to a single account/period export and try again.</div>"
            ),
            status_code=413,
        )
    try:
        res = audit.run(io.BytesIO(contents))
    except Exception as exc:
        return HTMLResponse(
            UPLOAD_FORM.format(
                error_block=f'<div class="err">Could not read this file: {exc}. '
                f"Make sure it's the raw export from your broker/platform, not a "
                f"reformatted copy.</div>"
            ),
            status_code=422,
        )
    subject = subject.strip() or "Account"
    return HTMLResponse(report.to_html(res, subject=subject))

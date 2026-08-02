# EdgeAudit free check — upload page

One route in, one route out. No accounts, no database, nothing written to
disk — the uploaded CSV lives in memory for the length of the request and
is discarded when the response is sent.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Open http://127.0.0.1:8000, drop in a CSV, get the report back in the
browser.

## Deploy

Stock ASGI app — runs unchanged anywhere that can run
`uvicorn app:app --host 0.0.0.0 --port $PORT`:

- **Render / Railway**: point a web service at this folder, build command
  `pip install -r requirements.txt`, start command
  `uvicorn app:app --host 0.0.0.0 --port $PORT`. Free tiers on both handle
  this fine — the bootstrap is CPU-bound for a few seconds per request, not
  memory-heavy.
- **Fly.io**: same start command inside a small Python Dockerfile.
- **A plain VM**: `uvicorn app:app --host 0.0.0.0 --port 80` behind
  anything (nginx, Caddy) for TLS.

Not deployable as a static site — the audit itself needs numpy/pandas/scipy
at request time, so this has to run as a server process, however small.

## What's deliberately not here yet

- Rate limiting. A public free-check page will get hit by scripts, not just
  people uploading real exports; the bootstrap resampling makes each
  request nontrivially expensive (a few CPU-seconds). Put a request cap
  (per-IP or global) in front of this before it's public — it isn't here
  because it's a platform-level decision (Cloudflare, a reverse proxy, or
  in-app) not a code-level one.
- Any capture of the uploaded data or the resulting verdict. If the plan is
  ever to build a funnel off free-check usage (e.g. "email me the report"),
  that's a real product/privacy decision, not something to add silently —
  flag it back before building it.
- HTTPS/TLS termination — expected to be handled by whatever platform this
  deploys behind, not by the app itself.

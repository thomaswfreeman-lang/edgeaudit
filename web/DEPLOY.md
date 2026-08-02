# Getting this online

**LIVE:** https://edgeaudit-freecheck.onrender.com (Render free tier, Oregon,
auto-deploys from `main`). Verified 2026-08-02 end-to-end: page renders,
unconsented POST refused with 400, consented audit returns a full report in
~17s, 52 glossary terms present.

Free tier sleeps when idle — first request after a quiet spell takes ~50s to
wake. $7/mo keeps it warm if Reddit traffic makes that annoying.

To forward consented payloads to the intake inbox, set three env vars on the
service (Dashboard → Environment). Unset, forwarding disables silently and
the audit still runs:

    INTAKE_EMAIL        where redacted payloads are sent
    GMAIL_USER          the sending account
    GMAIL_APP_PASSWORD  16-char app password (needs 2FA on that account)

Never commit these. The repo is public.

## The two paths below predate hosting, kept for reference

## Path A — no hosting at all (do this first)

`redactor.html` is a single file with no server, no build step and no network
calls. Email it, host it on a Gist, put it in a Google Drive link, whatever.
Someone opens it, drops their statement in, gets back a stripped CSV, and
emails you *that*.

You run the audit yourself:

```bash
python -m edgeaudit.cli their_file_redacted.csv --subject "Their label" -o report.html
```

and email the HTML back. Ten of those tells you more than any launch.

Verified: a statement run through `redactor.html` and the raw original produce
identical audits — 15,474 trades, same verdict, same 65/138/16 slice counts —
while the stripped file no longer contains the account number, balances,
transfers, dividends, interest or order IDs.

## Path B — the hosted version, when you actually want one

Any host that runs a Python web process. Render's free tier is the least
friction:

1. Put this repo on GitHub (private is fine).
2. Render → New → Web Service → connect the repo.
3. Root directory: `web`
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
6. Deploy.

Railway and Fly.io take the same two commands. A free tier will cold-start
after inactivity, so the first visitor waits ~30s — fine for ten people,
not fine for a front page.

`render.yaml` in this folder does steps 3–5 automatically if you'd rather
point Render at the file.

### Before it's genuinely public
- Put Cloudflare in front. The in-process rate limiter resets on every deploy
  and doesn't survive multiple workers.
- One audit is several CPU-seconds. A free instance handles a trickle, not a
  Reddit front page.

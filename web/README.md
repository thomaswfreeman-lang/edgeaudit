# EdgeAudit free check — upload page

One page in, one report out. The interesting part is not the upload; it's the
redaction that happens before it.

## Run it

```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Open http://127.0.0.1:8000.

## Deploy

Stock ASGI app — runs unchanged anywhere that can run
`uvicorn app:app --host 0.0.0.0 --port $PORT`: Render, Railway, Fly.io, or a
plain VM behind nginx/Caddy for TLS. Build command `pip install -r
requirements.txt`. Not deployable as a static site — the audit needs
numpy/pandas/scipy at request time.

## The privacy design, and why it's the product

The hardest objection to a trade-audit tool is not "is the maths right", it's
"why would I hand you my trading history". That objection is correct. A
privacy policy does not answer it; an architecture does.

**The file is redacted in the browser before anything is sent.** The JS in
`app.py` parses the CSV client-side and keeps only the rows the engine
actually reads — executed trades (`TRD`) and option expirations (`RAD`) —
with their fee columns. It drops:

- the account number (replaced with `[REDACTED]` in the title line)
- every running `BALANCE` column — this is what reveals account size
- the `AMOUNT` column in the equity ledger (the engine derives equity P&L
  from prices, so it never needs the cash flow)
- the broker `REF #` order identifiers — never read, needlessly correlatable
- every non-trade row: deposits, withdrawals, bank transfers, dividends,
  interest, cash sweeps, daily balance rows
- every section the engine doesn't read: Order History, Positions, Profits
  and Losses, Account Summary

On a real Schwab statement that removes **831 lines** — 259 cash movements,
366 balance rows, 206 other lines — and leaves 8,026 trade lines. What
arrives at the server cannot identify an account and does not reveal its
size.

The user is shown a count of exactly what was removed and can expand the
payload to read it, then check their own network tab. That is a verifiable
claim, which is worth more than a promise.

**Verified equivalence:** a raw statement and its browser-redacted payload
produce byte-identical audits — same verdict, same 6,279 trades, same
+1.051R with the same interval, same 32/78/7 slice counts. Redaction removes
everything sensitive and changes nothing about the answer.
`test_redacted_upload_shape_still_parses` locks that contract, so a future
parser change that starts depending on `AMOUNT` or `BALANCE` fails the suite
instead of silently breaking the page for every visitor.

The page also tells anyone who still doesn't want to upload how to run the
identical engine locally. Losing that visitor to the CLI is a better outcome
than losing them entirely.

## Rate limiting

In-process, per-IP: 12 audits per hour per IP, 240 globally. Each audit is
several CPU-seconds of bootstrap resampling, so an unthrottled public
endpoint is a free denial-of-service against yourself. This is good enough
for one box — put Cloudflare or a real reverse-proxy limiter in front before
it sees volume, since an in-process counter resets on deploy and doesn't
survive multiple workers.

## Deliberately not here

- **Any capture of uploaded data or results.** Nothing is written to disk;
  the text lives in memory for one request. If a funnel ever wants "email me
  the report", that is a product and privacy decision to make explicitly —
  and it would invalidate the claim on the page, which is the main asset.
- **Accounts, payments, sessions.** The free check exists to prove the
  engine, not to convert on the first visit.
- **TLS.** Expected from the deploy platform.

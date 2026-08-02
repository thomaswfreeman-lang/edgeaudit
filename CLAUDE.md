# EdgeAudit

Statistical audit of trading records. A trader uploads a raw broker export;
the engine tests every slice of it (instrument, session, day of week, hold
time, size, direction, sequence effects) and reports which findings survive
the fact that the trader went looking. The honest answer is usually "few"
and sometimes "none" — **saying so is the product**. Every competitor
(TradeZella, Tradervue, TraderSync, Edgewonk) sells edge-finding; this sells
edge-testing.

Who it's for: retail futures/prop traders first (r/FuturesTrading,
r/algotrading), prop-firm risk desks later — the outcome log of audits plus
60–90 day follow-ups is the asset that becomes a prop-firm scoring model.

## Non-negotiable statistical constraints

These are the product. Loosening any of them makes output more impressive
and the tool worthless. Do not trade them for engagement, conversion, or
prettier reports.

1. **Noise must never produce a verified slice.**
   `tests/test_engine.py::test_noise_yields_no_verified_slices` is the
   load-bearing test of the repo. If a change makes it fail, the change is
   wrong, whatever else it improves.
2. **No point estimates.** Every reported mean carries an interval.
3. **Block bootstrap, never i.i.d.** Trades are serially dependent
   (stationary bootstrap, Politis & Romano 1994). An i.i.d. bootstrap
   narrows intervals — that is how fake edges are manufactured.
4. **Two nulls per bucket.** Against zero AND against the trader's own
   global mean (label permutation). A bucket that beats zero because the
   whole book beats zero is not an edge. `verified` requires surviving both.
5. **Multiplicity is corrected.** Benjamini–Hochberg over the whole family
   of testable slices, q = 0.10. Untestable buckets (n < 30) never enter
   the family and are never reported as findings.
6. **Shrinkage stays.** Bucket means are shrunk toward the global mean
   (empirical Bayes). When spread is pure noise, buckets collapse onto the
   global mean — that collapse is the correct answer, not a display bug.
7. **Sharpe claims are deflated.** PSR for skew/kurtosis/sample length, DSR
   for the number of slices searched (Bailey & López de Prado).
8. **Honest limitations stay in the copy.** BH controls false discovery
   *rate*, not family-wise error: across 12 noise runs, 3 of 298 slices
   survived (1.0%, inside the bound, not zero). Never claim zero false
   positives — that claim is the tell of every "Edge Finder AI" this
   product exists to stand apart from.

## Reproducibility

`scripts/reproduce_post.py` regenerates every figure quoted in
`content/reddit-noise-post.md` from fixed seeds. **If a figure in the post
does not match the script's output, the post is wrong** — fix the post,
never the seed. Any engine change that alters these numbers requires
regenerating and re-checking the post before it ships.

`edgeaudit/survival.py` (prop-eval Monte Carlo) is a faithful-API rebuild
(2026-08-02) — the original module was lost with the final build artifact.
The post quotes no figures from it; its output is deterministic and its
invariants are pinned in `tests/test_survival.py`.

## Licensing / distribution

Contact for everything client-facing: **myedgeaudit@gmail.com** (never a
personal address — it appears in a public repo, a public form and a public
post). Same address is the app's INTAKE_EMAIL env var on Render.

Dual licence: AGPL-3.0 + commercial (see LICENSE). AGPL is deliberate — a
competitor embedding the engine in a closed SaaS must open their source or
pay. Disclosure norms for posts: authorship + commercial exploration stated
up front, no signup links, first ten audits free (CONSENT.md + DISCLAIMER.md
govern intake). Every real-file parser quirk fixed in `parsers.py` must gain
a regression test with a small anonymised fixture — the accumulated parser
library is the moat.

## Open items

- TOS multi-leg option structures (VERTICAL/BACKRATIO/CUSTOM): a handful of
  rows per statement, excluded and disclosed in --diagnose reconciliation;
  proper leg attribution needs an Account Trade History cross-reference

- Day-level loss-concentration aggregation (losses cluster in days;
  per-trade slices miss it) — partially covered by variance_concentration
  and drawdown_profile; a per-day bucket dimension is still open
- Calendar-month trend option in the report (regime_trend splits by count,
  not calendar)
- Reproducibility stamp on the report face
- Dedicated intake email (form draft: content/intake-form.md; outcome log
  live at clients/outcome_log.csv, gitignored)

Done and pinned by tests (do not re-open): per-instrument-class 1R with a
3x heterogeneity gate (test_r_unit_*); one-closing-decision-one-trade merge
(test_scale_out_is_one_trade_not_a_run) — both were shipped only after the
post's reproduction was verified byte-identical.

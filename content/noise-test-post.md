*Draft — content piece for the free-check landing page / launch post*

# We fed a trading journal 500 coin flips. It found an edge anyway.

Every trade journal on the market does the same thing: slice your trade history forty different ways — by instrument, by session, by day of week, by hold time — and show you which slices look good. Green cells, best setups, "your Tuesday CL trades are printing." It feels like insight. It's usually noise wearing a green highlight.

To prove it, we generated 500 trades from a fair coin flip, with the expectancy forced to exactly zero. Not "a bad trader." Zero. No edge exists in this data by construction — there is nothing to find.

We ran it through the kind of analysis every retail journal runs: test each slice against a flat baseline, flag anything with p < 0.05, done. Standard stuff, the stuff powering every "your best setup" widget you've seen.

**It found four winners.**

- CL overnight session: +0.541R average, "significant"
- GC afternoon session: +0.535R average, "significant"
- ES midday session: +0.414R average, "significant"
- Wednesdays: +0.356R average, "significant"

None of these are real. They can't be — the data has no edge in it. This is what happens when you test 32 slices at a 5% false-positive rate with no correction: math guarantees you'll find roughly one or two "significant" results before a single real effect exists. We found four, which is well within the noise you'd expect from pure chance. A journal that reports this uncorrected is not finding your edge. It's finding its own false-positive rate and calling it a setup.

**Then we ran the same 500 trades through EdgeAudit.**

Same data. Same 32 slices tested. Different question asked at every step: not "does this beat a flat baseline" but "does this survive being tested against zero, tested against your own overall average, corrected for the fact that we searched 32 places to find it, and shrunk toward the average when the spread between slices doesn't clear the bar for real signal."

Verdict: **not established.** Average trade -0.022R, 95% interval -0.150R to +0.099R — an interval that contains zero, meaning the honest answer is "we can't tell you have an edge or a leak, only that whichever it is, it's too small to see yet." Four slices still look significant on the naive pass. Zero survive correction. The report says so, on the page, next to the four that didn't make it.

That's the trap every other journal falls into, and it's not a subtle one: if you're already profitable overall, most slices will beat *zero* just by inheriting your global edge — the tool ends up crediting a "setup" for performance the setup didn't add. EdgeAudit tests every slice against zero *and* against your own average, so a slice only survives if it's doing something your whole book isn't already doing.

**The inverse case matters just as much.** We also built 600 trades with a real, uniform +0.18R edge baked into every trade — no slice special, the edge is just... everywhere. A naive scan finds nine "significant" slices and would hand you nine fake setups to chase. EdgeAudit confirms the edge globally and credits it to zero slices, because crediting one slice for an edge that lives in the whole book would be exactly as wrong as the coin-flip case in the other direction.

And when the edge actually *is* concentrated somewhere — we ran 600 trades where the entire edge lives in the 08:30–10:29 RTH-open window and nowhere else — the naive scan finds sixteen "significant" slices. EdgeAudit finds two, and both of them are the RTH-open window. Not sixteen. Two. The two that are real.

Three tests, three different true answers — zero edge, uniform edge, concentrated edge — and EdgeAudit gave the correct verdict on all three while the naive approach was wrong on every one that had a wrong answer to give.

**If your current journal has never told you "not established, come back in N trades,"** it isn't because your trading has always cleared the bar. It's because the tool isn't set up to say no. Statistically, some fraction of your slices will always clear an uncorrected significance test — that's not a discovery, that's arithmetic. The question worth asking about any trading tool that shows you a "best setup" is whether it checked if that setup would show up in random noise too. Most don't. This one does, out loud, on the same page as the result.

---

*Try it on your own export: [link to free-check page]. No template, no reformatting — drop in your Tradovate, NinjaTrader, TopstepX, Rithmic, IBKR, or TradeStation export as-is.*

# I generated 500 random trades with zero edge, then analysed them like a trading journal would. It found 4 profitable setups.

*(Title options at the bottom. Body below is the post.)*

---

I wanted to know how much of what a trading journal tells you is real, so I ran a control experiment.

I generated 500 fake trades. Win rate 45%, winners bigger than losers, spread across ES, NQ, CL and GC, across sessions, days of the week, position sizes and hold times. Then I subtracted the mean from every trade so the expectancy is exactly zero by construction. There is no edge in this data. There cannot be. It's a coin flip with a P&L column.

Then I sliced it the way a journal does: by instrument, by session, by day of week, by hold time, by position size, by long vs short, by first-trade-of-day vs later, by what happens right after a loss. 32 slices.

Four of them came back "profitable" at p < 0.05:

```
GC, afternoon session      n=32    +0.535R   p=0.004
ES, midday                 n=32    +0.414R   p=0.028
Wednesdays                 n=100   +0.356R   p=0.004
NQ (losing)                n=124   -0.266R   p=0.009
```

None of them exist. Every one is noise. If a journal had shown me that first line I'd have believed I had a gold afternoon setup, and I'd have sized up on it.

This isn't a bug in anyone's software. It's arithmetic. Test 32 slices at p < 0.05 and you expect about 1.6 false positives before any real effect exists. I ran it on 12 independent noise samples: 11 of the 12 produced at least one "significant" setup. The mean was 1.33 per sample against 1.24 expected by chance. Exactly what the maths predicts.

## The part I found more interesting

Second experiment. Same generator, but I added a genuine +0.18R edge to *every single trade*. Uniform. No setup, session or instrument is special — the edge is in the whole book.

Naive slicing reported **nine** significant setups: Mondays, longs, biggest 10% of positions, trades after a loss, ES, NQ, Wednesdays, and so on.

All nine are the same edge showing up nine times. None of them is a place where the edge is concentrated. If you respond to that report by trading only your "best setups", you cut your sample and gain nothing.

The fix is to stop testing slices against zero. If you're profitable overall, most of your slices beat zero — the test is answering "am I profitable", not "is my edge concentrated here". The right null is your own average: shuffle the slice labels and ask whether this bucket beats the rest of your book more than a random bucket of the same size would. Under that test, all nine drop out. Correctly.

## Does it ever find a real one

Third experiment: flat everywhere except a genuinely profitable 09:00–11:00 window.

Naive slicing found 16 "edges". After correction, 2 survived — the RTH open session and the ES-during-RTH-open cross. Which is the true effect, found twice from two angles.

So it isn't just a machine that says no.

## What I'd actually check on your own data

- **Confidence interval on your average trade, not the point estimate.** If it spans zero you don't have a result yet, however good the number looks.
- **Correct for the number of slices you looked at.** Benjamini–Hochberg is four lines of code. If your journal has 300 reports, you've run 300 tests.
- **Test slices against your own average**, not against zero.
- **Minimum track record length** (Bailey & López de Prado). Tells you how many trades you need before a Sharpe of your size, with your skew and your tails, is distinguishable from zero. Most retail samples are nowhere near it, and it's the fastest way to find out you're arguing about noise.
- **Don't shuffle trades when you bootstrap.** Trading is autocorrelated — tilt clusters, streaks are real. An i.i.d. bootstrap gives you intervals that are too narrow, which is how you manufacture confidence. Block bootstrap.

## Honest limitations, since someone will ask

The correction I used controls the false discovery rate at 10%, which bounds the expected *share* of your findings that are false — it does not guarantee zero. Across those 12 noise runs, 3 slices out of 298 tested survived correction: 1.0%, well inside the bound, but not zero. Two of the twelve runs produced phantom edges — one run had a single phantom, the other had two. Anyone claiming a method that never produces a false positive on noise is either lying or has set the threshold so high it can't detect anything real either.

The generator is also kinder than reality: no fees modelled beyond a flat commission, no slippage, no tilt, no revenge sizing. Real trading is worse.

## Code

The generator and the analysis are open source, seeds fixed, and there's a script that regenerates every number in this post. If the figures don't reproduce, the post is wrong and I'd like to know:

`[GITHUB LINK]`

Break it. I'd rather find out the method is flawed from someone here than from a customer.

**Disclosure:** I built this and I'm exploring whether people would pay for it as a service. I'm not selling anything in this post and there's no signup link. If you want me to run it on your own export — free, no strings, and I'll send you the report whether or not you ever talk to me again — comment or DM. I'd like about ten real datasets to find out where it breaks on live broker files.

---

## Title options

1. I generated 500 random trades with zero edge, then analysed them like a trading journal would. It found 4 profitable setups.
2. Your journal will find you an edge in pure noise. Here's the experiment.
3. I gave a trading journal 500 coin flips. It told me my Wednesday gold setup was profitable.
4. Tested 32 slices of zero-edge data. 4 came back significant. This is why "edge finder" features are dangerous.

## Where to post, in order

- **r/FuturesTrading** — closest to the buyer, less hostile to a first-time poster than r/algotrading.
- **r/algotrading** — the audience most likely to check the maths. Read the self-promo rules first; keep the link in a comment if the sidebar requires it.
- **r/Daytrading** — biggest reach, lowest signal. Post last, and expect the least useful replies.
- **EliteTrader (Technical Analysis or Journals)** — older, funded, the actual paying demographic.

Post once per sub, a few days apart. Never cross-post the same day.

## How to handle the comments

- The most valuable reply is someone who disagrees on method. Engage in detail, concede where they're right.
- If someone says "this is just p-hacking with extra steps" — they're most of the way to your point. Agree, then explain that the correction is what separates the two.
- Don't defend the product. Defend the experiment.
- Every "run it on mine" comment is a lead. Reply publicly, deliver privately.

"""
Regenerates every number quoted in the write-up. Fixed seeds, no cherry-picking.

    python scripts/reproduce_post.py

If a figure in the post does not match this output, the post is wrong.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from edgeaudit import audit, demo_data, survival  # noqa: E402

RESAMPLES = 4000


def line(k, v):
    print(f"  {k:<44} {v}")


def experiment(name, raw, note):
    res = audit.run(raw, resamples=RESAMPLES)
    print(f"\n{name}\n{'-' * 68}\n  {note}")
    line("trades", f"{res.n_trades:,}")
    line("true expectancy by construction", note.split('|')[-1].strip())
    line("measured mean trade", f"{res.global_stats['mean_r']:+.3f}R "
                                f"({res.global_stats['ci_lo']:+.3f} to "
                                f"{res.global_stats['ci_hi']:+.3f})")
    line("verdict", res.verdict)
    line("slices tested", res.family_size)
    line("significant at p<0.05, uncorrected", res.n_naive_significant)
    line("surviving BH correction (q=0.10)", res.n_survived)
    line("probabilistic Sharpe", f"{res.global_stats['psr']:.1%}")
    line("deflated Sharpe (after the search)", f"{res.global_stats['dsr']:.1%}")
    if res.n_naive_significant:
        naive = res.buckets[res.buckets["naive_significant"].fillna(False)]
        print("  slices a naive tool would have reported as edges:")
        for _, r in naive.iterrows():
            print(f"      {r['dimension']}: {r['label']:<34} "
                  f"n={int(r['n']):>4}  {r['mean_r']:+.3f}R  p={r['p_vs_zero']:.3f}")
    if res.n_survived:
        print("  slices that survived:")
        for _, r in res.buckets[res.buckets["verified"]].iterrows():
            print(f"      {r['dimension']}: {r['label']:<34} "
                  f"n={int(r['n']):>4}  {r['mean_r']:+.3f}R  q={r['q_vs_self']:.3f}")
    return res


print("=" * 68)
print("REPRODUCTION OF PUBLISHED FIGURES")
print("=" * 68)

r1 = experiment(
    "1. PURE NOISE — 500 coin-flip trades",
    demo_data.pure_noise(500, seed=9),
    "45% win rate, winners larger than losers | expectancy forced to exactly 0",
)

r2 = experiment(
    "2. UNIFORM EDGE — real edge, spread evenly over the whole book",
    demo_data.real_edge(500, seed=11),
    "no setup, session or instrument is special | +0.18R on every trade",
)

r3 = experiment(
    "3. CONCENTRATED EDGE — flat everywhere except 09:00-11:00",
    demo_data.concentrated_edge(600, seed=5),
    "one genuinely profitable window | edge exists in exactly one slice",
)

print("\n4. PROP EVALUATION ODDS FOR THE ZERO-EDGE TRADER")
print("-" * 68)
rules = survival.PRESETS["50k_trailing"]
print(f"  {rules.describe()}")
ev = survival.simulate_evaluation(r1.trades, rules, n_paths=4000)
line("pass", f"{ev.p_pass:.1%} (95% CI {ev.ci[0]:.1%} to {ev.ci[1]:.1%})")
line("breach drawdown", f"{ev.p_breach_drawdown:.1%}")
line("breach daily loss limit", f"{ev.p_breach_daily:.1%}")
line("expected attempts per pass", f"{ev.expected_attempts:.1f}")
line("expected cost per pass", f"${ev.expected_cost:,.0f} at ${rules.fee:,.0f} a reset")

print("\n5. FALSE-POSITIVE RATE ACROSS 12 INDEPENDENT NOISE SAMPLES")
print("-" * 68)
naive_hits, survived, families = [], [], []
for s in range(1, 13):
    r = audit.run(demo_data.pure_noise(400, seed=s), resamples=1500)
    naive_hits.append(r.n_naive_significant)
    survived.append(r.n_survived)
    families.append(r.family_size)
line("slices tested per sample (mean)", f"{np.mean(families):.1f}")
line("naive p<0.05 hits per sample (mean)", f"{np.mean(naive_hits):.2f}")
line("expected by chance alone at 5%", f"{np.mean(families) * 0.05:.2f}")
line("samples with >=1 naive 'edge'", f"{sum(h > 0 for h in naive_hits)} of 12")
line("samples with >=1 surviving 'edge'", f"{sum(h > 0 for h in survived)} of 12")
print("\n" + "=" * 68)

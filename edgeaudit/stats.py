"""
Statistical core for EdgeAudit.

Design rules enforced here (these are the product):

1. Nothing is reported as a point estimate. Every mean carries an interval.
2. Trades are not i.i.d. Bootstraps are *stationary block* bootstraps
   (Politis & Romano 1994) so serial dependence does not shrink the interval.
3. A bucket that looks good because the trader is globally good is not an
   edge. Every bucket is tested twice: against zero, and against the
   trader's own global mean (label-permutation test).
4. Slicing creates multiple comparisons. Every test enters one family and
   the family is corrected with Benjamini-Hochberg FDR.
5. Bucket means are shrunk toward the global mean (empirical Bayes), because
   the extreme buckets in any slice are extreme partly by luck.
6. Sharpe-style claims are adjusted for skew, kurtosis, sample length
   (Probabilistic Sharpe Ratio) and for the number of trials searched
   (Deflated Sharpe Ratio) -- Bailey & Lopez de Prado 2012/2014.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

EULER_MASCHERONI = 0.5772156649015329
DEFAULT_RESAMPLES = 10_000
DEFAULT_FDR_Q = 0.10
MIN_BUCKET_N = 30  # below this we refuse to test at all


# --------------------------------------------------------------------------
# resampling
# --------------------------------------------------------------------------

def _mean_block_length(n: int) -> float:
    """Politis-White style rule of thumb. Cheap, robust, good enough."""
    return max(1.0, float(n) ** (1.0 / 3.0))


def stationary_bootstrap_indices(
    n: int, resamples: int, rng: np.random.Generator, block_len: float | None = None
) -> np.ndarray:
    """
    Stationary bootstrap index matrix, shape (resamples, n).

    Geometric block lengths with mean `block_len`, wrapping at the end.
    Preserves short-range serial dependence (streaks, tilt-clusters,
    regime persistence) that an i.i.d. bootstrap would destroy -- and
    destroying it is exactly how you manufacture a fake edge.
    """
    if block_len is None:
        block_len = _mean_block_length(n)
    p = 1.0 / block_len
    idx = np.empty((resamples, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=resamples)
    new_block = rng.random((resamples, n)) < p
    starts = rng.integers(0, n, size=(resamples, n))
    for t in range(1, n):
        cont = (idx[:, t - 1] + 1) % n
        idx[:, t] = np.where(new_block[:, t], starts[:, t], cont)
    return idx


def bootstrap_mean(
    x: np.ndarray,
    resamples: int = DEFAULT_RESAMPLES,
    alpha: float = 0.05,
    seed: int = 0,
    block: bool = True,
) -> dict:
    """Bootstrap distribution of the mean, with percentile CI and p vs zero."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return {"n": 0, "mean": np.nan, "lo": np.nan, "hi": np.nan, "p_vs_zero": np.nan}
    rng = np.random.default_rng(seed)
    if block and n > 1:
        idx = stationary_bootstrap_indices(n, resamples, rng)
        boot = x[idx].mean(axis=1)
    else:
        boot = x[rng.integers(0, n, size=(resamples, n))].mean(axis=1)

    lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    # two-sided bootstrap p-value: how far must the interval widen to touch 0
    centred = boot - boot.mean()
    p = 2.0 * min(
        (centred >= -x.mean()).mean(),
        (centred <= -x.mean()).mean(),
    )
    return {
        "n": int(n),
        "mean": float(x.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "p_vs_zero": float(min(1.0, max(p, 1.0 / resamples))),
        "boot": boot,
    }


def permutation_vs_global(
    bucket_mask: np.ndarray,
    values: np.ndarray,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> float:
    """
    H0: this bucket's trades are exchangeable with the rest of the book.

    Shuffles bucket membership, not values. Answers the question that
    actually matters -- "is my edge *concentrated* here" -- rather than
    "am I profitable overall", which a vs-zero test silently conflates.
    """
    values = np.asarray(values, dtype=float)
    k = int(bucket_mask.sum())
    n = values.size
    if k == 0 or k == n:
        return np.nan
    observed = values[bucket_mask].mean() - values.mean()
    rng = np.random.default_rng(seed)
    null = np.empty(resamples)
    for i in range(resamples):
        sel = rng.choice(n, size=k, replace=False)
        null[i] = values[sel].mean() - values.mean()
    p = (np.abs(null) >= abs(observed)).mean()
    return float(min(1.0, max(p, 1.0 / resamples)))


# --------------------------------------------------------------------------
# multiplicity control
# --------------------------------------------------------------------------

def benjamini_hochberg(pvals: np.ndarray, q: float = DEFAULT_FDR_Q) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (rejected, qvalues). NaN p-values are never rejected and are
    excluded from the family size -- an untestable bucket should not make
    the surviving buckets look better.
    """
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    rejected = np.zeros_like(p, dtype=bool)
    qvals = np.full_like(p, np.nan, dtype=float)
    if m == 0:
        return rejected, qvals
    order = np.argsort(p[ok])
    p_sorted = p[ok][order]
    ranks = np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate((p_sorted * m / ranks)[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)
    rej_sorted = q_sorted <= q
    idx_ok = np.where(ok)[0]
    qvals[idx_ok[order]] = q_sorted
    rejected[idx_ok[order]] = rej_sorted
    return rejected, qvals


# --------------------------------------------------------------------------
# empirical Bayes shrinkage
# --------------------------------------------------------------------------

def shrink_toward_global(
    bucket_means: np.ndarray, bucket_ses: np.ndarray, global_mean: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    James-Stein / empirical-Bayes shrinkage.

    tau^2 (true between-bucket spread) = observed spread - average sampling
    noise, floored at zero. When the observed spread is fully explained by
    noise, tau^2 = 0 and every bucket collapses onto the global mean --
    which is the statistically correct answer to "which setup is my best",
    and the one no competitor will give.
    """
    m = np.asarray(bucket_means, dtype=float)
    se = np.asarray(bucket_ses, dtype=float)
    ok = np.isfinite(m) & np.isfinite(se) & (se > 0)
    shrunk = m.copy()
    weight = np.full_like(m, np.nan, dtype=float)
    if ok.sum() < 2:
        return shrunk, weight
    observed_var = np.var(m[ok], ddof=1)
    noise_var = np.mean(se[ok] ** 2)
    tau2 = max(0.0, observed_var - noise_var)
    w = tau2 / (tau2 + se[ok] ** 2)
    shrunk[ok] = global_mean + w * (m[ok] - global_mean)
    weight[ok] = w
    return shrunk, weight


# --------------------------------------------------------------------------
# Sharpe-family diagnostics (Bailey & Lopez de Prado)
# --------------------------------------------------------------------------

def sharpe_per_trade(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 and x.size > 1 else np.nan


def probabilistic_sharpe(x: np.ndarray, benchmark_sr: float = 0.0) -> float:
    """P(true per-trade Sharpe > benchmark), adjusted for skew and kurtosis."""
    x = np.asarray(x, dtype=float)
    n = x.size
    sr = sharpe_per_trade(x)
    if n < 3 or not np.isfinite(sr):
        return np.nan
    g3 = float(sps.skew(x, bias=False))
    g4 = float(sps.kurtosis(x, fisher=False, bias=False))
    denom = np.sqrt(max(1e-12, 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2))
    z = (sr - benchmark_sr) * np.sqrt(n - 1) / denom
    return float(sps.norm.cdf(z))


def min_track_record_length(x: np.ndarray, benchmark_sr: float = 0.0, conf: float = 0.95) -> float:
    """Trades required before the Sharpe claim clears `conf`. NaN if SR<=benchmark."""
    x = np.asarray(x, dtype=float)
    sr = sharpe_per_trade(x)
    if not np.isfinite(sr) or sr <= benchmark_sr or x.size < 3:
        return np.nan
    g3 = float(sps.skew(x, bias=False))
    g4 = float(sps.kurtosis(x, fisher=False, bias=False))
    z = sps.norm.ppf(conf)
    return float(1.0 + (1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2) * (z / (sr - benchmark_sr)) ** 2)


def deflated_sharpe(x: np.ndarray, trial_sharpes: np.ndarray) -> float:
    """
    PSR against the Sharpe you'd expect the *best of N random trials* to hit.

    This is the number that kills "I found my edge": if you sliced 40 ways,
    the best slice has to beat what 40 coin-flips would have produced.
    """
    trials = np.asarray([s for s in np.asarray(trial_sharpes, dtype=float) if np.isfinite(s)])
    n_trials = trials.size
    if n_trials < 2:
        return probabilistic_sharpe(x, 0.0)
    v = float(np.var(trials, ddof=1))
    if v <= 0:
        return probabilistic_sharpe(x, 0.0)
    e1 = sps.norm.ppf(1.0 - 1.0 / n_trials)
    e2 = sps.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    sr_expected_max = np.sqrt(v) * ((1.0 - EULER_MASCHERONI) * e1 + EULER_MASCHERONI * e2)
    return probabilistic_sharpe(x, float(sr_expected_max))


# --------------------------------------------------------------------------
# sample adequacy
# --------------------------------------------------------------------------

def trades_needed(x: np.ndarray, power: float = 0.80, alpha: float = 0.05) -> float:
    """n required to detect the *observed* effect at the given power."""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return np.nan
    sd = x.std(ddof=1)
    eff = abs(x.mean())
    if sd <= 0 or eff <= 0:
        return np.inf
    z_a = sps.norm.ppf(1 - alpha / 2)
    z_b = sps.norm.ppf(power)
    return float(np.ceil((z_a + z_b) ** 2 * sd**2 / eff**2))


def detectable_effect(n: int, sd: float, power: float = 0.80, alpha: float = 0.05) -> float:
    """Smallest mean R this sample size could reliably detect."""
    if n < 2 or sd <= 0:
        return np.nan
    z_a = sps.norm.ppf(1 - alpha / 2)
    z_b = sps.norm.ppf(power)
    return float((z_a + z_b) * sd / np.sqrt(n))


def independence_check(x: np.ndarray) -> dict:
    """Lag-1 autocorrelation + Wald-Wolfowitz runs test on the sign sequence."""
    x = np.asarray(x, dtype=float)
    n = x.size
    out = {"lag1_autocorr": np.nan, "runs_z": np.nan, "dependent": False}
    if n < 20:
        return out
    xc = x - x.mean()
    denom = float((xc**2).sum())
    if denom > 0:
        out["lag1_autocorr"] = float((xc[:-1] * xc[1:]).sum() / denom)
    signs = x > 0
    n1, n2 = int(signs.sum()), int((~signs).sum())
    if n1 > 0 and n2 > 0:
        runs = 1 + int((signs[1:] != signs[:-1]).sum())
        mu = 2 * n1 * n2 / n + 1
        var = (mu - 1) * (mu - 2) / (n - 1) if n > 1 else np.nan
        if var and var > 0:
            out["runs_z"] = float((runs - mu) / np.sqrt(var))
    out["dependent"] = bool(
        (np.isfinite(out["lag1_autocorr"]) and abs(out["lag1_autocorr"]) > 2 / np.sqrt(n))
        or (np.isfinite(out["runs_z"]) and abs(out["runs_z"]) > 1.96)
    )
    return out

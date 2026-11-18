"""
rl_stats.py
===========
Statistics for comparing offline RL agents over few seeds.
Referenced from Appendices B and C of "Offline RL: From Theory to Industrial Practice".

Every interval and test in Chapter 10 and in the companion paper comes from
these functions; paper/make_artifacts.py imports them.

Contents:
    1. t_interval       – mean and Student-t interval over per-seed values
    2. iqm, boot_iqm    – interquartile mean, with a percentile bootstrap over
                          seeds; paired (difference of IQMs, seeds resampled jointly)
    3. clopper_pearson  – exact interval for a failure rate
    4. mcnemar_exact    – exact test for paired failure counts
    5. sign_test        – exact test for paired wins
    6. coverage_demo    – how often t and bootstrap intervals cover the mean
                          over ten seeds, with and without rare catastrophes
    7. mcnemar_power    – power of the exact McNemar test for a seed budget,
                          from the probabilities of discordant seeds
    8. paired_t_seeds   – seeds needed to detect a paired difference
    9. winners_curse    – expected optimism of the best of k equal configurations

Usage:
    python rl_stats.py  # recomputes the worked examples of Appendices B and C
"""

from pathlib import Path
from typing import Optional, Sequence, Tuple
import json

import numpy as np
from scipy import stats

FAIL = -10.0         # Chapter 10: a seed fails when its mean return is below this
N_BOOT = 10_000


def t_interval(values: Sequence[float], level: float = 0.95) -> Tuple[float, float, float]:
    """Mean and Student-t interval over per-seed values (NaNs dropped).
    Exact for normally distributed seeds; meaningless when one seed is catastrophic."""
    v = np.asarray([x for x in values if not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return float('nan'), float('nan'), float('nan')
    if len(v) == 1:
        return float(v[0]), float('nan'), float('nan')
    half = stats.t.ppf(0.5 + level / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(v.mean() - half), float(v.mean() + half)


def iqm(values: Sequence[float]) -> float:
    """Interquartile mean: the mean of the middle half (the middle six of ten seeds)."""
    return float(stats.trim_mean(np.asarray(values, dtype=float), 0.25))


def boot_iqm(a: Sequence[float], b: Optional[Sequence[float]] = None, n_boot: int = N_BOOT,
             level: float = 0.95, seed: int = 0) -> Tuple[float, float, float]:
    """IQM of a, or IQM(a) - IQM(b) for paired seeds, with a percentile bootstrap
    interval. Seeds are resampled jointly, so a paired comparison keeps its pairing.
    With one task this is the stratified bootstrap of Agarwal et al. (2021)."""
    a = np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    est = lambda i: stats.trim_mean(a[i], 0.25, axis=-1) - (
        0 if b is None else stats.trim_mean(np.asarray(b, dtype=float)[i], 0.25, axis=-1))
    lo, hi = np.percentile(est(idx), [50 * (1 - level), 50 * (1 + level)])
    return float(est(np.arange(len(a)))), float(lo), float(hi)


def clopper_pearson(k: int, n: int, level: float = 0.95) -> Tuple[float, float]:
    """Exact (conservative) interval for a rate from k events in n trials."""
    a = 1 - level
    lo = stats.beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def mcnemar_exact(fail_a: Sequence[bool], fail_b: Sequence[bool]) -> Tuple[int, int, float, float]:
    """Exact McNemar test for paired binary outcomes, e.g. which seeds fail under
    two agents trained on the same data. Only discordant seeds carry information.
    Returns (only a fails, only b fails, two-sided p, one-sided p for 'a fails more')."""
    fa, fb = np.asarray(fail_a, bool), np.asarray(fail_b, bool)
    n10, n01 = int((fa & ~fb).sum()), int((~fa & fb).sum())
    if n10 + n01 == 0:
        return n10, n01, 1.0, 1.0
    two = stats.binomtest(n10, n10 + n01).pvalue
    one = stats.binomtest(n10, n10 + n01, alternative='greater').pvalue
    return n10, n01, float(two), float(one)


def sign_test(a: Sequence[float], b: Sequence[float]) -> Tuple[int, int, float]:
    """Exact two-sided sign test on paired values: (wins of a, non-tied pairs, p)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    wins, n = int((d > 0).sum()), int((d != 0).sum())
    return wins, n, float(stats.binomtest(wins, n).pvalue) if n else 1.0


def coverage_demo(n: int = 10, reps: int = 4000, n_boot: int = 2000, seed: int = 0):
    """Share of experiments, each with n seeds, whose interval contains the true mean:
    for well-behaved seeds, and for seeds that fail catastrophically 5% of the time."""
    rng = np.random.default_rng(seed)
    cases = {
        'normal seeds':        (lambda: rng.normal(-1.24, 0.06, n), -1.24),
        '5% catastrophic':     (lambda: np.where(rng.random(n) < 0.05, rng.normal(-224, 5, n),
                                                 rng.normal(-2, 0.3, n)), 0.95 * -2 + 0.05 * -224),
    }
    out = {}
    for name, (draw, mean) in cases.items():
        hit_t = hit_b = 0
        for _ in range(reps):
            x = draw()
            _, lo, hi = t_interval(x)
            hit_t += lo <= mean <= hi
            bm = x[rng.integers(0, n, (n_boot, n))].mean(1)
            blo, bhi = np.percentile(bm, [2.5, 97.5])
            hit_b += blo <= mean <= bhi
        out[name] = (hit_t / reps, hit_b / reps)
    return out


def mcnemar_power(p10: float, p01: float, n: int, alpha: float = 0.05,
                  reps: int = 20_000, seed: int = 0) -> float:
    """Probability that the exact two-sided McNemar test rejects at level alpha with
    n paired seeds, when a seed fails only under agent A with probability p10 and
    only under agent B with probability p01. The marginal failure rates alone do not
    fix the power: what matters is how often the agents fail on different seeds."""
    rng = np.random.default_rng(seed)
    u = rng.random((reps, n))
    n10 = (u < p10).sum(1)
    n01 = ((u >= p10) & (u < p10 + p01)).sum(1)
    # The p-value depends only on (n10, n10 + n01); cache it.
    cache = {}
    def p_value(a, d):
        if (a, d) not in cache:
            cache[(a, d)] = stats.binomtest(a, d).pvalue if d else 1.0
        return cache[(a, d)]
    return float(np.mean([p_value(a, a + b) < alpha for a, b in zip(n10, n01)]))


def paired_t_seeds(delta: float, sd_diff: float, power: float = 0.8, alpha: float = 0.05) -> int:
    """Smallest number of paired seeds for which a two-sided paired t-test detects a
    true mean difference delta with the given power, if the per-seed differences have
    standard deviation sd_diff (noncentral t, exact)."""
    for n in range(3, 10_000):
        df, nc = n - 1, delta / (sd_diff / np.sqrt(n))
        crit = stats.t.ppf(1 - alpha / 2, df)
        if stats.nct.sf(crit, df, nc) + stats.nct.cdf(-crit, df, nc) >= power:
            return n
    raise ValueError("no seed budget below 10,000 reaches this power")


def winners_curse(k: int, reps: int = 200_000, seed: int = 0) -> float:
    """Expected optimism, in standard errors, of the best of k configurations that are
    in truth equally good, each measured with the same noise: E[max of k N(0, 1)]."""
    rng = np.random.default_rng(seed)
    return float(rng.standard_normal((reps, k)).max(1).mean())


def _fmt(ci, d=2):
    m, lo, hi = ci
    return f"{m:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


if __name__ == '__main__':
    res = Path(__file__).resolve().parent / 'results'
    load = lambda name: json.loads((res / name).read_text())
    per_seed = lambda f, m: [r['results'][m]['reward'] for r in f['runs']]

    print("Coverage of nominal 95% intervals over ten seeds (t, percentile bootstrap):")
    for name, (t, b) in coverage_demo().items():
        print(f"  {name:18s} {t:.0%}  {b:.0%}")
    print(f"P(no catastrophe in ten seeds | 5% each) = {0.95 ** 10:.2f}")

    o = load('ch10_benchmark_oracle.json')
    v = per_seed(o, 'OracleMOReL-HALT')
    print("\nTrue dynamics with HALT (Chapter 10):")
    print(f"  mean, t-interval  {_fmt(t_interval(v))}")
    print(f"  IQM, bootstrap    {_fmt(boot_iqm(v))}")
    print(f"  HALT effect, learned model  {_fmt(boot_iqm(per_seed(o, 'HybridMOReL'), per_seed(o, 'HybridMOReL-noHALT')))}")
    print(f"  HALT effect, true dynamics  {_fmt(boot_iqm(v, per_seed(o, 'OracleMOReL')))}")

    print("\nFailure rates, 95% Clopper-Pearson:")
    for k, n in [(0, 10), (2, 10), (0, 20), (3, 20)]:
        lo, hi = clopper_pearson(k, n)
        print(f"  {k}/{n}: {lo:.1%} - {hi:.1%}")

    a, b = load('ch10_coverage_normal_1-10_noise_k20.json'), load('ch10_coverage_normal_1-10_noise_k20_fresh.json')
    nop = np.r_[per_seed(a, 'no pessimism'), per_seed(b, 'no pessimism')]
    dat = np.r_[per_seed(a, 'HALT p99, data kappa'), per_seed(b, 'HALT p99, data kappa')]
    n10, n01, two, one = mcnemar_exact(nop < FAIL, dat < FAIL)
    print(f"\nNarrow logs, 20-step rollouts, seeds 0-19: discordant {n10}/{n01}, "
          f"McNemar p = {two:.3f} (one-sided {one:.3f})")
    wins, n, p = sign_test(dat[:10], nop[:10])
    print(f"Seeds 0-9: data-sized HALT ahead of no pessimism on {wins} of {n}, sign test p = {p:.2f}")
    print(f"IQM difference, seeds 10-19 (held out): {_fmt(boot_iqm(dat[10:], nop[10:]))}")
    print(f"IQM difference, pooled (descriptive):   {_fmt(boot_iqm(dat, nop))}")

    print("\nSeeds for 80% power, failure rates 15% vs 2% (exact McNemar, alpha = 0.05):")
    for name, (p10, p01) in {'independent failures': (0.15 * 0.98, 0.85 * 0.02),
                             'B fails only where A fails': (0.13, 0.0),
                             'never on the same seed': (0.15, 0.02)}.items():
        row = ", ".join(f"{n}: {mcnemar_power(p10, p01, n):.2f}" for n in (10, 20, 40, 60, 80))
        print(f"  {name:28s} {row}")
    lagged = load('ch10_benchmark_lagged.json')
    d = np.array(per_seed(lagged, 'CQL+Physics')) - np.array(per_seed(lagged, 'CQL'))
    print(f"\nCQL+Physics - CQL, paired: sd of differences {d.std(ddof=1):.3f}; "
          f"seeds to detect +0.03 with 80% power: {paired_t_seeds(0.03, d.std(ddof=1))}")
    print("Winner's curse, best of k equal configurations, in standard errors:",
          ", ".join(f"k={k}: {winners_curse(k):.2f}" for k in (2, 5, 10, 20)))

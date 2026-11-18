---
layout: default
title: "Appendix B: Statistical Inference for Offline RL Experiments"
lang: en
ru_url: /ru/appendix_b/
prev_chapter:
  url: /en/appendix/
  title: "Algorithm Selection Guide"
next_chapter:
  url: /en/appendix_c/
  title: "Experimental Design and Validation in Offline RL"
permalink: "/offline-rl-book/en/appendix_b/"
---

# Appendix B: Statistical Inference for Offline RL Experiments

> *"Ten seeds can tell you that something happens. They rarely tell you how often."*

This appendix asks what can be concluded from the results of an offline RL experiment, and with how much uncertainty. It is written for a situation that general texts on statistics and on deep RL benchmarking rarely address together: **one task, ten or twenty training seeds, paired data and rare catastrophic failures.** That is the situation of most industrial offline RL studies, and of Chapter 10. [Appendix C](/offline-rl-book/en/appendix_c.html) takes the other half of the question: how to design an experiment so that these conclusions are justified at all. A good test does not repair a bad design.

Introductory hypothesis testing is not repeated here, and neither is off-policy evaluation, which Chapter 3 treats: there the task is to estimate one fixed policy from logged trajectories; here it is to describe the distribution of outcomes of a training procedure. The numerical examples use either Chapter 10's results or simulations specified in the text, and all can be recomputed with the code.

> 📄 Full code: [`rl_stats.py`](https://github.com/corba777/offline-rl-book/blob/main/code/rl_stats.py) – the functions that produced every interval and test in Chapter 10; `python code/rl_stats.py` recomputes the examples of Appendices B and C.

---

## Estimands and Replications

A **seed** in Chapter 10 fixes a whole training run: the logged dataset, the network initializations, the dynamics ensemble and the sampling inside SAC. Each trained policy is then evaluated on 20 episodes. The number that enters every comparison is the seed's mean return over those 20 episodes, and the seed is the unit of replication.

It is tempting to treat the 200 episodes of ten seeds as 200 observations. They are not independent: the 20 episodes of a seed share one policy, and that policy carries all the randomness of its training run. If the return of episode $j$ in seed $i$ is

$$R_{ij} = \mu + s_i + e_{ij}, \qquad \operatorname{Var}(s_i) = \sigma^2_{\text{seed}}, \quad \operatorname{Var}(e_{ij}) = \sigma^2_{\text{ep}},$$

then the mean over $n$ seeds with $m$ episodes each has variance

$$\operatorname{Var}(\bar R) = \frac{\sigma^2_{\text{seed}}}{n} + \frac{\sigma^2_{\text{ep}}}{n\,m}.$$

Treating all $n\,m$ episodes as independent replaces this with $(\sigma^2_{\text{seed}} + \sigma^2_{\text{ep}})/(n\,m)$. When seed-to-seed variation dominates – the usual case in offline RL, where a different dataset and initialization can change the policy entirely – the interval comes out up to $\sqrt{m} \approx 4.5$ times too narrow for $m = 20$. Ecologists call this *pseudoreplication* (Hurlbert, 1984). More evaluation episodes shrink only the second term; only more seeds shrink the first.

What is being estimated also has to be said, starting with what varies. A study can fix the logged data and repeat only the training, or draw new data as well:

$$\theta_A(D_0) = \mathbb{E}_U\big[J(A(D_0, U))\big] \qquad \text{or} \qquad \theta_A = \mathbb{E}_{D,\,U}\big[J(A(D, U))\big],$$

where $A$ is the training procedure, $D$ the dataset, $U$ the rest of its randomness and $J$ the return of the trained policy. The first describes how stable training is on one set of logs; the second describes the algorithm on logs like these. In Chapter 10 every seed draws a new dataset from the same simulated operators, so its intervals are about $\theta_A$. In industrial practice the difference is large: retraining ten times on one plant's logs says little about how the method will do on another plant's.

An agent that usually works and occasionally fails catastrophically has no single number that describes it, so Chapter 10 reports three **estimands**, each with its own uncertainty:

| Question | Estimate | Uncertainty |
|---|---|---|
| How well does the agent usually do? | Interquartile mean (IQM) over seeds | Bootstrap interval on the IQM; where no seed fails, the $t$-interval on the mean as a stand-in (below) |
| How often does it fail? | Number of failing seeds | Exact (Clopper–Pearson) interval; exact paired test between agents |
| What is the expected return? | Mean over seeds | Reported, but read descriptively |

---

## Heavy-Tailed Returns and Robust Statistics

The reason the mean cannot serve alone is arithmetic. Suppose an agent returns about −2 on 95% of seeds and −224 on the other 5%. Its expected return is $0.95 \cdot (-2) + 0.05 \cdot (-224) = -13.1$. But the chance that ten seeds contain no failure at all is $0.95^{10} = 0.60$. In six experiments out of ten, every seed lands near −2, the sample mean is near −2, and a narrow interval around it excludes the true mean of −13.1. In the other four, one failing seed drags the mean down and inflates the interval. No interval over ten seeds can fix this: the information about a 5% event is simply not in most samples of ten.

The **interquartile mean** – the mean of the middle half, the middle six of ten seeds – answers "how well does it usually do" well because it ignores the tails. The median does the same with even less efficiency. For the same reason both say nothing about failures, which are exactly the tail. A safeguard such as HALT exists to prevent failures, so judging it by IQM alone would miss its purpose; judging it by the mean alone lets one seed decide the verdict. Hence the three estimands, reported side by side. (With ten seeds, "the middle half" follows the convention of `scipy.stats.trim_mean(x, 0.25)` and of rliable: the number of seeds cut from each end is rounded down, so two of ten go from each side and six remain – strictly a 20% trim.)

A failure needs a definition. In Chapter 10 a seed fails when its mean return is below −10, an order of magnitude below every stable configuration. Such a line is a choice, so two checks belong with it. First, does the count depend on where the line is drawn? At the end point of the coverage study, the agent without pessimism has 3, 3 and 2 failing seeds at lines of −5, −10 and −20, and the data-sized HALT none at any of them. Second, does a return-defined failure mean a physical one? Of the 136 return-defined failures among all of Chapter 10's evaluations, 131 violate a process bound on more than 10% of the steps from disturbed starts, against 6 of the 759 non-failing ones (at most 30% of the steps). These evaluations share seeds and datasets across configurations, so the proportions describe the data rather than estimate a probability.

---

## Student-t, Bootstrap and Coverage

The **Student-$t$ interval** over $n$ per-seed values,

$$\bar x \pm t_{0.975,\,n-1}\, \frac{s}{\sqrt n},$$

is exact when the seeds are normally distributed, and with ten seeds it is the better default. A simulation in `rl_stats.py` draws 4,000 experiments of ten seeds each. For well-behaved seeds the $t$-interval covers the true mean 95% of the time, as it should; the **percentile bootstrap** covers it 91% of the time, because with ten observations the resampled means are too concentrated. Bootstrap intervals are often presented as the assumption-free option; with ten seeds they are the optimistic one.

Neither survives a catastrophic seed. With the 5% failure rate above, both intervals cover the true mean in only 42% of the experiments. Three different problems are at work here, and they should be kept apart: **poor coverage** (the interval misses the parameter more often than its nominal 5%), **poor informativeness** (it is too wide to say anything) and a **wrong estimand** (it describes the mean when the question is about typical performance or about failures). The true-dynamics agent with HALT in Chapter 10 has one seed at −224; the other nine lie between −5.7 and −1.8:

| Summary | Value |
|---|---|
| Mean, 95% $t$-interval | −25.6 [−75.5, +24.3] |
| IQM, 95% bootstrap interval | −3.8 [−41.1, −2.6] |

The $t$-interval reaches +24, a return this task cannot produce: every reward is a penalty. An interval that crosses the range of possible values is not wrong for that reason alone, but here it tells us almost nothing, and its normal-theory justification does not hold for a sample with one catastrophic seed. Chapter 10's tables use **one interval per cell**: a bootstrap interval on the IQM when a seed fails, with the mean shown alone as a description, and a $t$-interval on the mean when no seed fails. That is a compromise for compact tables, not a statistical protocol, and it has a flaw worth naming: the estimand changes with what was observed. Strictly, the $t$-interval is an interval for the mean, not for the IQM; it stands in for the typical return only because, without a failing seed, the two nearly coincide in these data (−1.24 and −1.23 for model-based SAC without pessimism). And "no failing seed among ten" does not mean the distribution has no failures – at a true rate of 5%, ten seeds show none 60% of the time. A protocol should fix its estimands **before** the analysis and report them in every cell, whatever the data turn out to be: for example, the IQM with a bootstrap interval and the failure count with an exact interval everywhere, and the mean as a description.

The bootstrap for the IQM resamples seeds with replacement, 10,000 times, recomputes the IQM of each resample and takes the 2.5th and 97.5th percentiles. The end of the interval on the side of the failure is the fragile one. A resample that draws the failing seed three or four times pushes it into the middle half, and that end stretches towards the failure: in the example above it is the lower end, −41.1, while the upper end, −2.6, barely moves. For a difference in which the failing agent is subtracted, the same thing happens to the upper end. The steadier end is not guaranteed to be right either – the coverage of a bootstrap interval with ten seeds remains in question – but it is not hostage to one seed. Read such an interval together with the failure count, never alone. More refined bootstrap intervals (BCa, studentized) do not change this: no resampling scheme can show events that the ten seeds did not contain. With one task, the stratified bootstrap that Agarwal et al. (2021) recommend for aggregating over tasks reduces to this plain resampling of seeds.

---

## Rare Failures

A count of failing seeds is an estimate of a failure rate, and with ten or twenty seeds it is a coarse one. The exact **Clopper–Pearson** interval for $k$ failures in $n$ seeds inverts the binomial test:

| Failures | 95% interval for the failure rate |
|---|---|
| 0 of 10 | 0% – 30.8% |
| 2 of 10 | 2.5% – 55.6% |
| 0 of 20 | 0% – 16.8% |
| 3 of 20 | 3.2% – 37.9% |

"No failures in ten seeds" is compatible with a failure rate of one in three. The *rule of three* gives the one-sided version quickly: after $n$ clean trials, the 95% upper bound on the rate is about $3/n$ (exactly 25.9% for $n = 10$ and 13.9% for $n = 20$).

Comparing two agents' failure counts calls for a paired analysis, because the same seeds are used. Each seed falls into one of four cells, with probabilities

| | Agent B fails | Agent B does not fail |
|---|---|---|
| **Agent A fails** | $p_{11}$ | $p_{10}$ (only A) |
| **Agent A does not fail** | $p_{01}$ (only B) | $p_{00}$ |

The two failure rates differ by $(p_{11} + p_{10}) - (p_{11} + p_{01}) = p_{10} - p_{01}$: seeds where both or neither fail cancel. **McNemar's test** uses only the discordant seeds. Given $d = n_{10} + n_{01}$ of them, $n_{10}$ is binomial with success probability $\theta = p_{10}/(p_{10} + p_{01})$, and "the agents fail equally often" is $\theta = 1/2$. At the end point of Chapter 10's coverage study, all three failures without pessimism fell on seeds where HALT with a data-sized $\kappa$ did not fail: $n_{10} = 3$, $n_{01} = 0$, and

$$p_{\text{two-sided}} = 2 \cdot 0.5^3 = 0.25, \qquad p_{\text{one-sided}} = 0.125.$$

The observed difference in failure rates, $(n_{10} - n_{01})/n = 3/20 = 15$ points, looks substantial; the test says it could easily be chance. The same conditional view gives an interval: the exact Clopper–Pearson interval for $\theta$ from 3 of 3 is [0.29, 1.00], and it contains $1/2$. An interval for the difference $p_{10} - p_{01}$ itself also exists (Newcombe, 1998), but with three discordant seeds it says the same thing less directly.

The arithmetic also shows a hard limit: with fewer than six discordant seeds, an exact two-sided McNemar test cannot reach $p < 0.05$ however lopsided they are ($2 \cdot 0.5^5 = 0.0625$). Appendix C turns this into a seed budget.

A tempting mistake is **Fisher's exact test** on "4 of 20 against 0 of 20", which treats the two agents' seeds as independent samples. For the comparison of HALT at the 80th percentile with no HALT in Chapter 10 it gives a one-sided $p = 0.053$; the correct paired test, with all four failures discordant, gives $p = 0.125$ two-sided and $0.0625$ one-sided. The conclusion – suggestive, not established – happens to survive, but the test must match the design.

---

## Paired Comparisons

When two agents are trained on the same data, compare them seed by seed. For per-seed returns $a_i$ and $b_i$,

$$\operatorname{Var}(a_i - b_i) = \operatorname{Var}(a_i) + \operatorname{Var}(b_i) - 2\operatorname{Cov}(a_i, b_i),$$

and the covariance is large exactly when the dataset drives much of the variation – a hard dataset is hard for both agents. The paired difference removes what the agents share.

Chapter 10's constraint penalty is the textbook case. On the Markov state, CQL reaches −1.4 [−1.5, −1.4] and CQL with the bounds penalty −1.4 [−1.4, −1.3]: the two intervals overlap, and side by side the penalty looks like it does nothing. The two agents' returns correlate at 0.87 across seeds, so the per-seed differences have a standard deviation of 0.047, against about 0.09 for either agent alone. The paired interval is 2.7 times narrower than an unpaired one would be, and the penalty improves the return by +0.07 [+0.03, +0.10]. Overlapping intervals of two means are not evidence of no difference.

The rules for paired data:

- **Intervals** on the per-seed differences: a $t$-interval when neither agent has a failing seed, otherwise the difference of IQMs with a bootstrap that **resamples seeds jointly** – the same seed indices for both agents. Resampling the two agents independently throws the pairing away.
- **Wins** can be counted with the exact **sign test**. Its null hypothesis is not "no difference" in general but that a positive difference is as likely as a negative one – the median of the per-seed differences is zero. Seven wins of ten give $p = 0.34$ – not evidence of anything on its own.
- The **median paired difference** is a robust companion to the mean difference when a seed or two misbehave.

One trap deserves its own line. For robust statistics, the summary of the differences is not the difference of the summaries:

$$\operatorname{IQM}(a - b) \neq \operatorname{IQM}(a) - \operatorname{IQM}(b)$$

in general, because the seeds that fall in the middle half of $a - b$ need not be those in the middle half of $a$ or of $b$. For the constraint penalty, $\operatorname{IQM}(a) - \operatorname{IQM}(b) = +0.086$, while the IQM of the per-seed differences is $+0.058$. Both use the pairing, but they answer different questions: the first compares the typical performance of the two agents, the second describes the typical gain on one seed. Chapter 10 reports the first, with a joint bootstrap; whichever is used, say which.

---

## Equivalence, Non-Inferiority and Risk

On the held-out seeds, HALT with a data-sized $\kappa$ and CQL at $\alpha = 0.1$ differ by −0.01 [−0.63, +0.62]. The interval contains zero, but it also contains differences of 0.6, which on this task is about a quarter of the return. "Not significantly different" means the data cannot tell the agents apart, not that they perform alike.

A claim of **equivalence** needs a margin $\delta$, fixed before looking: the largest difference that would not matter in practice. The *two one-sided tests* procedure (TOST; Schuirmann, 1987) declares equivalence at the 5% level when the 90% interval of the difference lies entirely inside $(-\delta, +\delta)$. Here the 90% $t$-interval of the mean paired difference is [−0.51, +0.50]; with a margin of, say, 0.25 the agents are not shown to be equivalent. This classical TOST is about the mean difference and rests on the $t$-interval's assumptions; equivalence in IQM would need an inference procedure calibrated for that estimand, which a ten-seed bootstrap is not. A claim of **non-inferiority** – the new method is not worse than the reference by more than $\delta$ – needs only the lower half: the lower end of the same 90% interval must lie above $-\delta$. That is the natural form for "a cheap safeguard costs little", and it too needs its margin before the data. Chapter 10 makes neither claim and writes "no resolved difference" instead of "level with".

The IQM deliberately ignores the tail, and the failure count reduces the tail to a yes or no. **Risk-sensitive summaries** describe the tail itself: the worst seed, or the conditional value at risk – the mean of the worst fraction of seeds. Pooled over the twenty seeds of the end point:

| Agent | Worst seed | Mean of worst 2 of 20 | Median | IQM |
|---|---|---|---|---|
| Model-based SAC, no pessimism | −227.1 | −162.1 | −3.0 | −3.0 |
| HALT $p_{99}$, data $\kappa$ | −4.7 | −4.2 | −2.2 | −2.2 |
| CQL, $\alpha = 0.1$ | −2.7 | −2.7 | −2.5 | −2.5 |

The table says something the IQM column does not: CQL has the best worst case even where it does not have the best typical return. With twenty seeds, "the worst 10%" is two seeds, so such summaries are descriptions of the sample, with no useful interval. They belong next to the three estimands, not instead of them.

---

## Diagnostics: Critic Checks and Rank Correlation

Diagnostics are estimates too, and they need the same care: a defined estimand and a measured noise.

**A critic against a simulator.** A SAC critic estimates a *soft* value: the TD target is $r + \gamma\big(\min Q'(s', a') - \alpha \log \pi(a' \mid s')\big)$. Its Monte Carlo counterpart, for a state $s$ and an action $a_0 \sim \pi(\cdot \mid s)$, starts the simulator in $s$, applies $a_0$, follows the stochastic policy, and averages

$$\hat Q(s, a_0) = \sum_{t=0}^{H-1} \gamma^t r_t \;-\; \alpha \sum_{t=1}^{H-1} \gamma^t \log \pi(a_t \mid s_t)$$

over several continuations – with the same $\gamma$, entropy weight, reward normalization and termination as in training. Dropping the entropy term, or using the deterministic policy, compares the critic with a different quantity. A finite horizon $H$ drops the terms beyond it, each discounted by at least $\gamma^H$ – about 0.01 for $H = 460$ at $\gamma = 0.99$. If every discounted term (reward and entropy) is bounded by $G_{\max}$ in magnitude, the truncation error is at most $\gamma^H G_{\max}/(1-\gamma)$; how large it is relative to the value depends on the trajectory, and it is small here because the far terms are strongly suppressed, not because it is exactly 1%. The estimate has its own noise: in Chapter 10 the spread of continuations from one start state, divided by the square root of the number of continuations and start states, gives a standard error of about 0.1 normalized units on the full logs, and rerunning the same agents with eight continuations instead of two moved the critic's extra optimism at the rollout end from +0.2 [−0.0, +0.4] to +0.2 [+0.1, +0.3].

Compare **differences** rather than levels. The critic in Chapter 10 is optimistic everywhere, by a factor of about five on the full logs; the question about edge-of-reach states is whether it is *more* optimistic at the end of rollouts than on the data. Subtracting each seed's gap at the dataset states from its gap at the rollout end cancels the overall bias, and the result is a paired comparison like any other.

**Does uncertainty track error?** The check needs a measure of association that does not depend on scale – disagreement and error are in different units and need not be linearly related – so Chapter 10 uses **Spearman's rank correlation** between the ensemble's disagreement and the true error, step by step along a rollout. It falls from about 0.4 at the first step to 0.05 by step 20 in closed loop: one step ahead the ranking is weakly informative, twenty steps ahead it is noise. Two companion summaries make it concrete: the error in the highest-uncertainty quintile against the lowest, and the error of transitions labelled KNOWN against UNKNOWN at the threshold actually used.

---

## Summary

| Question | Method | In `rl_stats.py` |
|---|---|---|
| Interval for the mean | Student-$t$ over per-seed means (Chapter 10's tables use it where no seed fails) | `t_interval` |
| Interval for the typical return | IQM with percentile bootstrap over seeds | `iqm`, `boot_iqm` |
| Difference between two agents on the same data | Paired $t$, or difference of IQMs with joint bootstrap | `t_interval`, `boot_iqm(a, b)` |
| How often does an agent fail? | Clopper–Pearson interval | `clopper_pearson` |
| Do two agents fail equally often? | Exact McNemar on discordant seeds | `mcnemar_exact` |
| How often does one agent win? | Exact sign test | `sign_test` |
| Are two agents equivalent, or one no worse, in mean? | TOST or non-inferiority with a margin fixed in advance | `t_interval(level=0.90)` |
| How reliable are these intervals at all? | Coverage simulation | `coverage_demo` |

The methods are standard; what this appendix argues for is the combination. With few seeds and rare failures, the honest report is not one number with an interval but three – how well the agent usually does, how often it fails, and what its mean is – each with the uncertainty that fits it. Whether those numbers mean what they seem to depends on how the experiment was designed, which is the subject of Appendix C.

---

## References

- Hurlbert, S.H. (1984). *Pseudoreplication and the Design of Ecological Field Experiments.* Ecological Monographs, 54(2), 187–211.
- Clopper, C.J., & Pearson, E.S. (1934). *The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial.* Biometrika, 26(4), 404–413.
- McNemar, Q. (1947). *Note on the Sampling Error of the Difference between Correlated Proportions or Percentages.* Psychometrika, 12(2), 153–157.
- Newcombe, R.G. (1998). *Improved Confidence Intervals for the Difference between Binomial Proportions Based on Paired Data.* Statistics in Medicine, 17(22), 2635–2650.
- Hanley, J.A., & Lippman-Hand, A. (1983). *If Nothing Goes Wrong, Is Everything All Right? Interpreting Zero Numerators.* JAMA, 249(13), 1743–1745. *(the rule of three)*
- Schuirmann, D.J. (1987). *A Comparison of the Two One-Sided Tests Procedure and the Power Approach for Assessing the Equivalence of Average Bioavailability.* Journal of Pharmacokinetics and Biopharmaceutics, 15(6), 657–680.
- Efron, B., & Tibshirani, R.J. (1993). *An Introduction to the Bootstrap.* Chapman & Hall.
- Agarwal, R., Schwarzer, M., Castro, P.S., Courville, A., & Bellemare, M.G. (2021). *Deep Reinforcement Learning at the Edge of the Statistical Precipice.* NeurIPS. [arXiv:2108.13264](https://arxiv.org/abs/2108.13264).

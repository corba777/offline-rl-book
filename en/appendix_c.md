---
layout: default
title: "Appendix C: Experimental Design and Validation in Offline RL"
lang: en
ru_url: /ru/appendix_c/
prev_chapter:
  url: /en/appendix_b/
  title: "Statistical Inference for Offline RL Experiments"
next_chapter: null
permalink: "/offline-rl-book/en/appendix_c/"
---

# Appendix C: Experimental Design and Validation in Offline RL

> *"A good test does not repair a bad design."*

[Appendix B](/offline-rl-book/en/appendix_b.html) asks what can be concluded from the data an experiment produced. This appendix asks the earlier question: how to set up the experiment so that those conclusions deserve trust. The two are different levels of reasoning. Take Chapter 10's end point, three failures without pessimism against none with a data-sized safeguard. The statistical question is why an exact paired test gives $p = 0.25$ for a difference that looks large. The design question is what it means that the configuration was found by searching on the first ten seeds, and why those seeds cannot then confirm it.

This is not a course in the design of experiments. It covers the few design decisions that offline RL studies actually face – paired seeds, factorial ablations, adaptive search, many comparisons, confirmation runs and seed budgets – with Chapter 10 as the running example.

> 📄 Full code: [`rl_stats.py`](https://github.com/corba777/offline-rl-book/blob/main/code/rl_stats.py) – including the power simulator for failure counts (`mcnemar_power`), the seed budget for paired comparisons (`paired_t_seeds`) and the winner's curse (`winners_curse`).

---

## Experimental Units and Randomization

The experimental unit in offline RL is the **training run**, identified by its seed. One seed fixes everything random about the run: the logged dataset, the initializations, the model ensemble, the minibatch order and the sampling inside the learner. Evaluation episodes are measurements of that unit, not new units (Appendix B shows how much interval width is at stake).

Three habits make the units honest:

- **Fix the seeds before running.** Chapter 10 uses seeds 0–9 for every study and 10–19 for confirmation. A seed list chosen after seeing which seeds behave is a selected sample.
- **Never drop a seed.** A seed that fails is a result, often the most important one. Chapter 10 reports every failing seed and names them where it matters (seeds 6 and 7 under HALT with the learned model; seed 6 with the true dynamics).
- **Keep the seed sets of different purposes disjoint.** Seeds that helped choose a configuration cannot confirm it (below).

Randomization in this setting is cheap: every source of randomness comes from the seed. What matters is to know which sources are randomized *per run* and which are *shared* between the agents being compared.

---

## Common Random Numbers and Paired Designs

Simulation studies have long used **common random numbers**: when comparing two systems, drive both with the same random inputs, so that the comparison is not drowned by noise both share. In offline RL the shared inputs are the dataset and the evaluation conditions. In Chapter 10, within a seed:

- every agent trains on the **same logged dataset**;
- every agent is evaluated on the **same 20 episodes** – the same start states and the same process noise;
- the model-based agents share **one dynamics ensemble**;
- each agent reseeds its own initialization and sampling from the seed before training, so an agent's result does not depend on which agents ran before it.

The last point has a practical payoff. Because every arm reseeds, an arm can be rerun on its own later – a new threshold, a new penalty – and it reproduces exactly what it would have produced in the full run. Chapter 10 checked this: an arm rerun alone matched the stored result bit for bit. New cells of a factorial design can then be added without losing the pairing. It is worth checking before relying on it; a single library update can break it.

The payoff in precision is large. For the constraint penalty in Chapter 10, the two agents' returns correlate at 0.87 across seeds. The paired interval is 2.7 times narrower than an unpaired one, and the seed budget follows: detecting a true improvement of 0.03 with 80% power takes about 22 paired seeds, against roughly 140 per agent if the agents were trained on independent data.

---

## Adaptive Search and the Winner's Curse

Much real research is a search: try a configuration, look, adjust. Chapter 10's coverage study is one. The logs were narrowed and the rollouts lengthened until the agent without pessimism failed. Any search that stops at the most interesting result selects for noise in that direction.

The size of the effect is easy to compute. If $k$ configurations are in truth equally good and each is measured with the same standard error, the best-looking one is, on average, this many standard errors too optimistic:

| Configurations compared | Expected optimism of the best |
|---|---|
| 2 | 0.57 SE |
| 5 | 1.17 SE |
| 10 | 1.54 SE |
| 20 | 1.87 SE |

The table assumes independent, normally distributed measurement errors of equal size; configurations measured on the same seeds have correlated errors. Positively correlated errors – the usual case when configurations share data – tend to reduce the effect, but correlation as such does not: with two perfectly negatively correlated errors the expected optimism is $\mathbb{E}[\max(Z, -Z)] = \sqrt{2/\pi} \approx 0.80$ SE, more than the 0.57 for independent ones. With ten or twenty configurations – a modest hyperparameter sweep – the winner is expected to look about one and a half to two standard errors better than it is, even if all are equal. This is the *winner's curse*, and it does not depend on any test: it is built into choosing the maximum. Gelman and Loken (2014) call the broader version the *garden of forking paths*: the analysis that was run depends on the data, even when no single decision feels like cherry-picking.

The same mechanism biases baselines in the other direction. Chapter 10 chose CQL's weight on evaluation returns over a grid of five values – a search that favours CQL – and then transferred it unchanged to the narrowed logs, where the termination settings were explored more widely. Neither procedure is a fair basis for ranking algorithms; Chapter 10 therefore calls its comparisons comparisons of fixed configurations. A fair ranking needs one tuning protocol for every method, with tuning done on validation runs and the ranking on runs that took no part in it (Cawley and Talbot, 2010).

---

## Exploration, Validation and Confirmation

The remedy for the winner's curse is to separate the runs that choose from the runs that judge. A complete design has three sets of seeds:

1. **Exploration** – pilots and searches, as free as needed, and documented. Chapter 10's coverage study started with a pilot on seeds 0–2 at two coverage levels; everything run after it is reported.
2. **Validation** – where settings are tuned, for every method under the same protocol.
3. **Confirmation** – fresh seeds, decided in advance, on which the chosen configuration is run once and judged.

Chapter 10 merges the first two and keeps the third: seeds 0–9 explored and selected, seeds 10–19 confirmed. The confirmation is then the only evidence that counts as confirmatory:

| Seeds | IQM difference, data-sized HALT − no pessimism |
|---|---|
| 0–9 (used in the search) | +0.81 [+0.19, +38.35] |
| 10–19 (held out) | +1.08 [−0.46, +20.17] |
| Pooled | +0.80 [+0.12, +3.34] |

The pooled interval excludes zero and the held-out one does not. Pooling looks like more data, but half of it chose the end point; the pooled row is a description, not a confirmation. And confirmation covers only what was repeated. Chapter 10 repeated the end point of its search on fresh seeds, not the boundaries between the regimes it passed through, so the regime map stays exploratory.

What to fix before the confirmation run, in writing if possible: the configuration, the seeds, the outcome measures, the failure line and the comparisons to be made. That is a light version of preregistration, and it is what makes the confirmation more than a second look.

---

## Multiple Comparisons and Selective Inference

Chapter 10's benchmark has about a dozen paired comparisons. If none of them reflected a real difference and the tests were independent, the chance that at least one would still clear $p < 0.05$ would be $1 - 0.95^{12} = 46\%$. Chapter 10's comparisons share seeds and baselines, so they are dependent, and with dependent tests the figure differs from 46% – without knowing the dependence, not necessarily downwards. The 46% is an illustration of scale, not an estimate. There are three honest responses (Bonferroni and Holm remain valid under any dependence):

- **Correct** for the number of tests. Bonferroni tests each of $k$ comparisons at $0.05/k$; Holm's step-down version (Holm, 1979) is uniformly more powerful and just as valid.
- **Say it.** Chapter 10 notes that a borderline result among many – the constraint penalty on the logged state – is expected by chance, and treats it as suggestive.
- **Report everything that was run.** The quieter form of the problem is selective reporting: showing only the cells that came out interesting. Intervals computed after such a selection no longer have their nominal coverage; *selective inference* (Taylor and Tibshirani, 2015) studies how to correct them, but the simple defence is to report every configuration, including the dull and the failed.

---

## Factorial Ablations and Interaction Effects

An ablation that removes one component at a time cannot tell whether two components matter jointly. A **factorial design** runs all combinations, and the effect of each factor is measured with the others held fixed. Chapter 10 uses two.

**Model versus termination.** Dynamics (learned or true) crossed with HALT (on or off), on the same seeds. The effect of HALT is measured within each dynamics, seed by seed, and the **interaction** is the difference between those effects:

$$\Delta_{\text{learned}} = R_{\text{learned, on}} - R_{\text{learned, off}}, \qquad \Delta_{\text{true}} = R_{\text{true, on}} - R_{\text{true, off}}, \qquad I = \Delta_{\text{learned}} - \Delta_{\text{true}}.$$

HALT lowers the IQM by 3.4 [1.5, 10.8] with the learned model and by 2.5 [1.4, 39.8] with the true dynamics; the interaction has a median of +0.02 and an IQM of −0.37 [−3.40, +33.35]. The design establishes that transition error is *not necessary* for HALT to do damage. It does not establish that model error never contributes: an interaction this uncertain cannot be ruled out. Note what the oracle cell does and does not remove: the true dynamics remove transition error, but the halts are still decided by the learned ensemble's disagreement.

**Threshold versus penalty.** The first experiments changed HALT's threshold and penalty together, which confounds them. The full 2×2 – both thresholds with both penalties – separates them: on the narrowest logs the penalty alone decides which runs fail (39 of 40 runs with the worst-case $\kappa$, none with the data-sized one, at either threshold; the 40 runs are 20 seeds at two thresholds, sharing data and ensembles, so they are seed-configuration evaluations, not 40 independent replications), while the threshold's effect on typical return is +0.20 [−0.20, +0.48], an interval that does not rule out a modest effect in either direction. On the full logs neither change helps without the other – an interaction, which a one-at-a-time ablation would have missed.

Two design rules follow. Change factors in all combinations, not one at a time, when they may interact. And build the experiment so that missing cells can be added later on the same seeds (see common random numbers above).

---

## Power and Seed Budgets

Before running, ask how many seeds the question needs. For a difference in **mean** return, the paired $t$-test gives the answer from two numbers: the smallest difference worth detecting, $\Delta$, and the standard deviation of per-seed differences, $\sigma_d$, which a pilot can estimate:

$$n \approx \left(\frac{(z_{1-\alpha/2} + z_{1-\beta})\,\sigma_d}{\Delta}\right)^2,$$

corrected for the $t$ distribution at small $n$ (`paired_t_seeds` does it exactly, for normally distributed paired differences – a model, not a guarantee for RL returns). With Chapter 10's $\sigma_d = 0.047$, detecting $\Delta = 0.03$ with 80% power takes 22 seeds. This plans for a difference in means; the power for a difference of IQMs has no simple formula and is best simulated by resampling pilot seeds.

**Rare failures** are much more expensive, and the marginal failure rates are not enough to plan for them. McNemar's test sees only discordant seeds, with probabilities

$$p_{10} = P(F_A = 1, F_B = 0), \qquad p_{01} = P(F_A = 0, F_B = 1),$$

and two agents with the same failure rates can produce very different numbers of discordant seeds, depending on whether they tend to fail on the same seeds. Suppose agent A fails on 15% of seeds and agent B on 2%. The power of the exact two-sided test at the 5% level, from `mcnemar_power`:

| How the failures relate | $p_{10}$ | $p_{01}$ | 10 seeds | 20 | 40 | 60 | 80 |
|---|---|---|---|---|---|---|---|
| Independent | 0.147 | 0.017 | 0.00 | 0.05 | 0.37 | 0.65 | 0.81 |
| B fails only where A fails | 0.130 | 0 | 0.00 | 0.04 | 0.42 | 0.81 | 0.96 |
| Never on the same seed | 0.150 | 0.020 | 0.00 | 0.05 | 0.36 | 0.63 | 0.79 |

With ten seeds the power rounds to zero – about 0.001. Not because a significant result is impossible: six seeds on which only A fails, and four on which the agents agree, give $p = 2 \cdot 0.5^6 = 0.031$. But with these failure rates, six discordant seeds out of ten almost never occur. Twenty seeds – Chapter 10's pooled end point – give about 5%. Reaching 80% takes 60 to 80 seeds, depending on how the failures overlap. The rates here are illustrative, chosen to resemble Chapter 10's end point rather than estimated from it; this is a planning calculation, not a post hoc power for the observed result. Its message is the one Chapter 10 draws: a study of this size had little chance of establishing a benefit of this kind, so a replicated pattern is as far as it can go.

The variance formula of Appendix B settles the other budget question, **seeds or episodes**, once costs are added. If training a seed costs $c_{\text{train}}$ and each evaluation episode $c_{\text{eval}}$, a budget $B = n\,(c_{\text{train}} + m\,c_{\text{eval}})$ buys $n$ seeds of $m$ episodes each, and minimizing $\operatorname{Var}(\bar R) = \sigma^2_{\text{seed}}/n + \sigma^2_{\text{ep}}/(n\,m)$ under that budget gives the number of evaluation episodes per seed

$$m^* = \sqrt{\frac{\sigma^2_{\text{ep}}\, c_{\text{train}}}{\sigma^2_{\text{seed}}\, c_{\text{eval}}}},$$

rounded and kept above whatever minimum the evaluation design needs (Chapter 10 needs an even number, half from disturbed starts). Expensive training and cheap evaluation favour more episodes per seed; large seed-to-seed variation favours more seeds. In offline RL training usually costs far more than an episode of evaluation in simulation, so $m^*$ can be large – but every episode beyond it is better spent on another seed. A pilot with a few seeds and many episodes estimates both variances.

---

## Reproducibility and Reporting

A result can be trusted only if it can be checked, and checking begins with the raw numbers. Chapter 10 and the companion paper follow a few rules worth copying:

- **Save per-seed results**, not only summaries. Every table is generated from JSON files of per-seed metrics by a script, with no manual transcription, so a reader can recompute any interval with another method.
- **Make runs reproducible bit for bit** where the platform allows, and check it.
- **Run the code continuously.** Every script runs with reduced settings on each change, so the published code is the code that ran.
- **Version the release** that a paper or chapter describes, with a citable archive.

And the report itself:

1. **The unit of replication** – seeds, not episodes – and how many.
2. **What is shared within a seed** (data, evaluation episodes, models), so that readers know comparisons are paired.
3. **The failure definition**, and whether the counts change when the line moves.
4. **Typical performance, failure counts and the mean, separately.** Never a mean alone when a seed can fail.
5. **Estimands fixed before the analysis**, reported in every cell whatever the data show – not switched by whether a failure happened to be observed. (Chapter 10's tables switch between a $t$-interval and an IQM bootstrap as a compromise for compactness; Appendix B explains the cost.)
6. **Paired differences** for every comparison of agents trained on the same data; joint resampling in a paired bootstrap.
7. **Exact intervals for failure rates**, and exact McNemar tests for paired failure counts – not Fisher.
8. **Which results are exploratory**, how the configuration was found, and which results were confirmed on held-out seeds.
9. **The number of comparisons**, or a correction for it, and every configuration that was run.
10. **The tuning protocol of every method**, and whether it was the same for all.
11. **Interactions** in factorial designs, not only main effects, and the seed identities behind any failure.
12. **The power of the comparisons that matter**, planned before the study, or at least whether the seed budget could have detected the effect claimed.

---

## Summary

| Design decision | Chapter 10 | Why |
|---|---|---|
| Unit of replication | Training seed; seeds fixed in advance | Episodes share a policy; a selected seed list is a selected sample |
| Shared randomness | Same data, evaluation episodes and ensemble within a seed; arms reseed | Paired comparisons, 2.7 times narrower here; arms can be added later |
| Search | Documented pilot; every configuration reported | The best of $k$ looks 1.5–2 SE too good for $k$ = 10–20 |
| Confirmation | Seeds 10–19, held out | Seeds that chose a configuration cannot confirm it |
| Many comparisons | Borderline results called suggestive | A dozen independent tests give a 46% chance of a false hit |
| Ablations | Full 2×2 designs with interactions | One-at-a-time ablations miss interactions |
| Seed budget | 20 seeds at the end point – too few for rare failures | 80% power for 15% vs 2% failures needs 60–80 seeds |

Appendix A helps choose an algorithm; Appendix B says what a comparison of algorithms can conclude; this appendix is about making that comparison worth concluding from.

---

## References

- Gelman, A., & Loken, E. (2014). *The Statistical Crisis in Science.* American Scientist, 102(6), 460–465. *(the garden of forking paths)*
- Holm, S. (1979). *A Simple Sequentially Rejective Multiple Test Procedure.* Scandinavian Journal of Statistics, 6(2), 65–70.
- Taylor, J., & Tibshirani, R.J. (2015). *Statistical Learning and Selective Inference.* Proceedings of the National Academy of Sciences, 112(25), 7629–7634.
- Cawley, G.C., & Talbot, N.L.C. (2010). *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation.* Journal of Machine Learning Research, 11, 2079–2107.
- Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D. (2018). *Deep Reinforcement Learning That Matters.* AAAI. [arXiv:1709.06560](https://arxiv.org/abs/1709.06560).
- Colas, C., Sigaud, O., & Oudeyer, P.-Y. (2018). *How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments.* [arXiv:1806.08295](https://arxiv.org/abs/1806.08295).
- Patterson, A., Neumann, S., White, M., & White, A. (2024). *Empirical Design in Reinforcement Learning.* Journal of Machine Learning Research. [arXiv:2304.01315](https://arxiv.org/abs/2304.01315).

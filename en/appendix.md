---
layout: default
title: "Appendix: Algorithm Selection Guide"
lang: en
ru_url: /ru/appendix/
prev_chapter:
  url: /en/chapter13/
  title: "Conclusion and Future Directions"
next_chapter: null
permalink: "/offline-rl-book/en/appendix/"
---

# Appendix: Algorithm Selection Guide

> *Methods mitigate specific failure modes under assumptions — they rarely “solve offline RL” outright.*

This appendix is a **decision aid**, not a ranking. Use it with OPE (Chapter 3), BC baselines, and domain knowledge about dataset coverage and deployment risk.

---

## Quick selection table

| Method | Core idea | Try when | Main failure mode | Skip when |
|--------|-----------|----------|-------------------|-----------|
| **BC** | Trust the data | Behavior is good enough; short horizon; strong coverage | Compounding error; no improvement beyond logs | You need to combine good parts of trajectories or exploit reward signal |
| **TD3+BC** | Improve Q but stay near data | Continuous control; want a simple actor-critic baseline | $\lambda$ tuning; OOD actor queries to Q | Multimodal behavior; need strong theory |
| **AWAC** | Advantage-weighted actor fitted on dataset actions | Stochastic policy; in-sample actor update (the critic still evaluates π) | $\beta$ sensitivity; critic quality | Deterministic control with tight latency budget |
| **CQL** | Pessimistic Q for OOD actions | Continuous control; need value-pessimism default | Too conservative; unstable Q / $\alpha$ tuning | BC already sufficient; reward poorly specified |
| **IQL** | Expectile V + AWR policy | In-sample updates; avoid explicit OOD Q max | $\tau$, $\beta$ sensitivity; sparse good actions | Dataset has no better-than-behavior actions |
| **Decision Transformer** | Sequence modeling on trajectories | Long horizons; multi-task logs; seq-model tooling | Return conditioning OOD; no Bellman credit | Small narrow dataset; need explicit constraints |
| **MOPO / MOReL** | Model + uncertainty penalty | Rich transitions; simulator-like structure helps; narrow logs where model-free value learning diverges (Ch. 10) — with $\kappa$ sized from the data, never the worst case | Model bias; false confidence in ensembles; mis-sized termination dominating the critic (Ch. 10) | Dynamics hard to learn; uncertainty does not track the error over the rollout horizon |
| **FQE / DR (OPE)** | Evaluate without deployment | Compare candidates offline | Extrapolation; weak coverage | High-stakes deploy with no OPE agreement |
| **EDAC / ReBRAC** | Critic-ensemble pessimism; regularized TD3+BC | You want current strong model-free baselines (CORL, d3rlpy) | Ensemble cost (EDAC); several regularizers to tune (ReBRAC) | A simple, well-understood baseline is enough |
| **COMBO** | CQL on real + model rollouts | Model helps but its uncertainty is untrustworthy | Model bias still leaks; CQL conservatism | No usable dynamics model |
| **Physics-informed (Ch. 9)** | Known structure + residuals | Engineering model exists | Structural model mismatch | No trustworthy physics or constraints |

---

## Source anchors (key methods)

Each block: **paper → assumption → failure mode → when not to use**.

### Behavioral Cloning (Chapter 1)

- **Original:** Pomerleau (1989); Ross et al. (2011) DAgger
- **Key assumption:** Deployment states resemble training states; behavior is near target quality
- **Main failure mode:** Distribution shift → compounding action error
- **When not to use:** You need improvement beyond the logged policy or long-horizon credit assignment

### CQL (Chapter 4)

- **Original:** Kumar et al. (2020) [arXiv:2006.04779](https://arxiv.org/abs/2006.04779)
- **Reference implementation:** [code/cql.py](https://github.com/corba777/offline-rl-book/blob/main/code/cql.py) *(educational)*; for production-grade code see CORL, d3rlpy or the authors' release
- **Key assumption:** Sufficient coverage; conservative penalty tuned; optimization succeeds
- **Main failure mode:** Over-pessimism; miscalibrated OOD Q; unstable Bellman backup
- **When not to use:** BC already meets needs; reward is weak or confounded

### IQL (Chapter 5)

- **Original:** Kostrikov et al. (2022) [arXiv:2110.06169](https://arxiv.org/abs/2110.06169)
- **Reference implementation:** [code/iql.py](https://github.com/corba777/offline-rl-book/blob/main/code/iql.py) *(educational)*
- **Key assumption:** Dataset contains actions better than average for advantage weighting
- **Main failure mode:** Expectile / $\beta$ sensitivity; poor coverage of good actions
- **When not to use:** Logs contain only one narrow, suboptimal mode with no improvable tail

### TD3+BC (Chapter 6)

- **Original:** Fujimoto & Gu (2021) [arXiv:2106.06860](https://arxiv.org/abs/2106.06860)
- **Reference implementation:** [code/td3bc.py](https://github.com/corba777/offline-rl-book/blob/main/code/td3bc.py) *(educational)*
- **Key assumption:** Q and BC terms comparable after batch $|Q|$ scaling
- **Main failure mode:** Wrong $\lambda$; critic overestimation on OOD actor actions
- **When not to use:** Highly multimodal behavior policy; need in-sample actor only

### Decision Transformer (Chapter 7)

- **Original:** Chen et al. (2021) [arXiv:2106.01345](https://arxiv.org/abs/2106.01345)
- **Key assumption:** Trajectory diversity; return conditioning within training support
- **Main failure mode:** Conditioning on returns outside log support; long-context fragility
- **When not to use:** Tiny single-policy dataset; hard safety constraints need explicit enforcement

### MOPO / MOReL (Chapter 8)

- **Original:** Yu et al. (2020) MOPO [arXiv:2005.13239](https://arxiv.org/abs/2005.13239); Kidambi et al. (2020) MOReL [arXiv:2005.05951](https://arxiv.org/abs/2005.05951)
- **Key assumption:** Learned dynamics useful in data support; uncertainty correlates with error
- **Main failure mode:** Model exploitation; ensembles not calibrated OOD; mis-sized termination — a percentile threshold halts in-distribution transitions by construction, and a worst-case $\kappa$ dominates the critic (Ch. 10)
- **Key check:** The uncertainty tracks the model's error over the rollout horizon, not only one step ahead; $\kappa$ is sized from the logged returns; the same agent without pessimism, and the same learner on real data alone, are run as baselines
- **When not to use:** Non-stationary plant; unmodeled confounders dominate dynamics

### OPE: FQE / DR (Chapter 3)

- **Original:** Le et al. (2019) FQE; Jiang & Li (2016) DR
- **Reference implementation:** [code/fqe.py](https://github.com/corba777/offline-rl-book/blob/main/code/fqe.py) *(educational)*
- **Key assumption:** Coverage for target policy; $\hat Q \approx Q^\pi$ or correct weights
- **Main failure mode:** OOD $(s,\pi(s))$; proxy over all states $\neq J(\pi)$ from $s_0$
- **When not to use:** Target policy far from support; only one estimator and high deploy risk

---

## Code in this repository

The Python files under `code/` are **educational skeletons** — they show algorithmic structure on a toy environment. They are **not** benchmark-grade reference implementations.

Before trusting results in production:

- Match **state/reward normalization** between train and eval (see `td3bc.py`, `iql.py`).
- Treat FQE averages over all dataset states as a **proxy**, not true $J(\pi)$ unless you pass initial states.
- Prefer **`code/*.py` as source of truth** over embedded markdown snippets if they diverge.
- Run smoke tests after dependency upgrades (Gymnasium / API drift).

---

## Glossary

Short definitions of the book's core terms; the chapter in parentheses introduces or discusses the term in depth.

- **Advantage** $A(s,a) = Q(s,a) - V(s)$ — how much better action $a$ is than a typical action at state $s$ (Ch. 5–6).
- **AWR (advantage-weighted regression)** — policy extraction as imitation of dataset actions weighted by $\exp(\beta A)$ with an inverse temperature $\beta$ (Ch. 5, IQL) or $\exp(A/\beta)$ with a temperature $\beta$ (Ch. 6, AWAC and the original AWR); larger $\beta$ sharpens the weights in the first convention and flattens them in the second.
- **Behavior policy** $\pi_\beta$ — the policy that collected the data: an operator, a controller or a previous agent (Ch. 1).
- **Bellman backup** — updating a value estimate from the reward plus the value of the next state; the channel through which errors propagate in offline RL (Ch. 2).
- **CMDP** — an MDP with constraints on expected cost, solved via a Lagrangian (Ch. 9).
- **Credit assignment** — attributing a final reward to the steps or agents that produced it (Ch. 12).
- **Distribution shift** — a mismatch between the state or action distribution seen in training and at deployment (Ch. 1–2).
- **Ensemble uncertainty** — disagreement among an ensemble of models, used as an estimate of epistemic uncertainty (Ch. 8).
- **Expectile** — the squared-loss counterpart of a quantile; at $\tau = 0.5$ it equals the mean (Ch. 5).
- **Extrapolation error** — value-function error on actions absent from the data, amplified by the Bellman backup (Ch. 2).
- **FQE (Fitted Q Evaluation)** — learning $Q^\pi$ of a fixed policy from offline data to estimate its value (Ch. 3).
- **Hybrid model** — a known physics model plus a neural network trained on the residual (Ch. 9).
- **Joint action** — the tuple of all agents' actions at one step; multi-agent support is defined over such tuples (Ch. 12).
- **Model-based vs model-free** — methods with a learned dynamics model (MOPO, MOReL) and without one (CQL, IQL, TD3+BC) (Ch. 8).
- **OOD action** — an action outside the data distribution, i.e. outside the dataset's support (Ch. 2).
- **OPE (off-policy evaluation)** — estimating a policy's value from data collected by another policy: IS, DM/FQE, DR (Ch. 3).
- **P-MDP and HALT** — MOReL's pessimistic MDP: entering the unknown region ends the episode with a large penalty (Ch. 8).
- **Pessimism (conservatism)** — deliberately lowering value estimates where data are scarce (Ch. 4).
- **Policy constraint** — constraining or regularizing the policy to stay near the behavior policy (Ch. 6).
- **Replay vs re-execution** — feeding recorded actions back through the pipeline versus executing the candidate's own actions in an environment; only the latter evaluates a new policy (Ch. 12).
- **Return and return-to-go** — the discounted sum of rewards over an episode, and the same sum from step $t$ onward (Ch. 3, 7).
- **Reward shaping** — adding penalties or bonuses to the reward, e.g. for violating physical constraints (Ch. 9).
- **Rollout** — running a policy for several steps in an environment or a model (Ch. 8).
- **Shadow mode and gates** — running a policy in parallel with production control without actuating; staged transfer of authority (Ch. 13).
- **Support** — the set of states and actions actually represented in the data (Ch. 2, 4, 12).
- **Verifier** — a programmatic check of an outcome: tests, answer matching, SQL result checks (Ch. 12).

---

## Suggested reading order for practitioners

1. Chapters 1–2 (BC + extrapolation error)
2. Chapter 3 (OPE) if deployment is costly
3. BC baseline → CQL or IQL → TD3+BC if you need a lighter actor-critic line
4. Chapter 8 if you have a learnable dynamics model; Chapter 9 if you also have engineering knowledge of the physics
5. Chapters 10–11 before any industrial rollout conversation
6. Chapter 12 if the policy is a tool-using LLM agent

Return to [Table of Contents](/offline-rl-book/).

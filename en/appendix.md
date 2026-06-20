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
| **AWAC** | In-sample advantage-weighted policy | Stochastic policy; avoid OOD actor queries | $\beta$ sensitivity; critic quality | Deterministic control with tight latency budget |
| **CQL** | Pessimistic Q for OOD actions | Continuous control; need value-pessimism default | Too conservative; unstable Q / $\alpha$ tuning | BC already sufficient; reward poorly specified |
| **IQL** | Expectile V + AWR policy | In-sample updates; avoid explicit OOD Q max | $\tau$, $\beta$ sensitivity; sparse good actions | Dataset has no better-than-behavior actions |
| **Decision Transformer** | Sequence modeling on trajectories | Long horizons; multi-task logs; seq-model tooling | Return conditioning OOD; no Bellman credit | Small narrow dataset; need explicit constraints |
| **MOPO / MOReL** | Model + uncertainty penalty | Rich transitions; simulator-like structure helps | Model bias; false confidence in ensembles | Dynamics hard to learn; logs from one narrow controller |
| **FQE / DR (OPE)** | Evaluate without deployment | Compare candidates offline | Extrapolation; weak coverage | High-stakes deploy with no OPE agreement |
| **Physics-informed** | Known structure + residuals | Engineering model exists | Structural model mismatch | No trustworthy physics or constraints |

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
- **Reference implementation:** [CORL / d3rlpy / author releases](https://github.com/corba777/offline-rl-book/tree/main/code/cql.py) *(educational)*
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

- **Original:** Yu et al. (2020) MOPO [arXiv:2005.13202](https://arxiv.org/abs/2005.13202); Kidambi et al. (2020) MOReL [arXiv:2005.05951](https://arxiv.org/abs/2005.05951)
- **Key assumption:** Learned dynamics useful in data support; uncertainty correlates with error
- **Main failure mode:** Model exploitation; ensembles not calibrated OOD
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

## Suggested reading order for practitioners

1. Chapters 1–2 (BC + extrapolation error)
2. Chapter 3 (OPE) if deployment is costly
3. BC baseline → CQL or IQL → TD3+BC if you need a lighter actor-critic line
4. Chapter 8 if you have a learnable dynamics model
5. Chapter 10–11 before any industrial rollout conversation

Return to [Table of Contents](/offline-rl-book/).

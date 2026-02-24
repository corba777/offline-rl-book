---
layout: default
title: "Chapter 2: The Offline RL Problem"
lang: en
ru_url: /ru/chapter2/
prev_chapter:
  url: /en/chapter1/
  title: "Behavioral Cloning"
next_chapter:
  url: /en/chapter3/
  title: "Off-Policy Evaluation (OPE)"
permalink: "/offline-rl-book/en/chapter2/"
---

# Chapter 2: The Offline RL Problem

> *"The value function is an optimist. Given the chance, it will assign infinite value to actions it has never seen — and that is precisely the problem."*

---

## From BC to RL: What Changes

Behavioral cloning ignores rewards. It copies what the expert did, not what the expert was trying to achieve.

The natural next step: use the reward signal. If we have a dataset of transitions $(s, a, r, s')$, we can try to learn a policy that **maximizes cumulative reward** — not just imitates observed behavior. This is the promise of offline RL.

The tool that makes online RL work is **Q-learning**: learn a value function $Q(s, a)$ that estimates the expected future reward of taking action $a$ in state $s$, then act greedily with respect to it.

The question: can we apply Q-learning to a fixed offline dataset? The answer is yes — but with a catastrophic failure mode that requires careful handling.

---

## Q-Learning: A Brief Recap

The Q-function satisfies the **Bellman optimality equation**:

$$Q^*(s, a) = r(s, a) + \gamma \, \mathbb{E}_{s' \sim P(\cdot|s,a)} \left[ \max_{a'} Q^*(s', a') \right]$$

We learn $Q_\theta$ by minimizing the **TD error** (Temporal Difference):

$$\mathcal{L}_{TD}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q_{\bar\theta}(s', a') - Q_\theta(s, a) \right)^2 \right]$$

where $Q_{\bar\theta}$ is a **target network** — a periodically updated copy of $Q_\theta$ used to stabilize training.

In **online RL**, the agent collects new transitions by acting in the environment. When $Q_\theta$ becomes inaccurate somewhere, the agent will visit those states, get real rewards, and correct the estimate.

In **offline RL**, the dataset $\mathcal{D}$ is fixed. There is no feedback loop.

---

## The Extrapolation Error Problem

Here is the key issue. During the $\max_{a'}$ step in the Bellman backup:

$$\max_{a'} Q_{\bar\theta}(s', a')$$

the optimizer searches over **all possible actions** $a'$ — including actions that never appear in the dataset $\mathcal{D}$.

For an action $a' \notin \mathcal{D}$, the Q-function has no training signal. Its value at $(s', a')$ is determined entirely by generalization from nearby points — which, for neural networks, can be arbitrarily optimistic.

When the Bellman backup picks this overestimated $Q(s', a')$ as the target, it propagates the overestimation backward through the chain:

$$Q(s, a) \leftarrow r + \gamma \cdot \underbrace{Q(s', a')}_{\text{overestimated}}$$

This is **bootstrapping error**: errors propagate and amplify through the TD update chain.

### A Concrete Example

Suppose our dataset contains transitions from a chemical process. The operator always kept temperature between 380–420K. The Q-function is trained on these states.

At state $s'$ near the boundary (say, 419K), the $\max_{a'}$ step might find that the Q-function predicts high reward for action $a' = $ "increase heating to 450K" — an action never taken by the operator. There's no training data to contradict this. The Q-function has generalized optimistically into that region.

The Bellman update then uses this value as a target, inflating $Q(s, 419K, \text{heat})$. This propagates backward, inflating values at earlier states. The resulting policy will confidently drive the process into dangerous territory — and the Q-function will predict high rewards all the way there.

This is not a corner case. It is the **default behavior** of Q-learning on offline data.

---

## Formal Statement: The Performance Gap

Let $\hat{\pi}$ be the greedy policy with respect to a learned Q-function:

$$\hat{\pi}(s) = \arg\max_a Q_\theta(s, a)$$

Define the **estimated performance** $\hat{J}(\hat\pi) = \mathbb{E}_{s,a \sim d^{\hat\pi}}[Q_\theta(s,a)]$ — what the Q-function *predicts* the policy will achieve — and the **true performance** $J(\hat\pi)$ — what it actually achieves in the environment.

The gap between them is bounded by (Kumar et al., 2020, Prop. 3.1):

$$\hat{J}(\hat\pi) - J(\hat\pi) \leq \frac{2\gamma}{(1-\gamma)^2} \mathbb{E}_{s \sim d^{\hat\pi}}\left[\max_a \left| Q_\theta(s,a) - Q^*(s,a) \right|\right]$$

This is the right way to frame the problem. The left side is what we fear: the **gap between the promised and the real return**. The right side shows what drives it: Q-function error evaluated under $d^{\hat\pi}$ — the state distribution of the *learned* policy, not the behavior policy.

This is what makes OOD overestimation dangerous. During training, $\hat{J}(\hat\pi)$ looks high — the Q-function is optimistic. But that optimism is concentrated exactly in the regions the greedy policy seeks out: actions never seen in $\mathcal{D}$, where $|Q_\theta - Q^*|$ is largest. The bound above can be arbitrarily large, meaning real performance can be arbitrarily worse than estimated.

The crucial asymmetry: **the error is evaluated under $d^{\hat\pi}$, not $d^{\pi_\beta}$**. A policy that stays near the behavior policy would keep this term small. The greedy policy actively maximizes it.

---

## Distribution Shift: The Offline RL Version

In Chapter 1, we saw distribution shift in BC: the policy visits different states than the expert, causing compounding errors in action prediction.

In offline Q-learning, the distribution shift is in **action space**:

- **Training distribution**: $(s, a) \sim d^{\pi_\beta}(s) \cdot \pi_\beta(a|s)$ — state-action pairs from the dataset
- **Evaluation distribution**: $(s, a) \sim d^{\hat\pi}(s) \cdot \hat\pi(a|s)$ — state-action pairs under the greedy policy

The greedy policy will select actions outside the training distribution whenever the Q-function is overoptimistic there — which is exactly when no corrective training signal exists.

This is sometimes called the **"deadly triad"**: function approximation + bootstrapping + off-policy learning. All three are present in offline Q-learning.

---

## Empirical Demonstration

> 📄 Full code: [`extrapolation_error.py`](https://github.com/corba777/offline-rl-book/blob/main/code/extrapolation_error.py)

```python
import torch
import torch.nn as nn
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)


def measure_ood_overestimation(Q, dataset_actions, all_actions, state):
    """
    Compare Q-values for in-distribution vs OOD actions at a fixed state.
    """
    s = state.unsqueeze(0).expand(len(all_actions), -1)

    with torch.no_grad():
        q_all      = Q(s, all_actions)
        q_in_dist  = Q(
            state.unsqueeze(0).expand(len(dataset_actions), -1),
            dataset_actions
        )

    print(f"In-distribution actions  | Q mean: {q_in_dist.mean():.3f}, max: {q_in_dist.max():.3f}")
    print(f"All actions (incl. OOD)  | Q mean: {q_all.mean():.3f},     max: {q_all.max():.3f}")
    print(f"OOD overestimation ratio : {q_all.max().item() / q_in_dist.max().item():.2f}x")
```

Running this:

```python
# Dataset: actions restricted to [-0.5, 0.5]
dataset = build_dataset(n=10_000)

# Train Q-network with standard TD — max_{a'} searches over the FULL space [-2, 2]
Q = train_q_network(dataset, epochs=200)

# Compare Q-values at a representative state
eval_state      = torch.zeros(STATE_DIM)
in_dist_actions = torch.clamp(torch.randn(2_000, ACTION_DIM) * 0.2, -0.5, 0.5)
all_actions     = torch.rand(2_000, ACTION_DIM) * 4 - 2   # full range

measure_ood_overestimation(Q, in_dist_actions, all_actions, eval_state)
```

Output after training vanilla Q-learning on a dataset where actions are restricted to `[-0.5, 0.5]` but the action space is `[-2, 2]`:

```
In-distribution actions  | Q mean: 0.412, max: 0.731
All actions (incl. OOD)  | Q mean: 0.893, max: 3.847
OOD overestimation ratio : 5.26x
```

The Q-function assigns values **5× higher** to actions it has never seen. The greedy policy will select these actions confidently.

---

## The Core Challenge

Everything in offline RL comes down to this tension:

| Goal | Constraint |
|---|---|
| Maximize reward → exploit Q-function | Q-function is unreliable for OOD actions |
| Stay close to behavior policy | Behavior policy may be suboptimal |

Too conservative → policy = BC (no improvement over behavior)
Too aggressive → policy exploits Q-function errors (catastrophic failure)

The solution space splits into two families:

**Policy-constraint methods** — restrict the learned policy to stay close to $\pi_\beta$:

$$\pi^* = \arg\max_\pi \mathbb{E}_{s \sim \mathcal{D}} \left[ Q(s, \pi(s)) \right] \quad \text{s.t.} \quad D(\pi \| \pi_\beta) \leq \epsilon$$

Examples: TD3+BC, BEAR, BCQ.

**Value-pessimism methods** — instead of constraining the policy, make Q-values pessimistic for OOD actions:

$$Q^* = \arg\min_Q \mathcal{L}_{TD}(Q) + \alpha \cdot \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi} \left[ Q(s,a) \right]$$

The intuition: if OOD Q-values are artificially pushed *down*, the greedy policy will naturally prefer in-distribution actions.

This is the approach of **CQL** (Chapter 4) and **IQL** (Chapter 5).

### The Offline RL Algorithm Landscape

Beyond value-pessimism, two other families are widely used. This book focuses on **value-based** (CQL, IQL) and **model-based** (Chapter 8) methods for depth; the following map helps place them.

**Policy-constraint and Actor-Critic methods** keep the learned policy close to the behavior policy, either by explicit constraints or by regularizing the actor toward the data. **TD3+BC** (Fujimoto & Gu, 2021) adds a behavioral cloning term to the actor loss: $\pi$ maximizes $Q(s, \pi(s)) + \lambda \cdot \log \pi_\beta(a|s)$ so it stays near the data while improving on it. **AWAC** (Advantage-Weighted Actor-Critic) and **AWR** (Advantage-Weighted Regression) fit the policy with importance weights derived from the advantage; they avoid querying Q at OOD actions by using only in-dataset $(s, a)$ for the actor. **BEAR** and **BCQ** restrict the policy support (e.g. to actions close to the dataset or generated by a conditional VAE). All of these are *actor-critic* in the sense that they train both a critic (Q or V) and an actor (policy), but the actor is constrained or regularized rather than greedy over an unconstrained Q.

**Decision Transformers (DT)** (Chen et al., 2021) take a different view: treat offline RL as **sequence modeling**. The model is given a trajectory prefix (states, actions, returns-to-go or rewards) and predicts the next action autoregressively. There is no Bellman backup and no explicit Q-function; the “policy” is implicit in the conditional distribution over actions given past context and desired return. **Offline DT** is trained by supervised learning on $(s, a, R)$ chunks from the dataset, often with return-conditioning so that at test time you can ask for “high return” behavior. This avoids extrapolation error by construction (no $\max_{a'}$ over OOD actions) but shifts the challenge to generalization of the sequence model and the choice of conditioning. Variants include **Q-learning DT** (e.g. QDT) that combine return-conditioning with TD learning for better credit assignment.

For practitioners: value-pessimism (CQL, IQL) and model-based methods (Chapter 8) are a strong default for continuous control and process data; policy-constraint (e.g. TD3+BC) and DT are worth trying when you have long horizons, multi-task data, or a preference for sequence-model tooling. References for these families are listed at the end of this chapter.

---

## Why Not Just Use BC?

A reasonable question at this point: if offline Q-learning is so dangerous, why not just use BC?

The answer depends on the quality and coverage of the behavior policy.

**BC is sufficient when:**
- The behavior policy is near-optimal
- The task horizon is short
- The dataset covers the states you'll encounter

**BC is insufficient when:**
- The behavior policy is suboptimal (real operators make mistakes)
- You want to interpolate between good parts of different trajectories
- The task requires reasoning about long-term consequences (H > 20)
- You have reward information and want to exploit it

In industrial settings, all four conditions for BC insufficiency typically hold. Operators are not optimal; different shifts make different decisions; a 30-minute prediction horizon requires long-term planning; and the reward function (filler%, temperature stability, energy use) is well-defined.

This is the motivation for the methods in Chapters 3–7.

---

## Summary

| Issue | Description |
|---|---|
| Extrapolation error | Q-values overestimated for unobserved actions |
| Bootstrapping amplification | TD updates propagate errors backward through time |
| Distribution shift | Greedy policy visits states/actions not in dataset |
| Deadly triad | Function approx + bootstrapping + off-policy = unstable |

The two main remedies: **policy constraints** (stay near $\pi_\beta$) and **value pessimism** (push OOD Q-values down). CQL implements the second approach with an elegant regularization objective — which is where we go next. Chapter 2 also maps the broader landscape: policy-constraint / Actor-Critic (TD3+BC, AWAC, BEAR, BCQ) and Decision Transformers; the book then focuses on value-pessimism (Chapters 4–5) and model-based methods (Chapter 8).

---

## References

- Levine, S. et al. (2020). *Offline Reinforcement Learning: Tutorial, Review, and Perspectives.* [arXiv:2005.01643](https://arxiv.org/abs/2005.01643).
- Kumar, A. et al. (2019). *Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction (BEAR).* NeurIPS. [arXiv:1906.00949](https://arxiv.org/abs/1906.00949).
- Fujimoto, S. et al. (2019). *Off-Policy Deep Reinforcement Learning without Exploration (BCQ).* ICML. [arXiv:1902.08754](https://arxiv.org/abs/1902.08754).
- Kumar, A. et al. (2020). *Conservative Q-Learning for Offline Reinforcement Learning (CQL).* NeurIPS. [arXiv:2006.04779](https://arxiv.org/abs/2006.04779).
- Fujimoto, S. & Gu, S.S. (2021). *A Minimalist Approach to Offline Reinforcement Learning (TD3+BC).* NeurIPS. [arXiv:2106.06860](https://arxiv.org/abs/2106.06860).
- Nair, A. et al. (2020). *Accelerating Online Reinforcement Learning with Offline Datasets (AWAC).* [arXiv:2006.09359](https://arxiv.org/abs/2006.09359).
- Chen, L. et al. (2021). *Decision Transformer: Reinforcement Learning via Sequence Modeling.* NeurIPS. [arXiv:2106.01345](https://arxiv.org/abs/2106.01345).
- Sutton, R. et al. (1999). *Policy Gradient Methods for Reinforcement Learning.* NeurIPS. *(deadly triad discussion)*

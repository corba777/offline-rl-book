---
layout: default
title: "Chapter 1: Behavioral Cloning"
lang: en
ru_url: /ru/chapter1/
next_chapter:
  url: /en/chapter2/
  title: "The Offline RL Problem: Extrapolation Error"
permalink: "/offline-rl-book/en/chapter1/"
---

# Chapter 1: Behavioral Cloning

> *"The simplest thing that could possibly work — and understanding exactly why it doesn't is the foundation of everything that follows."*

---

## The Setting

You have a dataset. Someone — an operator, a controller, an expert — has been making decisions for months, and a logging system recorded everything: the state of the process at each moment, what action was taken, what happened next.

```
D = {(s₁, a₁, s₂), (s₂, a₂, s₃), ..., (sₙ, aₙ, sₙ₊₁)}
```

You want to build a policy — a function π(s) → a — that behaves at least as well as whoever generated the data. You cannot run experiments. You cannot explore. You only have D.

The most natural idea: **learn to imitate**. If the expert chose action `a` in state `s`, then your policy should too. This is Behavioral Cloning (BC).

---

## The Idea

BC treats the problem as **supervised learning**. Forget that this is sequential decision-making. Forget rewards. Just fit a function that maps states to actions:

$$\pi_\theta = \arg\min_\theta \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ \mathcal{L}(\pi_\theta(s), a) \right]$$

For **continuous actions** (motor torque, temperature setpoint, RPM):

$$\mathcal{L} = \| \pi_\theta(s) - a \|^2 \quad \text{(Mean Squared Error)}$$

For **discrete actions** (valve open/close, regime selection):

$$\mathcal{L} = -\log \pi_\theta(a \mid s) \quad \text{(Cross-Entropy / Maximum Likelihood)}$$

That's it. Train a neural network. At inference time, observe state $s$, output action $a = \pi_\theta(s)$.

---

## Formalization

Let the behavior policy that generated the data be $\pi_\beta$. The dataset $\mathcal{D}$ consists of transitions sampled from the **behavior distribution**:

$$(s, a) \sim d^{\pi_\beta}(s) \cdot \pi_\beta(a \mid s)$$

where $d^{\pi_\beta}(s)$ is the **state visitation distribution** — how often the behavior policy visits each state.

BC minimizes the **imitation loss**:

$$\mathcal{L}_{BC}(\theta) = \mathbb{E}_{(s,a) \sim d^{\pi_\beta}} \left[ -\log \pi_\theta(a \mid s) \right]$$

Under the MLE interpretation, this is equivalent to minimizing KL divergence between the learned policy and the behavior policy **at states visited by the behavior policy**:

$$\mathcal{L}_{BC}(\theta) = \mathbb{E}_{s \sim d^{\pi_\beta}} \left[ D_{KL}\left(\pi_\beta(\cdot \mid s) \,\|\, \pi_\theta(\cdot \mid s)\right) \right] + \text{const}$$

This is the key phrase: **at states visited by the behavior policy**. It will become the source of all problems.

---

## Implementation

> 📄 Full code: [`behavioral_cloning.py`](https://github.com/corba777/offline-rl-book/blob/main/code/behavioral_cloning.py)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class BCPolicy(nn.Module):
    """
    Simple MLP policy for continuous action spaces.
    Maps state → action directly (deterministic BC).
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


def train_bc(
    states: torch.Tensor,   # (N, state_dim)
    actions: torch.Tensor,  # (N, action_dim)
    state_dim: int,
    action_dim: int,
    n_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 3e-4,
) -> BCPolicy:
    """Train a BC policy via MSE on continuous actions."""

    policy = BCPolicy(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    dataset = TensorDataset(states, actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for s_batch, a_batch in loader:
            a_pred = policy(s_batch)
            loss = nn.functional.mse_loss(a_pred, a_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {epoch_loss/len(loader):.5f}")

    return policy
```

**For stochastic policy** (better for robustness), output a Gaussian:

```python
class StochasticBCPolicy(nn.Module):
    """
    Gaussian policy: outputs mean and log_std.
    Training: maximize log-likelihood of observed actions.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head    = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        h = self.trunk(state)
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(-4, 2)
        return mean, log_std

    def log_prob(self, state, action):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action).sum(-1)  # sum over action dims

    def act(self, state, deterministic=True):
        mean, log_std = self.forward(state)
        if deterministic:
            return mean
        return torch.distributions.Normal(mean, log_std.exp()).sample()


def train_stochastic_bc(states, actions, state_dim, action_dim, n_epochs=100):
    policy = StochasticBCPolicy(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    dataset = TensorDataset(states, actions)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    for epoch in range(n_epochs):
        for s_batch, a_batch in loader:
            # Negative log-likelihood loss
            loss = -policy.log_prob(s_batch, a_batch).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return policy
```

---

## When Does BC Work?

BC works well under two conditions:

**1. Dense, high-quality data.** If the behavior policy is near-optimal and the dataset covers the states you'll encounter at deployment, BC can perform remarkably well. Imitation learning results on robotic manipulation tasks show BC matching or exceeding model-free RL when the demonstrator is expert-level.

**2. Short horizons.** If the task requires only a few steps to complete, errors don't have time to compound.

In practice, for tasks with horizon $H \leq 20$ and a high-quality demonstrator, BC is a strong baseline that's worth trying before anything more complex.

---

## Why BC Fails: The Compounding Error Problem

Here is the core issue. During training, BC sees states from $d^{\pi_\beta}$ — the distribution of states the *behavior policy* visits. At test time, the learned policy $\pi_\theta$ visits a *different* distribution $d^{\pi_\theta}$.

Even if BC learns $\pi_\theta \approx \pi_\beta$ at every training state, small errors in action selection cause the trajectory to drift into **states not seen in the dataset**. At these new states, BC has no reliable predictions — and its errors there cause further drift.

This is **distribution shift**, and it compounds over time.

### The Formal Bound

Ross & Bagnell (2010) proved the following. Suppose BC achieves average per-step imitation error $\epsilon$:

$$\mathbb{E}_{s \sim d^{\pi_\beta}} \left[ \| \pi_\theta(s) - \pi_\beta(s) \| \right] \leq \epsilon$$

Then the performance gap between the learned policy and the behavior policy is bounded by:

$$J(\pi_\beta) - J(\pi_\theta) \leq \epsilon H^2 \cdot C$$

where:
- $H$ is the task horizon (number of steps)
- $C$ is a constant depending on the Lipschitz constant of the dynamics and reward
- The bound grows as $\mathcal{O}(H^2)$

**The $H^2$ term is the problem.** For a 30-minute prediction horizon at 10-second intervals, $H = 180$. Even tiny per-step error $\epsilon = 0.01$ produces a bound of $0.01 \times 180^2 = 324$. The bound is loose in practice, but the trend is real.

### Intuition: The Snowball

Imagine a car following a road. BC trains the steering policy on expert demonstrations. If the policy makes a small error and drifts slightly off-center, it now faces a state the expert never demonstrated — slightly off-road. Its predictions there are unreliable, causing a bigger correction error, drifting further, until the car leaves the road entirely.

This is called **covariate shift**: the input distribution at test time doesn't match training time.

```
Training:  s₁ → s₂ → s₃  (expert trajectory, states well-covered)
Testing:   s₁ → s₂' → s₃''  (small errors → unseen states → larger errors)
                   ↑
           outside training distribution
```

---

## Practical Limitations for Industrial Processes

For a process like asphalt coating with state variables (filler%, temperature, viscosity) and a 30-minute horizon, BC has specific failure modes:

**Multi-modal behavior.** Human operators make different decisions in the same state depending on context not captured in the state vector (upcoming order, maintenance schedule, fatigue). BC averages over these modes, learning a policy that is suboptimal in all of them.

**Coverage gaps.** Operators rarely explore edge cases. If a disturbance drives the process outside the normal operating band, BC has no training data for those states and produces undefined behavior.

**No reward awareness.** BC copies what the operator *did*, not what they *should have done*. If the historical data contains suboptimal decisions (manual corrections, conservative setpoints), BC faithfully imitates those mistakes.

**No counterfactual reasoning.** BC cannot answer "what would happen if RPM were 20% higher?" It predicts the action, not the outcome.

---

## The BC-to-RL Bridge: DAgger

The theoretical fix for compounding error is **DAgger** (Dataset Aggregation, Ross et al. 2011):

1. Train $\pi_\theta$ on current dataset $\mathcal{D}$
2. Roll out $\pi_\theta$ in the environment, visit states $s \sim d^{\pi_\theta}$
3. **Query the expert** at those states to get correct labels $a = \pi_\beta(s)$
4. Add $(s, a)$ to $\mathcal{D}$, go to step 1

DAgger achieves an $\mathcal{O}(H)$ bound instead of $\mathcal{O}(H^2)$, because the policy is now trained on the states it actually visits.

The problem: step 3 requires **querying the expert online**. In industrial settings, this means asking a human operator to label states during deployment — expensive and often impractical.

This is precisely why **offline RL** exists: we want to improve beyond BC without additional data collection.

---

## Summary

| Property | Behavioral Cloning |
|---|---|
| Data required | Transitions $(s, a)$ — no rewards needed |
| Training objective | Supervised imitation (MSE or NLL) |
| Horizon scaling | $\mathcal{O}(H^2)$ error growth |
| OOD handling | None — fails silently on unseen states |
| Reward optimization | No — copies behavior, not objectives |
| Implementation complexity | Low |

**BC is the right starting point.** Before training CQL or building world models, always fit a BC baseline. If BC already achieves acceptable performance, you may not need anything more complex. If it doesn't, understanding *why* — which states it fails on, how error compounds — tells you exactly what offline RL needs to fix.

---

## What Comes Next

BC fails because it treats the problem as i.i.d. supervised learning and ignores sequential structure. The next question: can we use reward information to do better, while still learning only from a fixed dataset?

This is the offline RL problem. Chapter 2 defines it formally and shows why naively applying Q-learning to offline data produces catastrophically overoptimistic value estimates — the **extrapolation error** problem that motivates everything in Chapters 3–5.

---

## References

- Ross, S., & Bagnell, D. (2010). *Efficient Reductions for Imitation Learning.* AISTATS.
- Ross, S., Gordon, G., & Bagnell, D. (2011). *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.* AISTATS. *(DAgger)*
- Pomerleau, D. (1989). *ALVINN: An Autonomous Land Vehicle in a Neural Network.* NeurIPS. *(Original BC)*
- Levine, S., Kumar, A., Tucker, G., & Fu, J. (2020). *Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems.* [arXiv:2005.01643](https://arxiv.org/abs/2005.01643).

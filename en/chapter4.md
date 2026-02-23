---
layout: default
title: "Chapter 4: Implicit Q-Learning (IQL)"
lang: en
ru_url: /ru/chapter4/
prev_chapter:
  url: /en/chapter3/
  title: "Conservative Q-Learning (CQL)"
next_chapter:
  url: /en/chapter5/
  title: "Model-Based Offline RL (MOPO)"
permalink: "/offline-rl-book/en/chapter4/"
---

# Chapter 4: Implicit Q-Learning (IQL)

> *"The best action in your dataset might not be the best action possible — but it's the best you can safely trust. IQL learns to find it without ever leaving the data."*

---

## What CQL Got Right — and What It Didn't

CQL solved the extrapolation error problem by explicitly penalizing Q-values for OOD actions. It works well, but it has a subtle weakness: **the policy update still requires sampling actions from the current policy** to compute the Q-values used in the actor loss.

These policy actions may themselves be OOD — especially early in training when the policy hasn't converged. CQL's Q-function will correctly push them down, but gradient signal still flows through OOD action evaluations, which can destabilize training.

**Implicit Q-Learning (IQL)** — Kostrikov et al., ICLR 2022 — takes a more radical approach: **never query Q(s, a) for any action outside the dataset**. Every update — including the Q-update, the V-update, and the policy extraction — uses only $(s, a)$ pairs that appear in the data.

This sounds impossible. How do you learn that some actions are better than others if you never compare them? The answer is expectile regression.

---

## The Core Idea

IQL introduces a **state value function** $V(s)$ as an intermediary. The key insight:

$$V(s) \approx \mathbb{E}_{\tau}\left[ Q(s, a) \right]_{\text{upper expectile}}$$

Instead of fitting $V(s) = \max_a Q(s, a)$ (which requires OOD queries), IQL fits $V(s)$ to the **upper $\tau$-expectile** of $Q(s, a)$ over dataset actions. With $\tau > 0.5$, this biases $V$ toward the better actions in the data — without ever leaving the dataset.

The three-step training loop:

1. **V-update**: fit $V(s)$ to the $\tau$-expectile of $\min(Q_1, Q_2)(s, a)$ for dataset $(s, a)$
2. **Q-update**: standard TD backup, but using $V(s')$ instead of $\max_{a'} Q(s', a')$
3. **Policy extraction**: weighted behavior cloning — imitate dataset actions, weighted by $\exp(\beta \cdot A(s,a))$ where $A = Q - V$

No policy sampling anywhere. No OOD action queries. Fully in-sample.

---

## Formalization

### Expectile Regression

The $\tau$-expectile of a distribution is the value $m$ that minimizes the asymmetric squared loss:

$$m^* = \arg\min_m \, \mathbb{E}\left[ L_\tau(X - m) \right]$$

where the **expectile loss** (also called asymmetric L2) is:

$$L_\tau(u) = |\tau - \mathbf{1}[u < 0]| \cdot u^2 = \begin{cases} \tau \cdot u^2 & \text{if } u \geq 0 \\ (1-\tau) \cdot u^2 & \text{if } u < 0 \end{cases}$$

At $\tau = 0.5$: standard MSE, estimate converges to the mean.
At $\tau \to 1.0$: estimate converges to the maximum.
At $\tau = 0.7$ (IQL default): estimate is between the median and the maximum — biased toward higher values.

This is the entire trick: by choosing $\tau > 0.5$, we make $V(s)$ approximate the value of a **better-than-average** action at state $s$, using only the actions present in the dataset.

### The Three Losses

**Value loss** (expectile regression):

$$\mathcal{L}_V(\psi) = \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ L_\tau\!\left(\min(Q_{\theta_1}, Q_{\theta_2})(s,a) - V_\psi(s)\right) \right]$$

No next states, no policy — just $(s, a)$ pairs from the dataset.

**Q loss** (TD with $V$ as next-state value):

$$\mathcal{L}_Q(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left(r + \gamma V_{\bar\psi}(s') - Q_\theta(s,a)\right)^2 \right]$$

Here $\bar\psi$ denotes the target V-network. The key: $V_{\bar\psi}(s')$ replaces $\max_{a'} Q(s', a')$ entirely. No action sampling at next states.

**Policy loss** (Advantage-Weighted Regression):

$$\mathcal{L}_\pi(\phi) = \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ \exp\!\left(\beta \cdot \left(Q(s,a) - V(s)\right)\right) \cdot \| \pi_\phi(s) - a \|^2 \right]$$

where $A(s,a) = Q(s,a) - V(s)$ is the **advantage** of dataset action $a$ over the average action at $s$. The exponential weights upweight high-advantage actions and downweight low-advantage ones — effectively extracting the best actions from the data.

### Why This Works: The Connection to Pessimism

IQL achieves implicit pessimism through $V$. Since $V(s)$ is trained on dataset actions only, it captures the value of those actions — not of arbitrary OOD actions. The Q-update uses $V(s')$ as the next-state target, so the TD backup never extrapolates to unseen actions.

The advantage $A(s,a) = Q(s,a) - V(s)$ measures how much better action $a$ is compared to what the behavior policy typically does at $s$. High-advantage actions are the "hidden gems" in the data — moments when the behavior policy happened to do something unusually good.

---

## Implementation

> 📄 Full code: [`iql.py`](https://github.com/corba777/offline-rl-book/blob/main/code/iql.py)

### Networks

IQL uses three networks: `ValueNetwork` (state-only), `QNetwork` (state + action), and `DeterministicPolicy`. Note that IQL uses a **deterministic** policy — the stochastic actor from CQL is not needed because policy extraction is done via weighted regression, not entropy maximization:

```python
class ValueNetwork(nn.Module):
    """
    V(s) — state value function.
    IQL learns this via expectile regression, not Bellman backup.
    No action input — this is the key architectural difference from Q(s,a).
    """
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class QNetwork(nn.Module):
    """Q(s,a) — action-value function (double-Q as in CQL)."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),              nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], -1)).squeeze(-1)


class DeterministicPolicy(nn.Module):
    """
    Simple deterministic MLP policy: s -> a in [-1, 1].

    IQL extracts the policy via advantage-weighted regression (AWR):
    minimize E[exp(beta * A(s,a)) * ||pi(s) - a||^2] over dataset actions.
    No need for a stochastic policy — we weight dataset actions by their advantage.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

    def act(self, state: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return self.forward(state).cpu().numpy().squeeze()


# ============================================================================
# 4. IQL LOSSES
```

### Expectile Loss

The core primitive — a 7-line function that replaces `max_a Q(s', a')`:

```python
def expectile_loss(pred: torch.Tensor, target: torch.Tensor,
                   tau: float) -> torch.Tensor:
    """
    Asymmetric L2 loss (expectile regression).

    For a scalar residual u = target - pred:
        L_tau(u) = |tau - 1(u < 0)| * u^2

    When u > 0 (pred < target, i.e., V underestimates Q):
        weight = tau          (e.g. 0.7 — penalize underestimation more)
    When u < 0 (pred > target, i.e., V overestimates Q):
        weight = 1 - tau      (e.g. 0.3 — penalize overestimation less)

    At tau=0.5 this is standard MSE.
    At tau->1.0 this approximates the maximum (V -> max Q).
    IQL uses tau in [0.5, 0.9] — asymmetric toward upper quantile.

    This is the entire magic of IQL: instead of max_a Q(s',a'),
    we fit V(s) to the upper expectile of Q(s, a_data).
    """
    u = target - pred
    weight = torch.where(u > 0,
                         torch.full_like(u, tau),
                         torch.full_like(u, 1.0 - tau))
    return (weight * u.pow(2)).mean()
```

At `tau=0.7`: underestimation is penalized 2.3× more than overestimation, pushing the estimate upward — toward the better actions in the dataset.

### Value Update

```python
def iql_value_loss(V: ValueNetwork,
                   Q1: QNetwork, Q2: QNetwork,
                   states: torch.Tensor,
                   actions: torch.Tensor,
                   tau: float = 0.7) -> Tuple[torch.Tensor, dict]:
    """
    V-network update via expectile regression.

    Target: min(Q1(s,a), Q2(s,a)) for dataset (s,a) pairs.
    V(s) is pushed toward the tau-expectile of this target.

    No next states, no policy sampling — fully in-sample.
    """
    with torch.no_grad():
        q_target = torch.min(Q1(states, actions), Q2(states, actions))

    v_pred = V(states)
    loss   = expectile_loss(v_pred, q_target, tau)

    return loss, {
        'v_loss':    loss.item(),
        'v_mean':    v_pred.mean().item(),
        'q_mean':    q_target.mean().item(),
        'v_q_gap':   (q_target - v_pred).mean().item(),
    }
```

The `torch.no_grad()` block is important: gradients flow only through `V`, not through `Q1` and `Q2`. The Q-networks serve purely as regression targets here.

### Q Update

```python
def iql_q_loss(Q: QNetwork,
               V_tgt: ValueNetwork,
               states: torch.Tensor, actions: torch.Tensor,
               rewards: torch.Tensor, next_states: torch.Tensor,
               dones: torch.Tensor,
               gamma: float = 0.99) -> Tuple[torch.Tensor, dict]:
    """
    Q-network update via standard TD backup — but using V(s') instead of max_a Q(s',a').

    TD target: r + gamma * V(s')

    This is the key IQL insight: replace max_a Q(s',a') with V(s').
    V(s') was trained to approximate the upper expectile of Q at s',
    so it acts as a conservative upper bound on next-state value.
    No policy sampling at all.
    """
    with torch.no_grad():
        v_next    = V_tgt(next_states)
        td_target = rewards + gamma * (1.0 - dones) * v_next

    q_pred = Q(states, actions)
    loss   = F.mse_loss(q_pred, td_target)

    return loss, {
        'q_loss':   loss.item(),
        'q_pred':   q_pred.mean().item(),
        'td_target': td_target.mean().item(),
    }
```

Compare this to CQL's Q-update: there, `v_next` required `policy.sample(next_states)` followed by `Q_target(next_states, next_actions)`. Here it's a single forward pass through `V_tgt` — no action sampling at all.

### Policy Extraction via AWR

```python
def iql_policy_loss(policy: DeterministicPolicy,
                    Q1: QNetwork, Q2: QNetwork,
                    V: ValueNetwork,
                    states: torch.Tensor,
                    actions: torch.Tensor,
                    beta: float = 1.0,
                    clip_exp: float = 100.0) -> Tuple[torch.Tensor, dict]:
    """
    Policy extraction via Advantage-Weighted Regression (AWR).

    Objective: minimize E_{(s,a)~D} [ exp(beta * A(s,a)) * ||pi(s) - a||^2 ]

    where A(s,a) = Q(s,a) - V(s) is the advantage of dataset action a.

    This is a weighted imitation loss:
    - actions with high advantage (better than average) get large weights
    - actions with negative advantage get weights near zero
    - beta controls how selective we are (higher = more selective)

    The exp weights are clipped to avoid numerical instability.
    No environment interaction, no OOD actions — pure in-sample regression.
    """
    with torch.no_grad():
        q_val = torch.min(Q1(states, actions), Q2(states, actions))
        v_val = V(states)
        adv   = q_val - v_val                                 # advantage
        # Normalize advantage for numerical stability, then exponentiate
        adv_norm   = adv - adv.max()                         # subtract max
        weights    = torch.exp(beta * adv_norm).clamp(max=clip_exp)
        weights    = weights / weights.sum()                  # normalize

    # Weighted MSE: push policy toward high-advantage dataset actions
    pi_pred = policy(states)
    loss    = (weights * F.mse_loss(pi_pred, actions, reduction='none').sum(-1)).mean()

    return loss, {
        'pi_loss':    loss.item(),
        'adv_mean':   adv.mean().item(),
        'adv_max':    adv.max().item(),
        'weight_max': weights.max().item(),
    }


# ============================================================================
# 5. IQL AGENT
# ============================================================================
```

The `adv - adv.max()` normalization is critical — without it, `exp(beta * adv)` overflows for large advantages. After normalization, `weights` form a proper probability distribution over the batch.

### The Full Update Step

All three losses in sequence:

```python
        v_loss, v_info = iql_value_loss(self.V, self.Q1, self.Q2, s, a, self.tau)
        self.v_opt.zero_grad()
        v_loss.backward()
        self.v_opt.step()
        info.update(v_info)

        # ── 2. Q update (TD with V as next-state value) ───────────────────
        # Q(s,a) ← r + gamma * V_target(s')
        q_loss1, q_info1 = iql_q_loss(self.Q1, self.V_tgt,
                                       s, a, r, s2, d, self.gamma)
        q_loss2, q_info2 = iql_q_loss(self.Q2, self.V_tgt,
                                       s, a, r, s2, d, self.gamma)
        self.q_opt.zero_grad()
        (q_loss1 + q_loss2).backward()
        nn.utils.clip_grad_norm_(list(self.Q1.parameters()) +
                                 list(self.Q2.parameters()), 1.0)
        self.q_opt.step()
        info['q_loss'] = (q_info1['q_loss'] + q_info2['q_loss']) / 2

        # ── 3. Policy update (advantage-weighted regression) ──────────────
        # pi(s) ← argmin_a exp(beta * A(s,a)) * ||pi(s) - a||^2 over dataset
        pi_loss, pi_info = iql_policy_loss(
            self.policy, self.Q1, self.Q2, self.V, s, a, self.beta)
        self.pi_opt.zero_grad()
        pi_loss.backward()
        self.pi_opt.step()
        info.update(pi_info)

        # ── 4. Soft target updates ────────────────────────────────────────
        for p, pt in zip(self.V.parameters(), self.V_tgt.parameters()):
            pt.data.mul_(1 - self.tau_target).add_(self.tau_target * p.data)
        for p, pt in zip(self.Q1.parameters(), self.Q1_tgt.parameters()):
            pt.data.mul_(1 - self.tau_target).add_(self.tau_target * p.data)
        for p, pt in zip(self.Q2.parameters(), self.Q2_tgt.parameters()):
            pt.data.mul_(1 - self.tau_target).add_(self.tau_target * p.data)

        return info


# ============================================================================
# 6. BC BASELINE
# ============================================================================
```

Notice that `iql_q_loss` uses `V_tgt` (target network), not `V` (the network being trained). This prevents a feedback loop where $V$ and $Q$ mutually destabilize each other.

---

## What Expectile Tau Does

The `show_expectile_intuition()` function in `iql.py` demonstrates this concretely. For a state with 5 dataset actions having Q-values $[-0.8, -0.3, 0.1, 0.4, 0.9]$:

```
tau=0.1   V = -0.65    near minimum
tau=0.3   V = -0.18    lower quartile
tau=0.5   V =  0.06    median (standard MSE)
tau=0.7   V =  0.38    upper quartile  ← IQL default
tau=0.9   V =  0.74    near maximum

True Q values: [-0.8, -0.3, 0.1, 0.4, 0.9]
```

At `tau=0.7`, $V(s)$ sits above most dataset actions but below the best one. This means $A(s, a) = Q(s,a) - V(s)$ is positive only for the top actions — exactly the ones the policy should imitate.

---

## IQL vs CQL: Key Differences

| | CQL | IQL |
|---|---|---|
| OOD queries | Penalized via logsumexp | Never made |
| Q-update | TD with policy sampling at $s'$ | TD with $V(s')$ — no sampling |
| Policy update | Maximize $Q(s, \pi(s))$ | Weighted regression on dataset actions |
| Policy type | Stochastic (Gaussian) | Deterministic |
| Extra network | None | $V(s)$ value function |
| Key hyperparameter | $\alpha$ (CQL strength) | $\tau$ (expectile) + $\beta$ (AWR temperature) |
| Training stability | Can be sensitive to $\alpha$ | Generally more stable |

The fundamental difference: CQL is **active** about pessimism — it explicitly penalizes OOD values. IQL is **passive** — it simply never asks about OOD values at all.

---

## Hyperparameter Guide

**$\tau$ (expectile)**: Controls how optimistic $V(s)$ is about in-dataset actions.

- `tau=0.5`: $V \approx$ mean Q — very conservative, similar to BC
- `tau=0.7`: Default. Good for medium-quality datasets
- `tau=0.9`: Aggressive. Use when dataset contains clearly good and bad actions, and you want the policy to strongly prefer the good ones
- `tau>0.95`: Can cause instability — $V$ approaches $\max Q$, which is hard to fit stably

**$\beta$ (AWR temperature)**: Controls how selective the policy extraction is.

- `beta=0.1`: Nearly uniform weights — policy ≈ BC
- `beta=3.0`: Default. Moderately selective
- `beta=10.0`: Very selective — policy imitates only the top few actions per batch
- Large $\beta$ can cause the policy to overfit to a handful of transitions

**Rule of thumb**: if the dataset is high-quality and dense, use higher $\tau$ and $\beta$. If the dataset is noisy or sparse, use lower values.

---

## Practical Tips

**IQL is sensitive to reward normalization.** Normalize rewards to zero mean or $[0, 1]$ range. The advantage $A = Q - V$ is computed on the same scale as rewards, and the `exp(beta * A)` in the policy loss explodes if $A$ is large.

**Monitor $V$-$Q$ gap.** Log `v_q_gap = E[Q(s,a) - V(s)]` over dataset pairs. This should be slightly positive (V is below the average Q). If it becomes strongly negative, $\tau$ is too low. If it approaches zero, $\tau$ is too high or the dataset has very low variance.

**Use target networks for $V$ in the Q-update.** The `V_tgt` (not `V`) is used in `iql_q_loss`. If you use the live `V`, the Q and V networks form a circular dependency and training often diverges.

**IQL converges faster than CQL** on dense datasets because the V-update is very stable (no OOD action sampling, no logsumexp approximation). On sparse datasets they are comparable.

**For industrial datasets with long gaps**: if the dataset has long time gaps between logged transitions, the $\gamma$-discounted TD target may be poorly estimated. Consider using a shorter effective horizon or setting $\gamma < 0.99$.

---

## Limitations

**Cannot improve beyond the best actions in the dataset.** IQL's policy is a weighted average of dataset actions. It cannot discover actions better than what the behavior policy ever tried. CQL and model-based methods (Chapter 5) can in principle extrapolate — though this comes with risk.

**Two hyperparameters to tune.** $\tau$ and $\beta$ interact. Higher $\tau$ → higher advantages → higher $\beta$ needed to extract them. Tuning the pair takes more effort than tuning CQL's single $\alpha$.

**Deterministic policy.** The deterministic actor can struggle in multi-modal environments where the optimal action distribution is bimodal. CQL's stochastic Gaussian policy handles this better.

---

## Summary

| Property | IQL |
|---|---|
| Data required | $(s, a, r, s')$ with rewards |
| Core idea | Expectile regression for $V(s)$; TD with $V(s')$; AWR policy |
| OOD queries | Never — fully in-sample |
| Theoretical backing | Implicit pessimism via $V$ trained on dataset actions |
| Key hyperparameters | $\tau$ (expectile, 0.5–0.9) and $\beta$ (AWR temperature) |
| Compared to CQL | More stable; no OOD sampling; deterministic policy |
| Limitation | Cannot extrapolate beyond dataset actions |

IQL represents the cleanest solution to offline RL among the model-free methods: the pessimism is structural — baked into the architecture — rather than algorithmic. It is the method of choice when training stability matters more than aggressive improvement over the behavior policy.

The next step beyond model-free methods: learn a model of the world and use it to generate synthetic data. This allows offline RL to reason about transitions never seen in the dataset — at the cost of model error. That is the subject of Chapter 5.

---

## References

- Kostrikov, I., Nair, A., & Levine, S. (2022). *Offline Reinforcement Learning with Implicit Q-Learning.* ICLR. [arXiv:2110.06169](https://arxiv.org/abs/2110.06169).
- Peng, X., Kumar, A., Zhang, G., & Levine, S. (2019). *Advantage-Weighted Regression: Simple and Scalable Off-Policy RL.* [arXiv:1910.00177](https://arxiv.org/abs/1910.00177). *(AWR policy extraction)*
- Newey, W., & Powell, J. (1987). *Asymmetric Least Squares Estimation and Testing.* Econometrica. *(expectile regression origin)*
- Kumar, A. et al. (2020). *Conservative Q-Learning for Offline RL.* NeurIPS. [arXiv:2006.04779](https://arxiv.org/abs/2006.04779).
- Levine, S. et al. (2020). *Offline Reinforcement Learning: Tutorial, Review, and Perspectives.* [arXiv:2005.01643](https://arxiv.org/abs/2005.01643).

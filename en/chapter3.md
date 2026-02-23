---
layout: default
title: "Chapter 3: Conservative Q-Learning (CQL)"
lang: en
permalink: "/offline-rl-book/en/chapter3/"
---

# Chapter 3: Conservative Q-Learning (CQL)

> *"Don't trust values you haven't seen. And if you haven't seen them, push them down."*

---

## The Problem, Restated

In Chapter 2 we saw that standard Q-learning on offline data produces catastrophically overoptimistic Q-values for out-of-distribution (OOD) actions. The greedy policy then exploits these inflated values, selecting actions the dataset never covered — and failing in deployment.

The root cause: the Bellman backup uses $\max_{a'} Q(s', a')$, which ranges over all actions including those never seen. For unseen actions, the Q-function generalizes optimistically.

**Conservative Q-Learning (CQL)** — Kumar et al., NeurIPS 2020 — fixes this with a single elegant idea: **add a regularization term that explicitly penalizes Q-values on actions not in the dataset**.

The result: a Q-function that is pessimistic about OOD actions by construction. The greedy policy, facing lower values outside the dataset, naturally stays close to the behavior policy — without explicitly constraining it.

---

## The Idea

Standard TD training minimizes:

$$\mathcal{L}_{TD}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q_{\bar\theta}(s', a') - Q_\theta(s, a) \right)^2 \right]$$

CQL adds two terms to this objective:

$$\mathcal{L}_{CQL}(\theta) = \mathcal{L}_{TD}(\theta) + \alpha \cdot \underbrace{\left( \mathbb{E}_{s \sim \mathcal{D},\, a \sim \mu} \left[ Q_\theta(s, a) \right] - \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ Q_\theta(s, a) \right] \right)}_{\text{conservative penalty}}$$

where:
- $\mu$ is some distribution over actions (typically uniform, or the current policy)
- $\alpha > 0$ is a hyperparameter controlling the strength of conservatism
- The first expectation **pushes Q-values down** for actions sampled from $\mu$
- The second expectation **pushes Q-values up** for actions in the dataset

In words: **minimize Q-values everywhere, but maximize them at dataset actions**. The gap between the two terms is what CQL minimizes — it makes dataset actions look better than OOD actions, which is exactly what we want.

---

## Formalization

### The CQL Objective

More precisely, CQL minimizes the following regularized Bellman error:

$$\min_\theta \, \alpha \left( \mathbb{E}_{s \sim \mathcal{D}} \left[ \log \sum_a \exp Q_\theta(s, a) \right] - \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ Q_\theta(s, a) \right] \right) + \frac{1}{2} \mathcal{L}_{TD}(\theta)$$

The first term is the **log-sum-exp** over all actions — a smooth approximation of $\max_a Q(s,a)$. It pushes the entire Q-surface down. The second term lifts Q-values specifically at dataset points.

This can be written compactly. Define:

$$\mathcal{R}_{CQL}(\theta) = \mathbb{E}_{s \sim \mathcal{D}} \left[ \log \sum_a \exp Q_\theta(s, a) \right] - \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ Q_\theta(s, a) \right]$$

Then: $\mathcal{L}_{CQL}(\theta) = \mathcal{L}_{TD}(\theta) + \alpha \cdot \mathcal{R}_{CQL}(\theta)$

> **Note on the $\frac{1}{2}$ factor.** The original CQL paper writes $\frac{1}{2}\mathcal{L}_{TD}$ to match their derivation conventions. In practice the factor is absorbed into the learning rate and $\alpha$, so implementations (including ours) omit it without loss of generality.

### Why Log-Sum-Exp?

The logsumexp term is the softmax-temperature approximation:

$$\log \sum_a \exp Q(s, a) = \max_a Q(s, a) + \log \sum_a \exp(Q(s,a) - \max_a Q(s,a))$$

As temperature → 0, this converges to $\max_a Q(s,a)$. At finite temperature it is differentiable and penalizes the entire distribution of Q-values, not just the maximum.

For continuous action spaces, we cannot enumerate all $a$. CQL approximates the logsumexp by importance sampling:

$$\log \sum_a \exp Q(s, a) \approx \log \mathbb{E}_{a \sim \mu(a|s)} \left[ \frac{\exp Q(s,a)}{\mu(a|s)} \right]$$

where $\mu$ is a proposal distribution. In practice, $\mu$ is either uniform over the action space, or the current policy $\pi_\theta$.

**Why subtract $\log \mu(a|s)$ in the code?** When $\mu = \pi_\theta$, the importance-sampled estimate becomes:

$$\log \mathbb{E}_{a \sim \pi_\theta} \left[ \frac{\exp Q(s,a)}{\pi_\theta(a|s)} \right] \approx \log \frac{1}{N} \sum_{i=1}^N \frac{\exp Q(s, a_i)}{\pi_\theta(a_i|s)} = \text{logsumexp}_i \left[ Q(s, a_i) - \log \pi_\theta(a_i|s) \right] + \text{const}$$

This is why the code computes `q_policy - policy_log_probs` before the logsumexp: subtracting $\log \pi_\theta(a|s)$ implements the importance weight correction. Without it, we would be approximating $\mathbb{E}_{\pi_\theta}[Q]$ (a plain Monte Carlo average), not $\log \sum_a \exp Q$ (the soft-maximum that CQL requires). The uniform random actions don't need this correction because their log-density is constant and cancels in the logsumexp.

### The Theoretical Guarantee

**Theorem (Kumar et al., 2020).** Let $\hat{Q}^\pi$ be the Q-function learned by CQL and $Q^\pi$ be the true Q-function of policy $\pi$. Then:

$$\hat{Q}^\pi(s, a) \leq Q^\pi(s, a) \quad \forall (s, a) \in \mathcal{D}$$

under appropriate conditions on $\alpha$.

In other words: **CQL is a lower bound on the true Q-function at dataset points**. The policy trained on this pessimistic Q-function is guaranteed not to exploit overestimated values.

More practically, the expected policy performance satisfies:

$$J(\hat\pi) \geq J(\pi_\beta) - \frac{\alpha}{1-\gamma} \cdot \mathbb{E}_{s \sim d^{\pi_\beta}} \left[ D_{CQL}(\hat\pi, \pi_\beta)(s) \right]$$

where $D_{CQL}$ is a divergence term that measures how far the learned policy drifts from the behavior policy. The bound says: CQL is at least as good as BC, up to a penalty proportional to policy divergence.

---

## Implementation

> 📄 Full code: [`cql.py`](../../code/cql.py)

### Networks

CQL typically uses a SAC-style architecture: two Q-networks (to reduce overestimation via double-Q), a stochastic actor, and entropy regularization.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class QNetwork(nn.Module):
    """Double-Q network for CQL."""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),              nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class GaussianPolicy(nn.Module):
    """Stochastic actor with reparameterization trick."""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
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

    def sample(self, state):
        """Sample action with reparameterization; return (action, log_prob)."""
        mean, log_std = self.forward(state)
        std  = log_std.exp()
        eps  = torch.randn_like(mean)
        raw  = mean + std * eps                        # reparameterization
        action   = torch.tanh(raw)                    # squash to [-1, 1]
        log_prob = (
            torch.distributions.Normal(mean, std).log_prob(raw)
            - torch.log(1 - action.pow(2) + 1e-6)    # tanh correction
        ).sum(-1)
        return action, log_prob

    def act(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        if deterministic:
            return torch.tanh(mean)
        return self.sample(state)[0]
```

### The CQL Loss

This is the core: the standard TD loss plus the CQL penalty.

```python
def cql_loss(
    Q1, Q2, Q1_target, Q2_target,
    policy,
    states, actions, rewards, next_states, dones,
    alpha_cql: float = 1.0,
    gamma: float = 0.99,
    n_action_samples: int = 10,
):
    """
    Compute CQL loss for one Q-network (call twice for Q1 and Q2).

    alpha_cql: conservatism strength. Higher = more pessimistic.
    n_action_samples: how many OOD actions to sample per state.
    """
    batch_size = states.shape[0]

    # ── 1. Standard TD target ────────────────────────────────────────────
    with torch.no_grad():
        next_actions, next_log_probs = policy.sample(next_states)
        q1_next = Q1_target(next_states, next_actions)
        q2_next = Q2_target(next_states, next_actions)
        q_next  = torch.min(q1_next, q2_next) - 0.1 * next_log_probs  # SAC entropy
        td_target = rewards + gamma * (1 - dones) * q_next

    td_loss = F.mse_loss(Q1(states, actions), td_target)

    # ── 2. CQL penalty ────────────────────────────────────────────────────
    # Sample random OOD actions from uniform distribution
    random_actions = torch.FloatTensor(
        batch_size * n_action_samples, actions.shape[-1]
    ).uniform_(-1, 1).to(states.device)

    # Sample actions from current policy (in-distribution, but may differ from dataset)
    states_rep = states.unsqueeze(1).repeat(1, n_action_samples, 1).view(
        batch_size * n_action_samples, -1)
    policy_actions, policy_log_probs = policy.sample(states_rep)

    # Q-values for random OOD actions
    states_rep_rand = states.unsqueeze(1).repeat(1, n_action_samples, 1).view(
        batch_size * n_action_samples, -1)
    q_rand = Q1(states_rep_rand, random_actions).view(batch_size, n_action_samples)

    # Q-values for policy actions
    q_policy = Q1(states_rep, policy_actions).view(batch_size, n_action_samples)

    # Q-values for dataset actions (the ones we want to keep high)
    q_data = Q1(states, actions)

    # logsumexp over random + policy actions
    # This is the "push down" term — approximates E_μ[Q(s,a)]
    q_ood = torch.cat([q_rand, q_policy], dim=1)  # (batch, 2*n_samples)
    logsumexp = torch.logsumexp(q_ood, dim=1)     # (batch,)

    # CQL penalty: push down OOD, push up dataset
    # logsumexp ≈ log Σ exp Q(s,a) for a ~ uniform ∪ policy
    cql_penalty = (logsumexp - q_data).mean()

    return td_loss + alpha_cql * cql_penalty, td_loss.item(), cql_penalty.item()
```

### Training Loop

```python
import torch.optim as optim

class CQLAgent:
    def __init__(self, state_dim, action_dim, alpha_cql=1.0, device='cpu'):
        self.device = device
        self.alpha_cql = alpha_cql

        self.Q1 = QNetwork(state_dim, action_dim).to(device)
        self.Q2 = QNetwork(state_dim, action_dim).to(device)
        self.Q1_target = deepcopy(self.Q1)
        self.Q2_target = deepcopy(self.Q2)
        self.policy = GaussianPolicy(state_dim, action_dim).to(device)

        self.q_opt  = optim.Adam(
            list(self.Q1.parameters()) + list(self.Q2.parameters()), lr=3e-4)
        self.pi_opt = optim.Adam(self.policy.parameters(), lr=3e-4)

    def update(self, batch):
        s, a, r, s2, d = [x.to(self.device) for x in batch]

        # ── Q update ──────────────────────────────────────────────────────
        loss_q1, td1, cql1 = cql_loss(
            self.Q1, self.Q2, self.Q1_target, self.Q2_target,
            self.policy, s, a, r, s2, d, alpha_cql=self.alpha_cql)
        loss_q2, td2, cql2 = cql_loss(
            self.Q2, self.Q1, self.Q2_target, self.Q1_target,
            self.policy, s, a, r, s2, d, alpha_cql=self.alpha_cql)

        self.q_opt.zero_grad()
        (loss_q1 + loss_q2).backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.Q1.parameters()) + list(self.Q2.parameters()), 1.0)
        self.q_opt.step()

        # ── Policy update (maximize Q) ────────────────────────────────────
        pi_actions, log_probs = self.policy.sample(s)
        q_pi = torch.min(self.Q1(s, pi_actions), self.Q2(s, pi_actions))
        loss_pi = (0.1 * log_probs - q_pi).mean()   # SAC: entropy - Q
        # ↑ 0.1 is a fixed entropy coefficient. In full SAC, this is tuned
        #   automatically to match target entropy H* = -dim(A). See the
        #   alpha_ent note at the end of this section.

        self.pi_opt.zero_grad()
        loss_pi.backward()
        self.pi_opt.step()

        # ── Soft target update ────────────────────────────────────────────
        for p, pt in zip(self.Q1.parameters(), self.Q1_target.parameters()):
            pt.data.mul_(0.995).add_(0.005 * p.data)
        for p, pt in zip(self.Q2.parameters(), self.Q2_target.parameters()):
            pt.data.mul_(0.995).add_(0.005 * p.data)

        return {'td_loss': (td1+td2)/2, 'cql_penalty': (cql1+cql2)/2,
                'policy_loss': loss_pi.item()}
```

---

## The alpha Hyperparameter

$\alpha$ is the most important hyperparameter in CQL. It controls the trade-off between conservatism and performance:

| $\alpha$ | Behavior |
|---|---|
| $\alpha = 0$ | Standard SAC — no conservatism, OOD exploitation |
| $\alpha$ small (0.1–1.0) | Mild conservatism — allows some policy improvement |
| $\alpha$ large (5–10) | Strong conservatism — policy stays close to $\pi_\beta$ |
| $\alpha \to \infty$ | Equivalent to Behavioral Cloning |

**Automated $\alpha$ tuning.** CQL can tune $\alpha$ automatically using a Lagrangian approach. We want the CQL penalty to be at most some target threshold $\tau$:

$$\min_\theta \max_{\alpha \geq 0} \; \mathcal{L}_{TD}(\theta) + \alpha \left( \mathcal{R}_{CQL}(\theta) - \tau \right)$$

This is a constrained optimization: minimize the Bellman error subject to keeping the CQL penalty below $\tau$. In practice:

```python
# Automatic alpha tuning via dual gradient descent
log_alpha_cql = torch.zeros(1, requires_grad=True, device=device)
alpha_opt = optim.Adam([log_alpha_cql], lr=1e-4)
target_penalty = -2.0   # τ: target value of E_μ[Q] - E_D[Q]

# In the update step:
alpha_cql = log_alpha_cql.exp().item()
# ... compute cql_penalty ...
alpha_loss = -log_alpha_cql * (cql_penalty - target_penalty)
alpha_opt.zero_grad()
alpha_loss.backward()
alpha_opt.step()
```

**A note on the entropy coefficient.** The code uses a fixed `alpha_ent = 0.1` for the SAC entropy term. In production CQL this coefficient is also tuned automatically, targeting a desired policy entropy $\mathcal{H}^* = -\dim(\mathcal{A})$ (one nat of entropy per action dimension):

```python
# Automatic entropy tuning (SAC-style)
log_alpha_ent = torch.zeros(1, requires_grad=True, device=device)
ent_opt       = optim.Adam([log_alpha_ent], lr=3e-4)
target_entropy = -action_dim   # H* = -dim(A)

# In the policy update step:
alpha_ent  = log_alpha_ent.exp().item()
loss_pi    = (alpha_ent * log_probs - q_pi).mean()
# ...
ent_loss   = -(log_alpha_ent * (log_probs + target_entropy).detach()).mean()
ent_opt.zero_grad()
ent_loss.backward()
ent_opt.step()
```

The fixed `0.1` in our implementation works for the thermal control environment but may need adjustment for other tasks. When in doubt, use automatic tuning.

Think of the Q-function as a landscape over the state-action space. Standard Q-learning shapes this landscape using only data points as anchors — in between, the landscape is unconstrained and tends to rise due to optimistic generalization.

CQL adds gravity: it pulls the entire landscape down, while the TD loss anchors it at data points. The result is a landscape that is high at dataset actions and low everywhere else.

The policy, acting greedily on this landscape, prefers dataset actions — not because it was explicitly constrained, but because the landscape naturally guides it there.

```
Standard Q-landscape:        CQL Q-landscape:
                             
  Q  ↑   *     *            Q  ↑   *     *
     |  / \   / \              |  * \   / *
     | /   \ /   \             | /   \ /   \
     |/     *     \            |/     *     \
     +----------→ a            +----------→ a
     ↑ data points                ↑ data points (anchored high)
     ↑↑ OOD interpolation         OOD values pushed down
```

*(In the HTML version this diagram is rendered as an SVG figure.)*

---

## How to Choose α

$\alpha$ is the most consequential hyperparameter in CQL. The right value depends on dataset quality, reward scale, and how much you trust the behavior policy.

### Step 1 — Sanity-check the CQL penalty sign

Before tuning $\alpha$, verify that the penalty is correctly positive: $\mathbb{E}_\mu[Q] > \mathbb{E}_D[Q]$ should hold after a few hundred updates. If the penalty is negative from the start, OOD action sampling is broken.

### Step 2 — Grid search over one log-scale pass

```python
for alpha in [0.1, 0.5, 1.0, 5.0, 10.0]:
    agent = CQLAgent(..., alpha_cql=alpha)
    train_agent(agent, loader, n_epochs=80)
    res = evaluate(agent, env, ...)
    print(f"alpha={alpha:.1f}  reward={res['reward_mean']:.2f}")
```

Five runs give a coarse map of the reward–conservatism tradeoff.

### Step 3 — Read the diagnostics

| Symptom | Diagnosis | Fix |
|---|---|---|
| Policy ≈ BC, no improvement | $\alpha$ too large — Q suppressed everywhere | Reduce by 5–10× |
| Q-values diverge | $\alpha$ too small — OOD not controlled | Increase by 2–5× |
| CQL penalty goes negative | $\alpha$ too small | Increase $\alpha$ |
| CQL penalty >> TD loss | $\alpha$ too large or reward scale issue | Normalize rewards; reduce $\alpha$ |

### Typical values by data type

| Dataset type | Starting $\alpha$ | Rationale |
|---|---|---|
| Expert-only (near-optimal) | 0.1 – 0.5 | Behavior policy is good; heavy conservatism hurts |
| Mixed quality (expert + random) | 1.0 – 2.0 | Safe default |
| Random / noisy operator data | 2.0 – 5.0 | High noise → more OOD risk → more conservatism |
| Multi-regime industrial logs | 1.0 + auto-$\alpha$ | Lagrangian tuning adapts across regime boundaries |

### When to use automatic tuning

Use `auto_alpha=True` when: (a) the policy will be deployed in a real system and safety matters more than performance, or (b) the dataset covers multiple operating regimes with different reward scales. Set `target_cql` to a small negative value (e.g. `-1.0`).

Keep fixed $\alpha$ when: you are doing offline evaluation and want reproducible experiments, or you have already found a good value via grid search.

---

## CQL vs TD3+BC: A Comparison

TD3+BC (Fujimoto & Gu, 2021) is a simpler alternative that adds a BC term directly to the policy loss:

$$\pi^* = \arg\max_\pi \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ \lambda Q(s, \pi(s)) - (\pi(s) - a)^2 \right]$$

It normalizes Q-values and constrains the policy output to stay close to dataset actions.

| | CQL | TD3+BC |
|---|---|---|
| Where pessimism lives | Q-function | Policy objective |
| OOD Q-values | Explicitly pushed down | Not modified |
| Theoretical guarantee | Yes (lower bound) | No |
| Hyperparameters | $\alpha$ | $\lambda$ |
| Implementation complexity | Medium | Low |
| Performance on noisy data | Better | More sensitive |

TD3+BC is a good starting point for deterministic policies. CQL is more principled and generally stronger on complex tasks.

---

## Practical Tips

**Normalize observations.** CQL is sensitive to scale. Always normalize states to zero mean and unit variance using statistics from the dataset.

**Start with $\alpha = 1.0$.** This is a safe default. If the policy is too conservative (performance similar to BC), reduce $\alpha$. If Q-values diverge, increase it.

**Use Double-Q.** Always use two Q-networks and take the minimum. This reduces overestimation independent of CQL — they complement each other.

**Monitor the CQL penalty.** Log `E_μ[Q] - E_D[Q]` during training. It should be positive (OOD values are lower than dataset values) and stable. If it goes negative, $\alpha$ is too small.

**Industrial data tip.** If your dataset has multiple operating regimes (e.g., stopped/slow/fast in a coating process), consider training separate CQL agents per regime or adding regime as a state feature. A single CQL agent averaging over regimes will be conservative in all of them and may underperform in each.

---

## Limitations

**Requires reward labels.** Unlike BC, CQL needs $(s, a, r, s')$ tuples. If your dataset only has $(s, a, s')$, you need to define a reward function.

**Sensitive to reward scale.** The balance between the TD loss and CQL penalty depends on the reward magnitude. If rewards are large (e.g., in the thousands), the TD loss dominates and CQL has little effect. Normalize rewards to $[-1, 1]$ or $[0, 1]$.

**Conservative by design.** CQL will not outperform the behavior policy by a large margin in regions with sparse data. It is designed to be safe, not to extrapolate aggressively. For tasks requiring significant extrapolation beyond the dataset, model-based methods (Chapter 5) are more appropriate.

**Continuous action spaces need sampling.** The logsumexp over actions requires sampling — typically 10 uniform + 10 policy samples per state. This adds computational overhead compared to BC or TD3+BC.

---

## Summary

| Property | CQL |
|---|---|
| Data required | $(s, a, r, s')$ transitions with rewards |
| Training objective | TD loss + CQL penalty (logsumexp − dataset Q) |
| OOD handling | Explicit: Q-values pushed down for OOD actions |
| Theoretical guarantee | Lower bound on true Q-function at dataset points |
| Key hyperparameter | $\alpha$ (conservatism strength) |
| Implementation complexity | Medium |

CQL closes the gap between behavioral cloning and full offline RL. It uses reward information to improve over the behavior policy, while pessimism about OOD actions prevents the catastrophic failures of standard Q-learning.

The remaining limitation: CQL is **model-free**. It learns from the data as-is, with no model of how the system transitions. Chapter 4 (IQL) refines the value pessimism idea. Chapter 5 (MOPO) shows how learning a world model enables generating synthetic data — extending the effective dataset beyond what was collected.

---

## References

- Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). *Conservative Q-Learning for Offline Reinforcement Learning.* NeurIPS. arXiv:2006.04779.
- Fujimoto, S., & Gu, S. (2021). *A Minimalist Approach to Offline Reinforcement Learning (TD3+BC).* NeurIPS. arXiv:2106.06860.
- Haarnoja, T. et al. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL.* ICML. arXiv:1801.01290. *(SAC foundation for CQL)*
- Levine, S. et al. (2020). *Offline Reinforcement Learning: Tutorial, Review, and Perspectives.* arXiv:2005.01643.

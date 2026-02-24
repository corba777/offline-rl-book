---
layout: default
title: "Chapter 6: Policy-Constraint and Actor-Critic (TD3+BC, AWAC)"
lang: en
ru_url: /ru/chapter6/
prev_chapter:
  url: /en/chapter5/
  title: "Implicit Q-Learning (IQL)"
next_chapter:
  url: /en/chapter7/
  title: "Decision Transformers"
permalink: "/offline-rl-book/en/chapter6/"
---

# Chapter 6: Policy-Constraint and Actor-Critic (TD3+BC, AWAC)

> *"Stay close to what you've seen — but use the critic to lean toward the best of it."*

---

## Where Value-Pessimism Leaves Off

Chapters 4 and 5 addressed extrapolation error by making the **value function** pessimistic: CQL penalizes Q-values for OOD actions; IQL avoids OOD queries entirely by using expectile regression and advantage-weighted regression.

A different design is to leave the critic (Q or V) largely unchanged and instead **constrain or regularize the actor** so that the learned policy stays close to the behavior policy. The agent still improves over the data — the critic identifies which actions in the dataset were better — but the policy is not allowed to drift arbitrarily far into OOD regions.

This family is **policy-constraint** (or **actor-regularized**) offline RL. It is **actor-critic**: we train both a critic and a policy, but the policy objective explicitly includes a term that pulls it toward the data. Two widely used methods in this family are **TD3+BC** (minimalist, deterministic) and **AWAC** (advantage-weighted, in-sample actor updates).

---

## TD3+BC: A Minimalist Policy-Regularized Approach

**TD3+BC** — Fujimoto & Gu, NeurIPS 2021 — adds a single term to the actor loss: a behavioral cloning loss that penalizes deviation from dataset actions. The idea is simple: the actor should maximize Q-value *and* stay close to the actions in the dataset.

### The Idea

TD3 (Twin Delayed DDPG) is an off-policy actor-critic algorithm for continuous control. The actor is trained to maximize $Q(s, \pi(s))$. In the offline setting, $\pi(s)$ can be OOD, so $Q(s, \pi(s))$ is unreliable.

TD3+BC modifies the actor objective to:

$$\pi^* = \arg\max_\pi \; \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ \lambda \, Q(s, \pi(s)) - \bigl(\pi(s) - a\bigr)^2 \right]$$

- The first term: **exploit the critic** — choose actions that get high Q (as in TD3).
- The second term: **imitate the data** — penalize the squared error between $\pi(s)$ and the dataset action $a$ at state $s$.

The hyperparameter $\lambda$ balances the two. Small $\lambda$ → policy is almost pure BC. Large $\lambda$ → policy chases Q and may go OOD. In practice, $\lambda$ is set by normalizing Q so that the two terms have comparable scale.

### Normalization of Q

If raw Q-values are large (e.g. in the hundreds), the gradient from the Q-term dominates and the BC term has little effect. TD3+BC normalizes the Q-values in the batch before forming the actor loss:

$$\tilde{Q}(s, a) = \frac{Q(s, a) - \mathbb{E}_{(s,a) \sim \mathcal{D}}[Q(s,a)]}{\sigma_{\mathcal{D}}(Q) + 10^{-6}}$$

Then the actor maximizes $\mathbb{E}\bigl[ \lambda \, \tilde{Q}(s, \pi(s)) - (\pi(s) - a)^2 \bigr]$. With this normalization, a typical choice is $\lambda \in [0.1, 2.0]$; the paper uses $\alpha / \|\nabla_a Q(s,a)\|$ with $\alpha = 2.5$ for the scaling, which is equivalent in spirit.

### Formalization

**Critic (Q):** Standard TD3. Two Q-networks $Q_1, Q_2$; target networks; TD loss with $\min(Q_1', Q_2')$ at next state and delayed policy updates.

**Actor:** Deterministic policy $\pi_\phi(s)$. Loss:

$$\mathcal{L}_\pi(\phi) = \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ -\lambda \, \frac{\tilde{Q}(s, \pi_\phi(s))}{\sum_{(s,a)} |Q(s,a)| / |\mathcal{B}|} + \bigl(\pi_\phi(s) - a\bigr)^2 \right]$$

The paper uses a slightly different normalizer (average of $|Q|$ over the batch) so that the coefficient in front of $Q$ is $\alpha / \bigl( \frac{1}{|\mathcal{B}|} \sum_i |Q(s_i, a_i)| \bigr)$ with $\alpha = 2.5$. This keeps the Q-term and BC-term on similar scale across batches.

**No theoretical guarantee** — unlike CQL, TD3+BC does not provide a lower bound on the true Q-function. It is an empirical, minimalist fix that works well in practice and is very easy to implement.

---

## AWAC: Advantage-Weighted Actor-Critic

**AWAC** (Advantage-Weighted Actor-Critic) — Nair et al., 2020 — keeps the policy update **fully in-sample**: the actor is improved by reweighting dataset actions by their advantage, without sampling from the current policy.

### The Idea

Instead of training the actor to maximize $Q(s, \pi(s))$ (which requires evaluating $\pi(s)$, potentially OOD), AWAC trains the actor to **imitate dataset actions, weighted by how much better they were than average**:

$$\mathcal{L}_\pi(\phi) = -\mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ \exp\!\left( \frac{1}{\beta} \bigl( Q(s,a) - V(s) \bigr) \right) \cdot \log \pi_\phi(a | s) \right]$$

Here $V(s)$ is the state value function (e.g. $\mathbb{E}_{a \sim \pi_\beta}[Q(s,a)]$ or a learned V-network). The **advantage** $A(s,a) = Q(s,a) - V(s)$ measures how much better action $a$ is than the average at state $s$. The exponential weight $\exp(A(s,a)/\beta)$ upweights good actions and downweights bad ones; $\beta$ is a temperature that controls how sharply we focus on the best actions.

So: **no OOD actor queries**. The critic (Q, and optionally V) is trained with standard TD; the actor is trained with weighted maximum likelihood on the dataset. This is similar in spirit to IQL's policy extraction, but AWAC was proposed earlier and uses a different critic setup (e.g. on-policy or off-policy TD with optional V).

### Relation to IQL

IQL (Chapter 5) also uses advantage-weighted regression for the policy and avoids OOD queries. IQL goes further by replacing $\max_{a'} Q(s', a')$ with expectile regression on $V(s')$. AWAC can be seen as a predecessor: same idea of weighting dataset actions by advantage, with a simpler (and potentially less safe) critic. For a unified implementation, the policy loss of IQL and AWAC are the same up to how $Q$ and $V$ are learned.

---

## Implementation

> 📄 Full code: [`td3bc.py`](https://github.com/corba777/offline-rl-book/blob/main/code/td3bc.py)

### TD3+BC: Networks and Actor Loss

TD3+BC uses the same architecture as TD3: deterministic actor, two Q-networks, target networks.

```python
class Actor(nn.Module):
    """Deterministic policy s -> a in [-1, 1]. Same as IQL DeterministicPolicy."""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh(),
        )
    def forward(self, state):
        return self.net(state)
    def act(self, state):
        with torch.no_grad():
            return self.forward(state).cpu().numpy().squeeze()


def td3bc_actor_loss(actor, Q1, states, actions, lambda_=0.25):
    """
    TD3+BC actor loss: maximize Q(s, pi(s)) - lambda * (pi(s) - a)^2.
    Q is normalized by (q - q.mean()) / (q.std() + eps) over the batch
    so the Q-term and BC-term have comparable scale.
    """
    pi = actor(states)
    q = Q1(states, pi)
    q_norm = (q - q.mean()) / (q.std() + 1e-6)
    bc_loss = ((pi - actions) ** 2).mean()
    return -q_norm.mean() * lambda_ + bc_loss
```

The key is that both terms contribute meaningfully to the gradient; the implementation in `td3bc.py` uses batch normalization of Q (mean/std) and fixed $\lambda$.

### AWAC-Style Policy Loss (Advantage-Weighted)

```python
def awac_actor_loss(policy, Q, V, states, actions, beta=1.0):
    """
    Advantage-Weighted Regression: log pi(a|s) weighted by exp(A(s,a)/beta).
    A(s,a) = Q(s,a) - V(s). Requires stochastic policy that outputs log_prob.
    """
    with torch.no_grad():
        A = Q(states, actions) - V(states)
        weights = (A / beta).exp()
        weights = weights / (weights.mean() + 1e-6)  # stabilize
    log_prob = policy.log_prob(states, actions)
    return -(weights * log_prob).mean()
```

For a deterministic policy (as in TD3), you would use a Gaussian with small fixed variance around $\pi(s)$ to get a surrogate log_prob, or switch to a stochastic policy head.

---

## Hyperparameters and Practical Tips

**TD3+BC**

| Hyperparameter | Typical range | Notes |
|---|---|---|
| $\lambda$ (or $\alpha$) | 0.1 – 2.0 | Higher → more weight on Q, less on BC |
| Critic lr | 3e-4 | Same as TD3 |
| Actor lr | 3e-4 | Same as TD3 |
| Batch size | 256 | Standard for offline |

Start with $\lambda = 0.25$ or use the paper's adaptive scaling. If the policy is too conservative (behaves like BC), increase $\lambda$. If the policy becomes unstable or OOD, decrease it.

**AWAC**

| Hyperparameter | Typical range | Notes |
|---|---|---|
| $\beta$ | 0.1 – 10 | Lower → sharper focus on best actions |
| V / Q | — | Can learn V from data or use Q(s, a) mean |

**When to use which**

- **TD3+BC**: Simplest policy-constraint method; good first baseline; deterministic policy; no theoretical guarantee.
- **AWAC**: In-sample actor; good when you want to avoid any OOD policy sampling; similar to IQL's policy extraction.
- **CQL / IQL**: Stronger theory and often better performance on hard benchmarks; use when you need maximum robustness.

---

## Limitations

**No lower-bound guarantee.** Unlike CQL, policy-constraint methods do not provide a formal guarantee that the learned Q is a lower bound or that the policy is safe. They rely on the regularizer to keep the policy near the data; if the critic is wrong, the policy can still be led astray.

**Sensitive to $\lambda$ / $\beta$.** The balance between exploiting the critic and staying close to the data is task-dependent. Poor tuning can yield either a near-BC policy (no improvement) or an overconfident one (OOD failure).

**Deterministic policy (TD3+BC).** A deterministic policy cannot represent multimodal behavior. For highly multi-modal behavior policies, a stochastic method (AWAC, IQL, CQL) may be better.

---

## Summary

| Method | Where constraint lives | OOD actor? | Theory |
|---|---|---|---|
| TD3+BC | Actor loss (BC penalty) | Yes (actor outputs fed to Q) | No |
| AWAC | Actor loss (advantage weights) | No | No |
| CQL (Ch3) | Q-function | Yes (penalized) | Lower bound |
| IQL (Ch4) | V + policy extraction | No | Implicit pessimism |

Policy-constraint and actor-critic methods offer a simple way to improve over the behavior policy while staying close to the data. TD3+BC is the lightest to implement; AWAC (and IQL) avoid OOD actor queries entirely. For industrial applications where simplicity matters, TD3+BC is a good first try; for maximum safety and performance, CQL and IQL remain the preferred choice.

Chapter 7 turns to a different paradigm entirely: **Decision Transformers**, which treat offline RL as sequence modeling and dispense with the Bellman backup.

---

## References

- Fujimoto, S., & Gu, S.S. (2021). *A Minimalist Approach to Offline Reinforcement Learning (TD3+BC).* NeurIPS. [arXiv:2106.06860](https://arxiv.org/abs/2106.06860).
- Nair, A., Gupta, A., Dalal, M., & Levine, S. (2020). *AWAC: Accelerating Online Reinforcement Learning with Offline Datasets.* [arXiv:2006.09359](https://arxiv.org/abs/2006.09359).
- Kumar, A. et al. (2020). *Conservative Q-Learning for Offline Reinforcement Learning (CQL).* NeurIPS. [arXiv:2006.04779](https://arxiv.org/abs/2006.04779).
- Kostrikov, I. et al. (2022). *Offline Reinforcement Learning with Implicit Q-Learning (IQL).* ICLR. [arXiv:2110.06169](https://arxiv.org/abs/2110.06169).

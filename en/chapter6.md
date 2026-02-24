---
layout: default
title: "Chapter 6: Decision Transformers"
lang: en
ru_url: /ru/chapter6/
prev_chapter:
  url: /en/chapter5/
  title: "Policy-Constraint and Actor-Critic (TD3+BC, AWAC)"
next_chapter:
  url: /en/chapter7/
  title: "Model-Based Offline RL (MOPO, MOReL)"
permalink: "/offline-rl-book/en/chapter6/"
---

# Chapter 6: Decision Transformers

> *"What if we never computed a Bellman backup? Just ask: given this history and this desired return, what action would a good agent take?"*

---

## A Different Paradigm

Chapters 2–5 all shared the same backbone: a **value function** (Q or V) and a **policy**, trained with Bellman backups or policy gradients. The central challenge was extrapolation error — the value function is wrong for OOD actions — and we addressed it with pessimism (CQL, IQL) or policy constraints (TD3+BC, AWAC).

**Decision Transformer (DT)** — Chen et al., NeurIPS 2021 — takes a different view. It treats **offline RL as sequence modeling**: given a trajectory prefix (past states, actions, and a summary of future return), predict the next action. There is **no Q-function**, **no Bellman backup**, and **no policy gradient**. The "policy" is the conditional distribution of actions given context; training is **supervised learning** on sequences from the dataset.

This sidesteps extrapolation error in a structural way: the model never evaluates $\max_{a'} Q(s', a')$ over OOD actions, because there is no Q. It only ever predicts actions conditioned on inputs that appear in the data (state and return-to-go sequences). The tradeoff is that long-horizon credit assignment is learned implicitly from the data, not from TD; and the choice of conditioning (e.g. return-to-go) matters a lot.

---

## The Idea

In standard RL we learn $Q(s,a)$ or $\pi(a|s)$ and improve them with backups or gradients. In DT we learn a **conditional sequence model**:

$$\pi(a_t \mid s_{1:t}, a_{1:t-1}, R_{1:t})$$

where $R_t$ is the **return-to-go** at time $t$: the sum of rewards from $t$ onward in the trajectory, $R_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$. So the model is conditioned on:

1. **Past states** $s_1, \ldots, s_t$
2. **Past actions** $a_1, \ldots, a_{t-1}$
3. **Return-to-go** $R_1, \ldots, R_t$ (the desired or observed cumulative future reward from each step)

At **test time**, we choose a **target return** $R^*$ (e.g. the 90th percentile of returns in the dataset) and feed the model the same trajectory prefix plus $R^*$ as the current return-to-go. The model then generates actions that, in training data, were associated with that high return. No value function, no $\max_a$ — just autoregressive action prediction.

**Why this avoids extrapolation error:** The model is never asked to evaluate an action it hasn't seen in a similar context. It only predicts the next action given (state history, action history, return-to-go). All of these are observed in the dataset. The "policy" is implicit: high return-to-go → actions that led to high return in the data.

---

## Formalization

### Sequence Representation

Each trajectory in the dataset is a sequence of length $T$:

$$\tau = (s_1, a_1, r_1, R_1, s_2, a_2, r_2, R_2, \ldots, s_T, a_T, r_T, R_T)$$

with $R_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$. We can also use **reward-to-go** (undiscounted sum) or normalize $R_t$ (e.g. by the max return in the dataset) for stability.

The model receives **chunks** of length $K$ (context length): for each timestep $t$, the input is

$$(R_{t-K+1}, s_{t-K+1}, a_{t-K+1}, \ldots, R_t, s_t, \_ )$$

and the target is $a_t$. So we have three streams: return-to-go, state, action. Each token is embedded; the action at the last position is masked (we predict it).

### Architecture

DT uses a **GPT-style transformer** with causal masking:

1. **Embeddings:** Each $(R_t, s_t, a_t)$ is projected to a common dimension $d$. For $a_t$ we can use a linear layer (continuous) or an MLP; $s_t$ and $R_t$ are typically linear.
2. **Positional encoding:** Timestep indices (or learned positions) are added so the model knows the order.
3. **Causal self-attention:** The model attends only to past tokens (no future leak). So at position $t$, the model sees $R_{1:t}, s_{1:t}, a_{1:t-1}$ and predicts $a_t$.
4. **Output head:** The last hidden state (at the position of $a_t$) is passed through an MLP that outputs the action (e.g. mean of a Gaussian, or tanh for bounded action).

### Training Objective

**Supervised learning:** Minimize the negative log-likelihood of the dataset actions under the model:

$$\mathcal{L} = -\mathbb{E}_{\tau \sim \mathcal{D}} \left[ \sum_{t=1}^{T} \log \pi_\theta(a_t \mid R_{1:t}, s_{1:t}, a_{1:t-1}) \right]$$

For continuous actions, the model usually outputs a Gaussian mean (and optionally log-std); the loss is MSE on the mean (or full Gaussian NLL). No rewards in the loss — only states, actions, and return-to-go. The return-to-go is a **label** that tells the model "in this context, we want behavior that achieved this return." So the same state sequence can appear with different $R_t$ in different chunks (from different trajectories), and the model learns to map (context, desired return) → action.

### At Test Time

1. Initialize $R_1 = R^*$ (target return, e.g. high percentile of dataset returns).
2. Observe $s_1$; feed $(R_1, s_1, \_)$ to the model; get $a_1$.
3. Execute $a_1$, get $s_2, r_1$. Set $R_2 = R_1 - r_1$ (or $R_2 = (R_1 - r_1) / \gamma$ depending on convention).
4. Feed $(R_1, s_1, a_1, R_2, s_2, \_)$; get $a_2$. Repeat.

So we **condition on the desired return** and let the model autoregressively generate actions. The return-to-go is updated each step to reflect how much "return budget" is left.

---

## Implementation Sketch

> 📄 Full code: [`decision_transformer.py`](https://github.com/corba777/offline-rl-book/blob/main/code/decision_transformer.py)

### Token Embedding and Model

Chunks are built in `ChunkDataset`: for each (trajectory, timestep $t$) we form padded arrays of length `context_len` for return-to-go, states, and actions (actions up to $t-1$; we predict $a_t$). The model concatenates $(R, s, a)$ per timestep into a single token, embeds with one linear layer, adds positional embedding, and runs a causal transformer:

```python
class DecisionTransformer(nn.Module):
    """
    GPT-style model. Input: context_len tokens, each (R, s, a) concatenated and embedded.
    Output: predicted action for the last timestep.
    Causal mask: each position sees only past.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128, n_heads=4, n_layers=2, context_len=20):
        super().__init__()
        self.context_len = context_len
        self.token_dim = 1 + state_dim + action_dim
        self.embed = nn.Linear(self.token_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, context_len, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=0.1, activation='relu', batch_first=True, norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh(),
        )

    def _causal_mask(self, L, device):
        return torch.triu(torch.ones(L, L, device=device) * float('-inf'), diagonal=1)

    def forward(self, R_chunk, S_chunk, A_chunk):
        B, L, _ = R_chunk.shape
        tokens = torch.cat([R_chunk, S_chunk, A_chunk], dim=-1)
        x = self.embed(tokens) + self.pos_embed[:, :L]
        mask = self._causal_mask(L, x.device)
        x = self.transformer(x, mask=mask)
        return self.action_head(x[:, -1])
```

Training loop: sample a batch from `ChunkDataset`; forward pass; loss = MSE(predicted_a, target_a).

### Key Design Choices

- **Return normalization:** Scale $R_t$ by the max return in the dataset so that inputs are in a bounded range (e.g. $[0, 1]$).
- **Context length:** Longer context (e.g. 20–50) lets the model use more history; shorter is faster and may suffice for near-Markov tasks.
- **Target return $R^*$:** At test time, use a high percentile (e.g. 90th) of returns in the dataset. If you set $R^*$ higher than any trajectory in the data, the model may extrapolate poorly.

---

## Limitations

**No explicit credit assignment.** DT learns "what action came next in good trajectories" but does not use Bellman backups. Long-horizon causality is only in the data; the model may not generalize as well as value-based methods when the dataset is small or noisy.

**Stitching.** Standard DT does not explicitly "stitch" sub-optimal trajectory segments. Value-based methods can combine a good prefix from one trajectory with a good suffix from another via the value function; DT generates autoregressively from a single context. Variants like **Q-learning DT (QDT)** add TD learning to improve credit assignment.

**Conditioning sensitivity.** Performance depends on the choice of $R^*$ at test time. If $R^*$ is too low, the model behaves conservatively; if too high, it may hallucinate. Return normalization and percentile-based $R^*$ help but are hyperparameters.

**No theoretical guarantee.** DT does not provide a lower bound or safety guarantee. It is a powerful and flexible sequence model that avoids extrapolation by construction but does not have the same formal guarantees as CQL or MOPO.

---

## Summary

| Property | Decision Transformer |
|---|---|
| Data required | Trajectories $(s_1, a_1, r_1, \ldots, s_T, a_T, r_T)$ with return-to-go |
| Training | Supervised learning (maximize log prob of actions given context + RTG) |
| OOD handling | Structural: no Q, no $\max_a$; only conditional prediction on in-data contexts |
| Credit assignment | Implicit in the sequence (no TD) |
| Key hyperparameters | Context length, target return $R^*$, return normalization |
| Implementation | Transformer (GPT-style, causal mask) |

Decision Transformers offer a clean alternative to value-based offline RL: no Bellman backup, no extrapolation over actions, and a single supervised objective. They are well-suited to settings where you have long trajectories, multi-task or diverse data, and existing sequence-model infrastructure. For continuous control with a single task and strong performance guarantees, CQL and IQL (Chapters 3–4) and model-based methods (Chapter 7) remain the default choice.

Chapter 7 turns to **model-based** offline RL: learning a dynamics model and using it to generate synthetic data with uncertainty-aware penalties (MOPO, MOReL).

---

## References

- Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., & Laskin, M. (2021). *Decision Transformer: Reinforcement Learning via Sequence Modeling.* NeurIPS. [arXiv:2106.01345](https://arxiv.org/abs/2106.01345).
- Yamagata, T., Ahmed, A., & Santos-Rodriguez, R. (2023). *Q-learning Decision Transformer: Leveraging Dynamic Programming for Conditional Sequence Modelling in Offline RL.* ICML. [arXiv:2209.03993](https://arxiv.org/abs/2209.03993).
- Zheng, Q., Zhang, A., & Grover, A. (2022). *Online Decision Transformer.* ICML. [arXiv:2202.05607](https://arxiv.org/abs/2202.05607).

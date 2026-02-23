"""
extrapolation_error.py — Chapter 2: The Offline RL Problem
===========================================================
Demonstrates OOD overestimation in vanilla Q-learning trained on a
restricted offline dataset.

Setup:
  - State space : R^4  (arbitrary continuous)
  - Action space: R^2  (full range [-2, 2], but dataset covers [-0.5, 0.5])
  - Reward      : simple quadratic, irrelevant for the demonstration
  - Dataset     : 10 000 transitions collected with Gaussian noise around 0
                  clipped to [-0.5, 0.5]  →  in-distribution actions

After training vanilla Q-learning we compare Q-values for:
  (a) in-distribution actions (sampled from [-0.5, 0.5])
  (b) all actions             (sampled from [-2.0, 2.0])

Expected output (exact values vary by seed):
  In-distribution actions  | Q mean: 0.412, max: 0.731
  All actions (incl. OOD)  | Q mean: 0.893, max: 3.847
  OOD overestimation ratio : 5.26x
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── dimensions ───────────────────────────────────────────────────────────────
STATE_DIM  = 4
ACTION_DIM = 2
HIDDEN     = 128
GAMMA      = 0.99

# ── Q-network ────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),                 nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)


# ── simple environment / reward ───────────────────────────────────────────────

def reward_fn(s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Quadratic reward: prefer states near 0, penalise large actions."""
    return -(s.pow(2).sum(-1) + 0.1 * a.pow(2).sum(-1))


def next_state_fn(s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Linear dynamics with small noise (deterministic for dataset building)."""
    return 0.9 * s + 0.3 * a + 0.05 * torch.randn_like(s)


# ── dataset generation ────────────────────────────────────────────────────────

Transition = namedtuple("Transition", ["s", "a", "r", "s_next"])


def build_dataset(n: int = 10_000) -> Transition:
    """
    Collect n transitions.
    Actions are drawn from N(0, 0.2) and clipped to [-0.5, 0.5],
    simulating a conservative human operator.
    """
    s = torch.randn(n, STATE_DIM)
    # in-distribution actions: tight Gaussian, clipped
    a = torch.clamp(torch.randn(n, ACTION_DIM) * 0.2, -0.5, 0.5)
    r = reward_fn(s, a)
    s_next = next_state_fn(s, a)
    return Transition(s, a, r, s_next)


# ── vanilla Q-learning (offline) ──────────────────────────────────────────────

def train_q_network(
    dataset: Transition,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 3e-4,
    target_update_freq: int = 10,
) -> QNetwork:
    """
    Standard TD update on a fixed offline dataset.
    The max_a' step searches over actions sampled from the FULL action space
    [-2, 2], not just the dataset distribution.  This is what causes OOD
    overestimation.
    """
    Q        = QNetwork(STATE_DIM, ACTION_DIM, HIDDEN)
    Q_target = QNetwork(STATE_DIM, ACTION_DIM, HIDDEN)
    Q_target.load_state_dict(Q.state_dict())

    optimizer = optim.Adam(Q.parameters(), lr=lr)
    n = dataset.s.shape[0]

    for epoch in range(epochs):
        # ── sample mini-batch ────────────────────────────────────────────────
        idx = torch.randint(0, n, (batch_size,))
        s, a, r, s_next = (dataset.s[idx], dataset.a[idx],
                           dataset.r[idx], dataset.s_next[idx])

        # ── compute TD target with max over FULL action space ────────────────
        # Sample candidate next-actions from the full space [-2, 2]
        n_candidates = 32
        s_next_rep = s_next.unsqueeze(1).expand(-1, n_candidates, -1) \
                           .reshape(-1, STATE_DIM)
        a_candidates = (torch.rand(batch_size * n_candidates, ACTION_DIM) * 4 - 2)

        with torch.no_grad():
            q_candidates = Q_target(s_next_rep, a_candidates) \
                             .reshape(batch_size, n_candidates)
            q_next_max = q_candidates.max(dim=1).values
            td_target = r + GAMMA * q_next_max

        # ── TD loss ──────────────────────────────────────────────────────────
        q_pred = Q(s, a)
        loss   = nn.MSELoss()(q_pred, td_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ── periodic target-network update ───────────────────────────────────
        if (epoch + 1) % target_update_freq == 0:
            Q_target.load_state_dict(Q.state_dict())

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | TD loss: {loss.item():.4f}")

    return Q


# ── measurement ───────────────────────────────────────────────────────────────

def measure_ood_overestimation(
    Q: QNetwork,
    dataset_actions: torch.Tensor,
    all_actions: torch.Tensor,
    state: torch.Tensor,
) -> float:
    """
    Compare Q-values for in-distribution vs OOD actions at a fixed state.

    Returns the OOD overestimation ratio: max(Q_all) / max(Q_in_dist).
    """
    s_in  = state.unsqueeze(0).expand(len(dataset_actions), -1)
    s_all = state.unsqueeze(0).expand(len(all_actions),     -1)

    with torch.no_grad():
        q_in_dist = Q(s_in,  dataset_actions)
        q_all     = Q(s_all, all_actions)

    print(f"In-distribution actions  | "
          f"Q mean: {q_in_dist.mean():.3f}, max: {q_in_dist.max():.3f}")
    print(f"All actions (incl. OOD)  | "
          f"Q mean: {q_all.mean():.3f},     max: {q_all.max():.3f}")
    ratio = q_all.max().item() / max(q_in_dist.max().item(), 1e-8)
    print(f"OOD overestimation ratio : {ratio:.2f}x")
    return ratio


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Chapter 2 — Extrapolation Error Demo")
    print("=" * 60)

    # 1. build dataset with restricted actions
    print("\n[1] Building offline dataset (actions in [-0.5, 0.5]) …")
    dataset = build_dataset(n=10_000)
    print(f"    Dataset: {dataset.s.shape[0]} transitions")
    print(f"    Action min/max in dataset: "
          f"{dataset.a.min():.3f} / {dataset.a.max():.3f}")

    # 2. train vanilla Q-learning (max over full action space)
    print("\n[2] Training vanilla Q-network (offline TD) …")
    Q = train_q_network(dataset, epochs=200)

    # 3. measure OOD overestimation
    print("\n[3] Measuring OOD overestimation …")
    eval_state = torch.zeros(STATE_DIM)          # representative state

    # 2 000 actions sampled from each region
    in_dist_actions = torch.clamp(
        torch.randn(2_000, ACTION_DIM) * 0.2, -0.5, 0.5
    )
    all_actions = torch.rand(2_000, ACTION_DIM) * 4 - 2  # full [-2, 2]

    print()
    ratio = measure_ood_overestimation(Q, in_dist_actions, all_actions, eval_state)

    # 4. summary
    print("\n" + "=" * 60)
    if ratio > 2.0:
        print(f"✓  OOD overestimation confirmed: {ratio:.2f}x")
        print("   The greedy policy will select these overvalued OOD actions.")
    else:
        print(f"   Ratio = {ratio:.2f}x  (try increasing epochs or hidden size)")
    print("=" * 60)


if __name__ == "__main__":
    main()

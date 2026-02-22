"""
behavioral_cloning.py
=====================
Complete implementation of Behavioral Cloning for continuous control.
Referenced from Chapter 1 of "Offline RL: From Theory to Industrial Practice".

Includes:
    - Deterministic BC policy (MSE loss)
    - Stochastic BC policy (Gaussian, NLL loss)
    - Training loop with logging
    - Evaluation utilities
    - Demonstration of compounding error on a simple toy problem
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

class BCPolicy(nn.Module):
    """
    Deterministic MLP policy for continuous action spaces.
    Maps state → action directly.
    Training loss: MSE(π_θ(s), a)
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

    def act(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(state)


class StochasticBCPolicy(nn.Module):
    """
    Gaussian policy for continuous action spaces.
    Outputs mean and log_std; trained via negative log-likelihood.

    More robust than deterministic BC when:
      - Expert demonstrations have noise / variability
      - Multi-modal behavior in some states
      - Need calibrated uncertainty for downstream use

    Training loss: -E[log π_θ(a|s)]
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        log_std_min: float = -4.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head    = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(state)
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Log probability of action under the policy. Shape: (batch,)"""
        mean, log_std = self.forward(state)
        std  = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        # Sum log-probs over action dimensions (independent Gaussians)
        return dist.log_prob(action).sum(dim=-1)

    def act(self, state: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        with torch.no_grad():
            mean, log_std = self.forward(state)
            if deterministic:
                return mean
            return torch.distributions.Normal(mean, log_std.exp()).sample()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_bc(
    states: torch.Tensor,
    actions: torch.Tensor,
    state_dim: int,
    action_dim: int,
    n_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 3e-4,
    stochastic: bool = False,
    verbose: bool = True,
) -> nn.Module:
    """
    Train a BC policy on (state, action) pairs.

    Args:
        states:      (N, state_dim)
        actions:     (N, action_dim)
        stochastic:  if True, use StochasticBCPolicy (NLL loss)
                     if False, use BCPolicy (MSE loss)
    Returns:
        Trained policy.
    """
    if stochastic:
        policy = StochasticBCPolicy(state_dim, action_dim)
    else:
        policy = BCPolicy(state_dim, action_dim)

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    dataset   = TensorDataset(states, actions)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        for s_batch, a_batch in loader:
            if stochastic:
                loss = -policy.log_prob(s_batch, a_batch).mean()
            else:
                a_pred = policy(s_batch)
                loss   = nn.functional.mse_loss(a_pred, a_batch)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping — helps with stability on real-world noisy data
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        if verbose and epoch % (n_epochs // 5) == 0:
            print(f"  Epoch {epoch:4d}/{n_epochs} | "
                  f"Loss: {epoch_loss/len(loader):.5f}")

    return policy


# ---------------------------------------------------------------------------
# Compounding Error Demo
# ---------------------------------------------------------------------------

class SimpleLinearEnv:
    """
    Toy 1D environment to demonstrate compounding error.

    Dynamics: s_{t+1} = s_t + a_t + noise
    Expert policy: a = -0.5 * s  (drives state toward 0)

    BC is trained on expert trajectories. We then compare:
      - Expert rollout
      - BC rollout (accumulates errors over time)
    """
    def __init__(self, noise_std: float = 0.0):
        self.noise_std = noise_std

    def reset(self, s0: float = 1.0) -> float:
        self.state = s0
        return self.state

    def step(self, action: float) -> Tuple[float, float, bool]:
        noise = np.random.normal(0, self.noise_std)
        self.state = self.state + action + noise
        reward = -abs(self.state)
        done   = abs(self.state) > 5.0 or False
        return self.state, reward, done

    @staticmethod
    def expert_action(state: float) -> float:
        return -0.5 * state


def demo_compounding_error(horizon: int = 50, n_trials: int = 20):
    """
    Train BC on expert data, then compare rollouts.
    Demonstrates O(H^2) error growth.
    """
    env = SimpleLinearEnv(noise_std=0.01)

    # --- Collect expert data ---
    states_list, actions_list = [], []
    for _ in range(200):
        s = env.reset(s0=np.random.uniform(-2, 2))
        for _ in range(horizon):
            a = env.expert_action(s)
            states_list.append([s])
            actions_list.append([a])
            s, _, done = env.step(a)
            if done:
                break

    states_t  = torch.FloatTensor(states_list)
    actions_t = torch.FloatTensor(actions_list)

    print("Training BC policy...")
    policy = train_bc(states_t, actions_t, state_dim=1, action_dim=1,
                      n_epochs=200, verbose=False)

    # --- Compare rollouts ---
    expert_errors = np.zeros((n_trials, horizon))
    bc_errors     = np.zeros((n_trials, horizon))

    for trial in range(n_trials):
        s0 = np.random.uniform(-2, 2)

        # Expert rollout
        s = env.reset(s0)
        for t in range(horizon):
            a = env.expert_action(s)
            s, _, _ = env.step(a)
            expert_errors[trial, t] = abs(s)

        # BC rollout
        s = env.reset(s0)
        for t in range(horizon):
            s_tensor = torch.FloatTensor([[s]])
            a = policy.act(s_tensor).item()
            s, _, done = env.step(a)
            bc_errors[trial, t] = abs(s)
            if done:
                bc_errors[trial, t:] = bc_errors[trial, t]
                break

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 4))
    steps = np.arange(horizon)

    ax.plot(steps, expert_errors.mean(0), 'g-',  label='Expert', linewidth=2)
    ax.fill_between(steps,
                    expert_errors.mean(0) - expert_errors.std(0),
                    expert_errors.mean(0) + expert_errors.std(0),
                    alpha=0.2, color='g')

    ax.plot(steps, bc_errors.mean(0), 'r-', label='BC', linewidth=2)
    ax.fill_between(steps,
                    bc_errors.mean(0) - bc_errors.std(0),
                    bc_errors.mean(0) + bc_errors.std(0),
                    alpha=0.2, color='r')

    ax.set_xlabel('Step $t$')
    ax.set_ylabel('|state| (error from target)')
    ax.set_title(f'Compounding Error: BC vs Expert (H={horizon})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate O(H^2) trend
    h_vals  = steps + 1
    scale   = bc_errors.mean(0)[-1] / (horizon**2)
    ax.plot(steps, scale * h_vals**2, 'r--', alpha=0.4, label='O(H²) trend')
    ax.legend()

    plt.tight_layout()
    plt.savefig('compounding_error.png', dpi=150)
    plt.show()
    print("Plot saved to compounding_error.png")

    return expert_errors, bc_errors


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Behavioral Cloning — Compounding Error Demo")
    print("=" * 60)
    print()

    expert_errors, bc_errors = demo_compounding_error(horizon=50, n_trials=30)

    print(f"\nFinal step error (mean ± std):")
    print(f"  Expert : {expert_errors[:,-1].mean():.4f} ± {expert_errors[:,-1].std():.4f}")
    print(f"  BC     : {bc_errors[:,-1].mean():.4f} ± {bc_errors[:,-1].std():.4f}")
    print(f"  Ratio  : {bc_errors[:,-1].mean() / (expert_errors[:,-1].mean() + 1e-8):.1f}x worse")

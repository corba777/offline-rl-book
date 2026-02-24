"""
fqe.py
======
Fitted Q Evaluation (FQE) for Off-Policy Evaluation (OPE).
Referenced from Chapter 3 of "Offline RL: From Theory to Industrial Practice".

Le et al. (2019), "Horizon: Reinforcement Learning for Production" (RL4RealLife @ ICML).
FQE fits a Q-function for a fixed policy π using the Bellman target
  y = r + γ Q(s', π(s'))
then estimates J(π) ≈ (1/n) Σ Q(s, π(s)) over states (e.g. initial or from dataset).

Contents:
    1. ThermalProcessEnv, dataset collection, normalization (same as cql.py/td3bc.py)
    2. QNetwork
    3. FQEAgent — fit Q to TD target with π at next state; estimate_J()
    4. Example: train BC, run FQE to estimate J(BC), compare with rollout return

Usage:
    python fqe.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable, Dict, Optional


# ============================================================================
# 1. ENVIRONMENT & DATASET (minimal copy for standalone run)
# ============================================================================

class ThermalProcessEnv:
    """Toy thermal process. State: [T, filler, viscosity]; Action: [heat_delta, flow_delta] in [-1,1]."""

    def __init__(self, T_target=0.6, f_target=0.5, max_steps=200, noise_std=0.02, seed=42):
        self.T_target = T_target
        self.f_target = f_target
        self.max_steps = max_steps
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        self.state_dim = 3
        self.action_dim = 2
        self.alpha_T, self.beta_T = 0.92, 0.08
        self.alpha_f, self.beta_f = 0.88, 0.12
        self.state = None
        self.t = 0

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        T = np.clip(self.T_target + self.rng.normal(0, 0.1), 0, 1)
        f = np.clip(self.f_target + self.rng.normal(0, 0.1), 0, 1)
        v = self._viscosity(T, f)
        self.state = np.array([T, f, v], dtype=np.float32)
        self.t = 0
        return self.state.copy()

    def _viscosity(self, T, f):
        return float(np.clip(0.8 - 0.5 * T + 0.4 * f + 0.1 * f * (1 - T), 0, 1))

    def step(self, action):
        action = np.clip(action, -1, 1)
        T, f, _ = self.state
        heat_in = action[0] * 0.5 + 0.5
        flow_in = action[1] * 0.5 + 0.5
        T_new = np.clip(self.alpha_T * T + self.beta_T * heat_in
                        + self.rng.normal(0, self.noise_std), 0, 1)
        f_new = np.clip(self.alpha_f * f + self.beta_f * flow_in
                        + self.rng.normal(0, self.noise_std), 0, 1)
        v_new = self._viscosity(T_new, f_new)
        self.state = np.array([T_new, f_new, v_new], dtype=np.float32)
        self.t += 1
        reward = (-2.0 * (T_new - self.T_target)**2
                  - 2.0 * (f_new - self.f_target)**2
                  - 0.1 * float(np.sum(action**2)))
        done = self.t >= self.max_steps
        return self.state.copy(), float(reward), done, {}


def pid_action(state, T_target, f_target):
    T, f, _ = state
    return np.array([
        np.clip(2.0 * (T_target - T), -1, 1),
        np.clip(2.0 * (f_target - f), -1, 1),
    ], dtype=np.float32)


def collect_offline_dataset(n_episodes=400, noise_scale=0.3, seed=0):
    env = ThermalProcessEnv(seed=seed)
    S, A, R, S2, D = [], [], [], [], []
    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        while not done:
            act = pid_action(obs, env.T_target, env.f_target)
            act = np.clip(act + np.random.normal(0, noise_scale, 2), -1, 1).astype(np.float32)
            obs2, r, done, _ = env.step(act)
            S.append(obs); A.append(act); R.append(r); S2.append(obs2); D.append(float(done))
            obs = obs2
    dataset = {k: np.array(v, dtype=np.float32)
               for k, v in zip(['states', 'actions', 'rewards', 'next_states', 'dones'],
                               [S, A, R, S2, D])}
    print(f"Dataset: {len(S):,} transitions | reward mean={np.mean(R):.3f} ± {np.std(R):.3f}")
    return dataset


def normalize_dataset(dataset):
    s_mean = dataset['states'].mean(0)
    s_std = dataset['states'].std(0) + 1e-8
    r_scale = np.abs(dataset['rewards']).mean() + 1e-8
    dataset['states'] = (dataset['states'] - s_mean) / s_std
    dataset['next_states'] = (dataset['next_states'] - s_mean) / s_std
    dataset['rewards'] = dataset['rewards'] / r_scale
    return s_mean, s_std, r_scale


# ============================================================================
# 2. Q-NETWORK
# ============================================================================

class QNetwork(nn.Module):
    """Q(s, a) — same architecture as in cql.py / td3bc.py."""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], -1)).squeeze(-1)


# ============================================================================
# 3. FQE AGENT
# ============================================================================

class FQEAgent:
    """
    Fitted Q Evaluation: fit Q^π from off-policy data with Bellman target
      y = r + γ (1 - done) Q_tgt(s', π(s')).
    The policy π is fixed (e.g. a trained BC or TD3+BC actor). No importance weights.

    policy_fn: callable (states_tensor) -> actions_tensor, both on same device.
               Used to get π(s) and π(s') for TD target and for estimate_J.
    """

    def __init__(self, state_dim, action_dim, policy_fn: Callable,
                 hidden_dim=256, gamma=0.99, tau=0.005, lr=3e-4, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.policy_fn = policy_fn

        self.Q = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.Q_tgt = deepcopy(self.Q)
        for p in self.Q_tgt.parameters():
            p.requires_grad_(False)
        self.opt = optim.Adam(self.Q.parameters(), lr=lr)

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        with torch.no_grad():
            a_next = self.policy_fn(next_states)
            q_next = self.Q_tgt(next_states, a_next)
            td_target = rewards + self.gamma * (1 - dones) * q_next

        q = self.Q(states, actions)
        loss = F.mse_loss(q, td_target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # Soft target update
        for p, pt in zip(self.Q.parameters(), self.Q_tgt.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)

        return loss.item()

    def estimate_J(self, states, batch_size=4096) -> float:
        """
        FQE estimate: (1/n) Σ Q(s, π(s)) over the given states.
        Can be initial states, or all states in the dataset (as here).
        """
        states = torch.FloatTensor(states).to(self.device)
        self.Q.eval()
        with torch.no_grad():
            total = 0.0
            n = 0
            for i in range(0, len(states), batch_size):
                s = states[i:i + batch_size]
                a = self.policy_fn(s)
                q = self.Q(s, a)
                total += q.sum().item()
                n += q.numel()
        self.Q.train()
        return total / n if n else 0.0


# ============================================================================
# 4. BC POLICY (to evaluate with FQE)
# ============================================================================

class Actor(nn.Module):
    """Deterministic policy s -> a in [-1, 1] (same as td3bc)."""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh(),
        )

    def forward(self, state):
        return self.net(state)


class BCAgent:
    """Behavioral cloning; provides a policy we can evaluate with FQE."""
    def __init__(self, state_dim, action_dim, hidden_dim=256, device='cpu'):
        self.device = device
        self.policy = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.opt = optim.Adam(self.policy.parameters(), lr=3e-4)

    def update(self, states, actions, *_):
        s = torch.FloatTensor(states).to(self.device)
        a = torch.FloatTensor(actions).to(self.device)
        loss = F.mse_loss(self.policy(s), a)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def act(self, state):
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.policy(s).cpu().numpy().squeeze()

    def policy_fn_tensor(self, states_tensor):
        """For FQE: (B, state_dim) -> (B, action_dim) on same device."""
        return self.policy(states_tensor)


def evaluate(env, agent, n_episodes=20, seed=100, s_mean=None, s_std=None):
    """Rollout return (true J) for comparison with FQE estimate. Optional normalization."""
    denorm = (s_mean is not None and s_std is not None)
    returns = []
    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        total = 0.0
        done = False
        while not done:
            act = agent.act(obs if not denorm else (obs - s_mean) / s_std)
            obs, r, done, _ = env.step(act)
            total += r
        returns.append(total)
    return np.mean(returns), np.std(returns)


# ============================================================================
# 5. MAIN
# ============================================================================

def main():
    print("FQE (Chapter 3) — Off-Policy Evaluation on ThermalProcessEnv")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dim, action_dim = 3, 2
    batch_size = 256
    bc_epochs = 80
    fqe_epochs = 100

    dataset = collect_offline_dataset(n_episodes=400, noise_scale=0.3, seed=0)
    s_mean, s_std, r_scale = normalize_dataset(dataset)

    loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(dataset['states']),
            torch.FloatTensor(dataset['actions']),
            torch.FloatTensor(dataset['rewards']),
            torch.FloatTensor(dataset['next_states']),
            torch.FloatTensor(dataset['dones']),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    env = ThermalProcessEnv(seed=42)

    # 1) Train BC policy
    bc = BCAgent(state_dim, action_dim, device=device)
    for epoch in range(bc_epochs):
        for batch in loader:
            s, a, r, s2, d = [x.to(device) for x in batch]
            bc.update(s.cpu().numpy(), a.cpu().numpy())
    print("BC trained.")

    # 2) FQE: fit Q^π for π = BC, then estimate J
    policy_fn = lambda s: bc.policy_fn_tensor(s)
    fqe = FQEAgent(state_dim, action_dim, policy_fn, device=device)
    for epoch in range(fqe_epochs):
        for batch in loader:
            s, a, r, s2, d = [x.to(device) for x in batch]
            fqe.update(
                s.cpu().numpy(), a.cpu().numpy(),
                r.cpu().numpy(), s2.cpu().numpy(), d.cpu().numpy(),
            )
    J_fqe = fqe.estimate_J(dataset['states'])
    print(f"FQE estimate J(BC) = {J_fqe:.4f} (normalized reward scale)")

    # 3) True return via rollout (raw rewards, not normalized)
    mean_ret, std_ret = evaluate(env, bc, n_episodes=20, seed=100,
                                  s_mean=s_mean, s_std=s_std)
    print(f"Rollout return (true J): {mean_ret:.2f} ± {std_ret:.2f} (raw env reward)")

    print("\nNote: FQE value is on normalized rewards (dataset was scaled by r_scale).")
    print("Use FQE to rank policies (e.g. BC vs CQL) on the same scale; for absolute")
    print("comparison with env, run rollouts or scale J_FQE by r_scale.")


if __name__ == "__main__":
    main()

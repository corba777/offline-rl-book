"""
td3bc.py
========
TD3+BC (TD3 with Behavioral Cloning penalty) for offline continuous control.
Referenced from Chapter 5 of "Offline RL: From Theory to Industrial Practice".

Fujimoto & Gu, "A Minimalist Approach to Offline Reinforcement Learning (TD3+BC)", NeurIPS 2021.
arXiv:2106.06860

Idea: actor loss = -lambda * Q(s, pi(s)) + (pi(s) - a)^2.
      Q is normalized over the batch so the two terms are comparable.
No theoretical guarantee; simple and effective in practice.

Contents:
    1. ThermalProcessEnv  — toy environment (same as cql.py / iql.py)
    2. Dataset collection + normalization
    3. Actor (deterministic), QNetwork (double-Q)
    4. TD3+BC loss: TD for Q, BC-regularized actor
    5. TD3BCAgent — training loop
    6. run_comparison()  — train and evaluate vs BC

Usage:
    python td3bc.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict


# ============================================================================
# 1. ENVIRONMENT (same as cql.py for consistency)
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


# ============================================================================
# 2. DATASET
# ============================================================================

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
# 3. NETWORKS
# ============================================================================

class QNetwork(nn.Module):
    """Q(s,a) — same as CQL/IQL."""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], -1)).squeeze(-1)


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


# ============================================================================
# 4. TD3+BC LOSS
# ============================================================================

def td3bc_actor_loss(actor, Q1, states, actions, lambda_=0.25):
    """
    TD3+BC actor loss: maximize Q(s, pi(s)) - lambda * (pi(s) - a)^2.
    Per Fujimoto & Gu (2021), Q is normalized by mean absolute value over the batch:
    q_norm = q / (|B|^{-1} sum |Q(s,a)|), so the Q-term and BC-term have comparable scale.
    """
    pi = actor(states)
    q = Q1(states, pi)
    q_norm = q / (q.abs().mean() + 1e-6)
    bc_loss = ((pi - actions) ** 2).mean()
    return -q_norm.mean() * lambda_ + bc_loss


# ============================================================================
# 5. TD3BC AGENT
# ============================================================================

class TD3BCAgent:
    """
    TD3 with BC penalty. Critic: standard TD3 (double Q, target, delayed policy).
    Actor: TD3+BC loss with normalized Q.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256,
                 lambda_=0.25, gamma=0.99, tau=0.005, policy_delay=2,
                 device='cpu'):
        self.device = device
        self.lambda_ = lambda_
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self._policy_update_counter = 0

        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_tgt = deepcopy(self.actor)
        self.Q1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.Q2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.Q1_tgt = deepcopy(self.Q1)
        self.Q2_tgt = deepcopy(self.Q2)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.Q1_opt = optim.Adam(self.Q1.parameters(), lr=3e-4)
        self.Q2_opt = optim.Adam(self.Q2.parameters(), lr=3e-4)

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        with torch.no_grad():
            a_next = self.actor_tgt(next_states)
            q1_next = self.Q1_tgt(next_states, a_next)
            q2_next = self.Q2_tgt(next_states, a_next)
            q_next = torch.min(q1_next, q2_next)
            td_target = rewards + self.gamma * (1 - dones) * q_next

        q1 = self.Q1(states, actions)
        q2 = self.Q2(states, actions)
        loss_q1 = F.mse_loss(q1, td_target)
        loss_q2 = F.mse_loss(q2, td_target)
        self.Q1_opt.zero_grad()
        loss_q1.backward()
        self.Q1_opt.step()
        self.Q2_opt.zero_grad()
        loss_q2.backward()
        self.Q2_opt.step()

        self._policy_update_counter += 1
        if self._policy_update_counter % self.policy_delay == 0:
            loss_actor = td3bc_actor_loss(self.actor, self.Q1, states, actions, self.lambda_)
            self.actor_opt.zero_grad()
            loss_actor.backward()
            self.actor_opt.step()
            for p, pt in zip(self.actor.parameters(), self.actor_tgt.parameters()):
                pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
            for p, pt in zip(self.Q1.parameters(), self.Q1_tgt.parameters()):
                pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
            for p, pt in zip(self.Q2.parameters(), self.Q2_tgt.parameters()):
                pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)

    def act(self, state):
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.actor.act(s)


# ============================================================================
# 6. BC BASELINE & EVAL
# ============================================================================

class BCAgent:
    """Behavioral cloning: MLP that predicts action from state (same arch as Actor)."""
    def __init__(self, state_dim, action_dim, hidden_dim=256, device='cpu'):
        self.device = device
        self.policy = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.opt = optim.Adam(self.policy.parameters(), lr=3e-4)

    def update(self, states, actions, *_):
        s = torch.FloatTensor(states).to(self.device)
        a = torch.FloatTensor(actions).to(self.device)
        pred = self.policy(s)
        loss = F.mse_loss(pred, a)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def act(self, state):
        return self.policy.act(torch.FloatTensor(state).unsqueeze(0).to(self.device))


def evaluate(env, agent, n_episodes=20, seed=100):
    returns = []
    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        total = 0.0
        done = False
        while not done:
            act = agent.act(obs)
            obs, r, done, _ = env.step(act)
            total += r
        returns.append(total)
    return np.mean(returns), np.std(returns)


# ============================================================================
# 7. MAIN
# ============================================================================

def main():
    print("TD3+BC (Chapter 5) — Offline RL on ThermalProcessEnv")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dim, action_dim = 3, 2
    batch_size = 256
    epochs = 100

    dataset = collect_offline_dataset(n_episodes=400, noise_scale=0.3, seed=0)
    s_mean, s_std, r_scale = normalize_dataset(dataset)
    # Denormalize for env (eval runs in raw state space)
    def denorm(s):
        return s * s_std + s_mean

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
    td3bc = TD3BCAgent(state_dim, action_dim, lambda_=0.25, device=device)
    bc = BCAgent(state_dim, action_dim, device=device)

    for epoch in range(epochs):
        for batch in loader:
            s, a, r, s2, d = [x.to(device) for x in batch]
            td3bc.update(s.cpu().numpy(), a.cpu().numpy(), r.cpu().numpy(),
                         s2.cpu().numpy(), d.cpu().numpy())
            bc.update(s.cpu().numpy(), a.cpu().numpy(), r.cpu().numpy(),
                      s2.cpu().numpy(), d.cpu().numpy())
        if (epoch + 1) % 20 == 0:
            # Eval in raw env (states not normalized in env)
            mean_td3bc, std_td3bc = evaluate(env, td3bc, n_episodes=20, seed=100)
            mean_bc, std_bc = evaluate(env, bc, n_episodes=20, seed=100)
            print(f"Epoch {epoch+1} | TD3+BC return: {mean_td3bc:.2f} ± {std_td3bc:.2f} | "
                  f"BC return: {mean_bc:.2f} ± {std_bc:.2f}")

    print("Done. TD3+BC typically outperforms BC when lambda_ is tuned (e.g. 0.1--0.5).")


if __name__ == "__main__":
    main()

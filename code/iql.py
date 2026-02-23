"""
iql.py
======
Implicit Q-Learning (IQL) for offline continuous control.
Referenced from Chapter 4 of "Offline RL: From Theory to Industrial Practice".

Kostrikov et al., "Offline Reinforcement Learning with Implicit Q-Learning", ICLR 2022.
arXiv:2110.06169

Key idea: learn a value function V(s) via expectile regression, then extract
a policy that maximizes advantage A(s,a) = Q(s,a) - V(s) — without ever
querying the policy during Q-updates. This keeps all updates fully in-sample.

Contents:
    1. ThermalProcessEnv     — same toy environment as Chapter 3
    2. Dataset collection    — noisy PID behavior policy
    3. ValueNetwork          — V(s) critic
    4. QNetwork              — Q(s,a) critic (double-Q)
    5. DeterministicPolicy   — simple MLP actor
    6. expectile_loss()      — asymmetric L2 (core of IQL)
    7. IQLAgent              — full training loop
    8. BCAgent               — behavioral cloning baseline (from ch3)
    9. run_comparison()      — BC vs CQL-style vs IQL comparison

Usage:
    python iql.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, Optional


# ============================================================================
# 1. ENVIRONMENT  (identical to cql.py — reused for consistency)
# ============================================================================

class ThermalProcessEnv:
    """
    Toy thermal coating process (asphalt-manufacturing-inspired).

    State:  [temperature, filler_pct, viscosity]  — all in [0, 1]
    Action: [heat_delta, flow_delta]               — in [-1, 1]
    """

    def __init__(self, T_target=0.6, f_target=0.5, max_steps=200,
                 noise_std=0.02, seed=42):
        self.T_target  = T_target
        self.f_target  = f_target
        self.max_steps = max_steps
        self.noise_std = noise_std
        self.rng       = np.random.default_rng(seed)
        self.state_dim  = 3
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
# 2. DATASET COLLECTION
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
        obs  = env.reset(seed=seed + ep)
        done = False
        while not done:
            act = pid_action(obs, env.T_target, env.f_target)
            act = np.clip(act + np.random.normal(0, noise_scale, 2), -1, 1).astype(np.float32)
            obs2, r, done, _ = env.step(act)
            S.append(obs); A.append(act); R.append(r); S2.append(obs2); D.append(float(done))
            obs = obs2
    dataset = {k: np.array(v, dtype=np.float32)
               for k, v in zip(['states','actions','rewards','next_states','dones'],
                               [S, A, R, S2, D])}
    print(f"Dataset: {len(S):,} transitions | "
          f"reward mean={np.mean(R):.3f} ± {np.std(R):.3f}")
    return dataset


def normalize_dataset(dataset):
    s_mean  = dataset['states'].mean(0)
    s_std   = dataset['states'].std(0) + 1e-8
    r_scale = np.abs(dataset['rewards']).mean() + 1e-8
    dataset['states']      = (dataset['states']      - s_mean) / s_std
    dataset['next_states'] = (dataset['next_states'] - s_mean) / s_std
    dataset['rewards']     = dataset['rewards'] / r_scale
    return s_mean, s_std, r_scale


# ============================================================================
# 3. NETWORKS
# ============================================================================

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
# ============================================================================

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

    This is a weighted MSE loss:
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

class IQLAgent:
    """
    Full IQL agent.

    Three networks: V(s), Q1(s,a), Q2(s,a), policy pi(s).
    Three separate losses updated in sequence each step.

    Args:
        tau:  Expectile for V-function (0.5=MSE, 0.7=default, 0.9=aggressive).
        beta: Temperature for advantage-weighted policy extraction.
              Higher beta = more selective (closer to greedy over dataset).
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 256,
                 tau: float = 0.7,
                 beta: float = 3.0,
                 gamma: float = 0.99,
                 tau_target: float = 0.005,
                 device: str = 'cpu'):
        self.tau        = tau
        self.beta       = beta
        self.gamma      = gamma
        self.tau_target = tau_target
        self.device     = device

        # Networks
        self.V       = ValueNetwork(state_dim, hidden_dim).to(device)
        self.Q1      = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.Q2      = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy  = DeterministicPolicy(state_dim, action_dim, hidden_dim).to(device)

        # Targets for V and Q (used in TD backup)
        self.V_tgt   = deepcopy(self.V)
        self.Q1_tgt  = deepcopy(self.Q1)
        self.Q2_tgt  = deepcopy(self.Q2)
        for p in (list(self.V_tgt.parameters()) +
                  list(self.Q1_tgt.parameters()) +
                  list(self.Q2_tgt.parameters())):
            p.requires_grad_(False)

        # Optimizers — separate for each network
        self.v_opt  = optim.Adam(self.V.parameters(),      lr=3e-4)
        self.q_opt  = optim.Adam(list(self.Q1.parameters()) +
                                 list(self.Q2.parameters()), lr=3e-4)
        self.pi_opt = optim.Adam(self.policy.parameters(), lr=3e-4)

    def update(self, batch: Tuple[torch.Tensor, ...]) -> dict:
        s, a, r, s2, d = [x.to(self.device) for x in batch]
        info = {}

        # ── 1. V update (expectile regression) ───────────────────────────
        # V(s) ← tau-expectile of min(Q1(s,a), Q2(s,a)) for dataset (s,a)
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

class BCAgent:
    """Behavioral Cloning — deterministic MSE policy."""

    def __init__(self, state_dim, action_dim, hidden_dim=256, device='cpu'):
        self.device = device
        self.policy = DeterministicPolicy(state_dim, action_dim, hidden_dim).to(device)
        self.opt    = optim.Adam(self.policy.parameters(), lr=3e-4)

    def update(self, batch):
        s, a = batch[0].to(self.device), batch[1].to(self.device)
        loss = F.mse_loss(self.policy(s), a)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return {'bc_loss': loss.item()}


# ============================================================================
# 7. TRAINING & EVALUATION
# ============================================================================

def evaluate(agent, env: ThermalProcessEnv,
             s_mean: np.ndarray, s_std: np.ndarray,
             n_episodes: int = 20, device: str = 'cpu') -> dict:
    rewards, T_errs, f_errs = [], [], []
    for ep in range(n_episodes):
        obs  = env.reset(seed=9000 + ep)
        ep_r = 0.0
        done = False
        while not done:
            s = torch.FloatTensor((obs - s_mean) / s_std).unsqueeze(0).to(device)
            act = agent.policy.act(s)
            obs, r, done, _ = env.step(act)
            ep_r += r
            T_errs.append(abs(obs[0] - env.T_target))
            f_errs.append(abs(obs[1] - env.f_target))
        rewards.append(ep_r)
    return {'reward_mean': np.mean(rewards), 'reward_std': np.std(rewards),
            'T_err': np.mean(T_errs), 'f_err': np.mean(f_errs)}


def train_agent(agent, loader: DataLoader, n_epochs: int = 100,
                log_every: int = 25) -> None:
    for epoch in range(1, n_epochs + 1):
        info = {}
        for batch in loader:
            step = agent.update(batch)
            for k, v in step.items():
                info[k] = info.get(k, 0) + v
        if epoch % log_every == 0:
            n = len(loader)
            parts = [f"Epoch {epoch:3d}"] + [f"{k}={v/n:.4f}" for k, v in info.items()]
            print("  " + " | ".join(parts))


# ============================================================================
# 8. EXPECTILE VISUALIZATION  (shows what tau does)
# ============================================================================

def show_expectile_intuition():
    """
    Demonstrates how different tau values shape the V-function target.
    V(s) learns the tau-quantile of Q(s, a_data).
    """
    print("\n--- Expectile regression intuition ---")
    print(f"{'tau':<8} {'V estimate':>12} {'interpretation'}")
    print("-" * 50)

    # Simulate Q-values for a state: 5 dataset actions, different quality
    q_values = torch.tensor([-0.8, -0.3, 0.1, 0.4, 0.9])
    v_init   = torch.tensor([0.0], requires_grad=True)

    for tau in [0.1, 0.3, 0.5, 0.7, 0.9]:
        v = v_init.clone().detach().requires_grad_(True)
        opt = optim.Adam([v], lr=0.05)
        for _ in range(2000):
            loss = expectile_loss(v.expand(5), q_values, tau)
            opt.zero_grad(); loss.backward(); opt.step()
        interp = {0.1: "near minimum", 0.3: "lower quartile",
                  0.5: "median (MSE)", 0.7: "upper quartile (IQL default)",
                  0.9: "near maximum"}
        print(f"  tau={tau:.1f}   V={v.item():>8.3f}    {interp[tau]}")

    print(f"\n  True Q values: {q_values.tolist()}")
    print(f"  True min={q_values.min():.2f}, median={q_values.median():.2f},"
          f" max={q_values.max():.2f}")


# ============================================================================
# 9. MAIN
# ============================================================================

def run_comparison():
    print("=" * 60)
    print("Chapter 4: IQL vs BC on Thermal Process")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Show what expectile regression does before training
    show_expectile_intuition()
    print()

    # Data
    dataset = collect_offline_dataset(n_episodes=400, noise_scale=0.3)
    s_mean, s_std, _ = normalize_dataset(dataset)
    ds = TensorDataset(*[torch.FloatTensor(dataset[k])
                         for k in ['states','actions','rewards','next_states','dones']])
    loader = DataLoader(ds, batch_size=256, shuffle=True, drop_last=True)
    env = ThermalProcessEnv()

    # Behavior policy baseline (clean PID, no noise)
    pid_r = []
    for ep in range(20):
        obs = env.reset(seed=9000+ep); ep_r = 0; done = False
        while not done:
            act = pid_action(obs, env.T_target, env.f_target)
            obs, r, done, _ = env.step(act); ep_r += r
        pid_r.append(ep_r)
    print(f"Clean PID (oracle):        {np.mean(pid_r):.2f} ± {np.std(pid_r):.2f}")

    # Evaluate noisy PID (behavior policy)
    noisy_r = []
    rng = np.random.default_rng(42)
    for ep in range(20):
        obs = env.reset(seed=9000+ep); ep_r = 0; done = False
        while not done:
            act = pid_action(obs, env.T_target, env.f_target)
            act = np.clip(act + rng.normal(0, 0.3, 2), -1, 1).astype(np.float32)
            obs, r, done, _ = env.step(act); ep_r += r
        noisy_r.append(ep_r)
    print(f"Noisy PID (behavior data): {np.mean(noisy_r):.2f} ± {np.std(noisy_r):.2f}\n")

    # BC
    print("--- Behavioral Cloning ---")
    bc = BCAgent(3, 2, device=device)
    train_agent(bc, loader, n_epochs=100, log_every=25)
    bc_res = evaluate(bc, env, s_mean, s_std, device=device)

    # IQL with default tau=0.7
    print("\n--- IQL (tau=0.7, beta=3.0) ---")
    iql = IQLAgent(3, 2, tau=0.7, beta=3.0, device=device)
    train_agent(iql, loader, n_epochs=100, log_every=25)
    iql_res = evaluate(iql, env, s_mean, s_std, device=device)

    # IQL more aggressive: tau=0.9
    print("\n--- IQL (tau=0.9, beta=5.0) ---")
    iql_agg = IQLAgent(3, 2, tau=0.9, beta=5.0, device=device)
    train_agent(iql_agg, loader, n_epochs=100, log_every=25)
    agg_res = evaluate(iql_agg, env, s_mean, s_std, device=device)

    # Summary
    print("\n" + "=" * 60)
    print(f"{'Method':<28} {'Reward':>12}  {'T err':>8}  {'f err':>8}")
    print("-" * 60)
    print(f"{'Clean PID (oracle)':<28} {np.mean(pid_r):>8.2f}±{np.std(pid_r):.2f}"
          f"  {'—':>8}  {'—':>8}")
    print(f"{'Noisy PID (behavior data)':<28} {np.mean(noisy_r):>8.2f}±{np.std(noisy_r):.2f}"
          f"  {'—':>8}  {'—':>8}")
    for name, res in [("BC", bc_res),
                      ("IQL (τ=0.7, β=3.0)", iql_res),
                      ("IQL (τ=0.9, β=5.0)", agg_res)]:
        print(f"{name:<28} {res['reward_mean']:>8.2f}±{res['reward_std']:.2f}"
              f"  {res['T_err']:>8.4f}  {res['f_err']:>8.4f}")
    print("=" * 60)
    print(f"\nIQL vs BC: {iql_res['reward_mean'] - bc_res['reward_mean']:+.2f} reward")


if __name__ == '__main__':
    run_comparison()

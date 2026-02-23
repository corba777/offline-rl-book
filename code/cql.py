"""
cql.py
======
Conservative Q-Learning (CQL) for offline continuous control.
Referenced from Chapter 3 of "Offline RL: From Theory to Industrial Practice".

Kumar et al., "Conservative Q-Learning for Offline Reinforcement Learning", NeurIPS 2020.
arXiv:2006.04779

Contents:
    1. ThermalProcessEnv  — toy industrial environment (asphalt-inspired)
    2. Dataset collection — noisy PID behavior policy
    3. QNetwork           — double-Q critic
    4. GaussianPolicy     — SAC-style stochastic actor
    5. compute_cql_loss() — TD loss + CQL penalty
    6. CQLAgent           — full training loop with optional auto-alpha
    7. BCAgent            — behavioral cloning baseline
    8. run_comparison()   — train all, compare performance

Usage:
    python cql.py
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
# 1. ENVIRONMENT
# ============================================================================

class ThermalProcessEnv:
    """
    Toy thermal coating process (asphalt-manufacturing-inspired).

    State:  [temperature, filler_pct, viscosity]  — all in [0, 1]
    Action: [heat_delta, flow_delta]               — in [-1, 1]

    Dynamics (discrete-time, first-order):
        T_{t+1}      = alpha_T * T_t + beta_T * heat_input + noise
        filler_{t+1} = alpha_f * f_t + beta_f * flow_input + noise
        viscosity    = nonlinear(T, filler)

    Reward: penalize deviation from setpoints + action roughness
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
        # First-order dynamics parameters
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
        """Nonlinear: high temp -> low viscosity; more filler -> high viscosity."""
        return float(np.clip(0.8 - 0.5 * T + 0.4 * f + 0.1 * f * (1 - T), 0, 1))

    def step(self, action):
        action = np.clip(action, -1, 1)
        T, f, _ = self.state
        heat_in = action[0] * 0.5 + 0.5   # scale [-1,1] -> [0,1]
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
    """Proportional control — behavior policy."""
    T, f, _ = state
    return np.array([
        np.clip(2.0 * (T_target - T), -1, 1),
        np.clip(2.0 * (f_target - f), -1, 1),
    ], dtype=np.float32)


def collect_offline_dataset(n_episodes=400, noise_scale=0.3, seed=0):
    """
    Collect transitions using a noisy PID (suboptimal behavior policy).
    noise_scale=0.3 gives mixed-quality data — some good, some bad trajectories.
    """
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
    """Normalize states (zero mean, unit std) and rewards. Returns stats."""
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

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),              nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], -1)).squeeze(-1)


class GaussianPolicy(nn.Module):
    """SAC-style stochastic policy with tanh squashing to [-1, 1]."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head    = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def _dist(self, state):
        h       = self.trunk(state)
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(-4, 2)
        return mean, log_std.exp()

    def sample(self, state):
        """Returns (action in [-1,1], log_prob) via reparameterization."""
        mean, std = self._dist(state)
        raw    = mean + std * torch.randn_like(mean)
        action = torch.tanh(raw)
        lp     = (torch.distributions.Normal(mean, std).log_prob(raw)
                  - torch.log(1 - action.pow(2) + 1e-6)).sum(-1)
        return action, lp

    def act(self, state, deterministic=True):
        with torch.no_grad():
            mean, _ = self._dist(state)
            return (torch.tanh(mean) if deterministic else self.sample(state)[0]
                   ).cpu().numpy().squeeze()


# ============================================================================
# 4. CQL LOSS
# ============================================================================

def compute_cql_loss(Q, Q_other, Q_tgt, Q_other_tgt, policy,
                     states, actions, rewards, next_states, dones,
                     alpha_cql=1.0, alpha_ent=0.1, gamma=0.99, n_samples=10):
    """
    CQL loss = TD_loss + alpha_cql * CQL_penalty

    Note: Kumar et al. (2020) write (1/2)*TD_loss in their derivation.
    The factor is absorbed by the learning rate and alpha in practice,
    so we omit it here — consistent with most open-source implementations.

    CQL_penalty = E_s[ logsumexp_a Q(s,a) ] - E_{s,a~D}[ Q(s,a) ]

    logsumexp is approximated via importance sampling:
      - n_samples random actions  (uniform proposal, no IS correction needed)
      - n_samples policy actions  (proposal = pi_theta, IS correction applied)
    """
    B = states.shape[0]
    dev = states.device

    # ── TD target (standard SAC Bellman backup) ───────────────────────────
    with torch.no_grad():
        a_next, lp_next = policy.sample(next_states)
        v_next = (torch.min(Q_tgt(next_states, a_next),
                            Q_other_tgt(next_states, a_next))
                  - alpha_ent * lp_next)
        td_target = rewards + gamma * (1 - dones) * v_next

    q_data  = Q(states, actions)
    td_loss = F.mse_loss(q_data, td_target)

    # ── CQL penalty ───────────────────────────────────────────────────────
    # Expand states: (B, D) -> (B*n_samples, D)
    s_rep = states.unsqueeze(1).expand(-1, n_samples, -1).reshape(B * n_samples, -1)

    # Random OOD actions (uniform over [-1, 1])
    # Uniform proposal: log μ(a) = const  →  cancels in logsumexp, no IS correction needed.
    a_rand = torch.FloatTensor(B * n_samples, actions.shape[-1]).uniform_(-1, 1).to(dev)

    # Policy actions (on-policy, but may differ from behavior policy)
    a_pi, lp_pi = policy.sample(s_rep)

    q_rand = Q(s_rep, a_rand).reshape(B, n_samples)

    # Importance weight correction for policy proposal (μ = π_θ):
    #
    #   log E_{a~π}[ exp(Q(s,a)) / π(a|s) ]
    #   ≈  logsumexp_i[ Q(s, a_i) - log π(a_i|s) ]  + const
    #
    # Subtracting log π(a|s) turns a plain Monte Carlo average of Q into
    # the soft-maximum (logsumexp) that CQL requires.  Without this correction
    # we would compute E_π[Q], not log Σ_a exp Q(s,a).
    q_pi   = (Q(s_rep, a_pi) - lp_pi.detach()).reshape(B, n_samples)

    # logsumexp over all OOD action samples — "push down" term
    logsumexp   = torch.logsumexp(torch.cat([q_rand, q_pi], dim=1), dim=1)  # (B,)
    cql_penalty = (logsumexp - q_data).mean()

    loss = td_loss + alpha_cql * cql_penalty
    return loss, {
        'td_loss':     td_loss.item(),
        'cql_penalty': cql_penalty.item(),
        'q_data':      q_data.mean().item(),
        'q_ood':       logsumexp.mean().item(),
    }


# ============================================================================
# 5. AGENTS
# ============================================================================

class CQLAgent:
    """
    Full CQL agent with double-Q critics and optional automatic alpha tuning.

    alpha_cql controls conservatism:
        alpha=0   -> standard SAC (no conservatism)
        alpha=1   -> default, good starting point
        alpha>>1  -> approaches behavioral cloning

    alpha_ent is the SAC entropy coefficient.  The value 0.1 works for the
    thermal control task (action_dim=2).  For other environments, automatic
    tuning is recommended: target entropy H* = -dim(A) (one nat per action
    dimension).  To enable auto-tuning, subclass this agent and add a
    log_alpha_ent parameter analogous to log_alpha (the CQL dual variable).
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256,
                 alpha_cql=1.0, alpha_ent=0.1, gamma=0.99, tau=0.005,
                 auto_alpha=False, target_cql=-2.0, device='cpu'):
        self.alpha_ent  = alpha_ent
        self.gamma      = gamma
        self.tau        = tau
        self.auto_alpha = auto_alpha
        self.device     = device

        self.Q1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.Q2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.Q1_tgt = deepcopy(self.Q1)
        self.Q2_tgt = deepcopy(self.Q2)
        for p in list(self.Q1_tgt.parameters()) + list(self.Q2_tgt.parameters()):
            p.requires_grad_(False)

        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim).to(device)
        self.q_opt  = optim.Adam(list(self.Q1.parameters()) +
                                 list(self.Q2.parameters()), lr=3e-4)
        self.pi_opt = optim.Adam(self.policy.parameters(), lr=3e-4)

        if auto_alpha:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=1e-4)
            self.target_cql = target_cql
        else:
            self.log_alpha = torch.tensor(float(np.log(alpha_cql))).to(device)

    @property
    def alpha_cql(self):
        return self.log_alpha.exp().item()

    def update(self, batch):
        s, a, r, s2, d = [x.to(self.device) for x in batch]

        # Q update
        l1, i1 = compute_cql_loss(self.Q1, self.Q2, self.Q1_tgt, self.Q2_tgt,
                                   self.policy, s, a, r, s2, d,
                                   self.alpha_cql, self.alpha_ent, self.gamma)
        l2, i2 = compute_cql_loss(self.Q2, self.Q1, self.Q2_tgt, self.Q1_tgt,
                                   self.policy, s, a, r, s2, d,
                                   self.alpha_cql, self.alpha_ent, self.gamma)
        self.q_opt.zero_grad()
        (l1 + l2).backward()
        nn.utils.clip_grad_norm_(list(self.Q1.parameters()) +
                                 list(self.Q2.parameters()), 1.0)
        self.q_opt.step()

        # Auto alpha update (Lagrangian dual)
        if self.auto_alpha:
            pen = (i1['cql_penalty'] + i2['cql_penalty']) / 2
            al  = -self.log_alpha * (pen - self.target_cql)
            self.alpha_opt.zero_grad(); al.backward(); self.alpha_opt.step()

        # Policy update
        a_pi, lp = self.policy.sample(s)
        q_pi = torch.min(self.Q1(s, a_pi), self.Q2(s, a_pi))
        lpi  = (self.alpha_ent * lp - q_pi).mean()
        self.pi_opt.zero_grad(); lpi.backward(); self.pi_opt.step()

        # Soft target update
        for p, pt in zip(self.Q1.parameters(), self.Q1_tgt.parameters()):
            pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
        for p, pt in zip(self.Q2.parameters(), self.Q2_tgt.parameters()):
            pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return {'td': (i1['td_loss']+i2['td_loss'])/2,
                'cql': (i1['cql_penalty']+i2['cql_penalty'])/2,
                'q_data': (i1['q_data']+i2['q_data'])/2,
                'pi': lpi.item(), 'alpha': self.alpha_cql}


class BCAgent:
    """Behavioral Cloning baseline (NLL loss on Gaussian policy)."""

    def __init__(self, state_dim, action_dim, hidden_dim=256, device='cpu'):
        self.device = device
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim).to(device)
        self.opt    = optim.Adam(self.policy.parameters(), lr=3e-4)

    def update(self, batch):
        s, a = batch[0].to(self.device), batch[1].to(self.device)
        mean, std = self.policy._dist(s)
        raw  = torch.atanh(a.clamp(-0.999, 0.999))
        lp   = (torch.distributions.Normal(mean, std).log_prob(raw)
                - torch.log(1 - a.pow(2) + 1e-6)).sum(-1)
        loss = -lp.mean()
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return {'bc_loss': loss.item()}


# ============================================================================
# 6. TRAINING & EVALUATION
# ============================================================================

def evaluate(agent, env, s_mean, s_std, n_episodes=20, device='cpu'):
    rewards, T_errs, f_errs = [], [], []
    for ep in range(n_episodes):
        obs  = env.reset(seed=9000 + ep)
        ep_r = 0.0
        done = False
        while not done:
            s = torch.FloatTensor((obs - s_mean) / s_std).unsqueeze(0).to(device)
            act = agent.policy.act(s, deterministic=True)
            obs, r, done, _ = env.step(act)
            ep_r += r
            T_errs.append(abs(obs[0] - env.T_target))
            f_errs.append(abs(obs[1] - env.f_target))
        rewards.append(ep_r)
    return {'reward_mean': np.mean(rewards), 'reward_std': np.std(rewards),
            'T_err': np.mean(T_errs), 'f_err': np.mean(f_errs)}


def train_agent(agent, loader, n_epochs=80, log_every=20):
    for epoch in range(1, n_epochs + 1):
        info = {}
        for batch in loader:
            step_info = agent.update(batch)
            for k, v in step_info.items():
                info[k] = info.get(k, 0) + v
        if epoch % log_every == 0:
            n = len(loader)
            parts = [f"Epoch {epoch:3d}"] + [f"{k}={v/n:.4f}" for k, v in info.items()]
            print("  " + " | ".join(parts))


# ============================================================================
# 7. MAIN
# ============================================================================

def run_comparison():
    print("=" * 60)
    print("Chapter 3: CQL vs BC on Thermal Process")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    dataset = collect_offline_dataset(n_episodes=400, noise_scale=0.3)
    s_mean, s_std, _ = normalize_dataset(dataset)
    ds = TensorDataset(*[torch.FloatTensor(dataset[k])
                         for k in ['states','actions','rewards','next_states','dones']])
    loader = DataLoader(ds, batch_size=256, shuffle=True, drop_last=True)
    env = ThermalProcessEnv()

    # Behavior policy baseline
    class PIDAgent:
        policy = type('P', (), {'act': staticmethod(
            lambda s, det=True: pid_action(
                s.cpu().numpy().squeeze() * np.array([1,1,1]) + np.array([0,0,0]),
                0.6, 0.5))})()

    # We evaluate PID directly (no normalization needed for it)
    pid_r = []
    for ep in range(20):
        obs = env.reset(seed=9000+ep); ep_r = 0; done = False
        while not done:
            act = pid_action(obs, env.T_target, env.f_target)
            obs, r, done, _ = env.step(act); ep_r += r
        pid_r.append(ep_r)
    print(f"\nBehavior policy (noisy PID): {np.mean(pid_r):.2f} ± {np.std(pid_r):.2f}")

    # BC
    print("\n--- Behavioral Cloning ---")
    bc = BCAgent(3, 2, device=device)
    train_agent(bc, loader, n_epochs=80, log_every=20)
    bc_res = evaluate(bc, env, s_mean, s_std, device=device)

    # CQL fixed alpha
    print("\n--- CQL (alpha=1.0) ---")
    cql = CQLAgent(3, 2, alpha_cql=1.0, alpha_ent=0.1, device=device)
    train_agent(cql, loader, n_epochs=80, log_every=20)
    cql_res = evaluate(cql, env, s_mean, s_std, device=device)

    # CQL auto alpha
    print("\n--- CQL (auto alpha) ---")
    cql_auto = CQLAgent(3, 2, auto_alpha=True, target_cql=-1.0,
                        alpha_ent=0.1, device=device)
    train_agent(cql_auto, loader, n_epochs=80, log_every=20)
    auto_res = evaluate(cql_auto, env, s_mean, s_std, device=device)

    # Summary
    print("\n" + "=" * 60)
    print(f"{'Method':<20} {'Reward':>12}  {'T err':>8}  {'f err':>8}")
    print("-" * 60)
    print(f"{'Noisy PID (data)':<20} {np.mean(pid_r):>8.2f}±{np.std(pid_r):.2f}"
          f"  {'—':>8}  {'—':>8}")
    for name, res in [("BC", bc_res), ("CQL (α=1.0)", cql_res), ("CQL (auto α)", auto_res)]:
        print(f"{name:<20} {res['reward_mean']:>8.2f}±{res['reward_std']:.2f}"
              f"  {res['T_err']:>8.4f}  {res['f_err']:>8.4f}")
    print("=" * 60)
    print(f"\nCQL vs BC: {cql_res['reward_mean'] - bc_res['reward_mean']:+.2f} reward")


if __name__ == '__main__':
    run_comparison()

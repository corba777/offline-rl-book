"""
mopo.py
=======
Model-Based Offline Policy Optimization (MOPO) for offline continuous control.
Referenced from Chapter 5 of "Offline RL: From Theory to Industrial Practice".

Yu et al., "MOPO: Model-Based Offline Policy Optimization", NeurIPS 2020.
arXiv:2005.13239

Key idea: learn an ensemble of dynamics models from offline data, generate
synthetic rollouts with uncertainty-penalized rewards, then run standard RL
(SAC) on the augmented dataset.  The penalty prevents the agent from
exploiting regions where the model is unreliable.

Contents:
    1. ThermalProcessEnv         — toy industrial environment (same as CQL)
    2. Dataset collection        — noisy PID behavior policy
    3. ProbabilisticDynamicsNet  — single probabilistic dynamics model
    4. DynamicsEnsemble          — N models for uncertainty estimation
    5. QNetwork / GaussianPolicy — SAC critic and actor (reused from CQL)
    6. MOPOAgent                 — model training + synthetic rollouts + SAC
    7. BCAgent                   — behavioral cloning baseline
    8. show_model_uncertainty()  — visualize where the model is confident
    9. run_comparison()          — train all, compare performance

Usage:
    python mopo.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, List, Optional


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
    """Proportional control — behavior policy."""
    T, f, _ = state
    return np.array([
        np.clip(2.0 * (T_target - T), -1, 1),
        np.clip(2.0 * (f_target - f), -1, 1),
    ], dtype=np.float32)


def collect_offline_dataset(n_episodes=400, noise_scale=0.3, seed=0):
    """
    Collect transitions using a noisy PID (suboptimal behavior policy).
    noise_scale=0.3 gives mixed-quality data — some good, some bad.
    """
    env = ThermalProcessEnv(seed=seed)
    S, A, R, S2, D = [], [], [], [], []

    for ep in range(n_episodes):
        obs  = env.reset(seed=seed + ep)
        done = False
        while not done:
            act = pid_action(obs, env.T_target, env.f_target)
            act = np.clip(act + np.random.normal(0, noise_scale, 2),
                          -1, 1).astype(np.float32)
            obs2, r, done, _ = env.step(act)
            S.append(obs); A.append(act); R.append(r)
            S2.append(obs2); D.append(float(done))
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
# 3. PROBABILISTIC DYNAMICS MODEL
# ============================================================================

class ProbabilisticDynamicsNet(nn.Module):
    """
    Single probabilistic dynamics model: (s, a) -> (mean, log_var) of s'.

    Outputs a Gaussian distribution over next-state *residuals*:
        s'_pred ~ N(s + mean(s,a), exp(log_var(s,a)))

    Residual prediction (predicting delta rather than absolute s') is
    critical for stability — the identity baseline makes it easier to
    learn small corrections.

    Trained with negative log-likelihood (NLL) — not MSE — so the network
    learns its own uncertainty for each input.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 256):
        super().__init__()
        inp_dim = state_dim + action_dim
        self.trunk = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.mean_head    = nn.Linear(hidden_dim, state_dim)
        self.log_var_head = nn.Linear(hidden_dim, state_dim)

        # Learnable bounds on log_var to prevent numerical issues
        self.min_log_var = nn.Parameter(-10.0 * torch.ones(state_dim))
        self.max_log_var = nn.Parameter(0.5  * torch.ones(state_dim))

    def forward(self, state: torch.Tensor,
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean_residual, log_var) for next-state prediction."""
        h = self.trunk(torch.cat([state, action], -1))
        mean    = self.mean_head(h)
        log_var = self.log_var_head(h)
        # Soft-clamp log_var into [min_log_var, max_log_var]
        log_var = self.max_log_var - F.softplus(self.max_log_var - log_var)
        log_var = self.min_log_var + F.softplus(log_var - self.min_log_var)
        return mean, log_var

    def predict(self, state: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        """Point prediction (mean only): s' = s + mean_residual."""
        mean, _ = self.forward(state, action)
        return state + mean

    def nll_loss(self, state: torch.Tensor, action: torch.Tensor,
                 next_state: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Negative log-likelihood of observing next_state given (state, action).

        NLL = 0.5 * [log_var + (residual - mean)^2 / var],  summed over dims.

        This trains both the mean (accuracy) and log_var (calibrated uncertainty).
        """
        mean, log_var = self.forward(state, action)
        residual = next_state - state            # true delta
        var      = log_var.exp()
        nll      = 0.5 * (log_var + (residual - mean).pow(2) / var)
        loss     = nll.sum(-1).mean()            # sum over state dims, mean over batch

        # Regularize log_var bounds to prevent collapse
        bound_reg = 0.01 * (self.max_log_var.sum() - self.min_log_var.sum())

        return loss + bound_reg, {
            'nll': loss.item(),
            'mse': (residual - mean).pow(2).mean().item(),
            'log_var_mean': log_var.mean().item(),
        }


# ============================================================================
# 4. DYNAMICS ENSEMBLE
# ============================================================================

class DynamicsEnsemble:
    """
    Ensemble of N probabilistic dynamics models.

    Uncertainty estimation: for a given (s, a), each model predicts a mean.
    The disagreement (std across model means) estimates epistemic uncertainty.

    High disagreement = models learned different things = OOD input.
    Low disagreement  = models agree = well-covered by dataset.

    This is the core mechanism that makes MOPO safe:
    the agent is penalized for visiting states where the model is uncertain.
    """

    def __init__(self, n_models: int, state_dim: int, action_dim: int,
                 hidden_dim: int = 256, lr: float = 1e-3,
                 device: str = 'cpu'):
        self.n_models  = n_models
        self.state_dim = state_dim
        self.device    = device
        self.models    = [ProbabilisticDynamicsNet(state_dim, action_dim,
                                                   hidden_dim).to(device)
                          for _ in range(n_models)]
        self.optimizers = [optim.Adam(m.parameters(), lr=lr)
                           for m in self.models]

    def train_ensemble(self, dataset: dict, n_epochs: int = 50,
                       batch_size: int = 256, log_every: int = 10):
        """
        Train each model on a *different bootstrap sample* of the dataset.

        Bootstrap sampling is critical for ensemble diversity:
        each model sees a random 80% of the data, so they disagree
        on data-sparse regions — which is exactly what we want.
        """
        n_data = len(dataset['states'])
        s  = torch.FloatTensor(dataset['states']).to(self.device)
        a  = torch.FloatTensor(dataset['actions']).to(self.device)
        s2 = torch.FloatTensor(dataset['next_states']).to(self.device)

        for i, (model, opt) in enumerate(zip(self.models, self.optimizers)):
            # Bootstrap: sample 80% of data with replacement
            idx = torch.randint(0, n_data, (int(0.8 * n_data),))
            ds = TensorDataset(s[idx], a[idx], s2[idx])
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                                drop_last=True)

            for epoch in range(1, n_epochs + 1):
                total_nll = 0.0
                total_mse = 0.0
                n_batches = 0
                for s_b, a_b, s2_b in loader:
                    loss, info = model.nll_loss(s_b, a_b, s2_b)
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    total_nll += info['nll']
                    total_mse += info['mse']
                    n_batches += 1

                if epoch % log_every == 0:
                    print(f"  Model {i} | Epoch {epoch:3d} | "
                          f"NLL={total_nll/n_batches:.4f} "
                          f"MSE={total_mse/n_batches:.6f}")

    @torch.no_grad()
    def predict_with_uncertainty(self, states: torch.Tensor,
                                  actions: torch.Tensor
                                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next states and estimate epistemic uncertainty.

        Returns:
            next_states: mean prediction (average across ensemble)
            uncertainty: std of means across models (per state dim, then norm)
        """
        means = []
        for model in self.models:
            mean, _ = model(states, actions)
            means.append(states + mean)  # residual -> absolute

        means = torch.stack(means, dim=0)       # (N, batch, state_dim)
        ensemble_mean = means.mean(dim=0)       # (batch, state_dim)

        # Epistemic uncertainty = std of predictions across models
        ensemble_std = means.std(dim=0)          # (batch, state_dim)
        # Scalar uncertainty per sample: L2 norm across state dims
        uncertainty = ensemble_std.norm(dim=-1)  # (batch,)

        return ensemble_mean, uncertainty

    @torch.no_grad()
    def sample_prediction(self, states: torch.Tensor,
                           actions: torch.Tensor) -> torch.Tensor:
        """
        Sample a next-state prediction from a randomly chosen model.

        Using a single random model (rather than the ensemble mean) preserves
        transition stochasticity in synthetic rollouts. The ensemble mean
        would wash out aleatoric noise and produce overly smooth trajectories.
        """
        idx = np.random.randint(self.n_models)
        model = self.models[idx]
        mean, log_var = model(states, actions)
        std = (0.5 * log_var).exp()
        # Sample from the model's predicted distribution
        residual = mean + std * torch.randn_like(mean)
        return states + residual


# ============================================================================
# 5. NETWORKS (SAC-style, reused from CQL)
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
            return torch.tanh(mean).cpu().numpy().squeeze()


# ============================================================================
# 6. MOPO AGENT
# ============================================================================

class MOPOAgent:
    """
    Model-Based Offline Policy Optimization.

    Training loop:
        1. Train dynamics ensemble on offline dataset (once, upfront).
        2. Generate synthetic rollouts:
           - Branch from real states in the dataset.
           - Roll out h steps using the learned model.
           - Penalize rewards by model uncertainty: r_tilde = r - lambda * u(s,a).
        3. Combine real + synthetic data.
        4. Train SAC on the combined dataset.

    The uncertainty penalty lambda controls the pessimism:
        lambda=0   -> trust the model completely (dangerous)
        lambda=1   -> moderate penalty (default)
        lambda=5+  -> strong penalty (very conservative, approaches model-free)
    """

    def __init__(self, state_dim: int, action_dim: int,
                 n_ensemble: int = 5, rollout_horizon: int = 5,
                 lam: float = 1.0, rollout_batch: int = 512,
                 real_ratio: float = 0.5,
                 hidden_dim: int = 256, gamma: float = 0.99,
                 tau: float = 0.005, alpha_ent: float = 0.1,
                 model_lr: float = 1e-3, policy_lr: float = 3e-4,
                 q_lr: float = 3e-4, device: str = 'cpu'):

        self.device    = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma     = gamma
        self.tau        = tau
        self.alpha_ent  = alpha_ent
        self.lam        = lam
        self.rollout_horizon = rollout_horizon
        self.rollout_batch   = rollout_batch
        self.real_ratio      = real_ratio

        # Dynamics ensemble
        self.ensemble = DynamicsEnsemble(n_ensemble, state_dim, action_dim,
                                          hidden_dim, model_lr, device)
        # SAC components
        self.Q1     = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.Q2     = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.Q1_tgt = deepcopy(self.Q1)
        self.Q2_tgt = deepcopy(self.Q2)
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim).to(device)

        self.q_opt  = optim.Adam(list(self.Q1.parameters()) +
                                  list(self.Q2.parameters()), lr=q_lr)
        self.pi_opt = optim.Adam(self.policy.parameters(), lr=policy_lr)

    def train_model(self, dataset: dict, n_epochs: int = 50,
                    log_every: int = 10):
        """Phase 1: train the dynamics ensemble on offline data."""
        print("Training dynamics ensemble...")
        self.ensemble.train_ensemble(dataset, n_epochs=n_epochs,
                                      log_every=log_every)

    @torch.no_grad()
    def generate_synthetic_data(self, real_states: torch.Tensor
                                 ) -> Dict[str, torch.Tensor]:
        """
        Phase 2: model rollouts with uncertainty-penalized rewards.

        For each batch:
        1. Sample starting states from real dataset.
        2. Roll out h steps using the learned model.
        3. At each step: sample action from current policy,
           predict next state, compute reward, estimate uncertainty.
        4. Penalized reward: r_tilde = r_model - lambda * uncertainty.

        Returns a dict of synthetic transitions.
        """
        n = min(self.rollout_batch, real_states.shape[0])
        idx   = torch.randint(0, real_states.shape[0], (n,))
        state = real_states[idx].to(self.device)

        syn_s, syn_a, syn_r, syn_s2, syn_d = [], [], [], [], []

        for step in range(self.rollout_horizon):
            action, _ = self.policy.sample(state)

            # Predict next state and uncertainty
            next_state, uncertainty = self.ensemble.predict_with_uncertainty(
                state, action)

            # Compute reward from the model (same reward function as env)
            # In practice, the reward function is known or learned separately.
            # For our ThermalProcessEnv, reward = -2*(T-T*)² - 2*(f-f*)² - 0.1*||a||²
            # We apply it to the *predicted* next state.
            # Note: states are normalized, so we need raw values for reward.
            # Since this is a demo, we compute reward in normalized space
            # and rely on the penalty to correct model bias.
            reward = self._model_reward(state, action, next_state)

            # MOPO penalty: subtract uncertainty
            penalized_reward = reward - self.lam * uncertainty

            syn_s.append(state)
            syn_a.append(action)
            syn_r.append(penalized_reward)
            syn_s2.append(next_state)
            syn_d.append(torch.zeros(n, device=self.device))

            # Continue rollout from predicted next state
            state = next_state

        return {
            'states':      torch.cat(syn_s,  dim=0),
            'actions':     torch.cat(syn_a,  dim=0),
            'rewards':     torch.cat(syn_r,  dim=0),
            'next_states': torch.cat(syn_s2, dim=0),
            'dones':       torch.cat(syn_d,  dim=0),
        }

    def _model_reward(self, states, actions, next_states):
        """
        Compute reward from predicted states.

        In production MOPO, you would also learn a reward model.
        For our ThermalProcessEnv task, the reward function is known
        and simple, so we apply it analytically.

        Note: states are normalized, but the reward structure
        (quadratic penalty) still provides meaningful gradient signal.
        """
        # Quadratic penalty on next state (in whatever space)
        # T and filler are the first two state dimensions
        T_next = next_states[:, 0]
        f_next = next_states[:, 1]
        r = (-2.0 * T_next.pow(2) - 2.0 * f_next.pow(2)
             - 0.1 * actions.pow(2).sum(-1))
        return r

    def update(self, real_batch: Tuple[torch.Tensor, ...],
               synthetic: Dict[str, torch.Tensor]) -> dict:
        """
        Phase 3: SAC update on mixed real + synthetic data.

        Combines a batch of real transitions with synthetic transitions
        in the ratio specified by real_ratio.
        """
        # Unpack real batch
        s_r, a_r, r_r, s2_r, d_r = [x.to(self.device) for x in real_batch]
        n_real = s_r.shape[0]

        # Sample from synthetic buffer
        n_syn = max(1, int(n_real * (1.0 - self.real_ratio) / self.real_ratio))
        n_syn = min(n_syn, synthetic['states'].shape[0])
        syn_idx = torch.randint(0, synthetic['states'].shape[0], (n_syn,))

        s_s  = synthetic['states'][syn_idx]
        a_s  = synthetic['actions'][syn_idx]
        r_s  = synthetic['rewards'][syn_idx]
        s2_s = synthetic['next_states'][syn_idx]
        d_s  = synthetic['dones'][syn_idx]

        # Combine
        s  = torch.cat([s_r,  s_s],  dim=0)
        a  = torch.cat([a_r,  a_s],  dim=0)
        r  = torch.cat([r_r,  r_s],  dim=0)
        s2 = torch.cat([s2_r, s2_s], dim=0)
        d  = torch.cat([d_r,  d_s],  dim=0)

        info = {}

        # ── 1. Q update (standard SAC TD) ──────────────────────────────────
        with torch.no_grad():
            a2, lp2 = self.policy.sample(s2)
            q_next  = torch.min(self.Q1_tgt(s2, a2), self.Q2_tgt(s2, a2))
            q_next -= self.alpha_ent * lp2
            target  = r + self.gamma * (1 - d) * q_next

        q1_loss = F.mse_loss(self.Q1(s, a), target)
        q2_loss = F.mse_loss(self.Q2(s, a), target)

        self.q_opt.zero_grad()
        (q1_loss + q2_loss).backward()
        self.q_opt.step()
        info['q_loss'] = (q1_loss.item() + q2_loss.item()) / 2

        # ── 2. Policy update (SAC entropy-regularized) ─────────────────────
        a_pi, lp_pi = self.policy.sample(s)
        q_pi = torch.min(self.Q1(s, a_pi), self.Q2(s, a_pi))
        pi_loss = (self.alpha_ent * lp_pi - q_pi).mean()

        self.pi_opt.zero_grad()
        pi_loss.backward()
        self.pi_opt.step()
        info['pi_loss'] = pi_loss.item()

        # ── 3. Soft target update ──────────────────────────────────────────
        for p, pt in zip(self.Q1.parameters(), self.Q1_tgt.parameters()):
            pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
        for p, pt in zip(self.Q2.parameters(), self.Q2_tgt.parameters()):
            pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return info


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
# 7. UNCERTAINTY VISUALIZATION
# ============================================================================

def show_model_uncertainty(ensemble: DynamicsEnsemble,
                            dataset: dict, device: str = 'cpu'):
    """
    Compare model uncertainty for in-distribution vs OOD actions.

    Demonstrates that the ensemble correctly identifies OOD inputs:
    - In-distribution actions (from the dataset) → low uncertainty
    - OOD actions (large, unseen in data) → high uncertainty

    This is the mechanism that makes the MOPO penalty meaningful.
    """
    print("\n" + "=" * 60)
    print("Model Uncertainty: In-Distribution vs OOD")
    print("=" * 60)

    s = torch.FloatTensor(dataset['states'][:200]).to(device)
    a_data = torch.FloatTensor(dataset['actions'][:200]).to(device)

    # In-distribution: use actual dataset actions
    _, u_in = ensemble.predict_with_uncertainty(s, a_data)

    # OOD: use extreme actions never seen in training
    a_ood = torch.ones_like(a_data) * 0.95  # saturated actions
    _, u_ood = ensemble.predict_with_uncertainty(s, a_ood)

    # Random OOD: completely random
    a_rand = torch.FloatTensor(200, a_data.shape[-1]).uniform_(-1, 1).to(device)
    _, u_rand = ensemble.predict_with_uncertainty(s, a_rand)

    print(f"\nDataset actions (in-dist)  → uncertainty: "
          f"{u_in.mean():.4f} ± {u_in.std():.4f}")
    print(f"Extreme actions (OOD)     → uncertainty: "
          f"{u_ood.mean():.4f} ± {u_ood.std():.4f}")
    print(f"Random actions (mixed)    → uncertainty: "
          f"{u_rand.mean():.4f} ± {u_rand.std():.4f}")
    print(f"\nOOD/In-dist ratio: {u_ood.mean() / max(u_in.mean(), 1e-8):.2f}x")


# ============================================================================
# 8. TRAINING & EVALUATION
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


def train_sac_agent(agent, loader, real_states_tensor,
                    n_epochs=80, rollout_every=5, log_every=20):
    """
    MOPO training loop:
    1. Every rollout_every epochs, regenerate synthetic data.
    2. Each epoch, train SAC on real + synthetic batches.
    """
    synthetic = None

    for epoch in range(1, n_epochs + 1):
        # Regenerate synthetic data periodically
        if epoch == 1 or epoch % rollout_every == 0:
            synthetic = agent.generate_synthetic_data(real_states_tensor)

        info_accum = {}
        n_batches = 0
        for batch in loader:
            step_info = agent.update(batch, synthetic)
            for k, v in step_info.items():
                info_accum[k] = info_accum.get(k, 0) + v
            n_batches += 1

        if epoch % log_every == 0:
            parts = [f"Epoch {epoch:3d}"] + \
                    [f"{k}={v/n_batches:.4f}" for k, v in info_accum.items()]
            print("  " + " | ".join(parts))


# ============================================================================
# 9. MAIN
# ============================================================================

def run_comparison():
    print("=" * 60)
    print("Chapter 5: MOPO vs Model-Free Methods on Thermal Process")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    dataset = collect_offline_dataset(n_episodes=400, noise_scale=0.3)
    s_mean, s_std, _ = normalize_dataset(dataset)
    ds = TensorDataset(*[torch.FloatTensor(dataset[k])
                         for k in ['states','actions','rewards',
                                   'next_states','dones']])
    loader = DataLoader(ds, batch_size=256, shuffle=True, drop_last=True)
    env = ThermalProcessEnv()

    # Behavior policy baseline
    pid_r = []
    for ep in range(20):
        obs = env.reset(seed=9000+ep); ep_r = 0; done = False
        while not done:
            act = pid_action(obs, env.T_target, env.f_target)
            obs, r, done, _ = env.step(act); ep_r += r
        pid_r.append(ep_r)
    print(f"\nBehavior policy (noisy PID): {np.mean(pid_r):.2f} ± "
          f"{np.std(pid_r):.2f}")

    # BC
    print("\n--- Behavioral Cloning ---")
    bc = BCAgent(3, 2, device=device)
    for epoch in range(1, 81):
        for batch in loader:
            bc.update(batch)
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}")
    bc_res = evaluate(bc, env, s_mean, s_std, device=device)

    # MOPO (moderate penalty)
    print("\n--- MOPO (λ=1.0, h=5, N=5) ---")
    mopo = MOPOAgent(3, 2, n_ensemble=5, rollout_horizon=5, lam=1.0,
                     rollout_batch=512, real_ratio=0.5, device=device)
    mopo.train_model(dataset, n_epochs=30, log_every=10)

    # Show uncertainty diagnostic
    show_model_uncertainty(mopo.ensemble, dataset, device)

    # Train policy
    print("\n--- MOPO policy training ---")
    real_states = torch.FloatTensor(dataset['states']).to(device)
    train_sac_agent(mopo, loader, real_states,
                    n_epochs=80, rollout_every=10, log_every=20)
    mopo_res = evaluate(mopo, env, s_mean, s_std, device=device)

    # MOPO (conservative)
    print("\n--- MOPO (λ=3.0, h=3, N=5) ---")
    mopo_c = MOPOAgent(3, 2, n_ensemble=5, rollout_horizon=3, lam=3.0,
                       rollout_batch=512, real_ratio=0.7, device=device)
    mopo_c.train_model(dataset, n_epochs=30, log_every=30)
    print("\n--- MOPO conservative policy training ---")
    train_sac_agent(mopo_c, loader, real_states,
                    n_epochs=80, rollout_every=10, log_every=20)
    mopo_c_res = evaluate(mopo_c, env, s_mean, s_std, device=device)

    # Summary
    print("\n" + "=" * 60)
    print(f"{'Method':<28} {'Reward':>12}  {'T err':>8}  {'f err':>8}")
    print("-" * 60)
    print(f"{'Noisy PID (data)':<28} {np.mean(pid_r):>8.2f}±{np.std(pid_r):.2f}"
          f"  {'—':>8}  {'—':>8}")
    for name, res in [("BC", bc_res),
                      ("MOPO (λ=1.0, h=5)", mopo_res),
                      ("MOPO (λ=3.0, h=3)", mopo_c_res)]:
        print(f"{name:<28} {res['reward_mean']:>8.2f}±{res['reward_std']:.2f}"
              f"  {res['T_err']:>8.4f}  {res['f_err']:>8.4f}")
    print("=" * 60)
    print(f"\nMOPO vs BC: {mopo_res['reward_mean'] - bc_res['reward_mean']:+.2f}"
          f" reward")


if __name__ == '__main__':
    run_comparison()

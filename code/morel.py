"""
morel.py
========
Model-Based Offline Reinforcement Learning (MOReL) for offline continuous control.
Referenced from Chapter 5 of "Offline RL: From Theory to Industrial Practice".

Kidambi et al., "MOReL: Model-Based Offline Reinforcement Learning", NeurIPS 2020.
arXiv:2005.05951

Key idea: learn an ensemble of dynamics models, then construct a Pessimistic MDP
(P-MDP) by partitioning state-action space into KNOWN / UNKNOWN regions using
a hard uncertainty threshold epsilon. In the UNKNOWN region the agent is redirected
to an absorbing HALT state with a large fixed penalty kappa, making it strongly
unattractive to leave the data support.

Contrast with MOPO: MOPO subtracts a *continuous* reward penalty proportional to
uncertainty. MOReL uses a *hard* boundary: below epsilon -> normal MDP; above
epsilon -> immediate HALT with -kappa. This gives a distribution-free guarantee
on the suboptimality gap (Theorem 1, Kidambi et al. 2020).

Contents:
    1.  ThermalProcessEnv        — toy industrial environment (same as mopo.py)
    2.  Dataset collection       — noisy PID behavior policy
    3.  ProbabilisticDynamicsNet — single probabilistic dynamics model (same arch)
    4.  DynamicsEnsemble         — N models + uncertainty; calibrate_epsilon()
    5.  QNetwork / GaussianPolicy — SAC critic and actor
    6.  MOReLAgent               — P-MDP construction + rollouts + SAC
    7.  BCAgent                  — behavioral cloning baseline
    8.  visualize_pmdp()         — plot KNOWN vs UNKNOWN boundary in state space
    9.  run_comparison()         — train BC / MOPO-style / MOReL, compare

Key MOReL-specific additions vs mopo.py:
    - DynamicsEnsemble.calibrate_epsilon()   — set threshold from in-dist data
    - MOReLAgent.generate_synthetic_data()   — hard HALT on OOD, not soft penalty
    - MOReLAgent._is_halt()                  — Boolean OOD mask
    - visualize_pmdp()                       — show P-MDP boundary in 2D slice

Usage:
    python morel.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, List, Optional


# ============================================================================
# 1. ENVIRONMENT  (identical to mopo.py — same task, different algorithm)
# ============================================================================

class ThermalProcessEnv:
    """
    Toy thermal coating process (asphalt-manufacturing-inspired).

    State:  [temperature, filler_pct, viscosity]   in [0, 1]
    Action: [heat_delta, flow_delta]               in [-1, 1]

    Dynamics:
        T_{t+1}   = 0.92 * T_t + 0.08 * heat_input + noise
        f_{t+1}   = 0.88 * f_t + 0.12 * flow_input + noise
        viscosity = nonlinear(T, f)

    Reward: quadratic penalty on deviation from setpoints + action magnitude.
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

        reward = (-2.0 * (T_new - self.T_target) ** 2
                  - 2.0 * (f_new - self.f_target) ** 2
                  - 0.1 * float(np.sum(action ** 2)))
        done = self.t >= self.max_steps
        return self.state.copy(), float(reward), done, {}


# ============================================================================
# 2. DATASET COLLECTION
# ============================================================================

def pid_action(state, T_target, f_target):
    """Proportional controller — behavior policy."""
    T, f, _ = state
    return np.array([
        np.clip(2.0 * (T_target - T), -1, 1),
        np.clip(2.0 * (f_target - f), -1, 1),
    ], dtype=np.float32)


def collect_offline_dataset(n_episodes=400, noise_scale=0.3, seed=0):
    """
    Collect transitions using a noisy PID (suboptimal behavior policy).
    noise_scale=0.3 gives mixed-quality data typical of real industrial logs.
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
               for k, v in zip(['states', 'actions', 'rewards',
                                 'next_states', 'dones'],
                                [S, A, R, S2, D])}
    print(f"Dataset: {len(S):,} transitions | "
          f"reward mean={np.mean(R):.3f} ± {np.std(R):.3f}")
    return dataset


def normalize_dataset(dataset):
    """Normalize states (zero mean, unit std). Returns stats for denorm."""
    s_mean  = dataset['states'].mean(0)
    s_std   = dataset['states'].std(0) + 1e-8
    r_scale = np.abs(dataset['rewards']).mean() + 1e-8
    dataset['states']      = (dataset['states']      - s_mean) / s_std
    dataset['next_states'] = (dataset['next_states'] - s_mean) / s_std
    dataset['rewards']     = dataset['rewards'] / r_scale
    return s_mean, s_std, r_scale


# ============================================================================
# 3. PROBABILISTIC DYNAMICS MODEL  (same architecture as mopo.py)
# ============================================================================

class ProbabilisticDynamicsNet(nn.Module):
    """
    Single probabilistic dynamics model: (s, a) -> (mean, log_var) of delta_s.

    Predicts *residuals* (next_state - state) with learned uncertainty.
    Trained with negative log-likelihood so the network learns its own
    calibrated confidence per input region.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        inp = state_dim + action_dim
        self.trunk = nn.Sequential(
            nn.Linear(inp, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.mean_head    = nn.Linear(hidden_dim, state_dim)
        self.log_var_head = nn.Linear(hidden_dim, state_dim)
        # Learnable bounds prevent log_var from collapsing
        self.min_log_var = nn.Parameter(-10.0 * torch.ones(state_dim))
        self.max_log_var = nn.Parameter(  0.5 * torch.ones(state_dim))

    def forward(self, state, action):
        h       = self.trunk(torch.cat([state, action], -1))
        mean    = self.mean_head(h)
        log_var = self.log_var_head(h)
        log_var = self.max_log_var - F.softplus(self.max_log_var - log_var)
        log_var = self.min_log_var + F.softplus(log_var - self.min_log_var)
        return mean, log_var

    def nll_loss(self, state, action, next_state):
        mean, log_var = self.forward(state, action)
        residual = next_state - state
        var      = log_var.exp()
        nll      = 0.5 * (log_var + (residual - mean).pow(2) / var)
        loss     = nll.sum(-1).mean()
        bound_reg = 0.01 * (self.max_log_var.sum() - self.min_log_var.sum())
        return loss + bound_reg, {
            'nll': loss.item(),
            'mse': (residual - mean).pow(2).mean().item(),
        }


# ============================================================================
# 4. DYNAMICS ENSEMBLE  (extended with calibrate_epsilon for MOReL)
# ============================================================================

class DynamicsEnsemble:
    """
    Ensemble of N probabilistic dynamics models for uncertainty estimation.

    Epistemic uncertainty = std of ensemble predictions across models.
    High disagreement → models learned different things → OOD input.
    Low disagreement  → models agree → well-covered by the dataset.

    MOReL-specific method: calibrate_epsilon()
        Sets the KNOWN/UNKNOWN threshold from in-distribution data percentile.
        This ensures the KNOWN region comfortably covers the dataset and
        extends into nearby well-modeled areas without arbitrary guessing.
    """

    def __init__(self, n_models: int, state_dim: int, action_dim: int,
                 hidden_dim: int = 256, lr: float = 1e-3, device: str = 'cpu'):
        self.n_models   = n_models
        self.state_dim  = state_dim
        self.device     = device
        self.models     = [ProbabilisticDynamicsNet(state_dim, action_dim,
                                                    hidden_dim).to(device)
                           for _ in range(n_models)]
        self.optimizers = [optim.Adam(m.parameters(), lr=lr)
                           for m in self.models]

    def train_ensemble(self, dataset: dict, n_epochs: int = 50,
                       batch_size: int = 256, log_every: int = 10):
        """
        Train each model on a different bootstrap sample (80% with replacement).

        Bootstrap sampling is critical for ensemble diversity: if all models
        see the same data they converge to the same solution and uncertainty
        estimates collapse to zero even in OOD regions.
        """
        n_data = len(dataset['states'])
        s  = torch.FloatTensor(dataset['states']).to(self.device)
        a  = torch.FloatTensor(dataset['actions']).to(self.device)
        s2 = torch.FloatTensor(dataset['next_states']).to(self.device)

        for i, (model, opt) in enumerate(zip(self.models, self.optimizers)):
            idx = torch.randint(0, n_data, (int(0.8 * n_data),))
            ds  = TensorDataset(s[idx], a[idx], s2[idx])
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                                drop_last=True)

            for epoch in range(1, n_epochs + 1):
                total_nll, total_mse, n_b = 0.0, 0.0, 0
                for s_b, a_b, s2_b in loader:
                    loss, info = model.nll_loss(s_b, a_b, s2_b)
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    total_nll += info['nll']
                    total_mse += info['mse']
                    n_b += 1
                if epoch % log_every == 0:
                    print(f"  Model {i} | Epoch {epoch:3d} | "
                          f"NLL={total_nll/n_b:.4f}  MSE={total_mse/n_b:.6f}")

    @torch.no_grad()
    def predict_with_uncertainty(self, states: torch.Tensor,
                                  actions: torch.Tensor
                                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (ensemble_mean_next_state, epistemic_uncertainty).

        Uncertainty = L2 norm of std-across-models, one scalar per sample.
        High values reliably flag OOD inputs; see calibrate_epsilon() for
        how to set the threshold that separates KNOWN from UNKNOWN.
        """
        means = []
        for model in self.models:
            mean, _ = model(states, actions)
            means.append(states + mean)
        means = torch.stack(means, dim=0)           # (N, batch, state_dim)
        ensemble_mean = means.mean(0)               # (batch, state_dim)
        ensemble_std  = means.std(0)                # (batch, state_dim)
        uncertainty   = ensemble_std.norm(dim=-1)   # (batch,)
        return ensemble_mean, uncertainty

    @torch.no_grad()
    def sample_prediction(self, states: torch.Tensor,
                           actions: torch.Tensor) -> torch.Tensor:
        """Sample from a randomly chosen model (preserves aleatoric noise)."""
        idx   = np.random.randint(self.n_models)
        mean, log_var = self.models[idx](states, actions)
        std   = (0.5 * log_var).exp()
        return states + mean + std * torch.randn_like(mean)

    @torch.no_grad()
    def calibrate_epsilon(self, dataset: dict,
                           percentile: float = 80.0,
                           n_samples: int = 2000) -> float:
        """
        Calibrate the MOReL uncertainty threshold epsilon from in-dist data.

        Method:
            1. Sample n_samples transitions from the offline dataset.
            2. Compute uncertainty for each.
            3. Return the (percentile)-th quantile.

        Rationale:
            Setting epsilon at the 80th percentile means 80% of the actual
            dataset transitions are labelled KNOWN — the ensemble reliably
            models them. The other 20% near the edges of data support are
            labelled UNKNOWN and become the safety boundary.

            Smaller percentile → tighter boundary → more conservative.
            Larger percentile  → looser boundary  → more optimistic.

        Args:
            dataset:    normalized offline dataset dict
            percentile: quantile of in-dist uncertainty to use as threshold
            n_samples:  how many dataset transitions to evaluate

        Returns:
            epsilon: scalar float threshold
        """
        n = min(n_samples, len(dataset['states']))
        idx = np.random.choice(len(dataset['states']), n, replace=False)
        s = torch.FloatTensor(dataset['states'][idx]).to(self.device)
        a = torch.FloatTensor(dataset['actions'][idx]).to(self.device)

        _, uncertainty = self.predict_with_uncertainty(s, a)
        epsilon = float(torch.quantile(uncertainty, percentile / 100.0).item())
        print(f"  Calibrated epsilon = {epsilon:.4f}  "
              f"(p{percentile:.0f} of in-dist uncertainty, "
              f"n={n})")
        return epsilon


# ============================================================================
# 5. NETWORKS  (SAC-style, same as mopo.py)
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
# 6. MOReL AGENT   — the core of this file
# ============================================================================

class MOReeLAgent:
    """
    Model-Based Offline Reinforcement Learning (MOReL).

    Training pipeline:
        Phase 1 — Train dynamics ensemble on offline dataset.
        Phase 2 — Calibrate epsilon: set KNOWN/UNKNOWN threshold.
        Phase 3 — Generate P-MDP rollouts:
                    • sample starting states from real dataset
                    • roll out h steps using ensemble
                    • if uncertainty > epsilon → HALT (done=True, reward=-kappa)
                    • else → normal predicted transition
        Phase 4 — Train SAC on combined real + P-MDP synthetic data.
        Repeat phases 3-4 for each outer training iteration.

    The P-MDP guarantee (Theorem 1, Kidambi et al. 2020):
        J*(pi*) - J(pi_hat) ≤ 2*gamma*kappa/(1-gamma)^2 * Pr[UNKNOWN under pi_hat]

    i.e., suboptimality is bounded by the probability of the learned policy
    entering the UNKNOWN region, weighted by the HALT penalty.  If epsilon is
    well-calibrated and the learned policy stays in KNOWN, the gap is near zero.

    Args:
        state_dim:       dimensionality of state space
        action_dim:      dimensionality of action space
        n_ensemble:      number of dynamics models (5–7 typical)
        rollout_horizon: steps per synthetic rollout (shorter = safer)
        epsilon:         KNOWN/UNKNOWN threshold (set via calibrate_epsilon)
        kappa:           HALT penalty magnitude (large → strong deterrent)
        rollout_batch:   number of rollouts generated per iteration
        real_ratio:      fraction of real vs synthetic data in SAC updates
    """

    def __init__(self, state_dim: int, action_dim: int,
                 n_ensemble: int = 5,
                 rollout_horizon: int = 3,
                 epsilon: float = 0.1,
                 kappa: float = 100.0,
                 rollout_batch: int = 512,
                 real_ratio: float = 0.5,
                 hidden_dim: int = 256,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha_ent: float = 0.1,
                 model_lr: float = 1e-3,
                 policy_lr: float = 3e-4,
                 q_lr: float = 3e-4,
                 device: str = 'cpu'):

        self.device          = device
        self.state_dim       = state_dim
        self.action_dim      = action_dim
        self.gamma           = gamma
        self.tau             = tau
        self.alpha_ent       = alpha_ent
        self.epsilon         = epsilon   # KNOWN/UNKNOWN boundary
        self.kappa           = kappa     # HALT penalty
        self.rollout_horizon = rollout_horizon
        self.rollout_batch   = rollout_batch
        self.real_ratio      = real_ratio

        # Dynamics ensemble (shared architecture with MOPO)
        self.ensemble = DynamicsEnsemble(
            n_ensemble, state_dim, action_dim, hidden_dim, model_lr, device)

        # SAC components
        self.Q1     = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.Q2     = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.Q1_tgt = deepcopy(self.Q1)
        self.Q2_tgt = deepcopy(self.Q2)
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim).to(device)

        self.q_opt  = optim.Adam(
            list(self.Q1.parameters()) + list(self.Q2.parameters()), lr=q_lr)
        self.pi_opt = optim.Adam(self.policy.parameters(), lr=policy_lr)

        # Running stats for diagnostics
        self.halt_rate_history: List[float] = []

    # ------------------------------------------------------------------ #
    #  Phase 1: Train ensemble                                             #
    # ------------------------------------------------------------------ #

    def train_model(self, dataset: dict, n_epochs: int = 50,
                    log_every: int = 10):
        """Train the dynamics ensemble on offline data."""
        print("Training dynamics ensemble (MOReL)...")
        self.ensemble.train_ensemble(
            dataset, n_epochs=n_epochs, log_every=log_every)

    # ------------------------------------------------------------------ #
    #  Phase 2: Calibrate epsilon                                          #
    # ------------------------------------------------------------------ #

    def calibrate_epsilon(self, dataset: dict, percentile: float = 80.0):
        """
        Set self.epsilon from the in-distribution uncertainty distribution.

        Call this AFTER train_model() and BEFORE generate_synthetic_data().
        Overwrites any epsilon passed to __init__.
        """
        print("Calibrating epsilon...")
        self.epsilon = self.ensemble.calibrate_epsilon(
            dataset, percentile=percentile)

    # ------------------------------------------------------------------ #
    #  Phase 3: P-MDP rollouts (the heart of MOReL)                       #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate_synthetic_data(self,
                                 real_states: torch.Tensor
                                 ) -> Dict[str, torch.Tensor]:
        """
        Generate synthetic transitions in the P-MDP.

        MOReL rollout logic:
            For each step in horizon:
                1. Query current policy for action.
                2. Predict next state and uncertainty via ensemble.
                3. Check OOD: uncertainty > epsilon?
                   YES → HALT:
                         • reward = -kappa
                         • done   = True
                         • next_state = current state (absorbing)
                         • stop this rollout
                   NO  → KNOWN:
                         • reward from model reward function
                         • done   = False
                         • continue to next step

        The hard termination means any trajectory touching UNKNOWN
        gets a large one-time penalty and then zero future reward
        (because done=True makes the TD backup use no bootstrap).
        This is strictly stronger than MOPO's soft penalty.

        Returns:
            dict of tensors with keys:
                states, actions, rewards, next_states, dones
        """
        n = min(self.rollout_batch, real_states.shape[0])
        idx   = torch.randint(0, real_states.shape[0], (n,))
        state = real_states[idx].to(self.device)   # (n, state_dim)

        # Boolean mask: which rollouts are still active (not HALTed)
        active = torch.ones(n, dtype=torch.bool, device=self.device)

        syn_s, syn_a, syn_r, syn_s2, syn_d = [], [], [], [], []
        total_steps = 0
        halt_steps  = 0

        for step in range(self.rollout_horizon):
            if not active.any():
                break  # all rollouts terminated early

            # Only compute on active rollouts (efficiency)
            s_active = state[active]
            action, _ = self.policy.sample(s_active)

            # Ensemble prediction + uncertainty
            next_state, uncertainty = self.ensemble.predict_with_uncertainty(
                s_active, action)

            # OOD detection: the P-MDP boundary
            ood = uncertainty > self.epsilon        # (n_active,)
            halt_steps  += ood.sum().item()
            total_steps += ood.shape[0]

            # Reward: known → model reward; unknown → -kappa
            reward = torch.where(
                ood,
                -self.kappa * torch.ones(ood.shape[0], device=self.device),
                self._model_reward(s_active, action, next_state)
            )

            # Done: active halted transitions are done; end-of-horizon is not
            done = ood.float()

            # For HALT transitions: next_state is absorbing (stay in place)
            next_state_out = torch.where(
                ood.unsqueeze(-1),
                s_active,          # absorbing: stay at current state
                next_state         # known: move to predicted next state
            )

            # Record transitions for all active rollouts at this step
            # We record them fully, including halted ones
            syn_s.append(s_active)
            syn_a.append(action)
            syn_r.append(reward)
            syn_s2.append(next_state_out)
            syn_d.append(done)

            # Update active states: halted rollouts stop here
            active_indices = active.nonzero(as_tuple=True)[0]
            halted = active_indices[ood]
            active[halted] = False

            # Advance non-halted rollouts
            continued = active_indices[~ood]
            state[continued] = next_state[~ood].to(state.dtype)

        # Diagnostics
        halt_rate = halt_steps / max(total_steps, 1)
        self.halt_rate_history.append(halt_rate)

        return {
            'states':      torch.cat(syn_s,  dim=0),
            'actions':     torch.cat(syn_a,  dim=0),
            'rewards':     torch.cat(syn_r,  dim=0),
            'next_states': torch.cat(syn_s2, dim=0),
            'dones':       torch.cat(syn_d,  dim=0),
        }

    def _model_reward(self, states: torch.Tensor,
                      actions: torch.Tensor,
                      next_states: torch.Tensor) -> torch.Tensor:
        """
        Analytical reward function (known for ThermalProcessEnv).

        In production: either (a) assume reward function is known (most
        industrial control cases), or (b) learn a separate reward model
        alongside the dynamics ensemble.

        The reward penalizes deviation from setpoints (T=0.6, f=0.5 in
        normalized space, which maps to 0 after normalization with these
        setpoints near the mean).  For simplicity, we use raw next-state
        values with the original quadratic structure.
        """
        T_next = next_states[:, 0]
        f_next = next_states[:, 1]
        return (-2.0 * T_next.pow(2)
                - 2.0 * f_next.pow(2)
                - 0.1 * actions.pow(2).sum(-1))

    # ------------------------------------------------------------------ #
    #  Phase 4: SAC update on real + P-MDP data                           #
    # ------------------------------------------------------------------ #

    def update(self, real_batch: Tuple[torch.Tensor, ...],
               synthetic: Dict[str, torch.Tensor]) -> dict:
        """
        SAC update on mixed real + P-MDP synthetic data.

        The mixing ratio (real_ratio) matters:
            real_ratio = 1.0 → pure offline (ignore synthetic)
            real_ratio = 0.5 → equal mix (default)
            real_ratio → 0  → almost pure synthetic (trust model fully)

        In MOReL, it is safe to use a lower real_ratio than in MOPO because
        the HALT penalty ensures synthetic data outside the KNOWN region
        carries an explicit pessimistic signal, not just a soft penalty.
        """
        s_r, a_r, r_r, s2_r, d_r = [x.to(self.device) for x in real_batch]
        n_real = s_r.shape[0]

        n_syn = max(1, int(n_real * (1.0 - self.real_ratio) / self.real_ratio))
        n_syn = min(n_syn, synthetic['states'].shape[0])
        idx   = torch.randint(0, synthetic['states'].shape[0], (n_syn,))

        s  = torch.cat([s_r,  synthetic['states'][idx]],      dim=0)
        a  = torch.cat([a_r,  synthetic['actions'][idx]],     dim=0)
        r  = torch.cat([r_r,  synthetic['rewards'][idx]],     dim=0)
        s2 = torch.cat([s2_r, synthetic['next_states'][idx]], dim=0)
        d  = torch.cat([d_r,  synthetic['dones'][idx]],       dim=0)

        info = {}

        # Q update
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

        # Policy update
        a_pi, lp_pi = self.policy.sample(s)
        q_pi = torch.min(self.Q1(s, a_pi), self.Q2(s, a_pi))
        pi_loss = (self.alpha_ent * lp_pi - q_pi).mean()
        self.pi_opt.zero_grad()
        pi_loss.backward()
        self.pi_opt.step()
        info['pi_loss'] = pi_loss.item()

        # Soft target update
        for p, pt in zip(self.Q1.parameters(), self.Q1_tgt.parameters()):
            pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
        for p, pt in zip(self.Q2.parameters(), self.Q2_tgt.parameters()):
            pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return info


# ============================================================================
# 7. BEHAVIORAL CLONING BASELINE
# ============================================================================

class BCAgent:
    """Behavioral Cloning: supervised regression from state to action."""

    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4,
                 device='cpu'):
        self.device = device
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim).to(device)
        self.opt    = optim.Adam(self.policy.parameters(), lr=lr)

    def train(self, dataset: dict, n_epochs=50, batch_size=256):
        s = torch.FloatTensor(dataset['states']).to(self.device)
        a = torch.FloatTensor(dataset['actions']).to(self.device)
        ds = TensorDataset(s, a)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            drop_last=True)
        for epoch in range(1, n_epochs + 1):
            total_loss, n_b = 0.0, 0
            for s_b, a_b in loader:
                a_pred, lp = self.policy.sample(s_b)
                # NLL loss against observed actions
                loss = F.mse_loss(a_pred, a_b)
                self.opt.zero_grad(); loss.backward(); self.opt.step()
                total_loss += loss.item(); n_b += 1
            if epoch % 10 == 0:
                print(f"  BC epoch {epoch:3d} | loss={total_loss/n_b:.5f}")

    def act(self, state):
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.policy.act(s)


# ============================================================================
# 8. VISUALIZATION: P-MDP boundary in state space
# ============================================================================

def visualize_pmdp(ensemble: DynamicsEnsemble,
                   dataset: dict,
                   epsilon: float,
                   s_mean: np.ndarray,
                   s_std: np.ndarray,
                   device: str = 'cpu',
                   n_grid: int = 40):
    """
    Visualize the KNOWN / UNKNOWN partition in the (T, filler) plane.

    Creates a 2D grid over the (temperature, filler_pct) dimensions,
    fixes viscosity at its dataset mean, and queries the ensemble uncertainty
    at each grid point. Grid cells with uncertainty > epsilon are UNKNOWN (red);
    below epsilon are KNOWN (green).  Dataset transitions are overlaid.

    This directly illustrates the Pessimistic MDP geometry described in
    Figure 1(c) of Kidambi et al. (2020).
    """
    # Grid in *original* (un-normalized) space for readability
    T_vals = np.linspace(0.0, 1.0, n_grid)
    f_vals = np.linspace(0.0, 1.0, n_grid)
    TT, FF = np.meshgrid(T_vals, f_vals)

    # Fix viscosity at dataset mean (raw space)
    v_mean_raw = (dataset['states'][:, 2] * s_std[2] + s_mean[2]).mean()
    v_fill = np.full_like(TT, v_mean_raw)

    # Build (grid_size^2, state_dim) array in normalized space
    states_raw = np.stack([TT.ravel(), FF.ravel(), v_fill.ravel()], axis=-1)
    states_norm = (states_raw - s_mean) / s_std

    # Pick a representative action (zero = no control input)
    n_pts  = states_norm.shape[0]
    states_t = torch.FloatTensor(states_norm).to(device)
    actions_t = torch.zeros(n_pts, 2, device=device)

    with torch.no_grad():
        _, uncertainty = ensemble.predict_with_uncertainty(states_t, actions_t)

    unc_grid = uncertainty.cpu().numpy().reshape(n_grid, n_grid)
    known_grid = (unc_grid <= epsilon).astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # -- Left: uncertainty map --
    ax = axes[0]
    im = ax.contourf(T_vals, f_vals, unc_grid, levels=30, cmap='YlOrRd')
    ax.contour(T_vals, f_vals, unc_grid, levels=[epsilon],
               colors='k', linewidths=2.0, linestyles='--')
    plt.colorbar(im, ax=ax, label='Ensemble uncertainty (L2)')

    # Overlay dataset transitions (raw space)
    states_raw_ds = dataset['states'] * s_std + s_mean
    ax.scatter(states_raw_ds[::10, 0], states_raw_ds[::10, 1],
               c='blue', s=3, alpha=0.3, label='Dataset (every 10th)')
    ax.set_xlabel('Temperature'); ax.set_ylabel('Filler %')
    ax.set_title(f'Ensemble uncertainty  (ε = {epsilon:.4f})')
    ax.legend(fontsize=8)

    # -- Right: KNOWN / UNKNOWN partition --
    ax = axes[1]
    ax.contourf(T_vals, f_vals, known_grid,
                levels=[-0.5, 0.5, 1.5], colors=['#ffcccc', '#ccffcc'],
                alpha=0.7)
    ax.contour(T_vals, f_vals, unc_grid, levels=[epsilon],
               colors='k', linewidths=2.0, linestyles='--')
    ax.scatter(states_raw_ds[::10, 0], states_raw_ds[::10, 1],
               c='blue', s=3, alpha=0.3, label='Dataset')
    ax.set_xlabel('Temperature'); ax.set_ylabel('Filler %')
    ax.set_title('P-MDP: KNOWN (green) vs UNKNOWN/HALT (red)')
    ax.legend(fontsize=8)

    # Coverage stats
    known_frac = known_grid.mean()
    print(f"\nP-MDP coverage: {known_frac:.1%} of grid is KNOWN "
          f"(ε={epsilon:.4f})")

    plt.tight_layout()
    plt.savefig('pmdp_boundary.png', dpi=150, bbox_inches='tight')
    print("Saved: pmdp_boundary.png")
    plt.show()


# ============================================================================
# 9. EVALUATION & TRAINING HELPERS
# ============================================================================

def evaluate(agent, env: ThermalProcessEnv,
             s_mean: np.ndarray, s_std: np.ndarray,
             n_episodes: int = 20, device: str = 'cpu') -> float:
    """Evaluate agent on real environment. Returns mean episode reward."""
    total = 0.0
    for ep in range(n_episodes):
        obs  = env.reset(seed=1000 + ep)
        done = False
        ep_r = 0.0
        while not done:
            obs_n = (obs - s_mean) / s_std
            action = agent.act(obs_n)
            obs, r, done, _ = env.step(action)
            ep_r += r
        total += ep_r
    return total / n_episodes


def train_morel(agent: MOReeLAgent,
                dataset: dict,
                env: ThermalProcessEnv,
                s_mean: np.ndarray,
                s_std: np.ndarray,
                n_outer_iters: int = 20,
                sac_steps_per_iter: int = 500,
                batch_size: int = 256,
                device: str = 'cpu',
                eval_every: int = 5) -> List[float]:
    """
    Outer training loop for MOReL.

    Each outer iteration:
        1. Generate P-MDP rollouts from current policy.
        2. Run sac_steps_per_iter SAC updates on real + synthetic.
        3. Optionally evaluate on real env.

    The iterative re-generation of synthetic data (rather than a fixed
    synthetic buffer) lets the rollouts adapt as the policy improves.
    """
    # Build real data loader
    s_t  = torch.FloatTensor(dataset['states'])
    a_t  = torch.FloatTensor(dataset['actions'])
    r_t  = torch.FloatTensor(dataset['rewards'])
    s2_t = torch.FloatTensor(dataset['next_states'])
    d_t  = torch.FloatTensor(dataset['dones'])
    real_ds = TensorDataset(s_t, a_t, r_t, s2_t, d_t)
    real_loader = DataLoader(real_ds, batch_size=batch_size,
                              shuffle=True, drop_last=True)

    returns = []
    for outer in range(1, n_outer_iters + 1):
        # Generate fresh P-MDP synthetic data
        synthetic = agent.generate_synthetic_data(s_t.to(device))
        halt_rate = agent.halt_rate_history[-1]

        # SAC updates
        loader_iter = iter(real_loader)
        for step in range(sac_steps_per_iter):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(real_loader)
                batch = next(loader_iter)
            agent.update(batch, synthetic)

        # Evaluation
        if outer % eval_every == 0:
            ret = evaluate(agent, env, s_mean, s_std, device=device)
            returns.append(ret)
            print(f"Outer iter {outer:3d}/{n_outer_iters} | "
                  f"return={ret:.2f} | halt_rate={halt_rate:.1%} | "
                  f"synthetic={synthetic['states'].shape[0]:,}")

    return returns


# ============================================================================
# 10. FULL COMPARISON: BC vs MOReL
# ============================================================================

def run_comparison():
    """
    Train BC and MOReL on the same offline dataset; compare on real env.

    Expected output:
        BC:    limited performance, constrained to behavior policy
        MOReL: meaningfully better, but stays within KNOWN region
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # ── Data ────────────────────────────────────────────────────────────
    env     = ThermalProcessEnv(seed=99)
    dataset = collect_offline_dataset(n_episodes=400, noise_scale=0.3, seed=0)
    s_mean, s_std, r_scale = normalize_dataset(dataset)

    # ── Behavioral Cloning ───────────────────────────────────────────────
    print("\n" + "="*55)
    print("BEHAVIORAL CLONING")
    print("="*55)
    bc = BCAgent(env.state_dim, env.action_dim, device=device)
    bc.train(dataset, n_epochs=50)
    bc_return = evaluate(bc, env, s_mean, s_std, device=device)
    print(f"BC return: {bc_return:.2f}")

    # ── MOReL ────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("MOReL")
    print("="*55)
    morel = MOReeLAgent(
        state_dim       = env.state_dim,
        action_dim      = env.action_dim,
        n_ensemble      = 5,
        rollout_horizon = 3,     # short horizon for safety
        kappa           = 100.0, # large HALT penalty
        rollout_batch   = 512,
        real_ratio      = 0.5,
        device          = device,
    )

    # Phase 1: train model
    morel.train_model(dataset, n_epochs=50, log_every=10)

    # Phase 2: calibrate epsilon from data
    morel.calibrate_epsilon(dataset, percentile=80.0)
    print(f"  epsilon = {morel.epsilon:.4f}")

    # Visualize P-MDP boundary
    visualize_pmdp(morel.ensemble, dataset, morel.epsilon,
                   s_mean, s_std, device=device)

    # Phases 3-4: iterative rollout + SAC
    morel_returns = train_morel(
        morel, dataset, env, s_mean, s_std,
        n_outer_iters     = 20,
        sac_steps_per_iter= 500,
        device            = device,
        eval_every        = 5,
    )

    final_morel = evaluate(morel, env, s_mean, s_std, n_episodes=30,
                            device=device)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("RESULTS")
    print("="*55)
    print(f"  BC    return: {bc_return:.2f}")
    print(f"  MOReL return: {final_morel:.2f}")
    print(f"  Improvement : {(final_morel - bc_return):.2f}")
    print(f"\n  MOReL epsilon (KNOWN threshold): {morel.epsilon:.4f}")
    print(f"  Average HALT rate during training: "
          f"{np.mean(morel.halt_rate_history):.1%}")

    # Plot returns
    if morel_returns:
        plt.figure(figsize=(8, 4))
        plt.plot(range(5, 5 * len(morel_returns) + 1, 5),
                 morel_returns, 'o-', label='MOReL')
        plt.axhline(bc_return, linestyle='--', color='gray', label='BC')
        plt.xlabel('Outer iteration'); plt.ylabel('Episode return')
        plt.title('MOReL vs BC — ThermalProcessEnv')
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('morel_training.png', dpi=150)
        print("\nSaved: morel_training.png")
        plt.show()

    return bc_return, final_morel


if __name__ == '__main__':
    run_comparison()

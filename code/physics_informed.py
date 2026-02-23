"""
physics_informed.py
===================
Physics-Informed Offline RL — complete implementation for Chapter 6.
"Offline RL: From Theory to Industrial Practice"

Three modular components, each usable independently:

  Part I  — PhysicsRewardWrapper  : soft constraint penalties in reward
  Part II — HybridDynamicsNet     : physics prior + learned residual
             HybridEnsemble       : drop-in for Chapter 5 DynamicsEnsemble
  Part III— LagrangianPolicyOptimizer : hard constraints via primal-dual

All three components are designed to wrap existing Ch3-5 algorithms
without modification. See run_demo() at the bottom for a complete example.

References:
  Altman (1999) — Constrained Markov Decision Processes
  Banerjee et al. (2023) — Survey on Physics Informed RL
  Paternain et al. (2022) — Safe Policies for Factored MDPs
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable, Dict, List, Optional, Tuple


# ============================================================================
# 1. TOY ENVIRONMENT — generic continuous process
# ============================================================================

class ContinuousProcessEnv:
    """
    Generic continuous industrial process for demonstration.

    Anonymized three-variable process:
      s = (x0, x1, x2)   — three state variables
      a = (u0, u1)        — two control inputs

    True dynamics (unknown to the agent):
      x0: first-order with tau=15, K=0.9, plus nonlinear cross-coupling
      x1: first-order with tau=8,  K=0.7, plus transport delay effect
      x2: derived from x0, x1 via empirical relationship + noise

    Physics model (known to the agent — approximate):
      x0: first-order with tau=15, K=0.9  (good, 90% coverage)
      x1: first-order with tau=8,  K=0.7  (good, 85% coverage)
      x2: linear combination              (rough, 60% coverage)

    Reward: keep all variables near target with low control effort.
    Constraints: x0 in [0.2, 0.9], x1 in [0.1, 0.8], x2 in [0.0, 1.0]
    """

    # State bounds
    X0_BOUNDS = (0.2, 0.9)
    X1_BOUNDS = (0.1, 0.8)
    X2_BOUNDS = (0.0, 1.0)
    # Targets
    TARGET    = np.array([0.55, 0.45, 0.50])
    DT        = 1.0

    def __init__(self, noise_std: float = 0.02, seed: int = 0):
        self.noise_std = noise_std
        self.rng = np.random.RandomState(seed)
        self.state = None

    def reset(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        if state is not None:
            self.state = state.copy()
        else:
            # Start near target with some variability
            self.state = self.TARGET + self.rng.randn(3) * 0.1
            self.state = np.clip(self.state, [0.2, 0.1, 0.0], [0.9, 0.8, 1.0])
        return self.state.copy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        u0, u1 = np.clip(action, -1.0, 1.0)
        x0, x1, x2 = self.state

        # True dynamics (with cross-coupling the physics model doesn't know)
        dx0 = (self.DT / 15.0) * (0.9 * u0 - x0) + 0.04 * x1 * u1
        dx1 = (self.DT / 8.0)  * (0.7 * u1 - x1) + 0.02 * x0 * u0
        dx2 = 0.3 * (x0 + x1) - x2 + 0.05 * u0 * u1  # nonlinear

        noise = self.rng.randn(3) * self.noise_std
        self.state = self.state + np.array([dx0, dx1, dx2]) + noise
        self.state = np.clip(self.state, [-0.5, -0.5, -0.5], [1.5, 1.5, 1.5])

        reward = self._reward(self.state, action)
        done   = bool(np.any(self.state < -0.5) or np.any(self.state > 1.5))
        return self.state.copy(), reward, done

    def _reward(self, state: np.ndarray, action: np.ndarray) -> float:
        tracking = -np.sum((state - self.TARGET) ** 2)
        effort   = -0.01 * np.sum(action ** 2)
        return tracking + effort


# ============================================================================
# 2. DATASET COLLECTION (PID-like behavior policy)
# ============================================================================

def collect_offline_dataset(
    n_episodes: int = 300,
    episode_len: int = 50,
    noise_scale: float = 0.25,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Collect offline dataset from a noisy proportional controller.

    The behavior policy is suboptimal and covers only part of the state space —
    typical of historical industrial logs.
    """
    env = ContinuousProcessEnv(noise_std=0.02, seed=seed)
    rng = np.random.RandomState(seed)

    states, actions, rewards, next_states, dones = [], [], [], [], []

    for ep in range(n_episodes):
        s = env.reset()
        for _ in range(episode_len):
            # Proportional controller + exploration noise
            error = ContinuousProcessEnv.TARGET[:2] - s[:2]
            u     = 0.6 * error + rng.randn(2) * noise_scale
            u     = np.clip(u, -1.0, 1.0)

            s2, r, done = env.step(u)
            states.append(s.copy())
            actions.append(u.copy())
            rewards.append(r)
            next_states.append(s2.copy())
            dones.append(float(done))
            s = s2
            if done:
                break

    dataset = {
        'states':      np.array(states,      dtype=np.float32),
        'actions':     np.array(actions,     dtype=np.float32),
        'rewards':     np.array(rewards,     dtype=np.float32),
        'next_states': np.array(next_states, dtype=np.float32),
        'dones':       np.array(dones,       dtype=np.float32),
    }
    print(f"Dataset: {len(states)} transitions | "
          f"reward mean={dataset['rewards'].mean():.3f} ± {dataset['rewards'].std():.3f}")
    return dataset


def normalize_dataset(dataset: Dict) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Normalize states and rewards for stable training."""
    s_mean = dataset['states'].mean(0)
    s_std  = dataset['states'].std(0) + 1e-8
    r_mean = dataset['rewards'].mean()
    r_std  = dataset['rewards'].std() + 1e-8

    norm = {k: v.copy() for k, v in dataset.items()}
    norm['states']      = (dataset['states']      - s_mean) / s_std
    norm['next_states'] = (dataset['next_states'] - s_mean) / s_std
    norm['rewards']     = (dataset['rewards']     - r_mean) / r_std
    return norm, s_mean, s_std


# ============================================================================
# 3. PART I — PHYSICS REWARD SHAPING
# ============================================================================

class PhysicsRewardWrapper:
    """
    Wraps any reward function with physics-based soft penalty terms.

    Each constraint g(state, action, next_state) -> Tensor (batch,)
    returns violation magnitude ≥ 0. Zero = constraint satisfied.

    Compatible with any offline RL algorithm: apply to the dataset's
    reward column or inside synthetic rollouts.

    Theorem 6.1: the optimality gap ≤ 2*C_max / ((1-γ)*λ).
    Use calibrate_lambda() to set λ from data.
    """

    def __init__(self, base_reward: Callable,
                 constraints: List[Callable],
                 lambdas: List[float]):
        assert len(constraints) == len(lambdas)
        self.base_reward = base_reward
        self.constraints = constraints
        self.lambdas     = lambdas

    def __call__(self, state: torch.Tensor,
                 action: torch.Tensor,
                 next_state: torch.Tensor) -> torch.Tensor:
        r = self.base_reward(state, action, next_state)
        for g, lam in zip(self.constraints, self.lambdas):
            violation = g(state, action, next_state)   # (batch,) ≥ 0
            r = r - lam * violation
        return r

    def audit(self, state: torch.Tensor,
              action: torch.Tensor,
              next_state: torch.Tensor) -> Dict:
        """Per-constraint violation statistics for diagnostics."""
        with torch.no_grad():
            base = self.base_reward(state, action, next_state)
            info = {'base_reward': base.mean().item()}
            for i, (g, lam) in enumerate(zip(self.constraints, self.lambdas)):
                v = g(state, action, next_state)
                info[f'constraint_{i}_violation'] = v.mean().item()
                info[f'constraint_{i}_penalty']   = (lam * v).mean().item()
                info[f'constraint_{i}_active_pct']= (v > 0).float().mean().item()
        return info


def calibrate_lambda(
    dataset:        Dict,
    constraint_fn:  Callable,
    base_reward_fn: Callable,
    target_gap_fraction: float = 0.1,
    gamma:          float = 0.99,
    device:         str   = 'cpu',
) -> float:
    """
    Calibrate λ so Theorem 6.1 gap ≤ target_gap_fraction * mean_return.

    From the theorem: gap ≤ 2 * C_max / ((1-γ) * λ)
    Solving: λ = 2 * C_max / ((1-γ) * target_gap)
    """
    n  = min(2000, len(dataset['states']))
    s  = torch.FloatTensor(dataset['states'][:n]).to(device)
    a  = torch.FloatTensor(dataset['actions'][:n]).to(device)
    s2 = torch.FloatTensor(dataset['next_states'][:n]).to(device)

    with torch.no_grad():
        C_max    = constraint_fn(s, a, s2).max().item()
        mean_ret = base_reward_fn(s, a, s2).abs().mean().item() / (1 - gamma)

    target_gap = target_gap_fraction * mean_ret
    lam = 2.0 * C_max / max(target_gap, 1e-8)
    print(f"  calibrate_lambda: C_max={C_max:.4f} | "
          f"mean_return≈{mean_ret:.2f} | "
          f"λ={lam:.4f}  (gap ≤ {target_gap_fraction:.0%} of return)")
    return lam


# ------------- Concrete constraint functions --------------------------------

def bounds_constraint(state: torch.Tensor,
                      action: torch.Tensor,
                      next_state: torch.Tensor,
                      lower: List[float],
                      upper: List[float]) -> torch.Tensor:
    """
    Violation = sum of distances outside box [lower, upper].
    """
    lo = torch.FloatTensor(lower).to(next_state.device)
    hi = torch.FloatTensor(upper).to(next_state.device)
    v  = torch.clamp(lo - next_state, min=0.0) + torch.clamp(next_state - hi, min=0.0)
    return v.sum(-1)   # (batch,)


def mass_balance_constraint(state: torch.Tensor,
                             action: torch.Tensor,
                             next_state: torch.Tensor,
                             inflow_idx: int,
                             outflow_idx: int,
                             volume_idx:  int,
                             dt:    float = 1.0,
                             tol:   float = 0.05) -> torch.Tensor:
    """
    Mass balance: |V_{t+1} - V_t - dt*(q_in - q_out)| - tol.
    Violation > 0 when the model prediction violates conservation.
    """
    V_t   = state[:, volume_idx]
    V_tp1 = next_state[:, volume_idx]
    q_in  = action[:, inflow_idx]
    q_out = action[:, outflow_idx]
    balance_error = torch.abs(V_tp1 - V_t - dt * (q_in - q_out))
    return torch.clamp(balance_error - tol, min=0.0)


def first_order_constraint(state: torch.Tensor,
                            action: torch.Tensor,
                            next_state: torch.Tensor,
                            state_idx:  int   = 0,
                            action_idx: int   = 0,
                            tau:        float = 10.0,
                            K:          float = 0.8,
                            dt:         float = 1.0,
                            tol:        float = 0.05) -> torch.Tensor:
    """
    Penalty for dynamics inconsistent with first-order model:
        x_{t+1} ≈ x_t + dt/tau * (K*u_t - x_t)
    """
    x      = state[:, state_idx]
    u      = action[:, action_idx]
    x_phys = x + (dt / tau) * (K * u - x)
    x_pred = next_state[:, state_idx]
    return torch.clamp(torch.abs(x_pred - x_phys) - tol, min=0.0)


def monotone_constraint(state: torch.Tensor,
                         action: torch.Tensor,
                         next_state: torch.Tensor,
                         state_idx:  int   = 0,
                         direction:  int   = -1,   # -1=decreasing, +1=increasing
                         tol:        float = 0.01) -> torch.Tensor:
    """
    Penalize violations of a known monotone trend.
    direction=-1: x must be non-increasing (e.g. viscosity decreasing with T)
    direction=+1: x must be non-decreasing
    """
    dx = (next_state[:, state_idx] - state[:, state_idx]) * direction
    return torch.clamp(dx - tol, min=0.0)


# ============================================================================
# 4. PART II — HYBRID DYNAMICS MODEL
# ============================================================================

class HybridDynamicsNet(nn.Module):
    """
    Probabilistic hybrid dynamics: known physics prior + learned residual.

    Forward pass:
        mean_next_state = f_phys(state, action) + f_NN(state, action; θ)
        log_var         = f_logvar(state, action; θ)   [residual uncertainty]

    The physics term is computed with torch.no_grad() — no gradient flows
    through it. Only the residual parameters θ are trained.

    Training loss = NLL of next_state under predicted Gaussian.

    Proposition 6.2: hybrid error ≤ residual_error + physics_error,
    with residual_error < black_box_error on same data (smaller target amplitude).
    """

    def __init__(self, state_dim:  int,
                 action_dim: int,
                 physics_fn: Callable,
                 hidden_dim: int = 128):
        super().__init__()
        self.physics_fn = physics_fn
        self.state_dim  = state_dim

        inp = state_dim + action_dim
        self.trunk = nn.Sequential(
            nn.Linear(inp, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.mean_head    = nn.Linear(hidden_dim, state_dim)
        self.log_var_head = nn.Linear(hidden_dim, state_dim)

        # Tighter bounds than a pure black-box: residuals have smaller amplitude
        self.min_log_var = nn.Parameter(-8.0 * torch.ones(state_dim))
        self.max_log_var = nn.Parameter(-1.0 * torch.ones(state_dim))

    def forward(self, state: torch.Tensor,
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean_next_state, log_var)."""
        with torch.no_grad():
            phys = self.physics_fn(state, action)  # analytical, no gradient

        h       = self.trunk(torch.cat([state, action], -1))
        res     = self.mean_head(h)
        log_var = self.log_var_head(h)

        # Soft clamping to [min_log_var, max_log_var]
        log_var = self.max_log_var - F.softplus(self.max_log_var - log_var)
        log_var = self.min_log_var + F.softplus(log_var - self.min_log_var)

        return phys + res, log_var

    def nll_loss(self, state:      torch.Tensor,
                 action:     torch.Tensor,
                 next_state: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Gaussian NLL loss.
        Target = full next_state; physics is inside the mean prediction.
        """
        mean, log_var = self.forward(state, action)
        var  = log_var.exp()
        nll  = 0.5 * (log_var + (next_state - mean).pow(2) / var)
        loss = nll.sum(-1).mean()
        # Bound regularization (from Chua et al. 2018 PETS)
        loss = loss + 0.01 * (self.max_log_var.sum() - self.min_log_var.sum())

        with torch.no_grad():
            phys_mse = (next_state - self.physics_fn(state, action)).pow(2).mean()

        return loss, {
            'nll':          loss.item(),
            'full_mse':     (next_state - mean).pow(2).mean().item(),
            'physics_mse':  phys_mse.item(),   # what the NN must improve upon
        }

    @torch.no_grad()
    def sample(self, state: torch.Tensor,
               action: torch.Tensor) -> torch.Tensor:
        """Sample a next-state prediction."""
        mean, log_var = self.forward(state, action)
        std = log_var.exp().sqrt()
        return mean + std * torch.randn_like(mean)


def make_first_order_physics(tau, K, dt: float = 1.0) -> Callable:
    """
    Factory: physics function for first-order ODE  dx/dt = (K*u - x)/tau.
    Euler discretization: x_{t+1} = x_t + dt/tau * (K*u_t - x_t)

    tau, K: scalar or 1D tensor of length state_dim (per-dimension constants).
    """
    def physics_fn(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        u = action[:, :state.shape[1]]
        return state + (dt / tau) * (K * u - state)
    return physics_fn


def make_linear_combination_physics(weights: List[float],
                                     bias:    float = 0.0) -> Callable:
    """
    Physics function: x2 ≈ weights[0]*x0 + weights[1]*x1 + bias
    For a state variable derived from others via a known linear relationship.
    Returns a function that fills dimension 2 based on dims 0 and 1.
    """
    w = torch.FloatTensor(weights)

    def physics_fn(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Default: identity for first two dims, linear combination for third
        x2_phys = (state[:, :len(weights)] * w.to(state.device)).sum(-1) + bias
        phys    = state.clone()
        phys[:, len(weights)] = x2_phys
        return phys
    return physics_fn


class HybridEnsemble:
    """
    Ensemble of HybridDynamicsNet models.

    Drop-in replacement for Chapter 5's DynamicsEnsemble.
    Uncertainty = standard deviation of ensemble predictions over the residual.
    In OOD regions: physics term is active → physically meaningful mean prediction
    even when residuals are uncertain.

    Usage with MOPO:
        ensemble = HybridEnsemble(5, state_dim, action_dim, physics_fn)
        ensemble.train_ensemble(dataset)
        # Then pass ensemble.predict_with_uncertainty to MOPO rollout generator
    """

    def __init__(self, n_models:   int,
                 state_dim:  int,
                 action_dim: int,
                 physics_fn: Callable,
                 hidden_dim: int   = 128,
                 lr:         float = 1e-3,
                 device:     str   = 'cpu'):
        self.n_models  = n_models
        self.device    = device
        self.physics_fn = physics_fn
        self.models    = [
            HybridDynamicsNet(state_dim, action_dim,
                              physics_fn, hidden_dim).to(device)
            for _ in range(n_models)
        ]
        self.optimizers = [optim.Adam(m.parameters(), lr=lr)
                           for m in self.models]
        self._trained = False

    def train_ensemble(self, dataset:    Dict,
                       n_epochs:   int = 50,
                       batch_size: int = 256,
                       log_every:  int = 10) -> None:
        """Bootstrap training: each model sees a random 80% subset."""
        n  = len(dataset['states'])
        s  = torch.FloatTensor(dataset['states']).to(self.device)
        a  = torch.FloatTensor(dataset['actions']).to(self.device)
        s2 = torch.FloatTensor(dataset['next_states']).to(self.device)

        for i, (model, opt) in enumerate(zip(self.models, self.optimizers)):
            idx    = torch.randint(0, n, (int(0.8 * n),))
            loader = DataLoader(TensorDataset(s[idx], a[idx], s2[idx]),
                                batch_size=batch_size, shuffle=True,
                                drop_last=True)
            for epoch in range(1, n_epochs + 1):
                tot_nll, tot_phys, nb = 0.0, 0.0, 0
                for s_b, a_b, s2_b in loader:
                    loss, info = model.nll_loss(s_b, a_b, s2_b)
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    tot_nll  += info['nll']
                    tot_phys += info['physics_mse']
                    nb += 1
                if epoch % log_every == 0:
                    print(f"  HybridModel {i} | epoch {epoch:3d} | "
                          f"NLL={tot_nll/nb:.4f}  "
                          f"physics_MSE={tot_phys/nb:.6f}")
        self._trained = True

    @torch.no_grad()
    def predict_with_uncertainty(self,
                                  states:  torch.Tensor,
                                  actions: torch.Tensor
                                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (ensemble_mean, epistemic_uncertainty).
        uncertainty = L2 norm of per-dim standard deviation across models.
        """
        means = [m(states, actions)[0] for m in self.models]
        means = torch.stack(means, 0)          # (N_models, batch, state_dim)
        return means.mean(0), means.std(0).norm(dim=-1)

    @torch.no_grad()
    def sample_next_state(self, state: torch.Tensor,
                           action: torch.Tensor) -> torch.Tensor:
        """Sample from a randomly chosen ensemble member."""
        model = self.models[torch.randint(self.n_models, (1,)).item()]
        return model.sample(state, action)

    def diagnose_physics_coverage(self, dataset: Dict) -> np.ndarray:
        """
        Print how much variance the physics model explains per state dimension.
        Run BEFORE training to decide whether physics_fn is adequate.

        Interpretation:
          > 85%: excellent — residual network needs little capacity
          50–85%: good — standard residual capacity
          < 50%: ⚠ consider revising physics_fn or using more capacity
        """
        n  = min(1000, len(dataset['states']))
        s  = torch.FloatTensor(dataset['states'][:n]).to(self.device)
        a  = torch.FloatTensor(dataset['actions'][:n]).to(self.device)
        s2 = torch.FloatTensor(dataset['next_states'][:n]).to(self.device)

        with torch.no_grad():
            phys    = self.physics_fn(s, a)
            res_var = (s2 - phys).var(0).cpu().numpy()
            tot_var = s2.var(0).cpu().numpy()

        coverage = 1.0 - res_var / (tot_var + 1e-8)

        print("Physics model coverage (fraction of variance explained):")
        for j, cov in enumerate(coverage):
            filled = int(max(0.0, cov) * 20)
            bar    = '█' * filled + '░' * (20 - filled)
            print(f"  state dim {j}: {cov:6.1%}  {bar}")
        mc = coverage.mean()
        status = ("✓ strong" if mc > 0.85
                  else "⚠  consider revising physics_fn" if mc < 0.5
                  else "")
        print(f"  Overall  : {mc:6.1%}  {status}")
        return coverage

    def compare_to_blackbox(self, dataset: Dict,
                             blackbox_ensemble,
                             n_test: int = 500) -> Dict:
        """
        Compare prediction error: hybrid ensemble vs pure black-box ensemble.
        Returns dict with MSE for both on held-out test transitions.
        """
        n  = min(n_test, len(dataset['states']))
        s  = torch.FloatTensor(dataset['states'][:n]).to(self.device)
        a  = torch.FloatTensor(dataset['actions'][:n]).to(self.device)
        s2 = torch.FloatTensor(dataset['next_states'][:n]).to(self.device)

        with torch.no_grad():
            hybrid_mean, _ = self.predict_with_uncertainty(s, a)
            bb_mean,     _ = blackbox_ensemble.predict_with_uncertainty(s, a)
            hybrid_mse = (s2 - hybrid_mean).pow(2).mean().item()
            bb_mse     = (s2 - bb_mean).pow(2).mean().item()

        print(f"  Hybrid ensemble MSE : {hybrid_mse:.6f}")
        print(f"  Black-box ensemble  : {bb_mse:.6f}")
        print(f"  Improvement         : {(bb_mse - hybrid_mse)/bb_mse:.1%}")
        return {'hybrid_mse': hybrid_mse, 'blackbox_mse': bb_mse}


# ============================================================================
# 5. PART III — LAGRANGIAN POLICY OPTIMIZER
# ============================================================================

class QNetwork(nn.Module):
    """Simple Q-network for demonstration (reward and cost)."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),              nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], -1)).squeeze(-1)


class GaussianPolicy(nn.Module):
    """SAC-style stochastic policy."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head    = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h       = self.trunk(state)
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(-4, 2)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std  = log_std.exp()
        eps  = torch.randn_like(mean)
        raw  = mean + std * eps
        act  = torch.tanh(raw)
        lp   = (torch.distributions.Normal(mean, std).log_prob(raw)
                - torch.log(1 - act.pow(2) + 1e-6)).sum(-1)
        return act, lp

    def act(self, state: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            act, _ = self.sample(state)
        return act.cpu().numpy().squeeze()


class LagrangianPolicyOptimizer:
    """
    Constrained offline RL via primal-dual saddle-point updates.

    Theorem 6.3 (Altman, 1999): strong duality holds for CMDPs.
    The minimax over (policy, lambda) gives the exact constrained optimum.

    Primal step: maximize J_Q(θ) - Σ λᵢ (J_Cᵢ(θ) - dᵢ)
    Dual step:   λᵢ ← max(0, λᵢ + η_λ * (J_Cᵢ - dᵢ))

    Lambda is parameterized in log-space to ensure λ ≥ 0 at all times.

    Usage:
        lagrangian = LagrangianPolicyOptimizer(
            policy, Q_reward, [Q_cost_1, Q_cost_2],
            cost_thresholds=[0.1, 0.05])

        for batch in dataloader:
            s, a = batch
            lagrangian.policy_step(s, pi_optimizer)
            lagrangian.dual_step(s, a)
    """

    def __init__(self, policy:           nn.Module,
                 Q_reward:         QNetwork,
                 Q_costs:          List[QNetwork],
                 cost_thresholds:  List[float],
                 lambda_lr:        float = 1e-3,
                 lambda_init:      float = 0.1,
                 entropy_coef:     float = 0.1,
                 device:           str   = 'cpu'):
        assert len(Q_costs) == len(cost_thresholds)
        self.policy       = policy
        self.Q_reward     = Q_reward
        self.Q_costs      = Q_costs
        self.d            = torch.FloatTensor(cost_thresholds).to(device)
        self.entropy_coef = entropy_coef
        self.device       = device

        # One log-lambda per constraint → exp ensures positivity
        n = len(Q_costs)
        init_val = float(np.log(lambda_init))
        self.log_lam = nn.Parameter(torch.full((n,), init_val))
        self.lam_opt = optim.Adam([self.log_lam], lr=lambda_lr)

        # History for diagnostics
        self.lambda_history    = [[] for _ in range(n)]
        self.violation_history = [[] for _ in range(n)]

    @property
    def lambdas(self) -> torch.Tensor:
        return self.log_lam.exp()

    def policy_step(self, states: torch.Tensor,
                    pi_opt: optim.Optimizer) -> Dict:
        """
        Primal update: policy ascent on Lagrangian.
        L(θ,λ) = J_Q(θ) - Σ λᵢ(J_Cᵢ(θ) - dᵢ)
        """
        actions, log_probs = self.policy.sample(states)

        q_r = self.Q_reward(states, actions)                          # (batch,)
        q_c = torch.stack([Qc(states, actions) for Qc in self.Q_costs],
                           dim=1)                                     # (batch, k)

        # Lagrangian: reward - lambda * (cost - budget)
        violation_terms = q_c - self.d.unsqueeze(0)                  # (batch, k)
        lagrangian = q_r - (self.lambdas.detach() * violation_terms).sum(1)

        # Negate for minimization; subtract entropy (SAC-style)
        pi_loss = -(lagrangian - self.entropy_coef * log_probs).mean()
        pi_opt.zero_grad(); pi_loss.backward(); pi_opt.step()

        return {
            'lagrangian':    lagrangian.mean().item(),
            'policy_loss':   pi_loss.item(),
            'lambdas':       self.lambdas.detach().cpu().tolist(),
        }

    def dual_step(self, states:  torch.Tensor,
                  actions: torch.Tensor) -> Dict:
        """
        Dual update: ascend on violation (increase λ when cost > budget).
        Gradient of L w.r.t. λᵢ = -(J_Cᵢ - dᵢ) = dᵢ - J_Cᵢ
        → minimize dual = ascend on (J_Cᵢ - dᵢ).
        """
        with torch.no_grad():
            q_c = torch.stack([Qc(states, actions) for Qc in self.Q_costs],
                               dim=1)                               # (batch, k)

        violations = (q_c.mean(0) - self.d).detach()               # (k,)
        # Minimize dual: dual_loss = -Σ λᵢ*(J_Cᵢ - dᵢ)
        dual_loss = -(self.lambdas * violations).sum()
        self.lam_opt.zero_grad(); dual_loss.backward(); self.lam_opt.step()

        for i, v in enumerate(violations.tolist()):
            self.violation_history[i].append(v)
            self.lambda_history[i].append(self.lambdas[i].item())

        return {
            'violations': violations.tolist(),
            'lambdas':    self.lambdas.detach().cpu().tolist(),
        }

    def plot_training_history(self) -> None:
        """Plot lambda and constraint violation over training."""
        k = len(self.Q_costs)
        fig, axes = plt.subplots(k, 2, figsize=(10, 3 * k))
        if k == 1:
            axes = axes[np.newaxis, :]

        for i in range(k):
            axes[i, 0].plot(self.lambda_history[i])
            axes[i, 0].set_title(f'λ_{i} over training')
            axes[i, 0].set_xlabel('dual step')
            axes[i, 0].axhline(0, color='k', linestyle='--', alpha=0.3)

            axes[i, 1].plot(self.violation_history[i])
            axes[i, 1].set_title(f'Constraint {i} violation (J_C - d)')
            axes[i, 1].set_xlabel('dual step')
            axes[i, 1].axhline(0, color='r', linestyle='--', alpha=0.5,
                                label='budget')
            axes[i, 1].legend()

        plt.tight_layout()
        plt.savefig('lagrangian_history.png', dpi=150)
        plt.show()
        print("Saved to lagrangian_history.png")


# ============================================================================
# 6. VISUALIZATION
# ============================================================================

def visualize_reward_shaping(dataset:    Dict,
                               wrapper:    PhysicsRewardWrapper,
                               n_samples:  int = 500,
                               device:     str = 'cpu') -> None:
    """
    Compare base reward vs physics-penalized reward distribution.
    Shows how constraint violations shift reward downward.
    """
    n  = min(n_samples, len(dataset['states']))
    s  = torch.FloatTensor(dataset['states'][:n]).to(device)
    a  = torch.FloatTensor(dataset['actions'][:n]).to(device)
    s2 = torch.FloatTensor(dataset['next_states'][:n]).to(device)

    with torch.no_grad():
        r_base    = wrapper.base_reward(s, a, s2).cpu().numpy()
        r_penalized = wrapper(s, a, s2).cpu().numpy()
        info      = wrapper.audit(s, a, s2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(r_base,      bins=40, alpha=0.6, label='Base reward',  color='steelblue')
    axes[0].hist(r_penalized, bins=40, alpha=0.6, label='With penalty', color='tomato')
    axes[0].set_xlabel('Reward')
    axes[0].set_title('Reward distribution: base vs penalized')
    axes[0].legend()

    labels  = [k for k in info if k.endswith('_violation')]
    values  = [info[k] for k in labels]
    axes[1].bar(range(len(labels)), values, color='coral')
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels([l.replace('_violation', '') for l in labels],
                              rotation=15)
    axes[1].set_ylabel('Mean violation magnitude')
    axes[1].set_title('Per-constraint violation statistics')

    plt.tight_layout()
    plt.savefig('reward_shaping.png', dpi=150)
    plt.show()
    print(f"Saved to reward_shaping.png")
    print(f"Constraint audit: {info}")


def visualize_hybrid_vs_blackbox(hybrid_ensemble: HybridEnsemble,
                                  dataset:          Dict,
                                  state_dim:        int,
                                  n_test:           int = 200,
                                  device:           str = 'cpu') -> None:
    """
    Per-dimension prediction error: physics alone vs hybrid model.
    Illustrates Proposition 6.2.
    """
    n  = min(n_test, len(dataset['states']))
    s  = torch.FloatTensor(dataset['states'][:n]).to(device)
    a  = torch.FloatTensor(dataset['actions'][:n]).to(device)
    s2 = torch.FloatTensor(dataset['next_states'][:n]).to(device)

    with torch.no_grad():
        phys_pred,   _  = hybrid_ensemble.predict_with_uncertainty(s, a)
        phys_only       = hybrid_ensemble.physics_fn(s, a)

        hybrid_err = (s2 - phys_pred).abs().mean(0).cpu().numpy()
        physics_err= (s2 - phys_only).abs().mean(0).cpu().numpy()

    x   = np.arange(state_dim)
    w   = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w/2, physics_err, w, label='Physics alone', color='steelblue', alpha=0.8)
    ax.bar(x + w/2, hybrid_err,  w, label='Hybrid model',  color='darkorange', alpha=0.8)
    ax.set_xlabel('State dimension')
    ax.set_ylabel('Mean absolute error')
    ax.set_title('Prediction accuracy: physics alone vs hybrid model')
    ax.set_xticks(x)
    ax.set_xticklabels([f'dim {i}' for i in x])
    ax.legend()
    plt.tight_layout()
    plt.savefig('hybrid_vs_physics.png', dpi=150)
    plt.show()
    print("Saved to hybrid_vs_physics.png")


# ============================================================================
# 7. DEMO — full pipeline
# ============================================================================

def run_demo(device: str = 'cpu') -> None:
    """
    End-to-end demo:
      1. Collect offline dataset from noisy PID controller
      2. Build physics-penalized reward (Part I)
      3. Train hybrid ensemble and compare to physics alone (Part II)
      4. Set up Lagrangian optimizer (Part III — setup only, not full training)
    """
    print("=" * 65)
    print("Physics-Informed Offline RL — Chapter 6 Demo")
    print("=" * 65)

    STATE_DIM  = 3
    ACTION_DIM = 2

    # ── 1. Dataset ─────────────────────────────────────────────────────────
    print("\n[1] Collecting offline dataset...")
    dataset = collect_offline_dataset(n_episodes=200, episode_len=50)
    norm_dataset, s_mean, s_std = normalize_dataset(dataset)

    # ── 2. Physics reward shaping ──────────────────────────────────────────
    print("\n[2] Setting up physics reward shaping...")

    def base_reward(s, a, s2):
        target = torch.FloatTensor([0.55, 0.45, 0.50]).to(s.device)
        return -((s2 - target).pow(2).sum(-1) + 0.01 * a.pow(2).sum(-1))

    # Operating bounds constraint
    bounds_fn = lambda s, a, s2: bounds_constraint(
        s, a, s2,
        lower=[0.2, 0.1, 0.0],
        upper=[0.9, 0.8, 1.0],
    )
    # First-order consistency for state dim 0
    fo_fn_0 = lambda s, a, s2: first_order_constraint(
        s, a, s2, state_idx=0, action_idx=0, tau=15.0, K=0.9)
    fo_fn_1 = lambda s, a, s2: first_order_constraint(
        s, a, s2, state_idx=1, action_idx=1, tau=8.0,  K=0.7)

    # Calibrate lambda from data
    lam_bounds = calibrate_lambda(dataset, bounds_fn, base_reward,
                                   target_gap_fraction=0.1, device=device)
    lam_fo     = calibrate_lambda(dataset, fo_fn_0, base_reward,
                                   target_gap_fraction=0.2, device=device)

    reward_wrapper = PhysicsRewardWrapper(
        base_reward  = base_reward,
        constraints  = [bounds_fn, fo_fn_0, fo_fn_1],
        lambdas      = [lam_bounds, lam_fo, lam_fo],
    )
    visualize_reward_shaping(dataset, reward_wrapper, device=device)

    # ── 3. Hybrid ensemble ─────────────────────────────────────────────────
    print("\n[3] Training hybrid ensemble...")

    # Combined physics function for 3-variable process
    def process_physics(state, action):
        """
        First-order physics for dims 0 and 1.
        Linear combination approximation for dim 2.
        """
        u0 = action[:, 0:1]
        u1 = action[:, 1:2]
        dt = 1.0
        x0_new = state[:, 0:1] + (dt / 15.0) * (0.9 * u0 - state[:, 0:1])
        x1_new = state[:, 1:2] + (dt / 8.0)  * (0.7 * u1 - state[:, 1:2])
        x2_new = 0.3 * (state[:, 0:1] + state[:, 1:2])  # approximate
        return torch.cat([x0_new, x1_new, x2_new], dim=1)

    ensemble = HybridEnsemble(
        n_models   = 5,
        state_dim  = STATE_DIM,
        action_dim = ACTION_DIM,
        physics_fn = process_physics,
        hidden_dim = 64,
        lr         = 1e-3,
        device     = device,
    )

    print("\n  Diagnosing physics coverage before training:")
    coverage = ensemble.diagnose_physics_coverage(norm_dataset)

    print("\n  Training ensemble...")
    ensemble.train_ensemble(norm_dataset, n_epochs=30, log_every=10)

    print("\n  Post-training visualization:")
    visualize_hybrid_vs_blackbox(ensemble, norm_dataset, STATE_DIM, device=device)

    # ── 4. Lagrangian optimizer setup ──────────────────────────────────────
    print("\n[4] Setting up Lagrangian policy optimizer...")

    policy   = GaussianPolicy(STATE_DIM, ACTION_DIM)
    Q_reward = QNetwork(STATE_DIM, ACTION_DIM)
    Q_cost   = QNetwork(STATE_DIM, ACTION_DIM)

    lagrangian = LagrangianPolicyOptimizer(
        policy          = policy,
        Q_reward        = Q_reward,
        Q_costs         = [Q_cost],
        cost_thresholds = [0.1],      # maximum allowed expected constraint cost
        lambda_lr       = 1e-3,
        lambda_init     = 0.1,
        device          = device,
    )
    print(f"  Initial lambda: {lagrangian.lambdas.tolist()}")
    print("  (Full training loop would alternate policy_step / dual_step)")
    print("  See LagrangianPolicyOptimizer docstring for usage.")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Demo complete. Files saved:")
    print("  reward_shaping.png       — Part I visualization")
    print("  hybrid_vs_physics.png    — Part II: hybrid vs physics alone")
    print("  lagrangian_history.png   — Part III (generated during training)")
    print("=" * 65)


if __name__ == '__main__':
    run_demo(device='cpu')

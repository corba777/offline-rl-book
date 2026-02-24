"""
chapter10.py
===========
Industrial Applications of Offline RL — complete case study.
Chapter 7 of "Offline RL: From Theory to Industrial Practice"

This file integrates algorithms from Chapters 1–6 into a single
industrial pipeline. It does NOT reimplement existing algorithms;
instead it imports them and adds only what is new for Chapter 7:

  CoatingProcessEnv       : 5-variable thermal coating process
  IndustrialDataset       : realistic log collection with operating regimes
  IndustrialEvaluator     : DA, RMSE, constraint violation rate
  PhysicsInformedCQL      : CQL + physics reward shaping (Ch3 + Ch6)
  HybridMOReL             : MOReL with hybrid ensemble (Ch5 + Ch6)
  run_industrial_benchmark: full 4-algorithm comparison with plots

Algorithms imported (not reimplemented):
  Ch1: BCAgent            from behavioral_cloning.py / cql.py (BCAgent)
  Ch3: CQLAgent           from cql.py
  Ch5: MOReeLAgent        from morel.py
  Ch6: PhysicsRewardWrapper, HybridEnsemble, calibrate_lambda,
       bounds_constraint, first_order_constraint  from physics_informed.py

Expected comparison:
  BC             — strong within distribution, degrades on transitions
  CQL            — better generalization, occasional constraint violations
  CQL+Physics    — fewer violations, similar or better reward
  HybridMOReL    — best extrapolation, physically consistent dynamics
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from copy import deepcopy
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Imports from previous chapters
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from cql import CQLAgent, BCAgent as CQLBCAgent, GaussianPolicy, QNetwork
    from cql import train_agent as _train_cql_base
    _CQL_AVAILABLE = True
except ImportError:
    _CQL_AVAILABLE = False

try:
    from morel import MOReeLAgent, DynamicsEnsemble
    from morel import train_morel, evaluate as morel_evaluate
    _MOREL_AVAILABLE = True
except ImportError:
    _MOREL_AVAILABLE = False

try:
    from physics_informed import (
        PhysicsRewardWrapper, HybridEnsemble,
        calibrate_lambda, bounds_constraint, first_order_constraint,
    )
    _PHYSICS_AVAILABLE = True
except ImportError:
    _PHYSICS_AVAILABLE = False


# ============================================================================
# 1. ENVIRONMENT — THERMAL COATING PROCESS
# ============================================================================

class CoatingProcessEnv:
    """
    Anonymized thermal coating process — industrial case study for Chapter 7.

    Extends ThermalProcessEnv from Chapters 1–5 to 5 state variables,
    representative of a real multi-variable continuous manufacturing process.

    State (5 variables, all normalized to [0, 1]):
      s[0]  temperature     — process temperature in the coating zone
      s[1]  filler_fraction — fraction of filler material in the blend
      s[2]  viscosity       — blend viscosity (nonlinear fn of T and filler)
      s[3]  density         — bulk material density
      s[4]  level           — surge tank level (integrating dynamics)

    Actions (2 variables):
      a[0]  heat_input  — heater setpoint delta, in [-1, 1]
      a[1]  flow_input  — filler flow rate delta, in [-1, 1]

    True dynamics (unknown to agent):
      temperature:     first-order, tau=12, K=0.85, cross-coupling with filler
      filler_fraction: first-order, tau=8,  K=0.90, with 2-step transport delay
      viscosity:       nonlinear(T, filler) + noise
      density:         linear(T, filler) + small drift
      level:           integrating: L_{t+1} = L_t + dt*(inflow - outflow)

    Known physics model (available to agent):
      temperature:     first-order, tau=12, K=0.85     (~90% variance explained)
      filler_fraction: first-order, tau=8,  K=0.90     (~85%, delay ignored)
      viscosity:       linearized approximation         (~65%)
      density:         linear combination               (~80%)
      level:           Euler mass balance               (~92%)

    Hard operating constraints:
      temperature:     [0.35, 0.85]
      filler_fraction: [0.20, 0.75]
      viscosity:       [0.10, 0.90]
      density:         [0.30, 0.80]
      level:           [0.15, 0.85]

    The level variable is the key industrial challenge: integrating dynamics,
    no natural setpoint, drifts over time. Mass balance must be respected.
    """

    STATE_DIM  = 5
    ACTION_DIM = 2
    STATE_T_IDX = 0   # temperature
    STATE_F_IDX = 1   # filler_fraction

    T_TARGET     = 0.60
    F_TARGET     = 0.50
    LEVEL_TARGET = 0.50

    BOUNDS = [
        (0.35, 0.85),  # temperature
        (0.20, 0.75),  # filler_fraction
        (0.10, 0.90),  # viscosity
        (0.30, 0.80),  # density
        (0.15, 0.85),  # level
    ]

    TAU_T = 12.0
    TAU_F = 8.0
    K_T   = 0.85
    K_F   = 0.90
    DT    = 1.0

    def __init__(self, noise_std: float = 0.015,
                 max_steps: int = 200,
                 seed: int = 42):
        self.noise_std = noise_std
        self.max_steps = max_steps
        self.rng       = np.random.default_rng(seed)
        self.state     = None
        self.t         = 0
        self._filler_buffer = [self.F_TARGET, self.F_TARGET]

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        T = float(np.clip(self.T_TARGET + self.rng.normal(0, 0.08), 0.35, 0.85))
        f = float(np.clip(self.F_TARGET + self.rng.normal(0, 0.07), 0.20, 0.75))
        v = self._viscosity(T, f)
        d = self._density(T, f)
        L = float(np.clip(self.LEVEL_TARGET + self.rng.normal(0, 0.05), 0.15, 0.85))
        self.state = np.array([T, f, v, d, L], dtype=np.float32)
        self._filler_buffer = [f, f]
        self.t = 0
        return self.state.copy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        action  = np.clip(action, -1.0, 1.0)
        T, f, v, d, L = self.state
        heat_in = float(action[0]) * 0.5 + 0.5
        flow_in = float(action[1]) * 0.5 + 0.5

        # True dynamics with nonlinear cross-coupling
        T_new = float(np.clip(
            (1 - self.DT / self.TAU_T) * T
            + (self.DT / self.TAU_T) * self.K_T * heat_in
            + 0.03 * f * heat_in
            + self.rng.normal(0, self.noise_std),
            0.0, 1.0,
        ))
        # Filler with 2-step transport delay
        delayed = self._filler_buffer.pop(0)
        self._filler_buffer.append(flow_in)
        f_new = float(np.clip(
            (1 - self.DT / self.TAU_F) * f
            + (self.DT / self.TAU_F) * self.K_F * delayed
            + self.rng.normal(0, self.noise_std),
            0.0, 1.0,
        ))
        v_new = self._viscosity(T_new, f_new)
        d_new = self._density(T_new, f_new)
        # Integrating level — mass balance (outflow = 0.35*L - 0.05*heat: more heat → less outflow)
        inflow  = flow_in * 0.4
        outflow = L * 0.35 - 0.05 * heat_in
        L_new   = float(np.clip(
            L + self.DT * (inflow - outflow)
            + self.rng.normal(0, self.noise_std * 0.5),
            0.0, 1.0,
        ))

        self.state = np.array([T_new, f_new, v_new, d_new, L_new], dtype=np.float32)
        self.t    += 1
        reward     = self._reward(self.state, action)
        done       = (self.t >= self.max_steps)
        info       = {'constraint_violation': self._violation(self.state)}
        return self.state.copy(), float(reward), done, info

    def _reward(self, state: np.ndarray, action: np.ndarray) -> float:
        T, f, v, d, L = state
        return (
            -2.0 * (T - self.T_TARGET)**2
            - 2.0 * (f - self.F_TARGET)**2
            - 0.5 * (L - self.LEVEL_TARGET)**2
            - 0.05 * float(np.sum(action**2))
        )

    def _violation(self, state: np.ndarray) -> float:
        return sum(
            max(0.0, lo - state[i]) + max(0.0, state[i] - hi)
            for i, (lo, hi) in enumerate(self.BOUNDS)
        )

    def _viscosity(self, T: float, f: float) -> float:
        return float(np.clip(0.75 - 0.45 * T + 0.38 * f + 0.12 * f * (1 - T), 0.0, 1.0))

    def _density(self, T: float, f: float) -> float:
        return float(np.clip(0.55 + 0.25 * f - 0.10 * T, 0.0, 1.0))


# ============================================================================
# 2. DATASET COLLECTION — INDUSTRIAL OPERATING REGIMES
# ============================================================================

def collect_industrial_dataset(
    n_episodes:   int   = 500,
    episode_len:  int   = 100,
    noise_scale:  float = 0.30,
    seed:         int   = 0,
    regime_split: Tuple[float, float, float] = (0.60, 0.25, 0.15),
) -> Dict[str, np.ndarray]:
    """
    Simulate industrial operating logs with three regimes.

    Normal (60%):    near setpoint, moderate exploration noise
    Transition (25%): shifted starting states, higher noise
    Disturbance (15%): level disturbances, unusual initial conditions

    This distribution creates the characteristic industrial imbalance:
    dense data near nominal operation, sparse data near physical boundaries.
    Algorithms must handle this imbalance without explicit correction.
    """
    env = CoatingProcessEnv(noise_std=0.015, seed=seed)
    rng = np.random.default_rng(seed)

    states, actions, rewards, next_states, dones = [], [], [], [], []

    n_normal = int(n_episodes * regime_split[0])
    n_trans  = int(n_episodes * regime_split[1])
    n_dist   = n_episodes - n_normal - n_trans

    for ep_type, n_ep in [('normal', n_normal),
                           ('transition', n_trans),
                           ('disturbance', n_dist)]:
        for ep in range(n_ep):
            obs = env.reset(seed=seed * 1000 + ep)

            if ep_type == 'transition':
                noise_ep = noise_scale * 1.2
                env.state[0] = float(np.clip(
                    env.state[0] + rng.normal(0, 0.15), 0.35, 0.85))
                env.state[1] = float(np.clip(
                    env.state[1] + rng.normal(0, 0.12), 0.20, 0.75))
                obs = env.state.copy()
            elif ep_type == 'disturbance':
                noise_ep = noise_scale * 1.5
                env.state[4] = float(rng.uniform(0.15, 0.35))
                obs = env.state.copy()
            else:
                noise_ep = noise_scale * 0.8

            for _ in range(episode_len):
                # Proportional controller + exploration
                u_heat = float(np.clip(
                    1.8 * (env.T_TARGET - obs[env.STATE_T_IDX]) + rng.normal(0, noise_ep), -1, 1))
                u_flow = float(np.clip(
                    1.8 * (env.F_TARGET - obs[env.STATE_F_IDX]) + rng.normal(0, noise_ep), -1, 1))
                action = np.array([u_heat, u_flow], dtype=np.float32)

                s2, r, done, info = env.step(action)
                states.append(obs.copy())
                actions.append(action.copy())
                rewards.append(r)
                next_states.append(s2.copy())
                dones.append(float(done))
                obs = s2
                if done:
                    break

    dataset = {
        'states':      np.array(states,      dtype=np.float32),
        'actions':     np.array(actions,     dtype=np.float32),
        'rewards':     np.array(rewards,     dtype=np.float32),
        'next_states': np.array(next_states, dtype=np.float32),
        'dones':       np.array(dones,       dtype=np.float32),
    }
    n = len(states)
    print(f"Dataset: {n:,} transitions | "
          f"reward {dataset['rewards'].mean():.3f} ± {dataset['rewards'].std():.3f}")
    return dataset


def normalize_dataset(dataset: Dict) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Per-dimension state normalization. Reward scaled by std only."""
    s_mean = dataset['states'].mean(0)
    s_std  = dataset['states'].std(0) + 1e-8
    r_std  = dataset['rewards'].std() + 1e-8

    norm = {k: v.copy() for k, v in dataset.items()}
    norm['states']      = (dataset['states']      - s_mean) / s_std
    norm['next_states'] = (dataset['next_states'] - s_mean) / s_std
    norm['rewards']     = dataset['rewards'] / r_std
    return norm, s_mean, s_std


def make_dataloader(dataset: Dict,
                    batch_size: int = 256,
                    device:     str = 'cpu') -> DataLoader:
    s  = torch.FloatTensor(dataset['states'])
    a  = torch.FloatTensor(dataset['actions'])
    r  = torch.FloatTensor(dataset['rewards'])
    s2 = torch.FloatTensor(dataset['next_states'])
    d  = torch.FloatTensor(dataset['dones'])
    return DataLoader(TensorDataset(s, a, r, s2, d),
                      batch_size=batch_size, shuffle=True, drop_last=True)


# ============================================================================
# 3. EVALUATION — INDUSTRIAL METRICS
# ============================================================================

class IndustrialEvaluator:
    """
    Evaluate a trained policy with industrial-relevant metrics.

    Metrics computed:
      reward_mean / reward_std   — episode return statistics
      T_rmse, f_rmse, level_rmse — RMSE from setpoints
      da_T, da_f                 — Directional Accuracy per variable
      da_mean                    — mean DA across controlled variables
      constraint_viol_rate       — fraction of steps with any violation
      constraint_viol_severity   — mean violation magnitude (when violated)

    Directional Accuracy (DA):
      Fraction of steps where the state variable moved toward its setpoint.
      DA = 0.5 is random; DA > 0.8 is industrially acceptable; DA < 0.5
      means the policy is systematically pushing the variable the wrong way.
      DA is important because it is agnostic to action magnitude — a policy
      that always moves in the right direction but too slowly still has DA=1.
    """

    def __init__(self, env: CoatingProcessEnv,
                 s_mean:     np.ndarray,
                 s_std:      np.ndarray,
                 n_episodes: int = 30,
                 device:     str = 'cpu'):
        self.env        = env
        self.s_mean     = s_mean
        self.s_std      = s_std
        self.n_episodes = n_episodes
        self.device     = device

    def evaluate(self, agent) -> Dict[str, float]:
        """
        Run n_episodes and compute all metrics.
        agent must have .policy with .act(state_tensor, deterministic=True).
        """
        rewards, T_errs, f_errs, L_errs = [], [], [], []
        da_T_all, da_f_all = [], []
        viol_rates, viol_sevs = [], []

        for ep in range(self.n_episodes):
            obs  = self.env.reset(seed=8000 + ep)
            ep_r = 0.0
            done = False
            steps = viol_steps = 0
            viol_total = 0.0

            while not done:
                prev_T, prev_f = obs[self.env.STATE_T_IDX], obs[self.env.STATE_F_IDX]
                s_t = torch.FloatTensor(
                    (obs - self.s_mean) / self.s_std
                ).unsqueeze(0).to(self.device)

                act = agent.policy.act(s_t, deterministic=True)
                obs, r, done, info = self.env.step(act)
                ep_r += r

                T, f, _, _, L = obs
                T_errs.append((T - self.env.T_TARGET)**2)
                f_errs.append((f - self.env.F_TARGET)**2)
                L_errs.append((L - self.env.LEVEL_TARGET)**2)

                # DA: did the state move toward setpoint?
                da_T_all.append(
                    1.0 if (T - prev_T) * (self.env.T_TARGET - prev_T) >= 0 else 0.0)
                da_f_all.append(
                    1.0 if (f - prev_f) * (self.env.F_TARGET - prev_f) >= 0 else 0.0)

                viol = info['constraint_violation']
                steps += 1
                if viol > 0:
                    viol_steps += 1
                    viol_total += viol

            rewards.append(ep_r)
            viol_rates.append(viol_steps / max(steps, 1))
            viol_sevs.append(viol_total / max(viol_steps, 1))

        return {
            'reward_mean':            float(np.mean(rewards)),
            'reward_std':             float(np.std(rewards)),
            'T_rmse':                 float(np.sqrt(np.mean(T_errs))),
            'f_rmse':                 float(np.sqrt(np.mean(f_errs))),
            'level_rmse':             float(np.sqrt(np.mean(L_errs))),
            'da_T':                   float(np.mean(da_T_all)),
            'da_f':                   float(np.mean(da_f_all)),
            'da_mean':                float(np.mean(da_T_all + da_f_all)),
            'constraint_viol_rate':   float(np.mean(viol_rates)),
            'constraint_viol_severity': float(np.mean(viol_sevs)),
        }

    def print_results(self, name: str, metrics: Dict[str, float]) -> None:
        print(f"\n{'─'*52}")
        print(f"  {name}")
        print(f"{'─'*52}")
        print(f"  Reward        : {metrics['reward_mean']:+.3f} ± {metrics['reward_std']:.3f}")
        print(f"  T  RMSE / DA  : {metrics['T_rmse']:.4f}  /  {metrics['da_T']:.1%}")
        print(f"  f  RMSE / DA  : {metrics['f_rmse']:.4f}  /  {metrics['da_f']:.1%}")
        print(f"  Level RMSE    : {metrics['level_rmse']:.4f}")
        print(f"  DA (mean)     : {metrics['da_mean']:.1%}")
        print(f"  Constraint viol: {metrics['constraint_viol_rate']:.1%} of steps  "
              f"(severity={metrics['constraint_viol_severity']:.5f})")


# ============================================================================
# 4. PHYSICS MODEL FOR COATING PROCESS
# ============================================================================

def coating_physics_fn(state: torch.Tensor,
                        action: torch.Tensor) -> torch.Tensor:
    """
    Approximate first-principles physics model for the 5-variable coating process.

    Implements the known part of the dynamics; leaves cross-coupling, transport
    delay, and nonlinear viscosity corrections to the neural residual network.

    Physics coverage per dimension (approximate):
      temperature:     ~90%  (dominant first-order response)
      filler_fraction: ~80%  (delay model not included)
      viscosity:       ~65%  (linearized)
      density:         ~80%
      level:           ~92%  (mass balance is nearly exact)
    """
    TAU_T = CoatingProcessEnv.TAU_T
    TAU_F = CoatingProcessEnv.TAU_F
    K_T   = CoatingProcessEnv.K_T
    K_F   = CoatingProcessEnv.K_F
    DT    = CoatingProcessEnv.DT

    T  = state[:, 0]
    f  = state[:, 1]
    L  = state[:, 4]

    heat_in = action[:, 0] * 0.5 + 0.5
    flow_in = action[:, 1] * 0.5 + 0.5

    T_new = (1 - DT / TAU_T) * T + (DT / TAU_T) * K_T * heat_in
    f_new = (1 - DT / TAU_F) * f + (DT / TAU_F) * K_F * flow_in

    # Linearized viscosity: viscosity ≈ 0.75 - 0.45*T + 0.38*f
    v_new = (0.75 - 0.45 * T_new + 0.38 * f_new).clamp(0.0, 1.0)

    # Linear density
    d_new = (0.55 + 0.25 * f_new - 0.10 * T_new).clamp(0.0, 1.0)

    # Mass balance for level (outflow = 0.35*L - 0.05*heat per true dynamics)
    inflow  = flow_in * 0.4
    outflow = L * 0.35 - 0.05 * heat_in
    L_new   = (L + DT * (inflow - outflow)).clamp(0.0, 1.0)

    return torch.stack([T_new, f_new, v_new, d_new, L_new], dim=1)


def make_coating_constraints(dataset: Dict,
                              device: str = 'cpu') -> Tuple[List, List]:
    """
    Build physics constraint functions and calibrated lambdas for CQL+Physics.

    Returns:
      constraints: list of g(s, a, s') -> Tensor (batch,), violation ≥ 0
      lambdas:     list of floats, calibrated via Theorem 6.1

    Three constraints:
      1. Operating bounds: all state variables within hard limits
      2. Temperature first-order consistency
      3. Filler fraction first-order consistency (no delay — approximate)
    """
    lower = [lo for lo, hi in CoatingProcessEnv.BOUNDS]
    upper = [hi for lo, hi in CoatingProcessEnv.BOUNDS]

    def bounds_fn(s, a, s2):
        return bounds_constraint(s, a, s2, lower=lower, upper=upper)

    def temp_fn(s, a, s2):
        return first_order_constraint(
            s, a, s2, state_idx=0, action_idx=0,
            tau=CoatingProcessEnv.TAU_T, K=CoatingProcessEnv.K_T,
            dt=CoatingProcessEnv.DT, tol=0.04)

    def filler_fn(s, a, s2):
        return first_order_constraint(
            s, a, s2, state_idx=1, action_idx=1,
            tau=CoatingProcessEnv.TAU_F, K=CoatingProcessEnv.K_F,
            dt=CoatingProcessEnv.DT, tol=0.05)

    # Calibrate bounds lambda from Theorem 6.1
    def base_r(s, a, s2):
        return (
            -2.0 * (s2[:, 0] - CoatingProcessEnv.T_TARGET)**2
            - 2.0 * (s2[:, 1] - CoatingProcessEnv.F_TARGET)**2
        )

    lam_bounds = calibrate_lambda(
        dataset, bounds_fn, base_r, target_gap_fraction=0.10, device=device)

    constraints = [bounds_fn, temp_fn, filler_fn]
    lambdas     = [lam_bounds, 2.0, 2.0]
    return constraints, lambdas


# ============================================================================
# 5. ALGORITHM WRAPPERS
# ============================================================================

class PhysicsInformedCQL:
    """
    CQL with physics reward shaping (Chapter 3 + Chapter 6).

    The modification is minimal: apply PhysicsRewardWrapper to the reward
    before every CQL update. No changes to the CQL algorithm itself.
    This is the "one-line" physics integration: any offline RL agent can
    be made constraint-aware by modifying only its reward signal.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 constraints: List, lambdas: List,
                 alpha_cql:   float = 1.0,
                 hidden_dim:  int   = 256,
                 device:      str   = 'cpu'):
        self.device = device

        def _zero_r(s, a, s2):           # placeholder base reward
            return torch.zeros(s.shape[0], device=s.device)

        self.reward_wrapper = PhysicsRewardWrapper(
            base_reward=_zero_r,
            constraints=constraints,
            lambdas=lambdas,
        )
        self.cql    = CQLAgent(state_dim, action_dim, hidden_dim,
                               alpha_cql=alpha_cql, device=device)
        self.policy = self.cql.policy   # expose for evaluator

    def update(self, batch: Tuple) -> Dict:
        s, a, r, s2, d = [x.to(self.device) for x in batch]
        with torch.no_grad():
            penalty = sum(
                lam * g(s, a, s2)
                for g, lam in zip(self.reward_wrapper.constraints,
                                   self.reward_wrapper.lambdas)
            )
        return self.cql.update((s, a, r - penalty, s2, d))


class HybridMOReL:
    """
    MOReL with hybrid dynamics ensemble (Chapter 5 + Chapter 6).

    Replaces MOReL's pure neural ensemble with HybridEnsemble.
    All rollout generation, P-MDP construction, and SAC training
    logic is unchanged from Chapter 5 — only the dynamics model changes.

    The hybrid ensemble improves two things:
    1. Dynamics accuracy in OOD regions (physics provides a consistent prior)
    2. Calibrated uncertainty: the residual network has lower variance than
       a full black-box, so ensemble disagreement better reflects genuine
       epistemic uncertainty rather than fitting noise.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 physics_fn,
                 n_ensemble:  int   = 5,
                 hidden_dim:  int   = 128,
                 halt_thresh: float = 0.15,
                 horizon:     int   = 5,
                 device:      str   = 'cpu'):
        self.device = device

        self.hybrid_ensemble = HybridEnsemble(
            n_models   = n_ensemble,
            state_dim  = state_dim,
            action_dim = action_dim,
            physics_fn = physics_fn,
            hidden_dim = hidden_dim,
            device     = device,
        )
        self.morel_agent = MOReeLAgent(
            state_dim   = state_dim,
            action_dim  = action_dim,
            n_models    = n_ensemble,
            hidden_dim  = hidden_dim,
            halt_thresh = halt_thresh,
            horizon     = horizon,
            device      = device,
        )
        self.policy = self.morel_agent.policy

    def train_dynamics(self, dataset: Dict,
                       n_epochs: int = 50, batch_size: int = 256) -> None:
        """Train hybrid ensemble, then swap into MOReL."""
        print("\n  Physics coverage diagnosis:")
        self.hybrid_ensemble.diagnose_physics_coverage(dataset)
        print("\n  Training hybrid ensemble...")
        self.hybrid_ensemble.train_ensemble(
            dataset, n_epochs=n_epochs, batch_size=batch_size)
        # Swap: MOReL.generate_synthetic_data calls self.ensemble
        self.morel_agent.ensemble = self.hybrid_ensemble
        print("  ✓ HybridEnsemble installed in MOReL")

    def generate_synthetic_data(self, real_states: torch.Tensor) -> Dict:
        return self.morel_agent.generate_synthetic_data(real_states)

    def update(self, batch: Tuple, synthetic: Optional[Dict]) -> Dict:
        return self.morel_agent.update(batch, synthetic)


# ============================================================================
# 6. TRAINING LOOPS
# ============================================================================

def train_bc(agent, loader: DataLoader,
             n_epochs: int = 60, log_every: int = 20) -> None:
    for epoch in range(1, n_epochs + 1):
        total, nb = 0.0, 0
        for batch in loader:
            info = agent.update(batch)
            total += list(info.values())[0]
            nb    += 1
        if epoch % log_every == 0:
            key = list(info.keys())[0]
            print(f"  BC  | epoch {epoch:3d} | {key}={total/nb:.4f}")


def train_cql_agent(agent, loader: DataLoader,
                    n_epochs: int = 80, log_every: int = 20) -> None:
    for epoch in range(1, n_epochs + 1):
        acc, nb = {}, 0
        for batch in loader:
            info = agent.update(batch)
            for k, v in info.items():
                acc[k] = acc.get(k, 0.0) + v
            nb += 1
        if epoch % log_every == 0:
            parts = [f"epoch {epoch:3d}"] + [
                f"{k}={v/nb:.4f}" for k, v in acc.items()]
            print("  CQL | " + " | ".join(parts))


def train_hybrid_morel_agent(
    agent:              HybridMOReL,
    dataset:            Dict,
    norm_dataset:       Dict,
    env:                CoatingProcessEnv,
    s_mean:             np.ndarray,
    s_std:              np.ndarray,
    n_dynamics_epochs:  int = 40,
    n_outer_iters:      int = 20,
    sac_steps_per_iter: int = 500,
    batch_size:         int = 256,
    device:             str = 'cpu',
) -> List[float]:
    """Two-phase HybridMOReL training: dynamics first, then policy."""
    agent.train_dynamics(norm_dataset,
                         n_epochs=n_dynamics_epochs, batch_size=batch_size)
    print("\n  MOReL outer loop...")
    return train_morel(
        agent               = agent.morel_agent,
        dataset             = norm_dataset,
        env                 = env,
        s_mean              = s_mean,
        s_std               = s_std,
        n_outer_iters       = n_outer_iters,
        sac_steps_per_iter  = sac_steps_per_iter,
        batch_size          = batch_size,
        device              = device,
        eval_every          = 5,
    )


# ============================================================================
# 7. VISUALIZATION
# ============================================================================

def plot_comparison(results: Dict[str, Dict[str, float]],
                    save_path: str = 'ch7_comparison.png') -> None:
    """Four-panel comparison: reward, DA, RMSE, constraint violation rate."""
    names   = list(results.keys())
    colours = ['#5b9bd5', '#70ad47', '#ffc000', '#e04545'][:len(names)]
    x       = np.arange(len(names))

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Chapter 7: Industrial Benchmark — Thermal Coating Process',
                 fontsize=13, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    # Panel 1: Reward
    ax1 = fig.add_subplot(gs[0, 0])
    means = [results[n]['reward_mean'] for n in names]
    stds  = [results[n]['reward_std']  for n in names]
    bars  = ax1.bar(x, means, yerr=stds, capsize=5, color=colours,
                    edgecolor='white', error_kw={'elinewidth': 1.5})
    ax1.axhline(0, color='k', lw=0.7, ls='--', alpha=0.4)
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=15, ha='right')
    ax1.set_ylabel('Mean Episode Reward')
    ax1.set_title('Return (↑ better)')
    for bar, m in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + abs(bar.get_height()) * 0.02,
                 f'{m:.2f}', ha='center', va='bottom', fontsize=8)

    # Panel 2: Directional Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    da_T = [results[n]['da_T'] for n in names]
    da_f = [results[n]['da_f'] for n in names]
    w = 0.35
    ax2.bar(x - w/2, da_T, w, label='Temperature',     color='#4472c4', alpha=0.85)
    ax2.bar(x + w/2, da_f, w, label='Filler fraction', color='#ed7d31', alpha=0.85)
    ax2.axhline(0.5, color='r', lw=1.0, ls='--', alpha=0.6, label='Random (0.5)')
    ax2.axhline(0.8, color='g', lw=1.0, ls=':',  alpha=0.6, label='Good DA (0.8)')
    ax2.set_ylim(0, 1.05)
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=15, ha='right')
    ax2.set_ylabel('Directional Accuracy')
    ax2.set_title('DA (↑ better, >0.8 = industrially useful)')
    ax2.legend(fontsize=7, loc='lower right')

    # Panel 3: RMSE
    ax3 = fig.add_subplot(gs[1, 0])
    w2 = 0.25
    ax3.bar(x - w2, [results[n]['T_rmse']     for n in names], w2,
            label='Temperature',     color='#4472c4', alpha=0.85)
    ax3.bar(x,      [results[n]['f_rmse']     for n in names], w2,
            label='Filler fraction', color='#ed7d31', alpha=0.85)
    ax3.bar(x + w2, [results[n]['level_rmse'] for n in names], w2,
            label='Level',           color='#a9d18e', alpha=0.85)
    ax3.set_xticks(x); ax3.set_xticklabels(names, rotation=15, ha='right')
    ax3.set_ylabel('RMSE from setpoint')
    ax3.set_title('Tracking error (↓ better)')
    ax3.legend(fontsize=7)

    # Panel 4: Constraint violations
    ax4 = fig.add_subplot(gs[1, 1])
    viol = [results[n]['constraint_viol_rate'] * 100 for n in names]
    bars4 = ax4.bar(x, viol, color=colours, edgecolor='white')
    ax4.set_xticks(x); ax4.set_xticklabels(names, rotation=15, ha='right')
    ax4.set_ylabel('Constraint violation rate (%)')
    ax4.set_title('Physical constraint violations (↓ better)')
    for bar, v in zip(bars4, viol):
        ax4.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.2, f'{v:.1f}%',
                 ha='center', va='bottom', fontsize=8)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved: {save_path}")


def plot_episode_trajectory(
    agent,
    env:        CoatingProcessEnv,
    s_mean:     np.ndarray,
    s_std:      np.ndarray,
    agent_name: str = 'Agent',
    device:     str = 'cpu',
    seed:       int = 42,
    save_path:  str = 'ch7_trajectory.png',
) -> None:
    """Plot single-episode state trajectory with setpoints and constraint bounds."""
    obs  = env.reset(seed=seed)
    done = False
    T_h, f_h, L_h, viol_h = [], [], [], []

    while not done:
        s_t = torch.FloatTensor((obs - s_mean) / s_std).unsqueeze(0).to(device)
        act = agent.policy.act(s_t, deterministic=True)
        obs, r, done, info = env.step(act)
        T_h.append(obs[env.STATE_T_IDX]); f_h.append(obs[env.STATE_F_IDX])
        L_h.append(obs[4]); viol_h.append(info['constraint_violation'])

    t   = np.arange(len(T_h))
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    fig.suptitle(f'Episode trajectory — {agent_name}', fontsize=12)

    for ax, hist, tgt, lo, hi, name, col in [
        (axes[0], T_h, env.T_TARGET,     *env.BOUNDS[0], 'Temperature',     '#4472c4'),
        (axes[1], f_h, env.F_TARGET,     *env.BOUNDS[1], 'Filler fraction', '#ed7d31'),
        (axes[2], L_h, env.LEVEL_TARGET, *env.BOUNDS[4], 'Level',           '#70ad47'),
    ]:
        ax.plot(t, hist, col, lw=1.5)
        ax.axhline(tgt, color=col, lw=1.0, ls='--', alpha=0.7, label='setpoint')
        ax.axhline(lo,  color='red', lw=0.8, ls=':',  alpha=0.6)
        ax.axhline(hi,  color='red', lw=0.8, ls=':',  alpha=0.6)
        ax.axhspan(lo, lo + 0.015, alpha=0.12, color='red')
        ax.axhspan(hi - 0.015, hi, alpha=0.12, color='red')
        ax.set_ylabel(name, fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8, loc='upper right')

    axes[3].plot(t, viol_h, '#e04545', lw=1.2)
    axes[3].fill_between(t, 0, viol_h, alpha=0.3, color='#e04545')
    axes[3].set_ylabel('Violation', fontsize=9)
    axes[3].set_xlabel('Time step')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Trajectory plot saved: {save_path}")


# ============================================================================
# 8. RESULTS TABLE
# ============================================================================

def print_results_table(results: Dict[str, Dict[str, float]]) -> None:
    metrics = [
        ('reward_mean',            'Reward (mean)',    '{:+.3f}', 'max'),
        ('da_mean',                'DA (mean)',        '{:.1%}',  'max'),
        ('da_T',                   'DA temperature',   '{:.1%}',  'max'),
        ('da_f',                   'DA filler',        '{:.1%}',  'max'),
        ('T_rmse',                 'T RMSE',           '{:.4f}',  'min'),
        ('f_rmse',                 'f RMSE',           '{:.4f}',  'min'),
        ('level_rmse',             'Level RMSE',       '{:.4f}',  'min'),
        ('constraint_viol_rate',   'Violation rate',   '{:.1%}',  'min'),
    ]
    names  = list(results.keys())
    col_w  = max(len(n) for n in names) + 3
    mw     = 20

    hdr = f"{'Metric':{mw}}" + "".join(f"{n:>{col_w}}" for n in names)
    print("\n" + "=" * len(hdr))
    print("  CHAPTER 10 — INDUSTRIAL BENCHMARK RESULTS")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    for key, label, fmt, direction in metrics:
        vals     = [results[n][key] for n in names]
        best_idx = int(np.argmin(vals) if direction == 'min' else np.argmax(vals))
        row      = f"{label:{mw}}"
        for i, v in enumerate(vals):
            cell = fmt.format(v)
            tag  = '▶ ' if i == best_idx else '  '
            row += f"{tag + cell:>{col_w}}"
        print(row)

    print("=" * len(hdr))
    print("  ▶ = best value for this metric\n")


# ============================================================================
# 9. MAIN BENCHMARK
# ============================================================================

def run_industrial_benchmark(
    device:        str  = 'cpu',
    n_train_ep:    int  = 400,
    n_bc_epochs:   int  = 60,
    n_cql_epochs:  int  = 80,
    n_morel_iters: int  = 15,
    n_eval_ep:     int  = 20,
    batch_size:    int  = 256,
    quick_test:    bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Full 4-algorithm comparison on the industrial coating process:

      1. BC            (Chapter 1) — behavioral cloning baseline
      2. CQL           (Chapter 3) — conservative Q-learning
      3. CQL+Physics   (Chapter 3 + 6) — CQL with physics reward shaping
      4. HybridMOReL   (Chapter 5 + 6) — model-based + hybrid dynamics

    quick_test=True runs in minimal settings for CI / smoke testing.
    """
    if quick_test:
        n_train_ep     = 50
        n_bc_epochs    = 5
        n_cql_epochs   = 5
        n_morel_iters  = 3
        n_eval_ep      = 5

    print("=" * 60)
    print("Chapter 7: Industrial Offline RL Benchmark")
    print("  Process : Thermal Coating Line (anonymized)")
    print(f"  State   : {CoatingProcessEnv.STATE_DIM}D  "
          f"Action: {CoatingProcessEnv.ACTION_DIM}D")
    print("=" * 60)

    S   = CoatingProcessEnv.STATE_DIM
    A   = CoatingProcessEnv.ACTION_DIM
    env = CoatingProcessEnv(seed=0)

    # Data
    print(f"\n[1] Collecting dataset ({n_train_ep} episodes)...")
    dataset          = collect_industrial_dataset(n_episodes=n_train_ep, seed=0)
    norm_dataset, s_mean, s_std = normalize_dataset(dataset)
    loader           = make_dataloader(norm_dataset, batch_size, device)
    s_tensor         = torch.FloatTensor(norm_dataset['states'])
    evaluator        = IndustrialEvaluator(env, s_mean, s_std, n_eval_ep, device)

    results   = {}
    agent_map = {}

    # BC
    if _CQL_AVAILABLE:
        print(f"\n[2] Training BC ({n_bc_epochs} epochs)...")
        bc = CQLBCAgent(S, A, device=device)
        train_bc(bc, loader, n_epochs=n_bc_epochs)
        m = evaluator.evaluate(bc)
        evaluator.print_results("BC (Chapter 1)", m)
        results['BC'] = m; agent_map['BC'] = bc

    # CQL
    if _CQL_AVAILABLE:
        print(f"\n[3] Training CQL ({n_cql_epochs} epochs)...")
        cql = CQLAgent(S, A, alpha_cql=1.0, device=device)
        train_cql_agent(cql, loader, n_cql_epochs)
        m = evaluator.evaluate(cql)
        evaluator.print_results("CQL (Chapter 3)", m)
        results['CQL'] = m; agent_map['CQL'] = cql

    # CQL + Physics
    if _CQL_AVAILABLE and _PHYSICS_AVAILABLE:
        print(f"\n[4] Training CQL+Physics ({n_cql_epochs} epochs)...")
        constraints, lambdas = make_coating_constraints(dataset, device)
        cql_phys = PhysicsInformedCQL(
            S, A, constraints=constraints, lambdas=lambdas,
            alpha_cql=1.0, device=device)
        train_cql_agent(cql_phys, loader, n_cql_epochs)
        m = evaluator.evaluate(cql_phys)
        evaluator.print_results("CQL+Physics (Ch3+6)", m)
        results['CQL+Physics'] = m; agent_map['CQL+Physics'] = cql_phys

    # HybridMOReL
    if _MOREL_AVAILABLE and _PHYSICS_AVAILABLE:
        print(f"\n[5] Training HybridMOReL ({n_morel_iters} outer iters)...")
        hm = HybridMOReL(S, A, physics_fn=coating_physics_fn,
                         n_ensemble=4, hidden_dim=128,
                         halt_thresh=0.15, device=device)
        train_hybrid_morel_agent(
            agent              = hm,
            dataset            = dataset,
            norm_dataset       = norm_dataset,
            env                = env,
            s_mean             = s_mean,
            s_std              = s_std,
            n_dynamics_epochs  = 30 if not quick_test else 3,
            n_outer_iters      = n_morel_iters,
            sac_steps_per_iter = 300 if not quick_test else 20,
            batch_size         = batch_size,
            device             = device,
        )
        m = evaluator.evaluate(hm)
        evaluator.print_results("HybridMOReL (Ch5+6)", m)
        results['HybridMOReL'] = m; agent_map['HybridMOReL'] = hm

    if results:
        print_results_table(results)
        plot_comparison(results)
        # Trajectory for best policy by DA
        best = max(results, key=lambda k: results[k]['da_mean'])
        if best in agent_map:
            plot_episode_trajectory(agent_map[best], env, s_mean, s_std,
                                    agent_name=best, device=device)

    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--quick',  action='store_true')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    run_industrial_benchmark(device=args.device, quick_test=args.quick)

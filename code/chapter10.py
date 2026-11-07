"""
chapter10.py
===========
Industrial Applications of Offline RL — complete case study.
Chapter 10 of "Offline RL: From Theory to Industrial Practice"

This file integrates algorithms from Chapters 1–6 into a single
industrial pipeline. It does NOT reimplement existing algorithms;
instead it imports them and adds only what is new for Chapter 10:

  CoatingProcessEnv       : 5-variable thermal coating process
  IndustrialDataset       : realistic log collection with operating regimes
  IndustrialEvaluator     : DA outside the noise band, IAE, overshoot, violations
  PhysicsInformedCQL      : CQL + physics reward shaping (Ch3 + Ch6)
  HybridMOReL             : MOReL with hybrid ensemble (Ch5 + Ch6)
  run_industrial_benchmark: 4-algorithm comparison for one seed
  run_multi_seed          : several seeds, bootstrap CIs, JSON results

Algorithms imported (not reimplemented):
  Ch1: BCAgent            from behavioral_cloning.py / cql.py (BCAgent)
  Ch4: CQLAgent           from cql.py
  Ch8: MOReeLAgent        from morel.py
  Ch9: PhysicsRewardWrapper, HybridEnsemble, calibrate_lambda,
       bounds_constraint                         from physics_informed.py

Results are produced by run_multi_seed() and stored in
results/ch10_benchmark.json; the chapter quotes that file.
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
        calibrate_lambda, bounds_constraint,
    )
    _PHYSICS_AVAILABLE = True
except ImportError:
    _PHYSICS_AVAILABLE = False


# ============================================================================
# 1. ENVIRONMENT — THERMAL COATING PROCESS
# ============================================================================

class CoatingProcessEnv:
    """
    Anonymized thermal coating process — industrial case study for Chapter 10.

    Extends ThermalProcessEnv from Chapters 1–5 to 5 state variables,
    representative of a real multi-variable continuous manufacturing process.

    State (5 variables, all normalized to [0, 1]):
      s[0]  temperature     — process temperature in the coating zone
      s[1]  filler_fraction — fraction of filler material in the blend
      s[2]  viscosity       — blend viscosity (nonlinear fn of T and filler)
      s[3]  density         — bulk material density
      s[4]  level           — surge tank level (integrating: no self-regulation)

    Actions (2 variables):
      a[0]  heat_input  — heater setpoint delta, in [-1, 1]
      a[1]  flow_input  — filler flow rate delta, in [-1, 1]

    True dynamics (unknown to agent):
      temperature:     first-order, tau=12, K=0.85, cross-coupling with filler
      filler_fraction: first-order, tau=8,  K=0.90, with 2-step transport delay
      viscosity:       nonlinear(T, filler) + noise
      density:         linear(T, filler) + small drift
      level:           integrating: L_{t+1} = L_t + dt*(0.4*flow_in - Q_OUT)
                       Q_OUT is set by the downstream consumer and does not depend on L

    Known physics model (available to agent):
      temperature:     first-order, tau=12, K=0.85     (~90% variance explained)
      filler_fraction: first-order, tau=8,  K=0.90     (~85%, delay ignored)
      viscosity:       linearized approximation         (~65%)
      density:         linear combination               (~80%)
      level:           mass balance (exact)

    Hard operating constraints:
      temperature:     [0.35, 0.85]
      filler_fraction: [0.20, 0.75]
      viscosity:       [0.10, 0.90]
      density:         [0.30, 0.80]
      level:           [0.15, 0.85]

    The level variable is the key industrial challenge: it integrates the
    imbalance between filler inflow and downstream demand, has no natural
    equilibrium, and shares its only actuator (flow_in) with the filler fraction.
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
    # Downstream demand, chosen so that inflow balances outflow when the filler
    # loop sits at its setpoint (f* = K_F * flow_in*  =>  flow_in* = F_TARGET / K_F).
    Q_OUT = 0.4 * F_TARGET / K_F

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
        # Integrating level: inflow from the filler feed, outflow fixed by the
        # downstream consumer. No term pulls L back by itself.
        inflow  = flow_in * 0.4
        L_new   = float(np.clip(
            L + self.DT * (inflow - self.Q_OUT)
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
                # Proportional operators + exploration. The feed operator also
                # watches the surge tank and nudges the feed to keep it from
                # draining or overflowing: one actuator, two goals.
                u_heat = float(np.clip(
                    1.8 * (env.T_TARGET - obs[env.STATE_T_IDX]) + rng.normal(0, noise_ep), -1, 1))
                u_flow = float(np.clip(
                    1.8 * (env.F_TARGET - obs[env.STATE_F_IDX])
                    + 1.0 * (env.LEVEL_TARGET - obs[4]) + rng.normal(0, noise_ep), -1, 1))
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
    Evaluate a trained policy with industrial-relevant metrics, per episode.

    For temperature (T) and filler fraction (f):
      iae          — mean |x - x*| per step (integrated absolute error / T)
      rmse         — root-mean-square error from setpoint
      da           — directional accuracy, counted ONLY on steps where the
                     previous error exceeds a noise band (k * noise_std). Inside
                     the band the sign of the error is set by noise, and a
                     perfect regulator would score ~50%.
      overshoot    — fraction of those steps where the variable crosses the
                     setpoint and lands more than one band beyond it
    For the level: iae and rmse.
    Constraints: violation rate (fraction of steps with any bound violated)
                 and severity (mean violation magnitude when violated).

    Half of the episodes start near the setpoints; the other half start
    disturbed, like the 'disturbance' regime in the logs (low tank level,
    shifted temperature and filler). Constraints are only tested in those.
    """

    METRICS = ['reward', 'T_iae', 'f_iae', 'L_iae', 'T_rmse', 'f_rmse', 'L_rmse',
               'da_T', 'da_f', 'overshoot_T', 'overshoot_f',
               'viol_rate', 'viol_severity',
               'viol_rate_disturbed', 'L_iae_disturbed']

    def __init__(self, env: CoatingProcessEnv,
                 s_mean:     np.ndarray,
                 s_std:      np.ndarray,
                 n_episodes: int = 20,
                 band_k:     float = 2.0,
                 seed_base:  int = 8000,
                 device:     str = 'cpu'):
        self.env        = env
        self.s_mean     = s_mean
        self.s_std      = s_std
        self.n_episodes = n_episodes
        self.band       = band_k * env.noise_std
        self.seed_base  = seed_base
        self.device     = device

    def _episode(self, agent, seed: int, disturbed: bool = False) -> Dict[str, float]:
        env  = self.env
        obs  = env.reset(seed=seed)
        if disturbed:
            rng = np.random.default_rng(seed + 50_000)
            env.state[4] = float(rng.uniform(0.15, 0.25))
            env.state[0] = float(np.clip(env.state[0] + rng.choice([-1, 1]) * 0.12, 0.35, 0.85))
            env.state[1] = float(np.clip(env.state[1] + rng.choice([-1, 1]) * 0.10, 0.20, 0.75))
            obs = env.state.copy()
        tgt  = {'T': env.T_TARGET, 'f': env.F_TARGET}
        idx  = {'T': env.STATE_T_IDX, 'f': env.STATE_F_IDX}
        err  = {k: [] for k in ('T', 'f', 'L')}
        da   = {'T': [], 'f': []}
        over = {'T': [], 'f': []}
        ep_r, steps, viol_steps, viol_total, done = 0.0, 0, 0, 0.0, False

        while not done:
            prev = {k: float(obs[idx[k]]) for k in ('T', 'f')}
            s_t  = torch.FloatTensor((obs - self.s_mean) / self.s_std).unsqueeze(0).to(self.device)
            act  = agent.policy.act(s_t, deterministic=True)
            obs, r, done, info = env.step(act)
            ep_r += r
            for k in ('T', 'f'):
                e_prev = tgt[k] - prev[k]
                e_new  = tgt[k] - float(obs[idx[k]])
                err[k].append(e_new)
                if abs(e_prev) > self.band:
                    da[k].append(1.0 if (float(obs[idx[k]]) - prev[k]) * e_prev > 0 else 0.0)
                    over[k].append(1.0 if (e_new * e_prev < 0 and abs(e_new) > self.band) else 0.0)
            err['L'].append(env.LEVEL_TARGET - float(obs[4]))
            steps += 1
            v = info['constraint_violation']
            if v > 0:
                viol_steps += 1
                viol_total += v

        m = {'reward': ep_r}
        for k in ('T', 'f', 'L'):
            e = np.abs(np.array(err[k]))
            m[f'{k}_iae']  = float(e.mean())
            m[f'{k}_rmse'] = float(np.sqrt((e ** 2).mean()))
        for k in ('T', 'f'):
            m[f'da_{k}']        = float(np.mean(da[k])) if da[k] else float('nan')
            m[f'overshoot_{k}'] = float(np.mean(over[k])) if over[k] else float('nan')
        m['viol_rate']     = viol_steps / max(steps, 1)
        m['viol_severity'] = viol_total / viol_steps if viol_steps else 0.0
        m['disturbed']     = disturbed
        return m

    def evaluate(self, agent) -> Dict[str, object]:
        """Returns per-metric means and the per-episode values (key 'episodes')."""
        eps = [self._episode(agent, self.seed_base + ep, disturbed=(ep % 2 == 1))
               for ep in range(self.n_episodes)]
        dist = [e for e in eps if e['disturbed']]
        out = {k: float(np.nanmean([e[k] for e in eps]))
               for k in self.METRICS if not k.endswith('_disturbed')}
        out['viol_rate_disturbed'] = float(np.mean([e['viol_rate'] for e in dist]))
        out['L_iae_disturbed']     = float(np.mean([e['L_iae'] for e in dist]))
        out['episodes'] = eps
        return out

    def print_results(self, name: str, m: Dict[str, float]) -> None:
        print(f"\n{'─'*60}\n  {name}\n{'─'*60}")
        print(f"  Return         : {m['reward']:+.2f}")
        print(f"  T  IAE / DA / overshoot : {m['T_iae']:.4f} / {m['da_T']:.1%} / {m['overshoot_T']:.1%}")
        print(f"  f  IAE / DA / overshoot : {m['f_iae']:.4f} / {m['da_f']:.1%} / {m['overshoot_f']:.1%}")
        print(f"  Level IAE      : {m['L_iae']:.4f}")
        print(f"  Violations     : {m['viol_rate']:.1%} of steps (severity {m['viol_severity']:.4f}); "
              f"disturbed starts: {m['viol_rate_disturbed']:.1%}, level IAE {m['L_iae_disturbed']:.4f}")


# ============================================================================
# 4. PHYSICS MODEL FOR COATING PROCESS
# ============================================================================

def coating_physics_fn(state: torch.Tensor,
                        action: torch.Tensor) -> torch.Tensor:
    """
    Approximate first-principles physics model for the 5-variable coating process.

    Implements the known part of the dynamics; leaves cross-coupling, transport
    delay, and nonlinear viscosity corrections to the neural residual network.

    Omitted on purpose: the 0.03*f*u_T cross-coupling, the 2-step filler delay,
    and the 0.12*f*(1-T) viscosity term. Measure how much each dimension's
    increment the model explains with HybridEnsemble.diagnose_physics_coverage.
      level:           mass balance (exact up to noise)
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

    # Mass balance for level: inflow from the feed, outflow set downstream
    inflow  = flow_in * 0.4
    L_new   = (L + DT * (inflow - CoatingProcessEnv.Q_OUT)).clamp(0.0, 1.0)

    return torch.stack([T_new, f_new, v_new, d_new, L_new], dim=1)


def coating_reward_torch(next_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """The environment's reward, in raw units, for batches of tensors."""
    T, f, L = next_state[:, 0], next_state[:, 1], next_state[:, 4]
    return (-2.0 * (T - CoatingProcessEnv.T_TARGET) ** 2
            - 2.0 * (f - CoatingProcessEnv.F_TARGET) ** 2
            - 0.5 * (L - CoatingProcessEnv.LEVEL_TARGET) ** 2
            - 0.05 * action.pow(2).sum(-1))


class Units:
    """
    Converts between the normalized space the agents train in and the raw
    physical units in which bounds, physics and rewards are defined.
    Everything physical must be evaluated in raw units.
    """
    def __init__(self, s_mean: np.ndarray, s_std: np.ndarray, r_std: float, device: str = 'cpu'):
        self.mean  = torch.FloatTensor(s_mean).to(device)
        self.std   = torch.FloatTensor(s_std).to(device)
        self.r_std = float(r_std)

    def raw(self, s_norm: torch.Tensor) -> torch.Tensor:
        return s_norm * self.std + self.mean

    def norm(self, s_raw: torch.Tensor) -> torch.Tensor:
        return (s_raw - self.mean) / self.std

    def physics_fn(self, s_norm: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.norm(coating_physics_fn(self.raw(s_norm), a))

    def reward_fn(self, s_norm, a, s2_norm) -> torch.Tensor:
        return coating_reward_torch(self.raw(s2_norm), a) / self.r_std


def make_bounds_penalty(dataset_raw: Dict, units: Units,
                        cost_ratio: float = 10.0, device: str = 'cpu'):
    """
    Penalty for leaving the hard operating bounds, in normalized-reward units.

    Only the bounds constraint is used. First-order consistency checks are
    about the dynamics model, not the policy: on real logs they would penalize
    transitions in which the true (unmodeled) delay and coupling show up.
    """
    lower = [lo for lo, hi in CoatingProcessEnv.BOUNDS]
    upper = [hi for lo, hi in CoatingProcessEnv.BOUNDS]

    def bounds_raw(s, a, s2):
        return bounds_constraint(s, a, s2, lower=lower, upper=upper)

    def reward_raw(s, a, s2):
        return coating_reward_torch(s2, a)

    lam = calibrate_lambda(dataset_raw, bounds_raw, reward_raw,
                           cost_ratio=cost_ratio, device=device)

    def penalty(s_norm, a, s2_norm):
        g = bounds_raw(units.raw(s_norm), a, units.raw(s2_norm))
        return lam * g / units.r_std

    return penalty, lam


# ============================================================================
# 5. ALGORITHM WRAPPERS
# ============================================================================

class PhysicsInformedCQL:
    """
    CQL with a constraint penalty subtracted from the reward (Chapter 4 + Chapter 9).
    The CQL algorithm itself is unchanged; only its reward signal is.
    """

    def __init__(self, state_dim: int, action_dim: int, penalty_fn,
                 alpha_cql: float = 1.0, hidden_dim: int = 256, device: str = 'cpu'):
        self.device  = device
        self.penalty = penalty_fn
        self.cql     = CQLAgent(state_dim, action_dim, hidden_dim,
                                alpha_cql=alpha_cql, device=device)
        self.policy  = self.cql.policy

    def update(self, batch: Tuple) -> Dict:
        s, a, r, s2, d = [x.to(self.device) for x in batch]
        with torch.no_grad():
            pen = self.penalty(s, a, s2)
        return self.cql.update((s, a, r - pen, s2, d))


class HybridMOReL:
    """
    MOReL with a hybrid dynamics ensemble (Chapter 8 + Chapter 9).

    The ensemble predicts physics + learned residual. Caveat (Chapter 9): every
    member shares the same physics, so where the physics is wrong the members
    agree and the uncertainty used for HALT is too low.
    """

    def __init__(self, state_dim: int, action_dim: int, units: Units,
                 kappa: float, n_ensemble: int = 4, hidden_dim: int = 128,
                 rollout_horizon: int = 5, epsilon_percentile: float = 80.0,
                 device: str = 'cpu'):
        self.device = device
        self.epsilon_percentile = epsilon_percentile
        self.hybrid_ensemble = HybridEnsemble(
            n_models=n_ensemble, state_dim=state_dim, action_dim=action_dim,
            physics_fn=units.physics_fn, hidden_dim=hidden_dim, device=device)
        self.morel_agent = MOReeLAgent(
            state_dim, action_dim, n_ensemble=n_ensemble,
            rollout_horizon=rollout_horizon, kappa=kappa, device=device)
        # The coating reward (raw units, then scaled like the dataset rewards)
        self.morel_agent._model_reward = units.reward_fn
        self.policy = self.morel_agent.policy

    def train_dynamics(self, dataset: Dict, n_epochs: int = 30, batch_size: int = 256):
        print("\n  Physics coverage (increment variance explained):")
        coverage = self.hybrid_ensemble.diagnose_physics_coverage(dataset)
        self.hybrid_ensemble.train_ensemble(dataset, n_epochs=n_epochs,
                                            batch_size=batch_size, log_every=n_epochs)
        self.morel_agent.ensemble = self.hybrid_ensemble
        self.morel_agent.epsilon = self.hybrid_ensemble.calibrate_epsilon(
            dataset, percentile=self.epsilon_percentile)
        return coverage


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


def train_hybrid_morel_agent(agent: HybridMOReL, norm_dataset: Dict, env: CoatingProcessEnv,
                             s_mean: np.ndarray, s_std: np.ndarray,
                             n_dynamics_epochs: int = 30, n_outer_iters: int = 15,
                             sac_steps_per_iter: int = 300, batch_size: int = 256,
                             device: str = 'cpu'):
    """Two-phase HybridMOReL training: dynamics first, then the P-MDP policy loop."""
    coverage = agent.train_dynamics(norm_dataset, n_epochs=n_dynamics_epochs,
                                    batch_size=batch_size)
    train_morel(agent=agent.morel_agent, dataset=norm_dataset, env=env,
                s_mean=s_mean, s_std=s_std, n_outer_iters=n_outer_iters,
                sac_steps_per_iter=sac_steps_per_iter, batch_size=batch_size,
                device=device, eval_every=n_outer_iters + 1)   # no env peeking
    return coverage


# ============================================================================
# 7. VISUALIZATION
# ============================================================================

def plot_episode_trajectory(
    agent,
    env:        CoatingProcessEnv,
    s_mean:     np.ndarray,
    s_std:      np.ndarray,
    agent_name: str = 'Agent',
    device:     str = 'cpu',
    seed:       int = 42,
    save_path:  str = 'ch10_trajectory.png',
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

def run_industrial_benchmark(seed: int = 0, device: str = 'cpu',
                             n_train_ep: int = 400, n_bc_epochs: int = 60,
                             n_cql_epochs: int = 80, n_morel_iters: int = 15,
                             n_eval_ep: int = 20, batch_size: int = 256,
                             quick_test: bool = False) -> Dict[str, object]:
    """
    One seed of the 4-algorithm comparison. The seed controls the logged
    dataset, network initialization and the evaluation episodes.
    """
    if quick_test:
        n_train_ep, n_bc_epochs, n_cql_epochs, n_morel_iters, n_eval_ep = 40, 2, 2, 2, 3

    torch.manual_seed(seed)
    np.random.seed(seed)
    S, A = CoatingProcessEnv.STATE_DIM, CoatingProcessEnv.ACTION_DIM
    env  = CoatingProcessEnv(seed=seed)

    dataset = collect_industrial_dataset(n_episodes=n_train_ep, seed=seed)
    norm_dataset, s_mean, s_std = normalize_dataset(dataset)
    r_std  = float(dataset['rewards'].std() + 1e-8)
    units  = Units(s_mean, s_std, r_std, device)
    loader = make_dataloader(norm_dataset, batch_size, device)
    evaluator = IndustrialEvaluator(env, s_mean, s_std, n_eval_ep,
                                    seed_base=8000 + 100 * seed, device=device)
    out = {'seed': seed, 'results': {}, 'extra': {}}

    print(f"\n[seed {seed}] BC")
    bc = CQLBCAgent(S, A, device=device)
    train_bc(bc, loader, n_epochs=n_bc_epochs)
    out['results']['BC'] = evaluator.evaluate(bc)

    print(f"\n[seed {seed}] CQL")
    cql = CQLAgent(S, A, alpha_cql=1.0, device=device)
    train_cql_agent(cql, loader, n_cql_epochs)
    out['results']['CQL'] = evaluator.evaluate(cql)

    print(f"\n[seed {seed}] CQL+Physics")
    penalty, lam = make_bounds_penalty(dataset, units, device=device)
    cql_p = PhysicsInformedCQL(S, A, penalty, alpha_cql=1.0, device=device)
    train_cql_agent(cql_p, loader, n_cql_epochs)
    out['results']['CQL+Physics'] = evaluator.evaluate(cql_p)
    out['extra']['lambda_bounds'] = lam

    print(f"\n[seed {seed}] HybridMOReL")
    # HALT must outweigh staying forever in the worst-rewarded state (Chapter 8).
    r_min = float(np.abs(norm_dataset['rewards']).max())
    kappa = max(100.0, 2.0 * r_min / (1 - 0.99))
    hm = HybridMOReL(S, A, units, kappa=kappa, device=device)
    coverage = train_hybrid_morel_agent(
        hm, norm_dataset, env, s_mean, s_std,
        n_dynamics_epochs=30 if not quick_test else 2,
        n_outer_iters=n_morel_iters,
        sac_steps_per_iter=300 if not quick_test else 20,
        batch_size=batch_size, device=device)
    out['results']['HybridMOReL'] = evaluator.evaluate(hm)
    out['extra'].update({'kappa': kappa, 'physics_coverage': [float(c) for c in coverage],
                         'halt_rate_last': float(hm.morel_agent.halt_rate_history[-1])
                         if hm.morel_agent.halt_rate_history else None,
                         'n_transitions': int(len(dataset['states']))})

    for name, m in out['results'].items():
        evaluator.print_results(f"{name} (seed {seed})", m)
    return out


def bootstrap_ci(values, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI of the mean."""
    v = np.asarray([x for x in values if not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(1)
    return float(v.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def summarize(runs: List[Dict]) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    """
    Hierarchical bootstrap over seeds: per-seed means are resampled, so the
    interval reflects seed-to-seed variation (dataset + training), which is
    larger than the episode-to-episode noise within one trained policy.
    """
    methods = list(runs[0]['results'].keys())
    summary = {}
    for mth in methods:
        summary[mth] = {k: bootstrap_ci([r['results'][mth][k] for r in runs])
                        for k in IndustrialEvaluator.METRICS}
    return summary


def print_summary(summary) -> None:
    rows = [('reward', 'Return', '{:+.1f}'), ('T_iae', 'T IAE', '{:.4f}'),
            ('f_iae', 'f IAE', '{:.4f}'), ('L_iae', 'Level IAE', '{:.4f}'),
            ('da_T', 'DA temperature', '{:.1%}'), ('da_f', 'DA filler', '{:.1%}'),
            ('overshoot_T', 'Overshoot T', '{:.1%}'), ('overshoot_f', 'Overshoot f', '{:.1%}'),
            ('viol_rate', 'Violation rate', '{:.1%}'),
            ('viol_rate_disturbed', 'Viol. (disturbed)', '{:.1%}'),
            ('L_iae_disturbed', 'Level IAE (dist.)', '{:.4f}')]
    names = list(summary.keys())
    print("\n" + "=" * 100)
    print("  CHAPTER 10 — mean over seeds [95% bootstrap CI]")
    print("=" * 100)
    print(f"{'Metric':16}" + "".join(f"{n:>21}" for n in names))
    for key, label, fmt in rows:
        cells = []
        for n in names:
            m, lo, hi = summary[n][key]
            cells.append(f"{fmt.format(m)} [{fmt.format(lo)}, {fmt.format(hi)}]")
        print(f"{label:16}" + "".join(f"{c:>21}" for c in cells))


def run_multi_seed(seeds=(0, 1, 2, 3, 4), out_path: Optional[str] = None,
                   quick_test: bool = False, device: str = 'cpu') -> Dict:
    runs = [run_industrial_benchmark(seed=s, device=device, quick_test=quick_test) for s in seeds]
    summary = summarize(runs)
    print_summary(summary)
    if out_path:
        import json
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        slim = [{'seed': r['seed'], 'extra': r['extra'],
                 'results': {m: {k: v for k, v in res.items() if k != 'episodes'}
                             for m, res in r['results'].items()}} for r in runs]
        with open(out_path, 'w') as fh:
            json.dump({'seeds': list(seeds), 'runs': slim,
                       'summary': {m: {k: list(v) for k, v in d.items()} for m, d in summary.items()}},
                      fh, indent=2)
        print(f"\nSaved {out_path}")
    return {'runs': runs, 'summary': summary}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--quick', action='store_true')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--out', default=os.path.join(_HERE, 'results', 'ch10_benchmark.json'))
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    run_multi_seed(tuple(args.seeds), out_path=None if args.quick else args.out,
                   quick_test=args.quick, device=args.device)

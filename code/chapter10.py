"""
chapter10.py
===========
Industrial Applications of Offline RL – complete case study.
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
# 1. ENVIRONMENT – THERMAL COATING PROCESS
# ============================================================================

class CoatingProcessEnv:
    """
    Anonymized thermal coating process – industrial case study for Chapter 10.

    Extends ThermalProcessEnv from Chapters 1–5 to 5 state variables,
    representative of a real multi-variable continuous manufacturing process.

    State (5 variables, all normalized to [0, 1]):
      s[0]  temperature     – process temperature in the coating zone
      s[1]  filler_fraction – fraction of filler material in the blend
      s[2]  viscosity       – blend viscosity (nonlinear fn of T and filler)
      s[3]  density         – bulk material density
      s[4]  level           – surge tank level (integrating: no self-regulation)

    Actions (2 variables):
      a[0]  heat_input  – heater setpoint delta, in [-1, 1]
      a[1]  flow_input  – filler flow rate delta, in [-1, 1]

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
                 seed: int = 42,
                 lagged_flow: bool = False):
        self.noise_std = noise_std
        self.max_steps = max_steps
        # The filler responds to the feed two steps late, so the 5 process
        # variables alone are not a Markov state. With lagged_flow=True the
        # observation also carries the two feed commands still in transit.
        self.lagged_flow = lagged_flow
        self.obs_dim   = self.STATE_DIM + (2 if lagged_flow else 0)
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
        return self.observe()

    def observe(self) -> np.ndarray:
        """Process variables, plus [u_f(t-2), u_f(t-1)] when lagged_flow is on."""
        if not self.lagged_flow:
            return self.state.copy()
        return np.concatenate([self.state, np.array(self._filler_buffer, dtype=np.float32)])

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
        return self.observe(), float(reward), done, info

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
# 2. DATASET COLLECTION – INDUSTRIAL OPERATING REGIMES
# ============================================================================

def collect_industrial_dataset(
    n_episodes:   int   = 500,
    episode_len:  int   = 100,
    noise_scale:  float = 0.30,
    seed:         int   = 0,
    regime_split: Tuple[float, float, float] = (0.60, 0.25, 0.15),
    lagged_flow:  bool  = False,
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
    env = CoatingProcessEnv(noise_std=0.015, seed=seed, lagged_flow=lagged_flow)
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
                obs = env.observe()
            elif ep_type == 'disturbance':
                noise_ep = noise_scale * 1.5
                env.state[4] = float(rng.uniform(0.15, 0.35))
                obs = env.observe()
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
                # Episodes end on a time limit, not in a terminal state, so the
                # stored flag stays 0 and TD targets keep bootstrapping.
                dones.append(0.0)
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
# 3. EVALUATION – INDUSTRIAL METRICS
# ============================================================================

class IndustrialEvaluator:
    """
    Evaluate a trained policy with industrial-relevant metrics, per episode.

    For temperature (T) and filler fraction (f):
      iae          – mean |x - x*| per step (integrated absolute error / T)
      rmse         – root-mean-square error from setpoint
      da           – directional accuracy, counted ONLY on steps where the
                     previous error exceeds a noise band (k * noise_std). Inside
                     the band the sign of the error is set by noise, and a
                     perfect regulator would score ~50%.
      overshoot    – fraction of those steps where the variable crosses the
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
               'viol_rate', 'viol_severity', 'viol_max',
               'reward_nominal', 'reward_disturbed', 'L_iae_nominal',
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
            obs = env.observe()
        tgt  = {'T': env.T_TARGET, 'f': env.F_TARGET}
        idx  = {'T': env.STATE_T_IDX, 'f': env.STATE_F_IDX}
        err  = {k: [] for k in ('T', 'f', 'L')}
        da   = {'T': [], 'f': []}
        over = {'T': [], 'f': []}
        ep_r, steps, viol_steps, viol_total, viol_max, done = 0.0, 0, 0, 0.0, 0.0, False

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
                viol_max = max(viol_max, v)

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
        m['viol_max']      = viol_max
        m['disturbed']     = disturbed
        return m

    def evaluate(self, agent) -> Dict[str, object]:
        """Returns per-metric means and the per-episode values (key 'episodes')."""
        eps = [self._episode(agent, self.seed_base + ep, disturbed=(ep % 2 == 1))
               for ep in range(self.n_episodes)]
        dist = [e for e in eps if e['disturbed']]
        nom  = [e for e in eps if not e['disturbed']]
        split = ('_disturbed', '_nominal')
        out = {k: float(np.nanmean([e[k] for e in eps]))
               for k in self.METRICS if not k.endswith(split) and k != 'viol_max'}
        out['viol_max']            = float(np.max([e['viol_max'] for e in eps]))
        out['reward_nominal']      = float(np.mean([e['reward'] for e in nom]))
        out['reward_disturbed']    = float(np.mean([e['reward'] for e in dist]))
        out['L_iae_nominal']       = float(np.mean([e['L_iae'] for e in nom]))
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

    Omitted on purpose: the 0.03*f*u_T cross-coupling and the 0.12*f*(1-T)
    viscosity term. The 2-step filler delay is omitted too when the state has
    only the 5 process variables; with the lagged feed commands appended
    (lagged_flow=True, 7 columns) the delay is applied exactly. Measure how much each dimension's
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

    lagged  = state.shape[1] > CoatingProcessEnv.STATE_DIM
    feed_f  = state[:, 5] if lagged else flow_in      # the feed that reaches the zone now

    T_new = (1 - DT / TAU_T) * T + (DT / TAU_T) * K_T * heat_in
    f_new = (1 - DT / TAU_F) * f + (DT / TAU_F) * K_F * feed_f

    # Linearized viscosity: viscosity ≈ 0.75 - 0.45*T + 0.38*f
    v_new = (0.75 - 0.45 * T_new + 0.38 * f_new).clamp(0.0, 1.0)

    # Linear density
    d_new = (0.55 + 0.25 * f_new - 0.10 * T_new).clamp(0.0, 1.0)

    # Mass balance for level: inflow from the feed, outflow set downstream
    inflow  = flow_in * 0.4
    L_new   = (L + DT * (inflow - CoatingProcessEnv.Q_OUT)).clamp(0.0, 1.0)

    out = [T_new, f_new, v_new, d_new, L_new]
    if lagged:
        out += [state[:, 6], flow_in]                  # shift the feed commands in transit
    return torch.stack(out, dim=1)


def coating_reward_torch(next_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """The environment's reward, in raw units, for batches of tensors."""
    T, f, L = next_state[:, 0], next_state[:, 1], next_state[:, 4]
    return (-2.0 * (T - CoatingProcessEnv.T_TARGET) ** 2
            - 2.0 * (f - CoatingProcessEnv.F_TARGET) ** 2
            - 0.5 * (L - CoatingProcessEnv.LEVEL_TARGET) ** 2
            - 0.05 * action.pow(2).sum(-1))


def coating_true_dynamics_torch(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    The environment's own dynamics without noise, for batches in raw units.
    Needs the lagged state (7 columns): the filler responds to the feed command
    issued two steps earlier. Used only for the oracle ablation.
    """
    E = CoatingProcessEnv
    T, f, L, u_lag2, u_lag1 = state[:, 0], state[:, 1], state[:, 4], state[:, 5], state[:, 6]
    heat_in = action[:, 0].clamp(-1, 1) * 0.5 + 0.5
    flow_in = action[:, 1].clamp(-1, 1) * 0.5 + 0.5
    T_new = ((1 - E.DT / E.TAU_T) * T + (E.DT / E.TAU_T) * E.K_T * heat_in
             + 0.03 * f * heat_in).clamp(0.0, 1.0)
    f_new = ((1 - E.DT / E.TAU_F) * f + (E.DT / E.TAU_F) * E.K_F * u_lag2).clamp(0.0, 1.0)
    v_new = (0.75 - 0.45 * T_new + 0.38 * f_new + 0.12 * f_new * (1 - T_new)).clamp(0.0, 1.0)
    d_new = (0.55 + 0.25 * f_new - 0.10 * T_new).clamp(0.0, 1.0)
    L_new = (L + E.DT * (0.4 * flow_in - E.Q_OUT)).clamp(0.0, 1.0)
    return torch.stack([T_new, f_new, v_new, d_new, L_new, u_lag1, flow_in], dim=1)


class OracleDynamics:
    """
    Stand-in for a dynamics ensemble that knows the true (noise-free) dynamics:
    same interface as DynamicsEnsemble, zero uncertainty, so MOReL never halts.
    """
    def __init__(self, units: 'Units'):
        self.units = units

    def predict_with_uncertainty(self, s_norm: torch.Tensor, a: torch.Tensor):
        with torch.no_grad():
            nxt = self.units.norm(coating_true_dynamics_torch(self.units.raw(s_norm), a))
        return nxt, torch.zeros(len(s_norm), device=s_norm.device)


class OracleWithLearnedUncertainty:
    """True dynamics for the predictions, the hybrid ensemble's u for HALT:
    the oracle arm of the 2x2 ablation with the same termination mechanism."""
    def __init__(self, units: 'Units', ensemble):
        self.oracle, self.ensemble = OracleDynamics(units), ensemble

    def predict_with_uncertainty(self, s_norm: torch.Tensor, a: torch.Tensor):
        nxt, _ = self.oracle.predict_with_uncertainty(s_norm, a)
        with torch.no_grad():
            _, u = self.ensemble.predict_with_uncertainty(s_norm, a)
        return nxt, u


def rollout_error_analysis(ensemble, units: 'Units', norm_dataset: Dict, policy, epsilon: float,
                           horizons=(1, 2, 5, 10, 20), n_starts: int = 400,
                           episode_len: int = 100, seed: int = 0) -> Dict[str, Dict[str, Dict]]:
    """
    Multi-step error of the hybrid model against the true noise-free dynamics,
    from the same start states. Open loop: both follow the same logged action
    sequence. Closed loop: the trained policy acts on its own trajectory in
    each – on the model's predicted states in one, on the true states in the
    other – which is what MOReL's synthetic rollouts look like. For each
    horizon k: RMS error over the five process variables (normalized units),
    the mean uncertainty u at step k, the share of rollouts whose u has
    exceeded epsilon by step k (where MOReL would have halted), and the rank
    correlation of u with the error at step k.
    """
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    H = max(horizons)
    n_ep = len(norm_dataset['states']) // episode_len
    ep = rng.integers(0, n_ep, size=n_starts)
    off = rng.integers(0, episode_len - H, size=n_starts)
    start = ep * episode_len + off
    P = CoatingProcessEnv.STATE_DIM
    out = {}
    for mode in ('open loop', 'closed loop'):
        s_mod = torch.FloatTensor(norm_dataset['states'][start])
        s_true = s_mod.clone()
        halted = torch.zeros(n_starts, dtype=torch.bool)
        rows = {}
        for k in range(1, H + 1):
            with torch.no_grad():
                if mode == 'open loop':
                    a_mod = a_true = torch.FloatTensor(norm_dataset['actions'][start + k - 1])
                else:
                    a_mod = torch.as_tensor(policy.act(s_mod, deterministic=True),
                                            dtype=torch.float32).reshape(n_starts, -1)
                    a_true = torch.as_tensor(policy.act(s_true, deterministic=True),
                                             dtype=torch.float32).reshape(n_starts, -1)
                s_mod, u = ensemble.predict_with_uncertainty(s_mod, a_mod)
                s_true = units.norm(coating_true_dynamics_torch(units.raw(s_true), a_true))
            halted |= u > epsilon
            if k in horizons:
                err = (s_mod - s_true)[:, :P].norm(dim=-1)
                rows[k] = {'rmse': float(err.pow(2).mean().sqrt()),
                           'mean_u': float(u.mean()),
                           'halted_by_k': float(halted.float().mean()),
                           'spearman_u_err': float(spearmanr(u.numpy(), err.numpy())[0])}
        out[mode] = rows
    return out


def uncertainty_calibration(ensemble, units: 'Units', norm_dataset: Dict, policy=None,
                            n: int = 4000, seed: int = 0) -> Dict[str, Dict[str, float]]:
    """
    Does the ensemble's uncertainty u predict its actual one-step error? The
    oracle gives the true next state anywhere, so the error can be measured on
    dataset actions, on the trained policy's actions and on random actions.
    Reports the Spearman correlation of u with the error, the mean error in the
    lowest and highest quintile of u, and the error among pairs MOReL would call
    KNOWN (u <= epsilon) versus UNKNOWN.
    """
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(norm_dataset['states']), size=min(n, len(norm_dataset['states'])), replace=False)
    s = torch.FloatTensor(norm_dataset['states'][idx])
    sources = {'dataset actions': torch.FloatTensor(norm_dataset['actions'][idx]),
               'random actions': torch.FloatTensor(rng.uniform(-1, 1, size=(len(idx), 2)))}
    if policy is not None:
        with torch.no_grad():
            sources['policy actions'] = torch.as_tensor(policy.act(s, deterministic=True),
                                                        dtype=torch.float32).reshape(len(idx), -1)
    eps = getattr(ensemble, 'epsilon_used', None)
    out = {}
    for name, a in sources.items():
        with torch.no_grad():
            pred, u = ensemble.predict_with_uncertainty(s, a)
            true = units.norm(coating_true_dynamics_torch(units.raw(s), a))
        err = (pred - true)[:, :CoatingProcessEnv.STATE_DIM].norm(dim=-1).numpy()
        u = u.numpy()
        q = np.quantile(u, [0.2, 0.8])
        row = {'spearman': float(spearmanr(u, err)[0]),
               'err_low_u': float(err[u <= q[0]].mean()),
               'err_high_u': float(err[u >= q[1]].mean()),
               'mean_u': float(u.mean()), 'mean_err': float(err.mean())}
        if eps is not None:
            known = u <= eps
            row['known_frac'] = float(known.mean())
            row['err_known'] = float(err[known].mean()) if known.any() else float('nan')
            row['err_unknown'] = float(err[~known].mean()) if (~known).any() else float('nan')
        out[name] = row
    return out


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

    n = CoatingProcessEnv.STATE_DIM                    # bounds apply to the process variables

    def bounds_raw(s, a, s2):
        return bounds_constraint(s[:, :n], a, s2[:, :n], lower=lower, upper=upper)

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
                 device: str = 'cpu', physics: bool = True):
        self.device = device
        self.epsilon_percentile = epsilon_percentile
        # physics=False: persistence (s' = s) in place of the physics model, so
        # the residual learns the whole increment -- a black-box ensemble.
        self.hybrid_ensemble = HybridEnsemble(
            n_models=n_ensemble, state_dim=state_dim, action_dim=action_dim,
            physics_fn=units.physics_fn if physics else (lambda s, a: s),
            hidden_dim=hidden_dim, device=device)
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
    fig.suptitle(f'Episode trajectory – {agent_name}', fontsize=12)

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
                             lagged_flow: bool = False,
                             quick_test: bool = False) -> Dict[str, object]:
    """
    One seed of the 4-algorithm comparison. The seed controls the logged
    dataset, network initialization and the evaluation episodes.
    """
    if quick_test:
        n_train_ep, n_bc_epochs, n_cql_epochs, n_morel_iters, n_eval_ep = 40, 2, 2, 2, 3

    torch.manual_seed(seed)
    np.random.seed(seed)
    env  = CoatingProcessEnv(seed=seed, lagged_flow=lagged_flow)
    S, A = env.obs_dim, CoatingProcessEnv.ACTION_DIM

    dataset = collect_industrial_dataset(n_episodes=n_train_ep, seed=seed, lagged_flow=lagged_flow)
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


def run_oracle_ablation(seed: int = 0, device: str = 'cpu', n_train_ep: int = 400,
                        n_morel_iters: int = 15, n_eval_ep: int = 20, batch_size: int = 256,
                        quick_test: bool = False) -> Dict[str, object]:
    """
    One seed of the oracle ablation, on the Markov (lagged) state: the same
    MOReL policy optimizer (SAC, rollout horizon, reward, budget, policy
    initialization) trained in a 2x2 design – learned hybrid or true
    noise-free dynamics, with HALT driven by the hybrid ensemble's uncertainty
    or with no HALT. Also measures how well that uncertainty predicts the
    model's one-step error and how the error grows over multi-step rollouts.
    """
    if quick_test:
        n_train_ep, n_morel_iters, n_eval_ep = 40, 2, 3
    torch.manual_seed(seed)
    np.random.seed(seed)
    env  = CoatingProcessEnv(seed=seed, lagged_flow=True)
    S, A = env.obs_dim, CoatingProcessEnv.ACTION_DIM
    dataset = collect_industrial_dataset(n_episodes=n_train_ep, seed=seed, lagged_flow=True)
    norm_dataset, s_mean, s_std = normalize_dataset(dataset)
    units = Units(s_mean, s_std, float(dataset['rewards'].std() + 1e-8), device)
    evaluator = IndustrialEvaluator(env, s_mean, s_std, n_eval_ep,
                                    seed_base=8000 + 100 * seed, device=device)
    kappa = max(100.0, 2.0 * float(np.abs(norm_dataset['rewards']).max()) / (1 - 0.99))
    sac = dict(n_outer_iters=n_morel_iters, sac_steps_per_iter=300 if not quick_test else 20)
    out = {'seed': seed, 'results': {}, 'extra': {}}

    print(f"\n[seed {seed}] HybridMOReL (lagged state)")
    torch.manual_seed(seed + 1)
    hm = HybridMOReL(S, A, units, kappa=kappa, device=device)
    train_hybrid_morel_agent(hm, norm_dataset, env, s_mean, s_std,
                             n_dynamics_epochs=30 if not quick_test else 2,
                             batch_size=batch_size, device=device, **sac)
    out['results']['HybridMOReL'] = evaluator.evaluate(hm)
    hm.hybrid_ensemble.epsilon_used = hm.morel_agent.epsilon
    out['extra']['calibration'] = uncertainty_calibration(
        hm.hybrid_ensemble, units, norm_dataset, policy=hm.policy, seed=seed)
    out['extra']['halt_rate_last'] = float(hm.morel_agent.halt_rate_history[-1])

    print(f"\n[seed {seed}] OracleMOReL (true dynamics, same optimizer)")
    torch.manual_seed(seed + 1)                        # same policy initialization
    om = MOReeLAgent(S, A, n_ensemble=4, rollout_horizon=5, kappa=kappa, device=device)
    om._model_reward = units.reward_fn
    om.ensemble = OracleDynamics(units)
    om.epsilon = 1.0                                   # uncertainty is 0: never HALT
    train_morel(agent=om, dataset=norm_dataset, env=env, s_mean=s_mean, s_std=s_std,
                batch_size=batch_size, device=device, eval_every=n_morel_iters + 1, **sac)
    out['results']['OracleMOReL'] = evaluator.evaluate(om)

    # Second half of the 2x2 design: each model with the other termination rule.
    print(f"\n[seed {seed}] HybridMOReL without HALT (learned dynamics, never halts)")
    torch.manual_seed(seed + 1)
    hn = MOReeLAgent(S, A, n_ensemble=4, rollout_horizon=5, kappa=kappa, device=device)
    hn._model_reward = units.reward_fn
    hn.ensemble = hm.hybrid_ensemble
    hn.epsilon = float('inf')
    train_morel(agent=hn, dataset=norm_dataset, env=env, s_mean=s_mean, s_std=s_std,
                batch_size=batch_size, device=device, eval_every=n_morel_iters + 1, **sac)
    out['results']['HybridMOReL-noHALT'] = evaluator.evaluate(hn)

    print(f"\n[seed {seed}] OracleMOReL with HALT (true dynamics, the hybrid ensemble's u)")
    torch.manual_seed(seed + 1)
    oh = MOReeLAgent(S, A, n_ensemble=4, rollout_horizon=5, kappa=kappa, device=device)
    oh._model_reward = units.reward_fn
    oh.ensemble = OracleWithLearnedUncertainty(units, hm.hybrid_ensemble)
    oh.epsilon = hm.morel_agent.epsilon
    train_morel(agent=oh, dataset=norm_dataset, env=env, s_mean=s_mean, s_std=s_std,
                batch_size=batch_size, device=device, eval_every=n_morel_iters + 1, **sac)
    out['results']['OracleMOReL-HALT'] = evaluator.evaluate(oh)

    out['extra']['rollout_error'] = rollout_error_analysis(
        hm.hybrid_ensemble, units, norm_dataset, hm.policy, hm.morel_agent.epsilon, seed=seed)
    for name, m in out['results'].items():
        evaluator.print_results(f"{name} (seed {seed})", m)
    return out


def run_halt_study(seed: int = 0, device: str = 'cpu', lagged_flow: bool = True,
                   n_train_ep: int = 400, n_morel_iters: int = 15, n_eval_ep: int = 20,
                   batch_size: int = 256, quick_test: bool = False) -> Dict[str, object]:
    """
    One seed of the HALT study: one hybrid ensemble, then the same MOReL agent
    (same policy initialization, SAC budget, kappa) with three termination
    rules – epsilon at the 80th percentile of the ensemble's uncertainty on the
    data, at the 99th, or no HALT. On the lagged state the first arm repeats
    HybridMOReL of the oracle ablation exactly.
    """
    if quick_test:
        n_train_ep, n_morel_iters, n_eval_ep = 40, 2, 3
    torch.manual_seed(seed)
    np.random.seed(seed)
    env  = CoatingProcessEnv(seed=seed, lagged_flow=lagged_flow)
    S, A = env.obs_dim, CoatingProcessEnv.ACTION_DIM
    dataset = collect_industrial_dataset(n_episodes=n_train_ep, seed=seed, lagged_flow=lagged_flow)
    norm_dataset, s_mean, s_std = normalize_dataset(dataset)
    units = Units(s_mean, s_std, float(dataset['rewards'].std() + 1e-8), device)
    evaluator = IndustrialEvaluator(env, s_mean, s_std, n_eval_ep,
                                    seed_base=8000 + 100 * seed, device=device)
    kappa = max(100.0, 2.0 * float(np.abs(norm_dataset['rewards']).max()) / (1 - 0.99))
    sac = dict(n_outer_iters=n_morel_iters, sac_steps_per_iter=300 if not quick_test else 20)
    out = {'seed': seed, 'results': {}, 'extra': {'lagged_flow': lagged_flow}}

    print(f"\n[seed {seed}] HALT at the 80th percentile")
    torch.manual_seed(seed + 1)
    hm = HybridMOReL(S, A, units, kappa=kappa, device=device)
    train_hybrid_morel_agent(hm, norm_dataset, env, s_mean, s_std,
                             n_dynamics_epochs=30 if not quick_test else 2,
                             batch_size=batch_size, device=device, **sac)
    out['results']['HALT p80'] = evaluator.evaluate(hm)
    out['extra']['halt_rate p80'] = float(np.mean(hm.morel_agent.halt_rate_history))

    eps99 = hm.hybrid_ensemble.calibrate_epsilon(norm_dataset, percentile=99.0)
    for name, eps in [('HALT p99', eps99), ('no HALT', float('inf'))]:
        print(f"\n[seed {seed}] {name}")
        torch.manual_seed(seed + 1)
        np.random.seed(seed + 1)
        ag = MOReeLAgent(S, A, n_ensemble=4, rollout_horizon=5, kappa=kappa, device=device)
        ag._model_reward = units.reward_fn
        ag.ensemble = hm.hybrid_ensemble
        ag.epsilon = eps
        train_morel(agent=ag, dataset=norm_dataset, env=env, s_mean=s_mean, s_std=s_std,
                    batch_size=batch_size, device=device, eval_every=n_morel_iters + 1, **sac)
        out['results'][name] = evaluator.evaluate(ag)
        out['extra'][f'halt_rate {name.split()[-1]}'] = float(np.mean(ag.halt_rate_history))
    # The same 99th-percentile threshold with a penalty taken from the data
    # instead of the worst case: the magnitude of the most negative discounted
    # return-to-go in the logs, i.e. as bad as the worst logged future.
    r = norm_dataset['rewards'].reshape(-1, 100)            # 100-step logged episodes
    g, rtg = np.zeros(len(r)), np.zeros_like(r)
    for k in range(r.shape[1] - 1, -1, -1):
        g = r[:, k] + 0.99 * g
        rtg[:, k] = g
    kappa_data = float(np.abs(rtg.min()))
    print(f"\n[seed {seed}] HALT p99, data kappa = {kappa_data:.1f} (worst-case kappa = {kappa:.1f})")
    torch.manual_seed(seed + 1)
    np.random.seed(seed + 1)
    ag = MOReeLAgent(S, A, n_ensemble=4, rollout_horizon=5, kappa=kappa_data, device=device)
    ag._model_reward = units.reward_fn
    ag.ensemble = hm.hybrid_ensemble
    ag.epsilon = eps99
    train_morel(agent=ag, dataset=norm_dataset, env=env, s_mean=s_mean, s_std=s_std,
                batch_size=batch_size, device=device, eval_every=n_morel_iters + 1, **sac)
    out['results']['HALT p99, data kappa'] = evaluator.evaluate(ag)
    out['extra']['halt_rate p99 data kappa'] = float(np.mean(ag.halt_rate_history))
    out['extra'].update({'epsilon p80': float(hm.morel_agent.epsilon), 'epsilon p99': eps99,
                         'kappa': kappa, 'kappa_data': kappa_data})
    for name, m in out['results'].items():
        evaluator.print_results(f"{name} (seed {seed})", m)
    return out


@torch.no_grad()
def mean_q_on_data(agent, norm_dataset: Dict, device: str = 'cpu') -> float:
    """Mean min(Q1, Q2)(s, pi(s)) over the dataset states: a value that grows
    far beyond the logged returns would be the overestimation of Sims et al."""
    s = torch.FloatTensor(norm_dataset['states']).to(device)
    a, _ = agent.policy.sample(s)
    return float(torch.min(agent.Q1(s, a), agent.Q2(s, a)).mean())


def run_baselines_study(seed: int = 0, device: str = 'cpu', n_train_ep: int = 400,
                        n_morel_iters: int = 15, n_cql_epochs: int = 80, n_eval_ep: int = 20,
                        batch_size: int = 256, cql_alphas=(0.1, 0.3, 3.0, 10.0),
                        quick_test: bool = False) -> Dict[str, object]:
    """
    One seed of the baselines study, on the Markov (lagged) state and the same
    datasets as the benchmark. Does the model-based agent without HALT gain from
    its synthetic data at all (the same SAC on real data only), does it collapse
    with a longer budget (three times the SAC steps), and is CQL with alpha = 1
    a fair baseline (an alpha grid)? The hybrid ensemble and the no-HALT arm
    repeat those of the HALT study exactly.
    """
    if quick_test:
        n_train_ep, n_morel_iters, n_cql_epochs, n_eval_ep = 40, 2, 2, 3
    torch.manual_seed(seed)
    np.random.seed(seed)
    env  = CoatingProcessEnv(seed=seed, lagged_flow=True)
    S, A = env.obs_dim, CoatingProcessEnv.ACTION_DIM
    dataset = collect_industrial_dataset(n_episodes=n_train_ep, seed=seed, lagged_flow=True)
    norm_dataset, s_mean, s_std = normalize_dataset(dataset)
    units = Units(s_mean, s_std, float(dataset['rewards'].std() + 1e-8), device)
    evaluator = IndustrialEvaluator(env, s_mean, s_std, n_eval_ep,
                                    seed_base=8000 + 100 * seed, device=device)
    kappa = max(100.0, 2.0 * float(np.abs(norm_dataset['rewards']).max()) / (1 - 0.99))
    steps = 300 if not quick_test else 20
    out = {'seed': seed, 'results': {}, 'extra': {}}

    torch.manual_seed(seed + 1)                       # as in the HALT study
    hm = HybridMOReL(S, A, units, kappa=kappa, device=device)
    hm.train_dynamics(norm_dataset, n_epochs=30 if not quick_test else 2, batch_size=batch_size)

    for name, iters, real_ratio in [('no HALT', n_morel_iters, 0.5),
                                    ('no HALT, 3x budget', 3 * n_morel_iters, 0.5),
                                    ('SAC, real data only', n_morel_iters, 1.0)]:
        print(f"\n[seed {seed}] {name}")
        torch.manual_seed(seed + 1)
        np.random.seed(seed + 1)
        ag = MOReeLAgent(S, A, n_ensemble=4, rollout_horizon=5, kappa=kappa,
                         real_ratio=real_ratio, device=device)
        ag._model_reward = units.reward_fn
        ag.ensemble = hm.hybrid_ensemble if real_ratio < 1.0 else OracleDynamics(units)
        ag.epsilon = float('inf')
        train_morel(agent=ag, dataset=norm_dataset, env=env, s_mean=s_mean, s_std=s_std,
                    n_outer_iters=iters, sac_steps_per_iter=steps, batch_size=batch_size,
                    device=device, eval_every=iters + 1)
        out['results'][name] = evaluator.evaluate(ag)
        out['extra'][f'q_data {name}'] = mean_q_on_data(ag, norm_dataset, device)

    loader = make_dataloader(norm_dataset, batch_size, device)
    for alpha in cql_alphas:
        name = f'CQL alpha={alpha:g}'
        print(f"\n[seed {seed}] {name}")
        torch.manual_seed(seed + 2)
        cql = CQLAgent(S, A, alpha_cql=alpha, device=device)
        train_cql_agent(cql, loader, n_cql_epochs)
        out['results'][name] = evaluator.evaluate(cql)
    for name, m in out['results'].items():
        evaluator.print_results(f"{name} (seed {seed})", m)
    return out


def run_edge_study(seed: int = 0, device: str = 'cpu', n_train_ep: int = 400,
                   n_morel_iters: int = 15, n_eval_ep: int = 20, batch_size: int = 256,
                   quick_test: bool = False) -> Dict[str, object]:
    """
    One seed of the edge-of-reach check (Sims et al., 2024), on the Markov state:
    model-based SAC without pessimism on the true dynamics, with half of every
    batch real data (as in the factorial ablation) or 5%, as in their
    HalfCheetah runs, and with three times the SAC budget. Edge-of-reach states
    hurt only through bootstrapping from states that are never updated; real
    data and dataset-covered states update them anyway.
    """
    if quick_test:
        n_train_ep, n_morel_iters, n_eval_ep = 40, 2, 3
    torch.manual_seed(seed)
    np.random.seed(seed)
    env  = CoatingProcessEnv(seed=seed, lagged_flow=True)
    S, A = env.obs_dim, CoatingProcessEnv.ACTION_DIM
    dataset = collect_industrial_dataset(n_episodes=n_train_ep, seed=seed, lagged_flow=True)
    norm_dataset, s_mean, s_std = normalize_dataset(dataset)
    units = Units(s_mean, s_std, float(dataset['rewards'].std() + 1e-8), device)
    evaluator = IndustrialEvaluator(env, s_mean, s_std, n_eval_ep,
                                    seed_base=8000 + 100 * seed, device=device)
    kappa = max(100.0, 2.0 * float(np.abs(norm_dataset['rewards']).max()) / (1 - 0.99))
    steps = 300 if not quick_test else 20
    out = {'seed': seed, 'results': {}, 'extra': {}}
    for name, iters, real_ratio in [('true, 50% real', n_morel_iters, 0.5),
                                    ('true, 5% real', n_morel_iters, 0.05),
                                    ('true, 5% real, 3x budget', 3 * n_morel_iters, 0.05)]:
        print(f"\n[seed {seed}] {name}")
        torch.manual_seed(seed + 1)
        np.random.seed(seed + 1)
        ag = MOReeLAgent(S, A, n_ensemble=4, rollout_horizon=5, kappa=kappa,
                         real_ratio=real_ratio, device=device)
        ag._model_reward = units.reward_fn
        ag.ensemble = OracleDynamics(units)
        ag.epsilon = float('inf')
        # 5% of a ~256 batch: 13 real + 247 synthetic transitions per update.
        real_batch = batch_size if real_ratio >= 0.5 else max(1, round(batch_size * real_ratio))
        train_morel(agent=ag, dataset=norm_dataset, env=env, s_mean=s_mean, s_std=s_std,
                    n_outer_iters=iters, sac_steps_per_iter=steps, batch_size=real_batch,
                    device=device, eval_every=iters + 1)
        out['results'][name] = evaluator.evaluate(ag)
        out['extra'][f'q_data {name}'] = mean_q_on_data(ag, norm_dataset, device)
    for name, m in out['results'].items():
        evaluator.print_results(f"{name} (seed {seed})", m)
    return out


# Coverage ladder for the coverage study: (regime split, operator noise scale).
# 'full' is the dataset of every other experiment; the others keep only normal
# operation and shrink the operators' exploration, so the logs cover ever less
# of what a policy evaluated from disturbed starts will visit.
COVERAGE = {
    'full':          ((0.60, 0.25, 0.15), 0.30),
    'normal':        ((1.00, 0.00, 0.00), 0.30),
    'normal, 1/3 noise':  ((1.00, 0.00, 0.00), 0.10),
    'normal, 1/10 noise': ((1.00, 0.00, 0.00), 0.03),
}


def run_coverage_study(seed: int = 0, device: str = 'cpu', coverage: str = 'full',
                       rollout_horizon: int = 5, physics: bool = True, n_train_ep: int = 400,
                       n_morel_iters: int = 15, n_cql_epochs: int = 80, n_eval_ep: int = 20,
                       batch_size: int = 256, quick_test: bool = False) -> Dict[str, object]:
    """
    One seed of the coverage study, on the Markov state: the same agents as the
    rest of the chapter on logs of decreasing coverage, to find where some form
    of pessimism starts to pay. Arms: SAC on real data only; model-based SAC
    without pessimism; HALT at p80 with worst-case kappa (the original
    configuration); HALT at p99 with kappa from the data; CQL at alpha = 1 and
    alpha = 0.1. All model-based arms share one hybrid ensemble. With
    physics=False the ensemble is black-box and only the model-based arms run.
    """
    if quick_test:
        n_train_ep, n_morel_iters, n_cql_epochs, n_eval_ep = 40, 2, 2, 3
    split, noise = COVERAGE[coverage]
    torch.manual_seed(seed)
    np.random.seed(seed)
    env  = CoatingProcessEnv(seed=seed, lagged_flow=True)
    S, A = env.obs_dim, CoatingProcessEnv.ACTION_DIM
    dataset = collect_industrial_dataset(n_episodes=n_train_ep, seed=seed, lagged_flow=True,
                                         regime_split=split, noise_scale=noise)
    norm_dataset, s_mean, s_std = normalize_dataset(dataset)
    units = Units(s_mean, s_std, float(dataset['rewards'].std() + 1e-8), device)
    evaluator = IndustrialEvaluator(env, s_mean, s_std, n_eval_ep,
                                    seed_base=8000 + 100 * seed, device=device)
    kappa = max(100.0, 2.0 * float(np.abs(norm_dataset['rewards']).max()) / (1 - 0.99))
    r = norm_dataset['rewards'].reshape(-1, 100)
    g, rtg = np.zeros(len(r)), np.zeros_like(r)
    for k in range(r.shape[1] - 1, -1, -1):
        g = r[:, k] + 0.99 * g
        rtg[:, k] = g
    kappa_data = float(np.abs(rtg.min()))
    steps = 300 if not quick_test else 20
    out = {'seed': seed, 'results': {},
           'extra': {'coverage': coverage, 'rollout_horizon': rollout_horizon, 'physics': physics,
                     'kappa': kappa, 'kappa_data': kappa_data,
                     'state_std': [float(x) for x in dataset['states'].std(0)]}}

    torch.manual_seed(seed + 1)
    hm = HybridMOReL(S, A, units, kappa=kappa, device=device, physics=physics)
    hm.train_dynamics(norm_dataset, n_epochs=30 if not quick_test else 2, batch_size=batch_size)
    ens = hm.hybrid_ensemble
    eps = {'p80': ens.calibrate_epsilon(norm_dataset, percentile=80.0),
           'p99': ens.calibrate_epsilon(norm_dataset, percentile=99.0)}

    for name, model, real_ratio, epsilon, kap in [
            ('SAC, real data only', None, 1.0, float('inf'), kappa),
            ('no pessimism', ens, 0.5, float('inf'), kappa),
            ('HALT p80, worst-case kappa', ens, 0.5, eps['p80'], kappa),
            ('HALT p99, data kappa', ens, 0.5, eps['p99'], kappa_data)]:
        if model is None and not physics:
            continue
        print(f"\n[seed {seed}, {coverage}, k={rollout_horizon}] {name}")
        torch.manual_seed(seed + 1)
        np.random.seed(seed + 1)
        ag = MOReeLAgent(S, A, n_ensemble=4, rollout_horizon=rollout_horizon, kappa=kap,
                         real_ratio=real_ratio, device=device)
        ag._model_reward = units.reward_fn
        ag.ensemble = model if model is not None else OracleDynamics(units)
        ag.epsilon = epsilon
        train_morel(agent=ag, dataset=norm_dataset, env=env, s_mean=s_mean, s_std=s_std,
                    n_outer_iters=n_morel_iters, sac_steps_per_iter=steps,
                    batch_size=batch_size, device=device, eval_every=n_morel_iters + 1)
        out['results'][name] = evaluator.evaluate(ag)
        out['extra'][f'q_data {name}'] = mean_q_on_data(ag, norm_dataset, device)
        out['extra'][f'halt_rate {name}'] = float(np.mean(ag.halt_rate_history))

    loader = make_dataloader(norm_dataset, batch_size, device)
    for alpha in ((1.0, 0.1) if physics else ()):
        name = f'CQL alpha={alpha:g}'
        print(f"\n[seed {seed}, {coverage}] {name}")
        torch.manual_seed(seed + 2)
        cql = CQLAgent(S, A, alpha_cql=alpha, device=device)
        train_cql_agent(cql, loader, n_cql_epochs)
        out['results'][name] = evaluator.evaluate(cql)
    for name, m in out['results'].items():
        evaluator.print_results(f"{name} (seed {seed})", m)
    return out


def t_interval(values, alpha: float = 0.05):
    """Mean and Student-t confidence interval over per-seed values. With few
    seeds this is more honest than a percentile bootstrap, which undercovers."""
    from scipy import stats
    v = np.asarray([x for x in values if not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return float('nan'), float('nan'), float('nan')
    if len(v) == 1:
        return float(v[0]), float('nan'), float('nan')
    half = stats.t.ppf(1 - alpha / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(v.mean() - half), float(v.mean() + half)


PAIRS = [('CQL', 'BC'), ('CQL+Physics', 'CQL'), ('HybridMOReL', 'BC')]
PAIR_METRICS = ['reward', 'L_iae', 'f_iae', 'viol_rate_disturbed']


def summarize(runs: List[Dict]) -> Dict[str, Dict]:
    """
    Per method and metric: mean over seeds with a t-interval, plus the worst
    and best seed. Per pair of methods: the paired per-seed difference, which
    removes the variation that both methods share through the same dataset.
    """
    methods = list(runs[0]['results'].keys())
    summary = {'methods': {}, 'pairs': {}}
    for mth in methods:
        summary['methods'][mth] = {}
        for k in IndustrialEvaluator.METRICS:
            vals = [r['results'][mth][k] for r in runs]
            m, lo, hi = t_interval(vals)
            summary['methods'][mth][k] = {'mean': m, 'lo': lo, 'hi': hi,
                                          'min': float(np.nanmin(vals)), 'max': float(np.nanmax(vals))}
    pairs = [(a, b) for a, b in PAIRS if a in methods and b in methods]
    if 'no pessimism' in methods:
        pairs = [('no pessimism', 'SAC, real data only'), ('HALT p80, worst-case kappa', 'no pessimism'),
                 ('HALT p99, data kappa', 'no pessimism'), ('HALT p99, data kappa', 'HALT p80, worst-case kappa'),
                 ('CQL alpha=1', 'no pessimism'), ('CQL alpha=0.1', 'no pessimism')]
        pairs = [(a, b) for a, b in pairs if a in methods and b in methods]
    elif 'true, 5% real' in methods:
        pairs = [('true, 5% real', 'true, 50% real'), ('true, 5% real, 3x budget', 'true, 5% real')]
    elif 'SAC, real data only' in methods:
        pairs = [('no HALT', 'SAC, real data only'), ('no HALT, 3x budget', 'no HALT')]
    elif 'HALT p80' in methods:
        pairs = [('HALT p99', 'HALT p80'), ('no HALT', 'HALT p80'), ('no HALT', 'HALT p99'),
                 ('HALT p99, data kappa', 'HALT p99'), ('no HALT', 'HALT p99, data kappa')]
        pairs = [(a, b) for a, b in pairs if a in methods and b in methods]
    elif 'OracleMOReL' in methods:
        pairs = [('OracleMOReL', 'HybridMOReL'), ('HybridMOReL-noHALT', 'HybridMOReL'),
                 ('OracleMOReL-HALT', 'HybridMOReL'), ('OracleMOReL', 'OracleMOReL-HALT')]
        pairs = [(a, b) for a, b in pairs if a in methods and b in methods]
    for a, b in pairs:
        summary['pairs'][f'{a} - {b}'] = {}
        for k in PAIR_METRICS:
            diff = [r['results'][a][k] - r['results'][b][k] for r in runs]
            m, lo, hi = t_interval(diff)
            summary['pairs'][f'{a} - {b}'][k] = {'mean': m, 'lo': lo, 'hi': hi}
    return summary


def print_summary(summary) -> None:
    rows = [('reward', 'Return', '{:+.1f}'), ('T_iae', 'T IAE', '{:.4f}'),
            ('f_iae', 'f IAE', '{:.4f}'), ('L_iae', 'Level IAE', '{:.4f}'),
            ('da_T', 'DA temperature', '{:.1%}'), ('da_f', 'DA filler', '{:.1%}'),
            ('overshoot_T', 'Overshoot T', '{:.1%}'), ('overshoot_f', 'Overshoot f', '{:.1%}'),
            ('viol_rate', 'Violation rate', '{:.1%}'),
            ('viol_rate_disturbed', 'Viol. (disturbed)', '{:.1%}'),
            ('viol_max', 'Max excursion', '{:.3f}'),
            ('reward_nominal', 'Return (nominal)', '{:+.1f}'),
            ('reward_disturbed', 'Return (disturbed)', '{:+.1f}'),
            ('L_iae_disturbed', 'Level IAE (dist.)', '{:.4f}')]
    methods = summary['methods']
    names = list(methods.keys())
    print("\n" + "=" * 100)
    print("  CHAPTER 10 – mean over seeds [95% t-interval]")
    print("=" * 100)
    print(f"{'Metric':16}" + "".join(f"{n:>21}" for n in names))
    for key, label, fmt in rows:
        cells = []
        for n in names:
            c = methods[n][key]
            cells.append(f"{fmt.format(c['mean'])} [{fmt.format(c['lo'])}, {fmt.format(c['hi'])}]")
        print(f"{label:16}" + "".join(f"{c:>21}" for c in cells))
    print("\n  Paired per-seed differences [95% t-interval]")
    for pair, d in summary['pairs'].items():
        print(f"  {pair:24}" + "  ".join(
            f"{k}={v['mean']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]" for k, v in d.items()))


def run_multi_seed(seeds=(0, 1, 2, 3, 4), out_path: Optional[str] = None,
                   quick_test: bool = False, device: str = 'cpu',
                   lagged_flow: bool = False, oracle_ablation: bool = False,
                   halt_study: bool = False, baselines_study: bool = False,
                   edge_study: bool = False, coverage: Optional[str] = None,
                   rollout_horizon: int = 5, physics: bool = True) -> Dict:
    if coverage is not None:
        runs = [run_coverage_study(seed=s, device=device, coverage=coverage,
                                   rollout_horizon=rollout_horizon, physics=physics,
                                   quick_test=quick_test)
                for s in seeds]
    elif edge_study:
        runs = [run_edge_study(seed=s, device=device, quick_test=quick_test) for s in seeds]
    elif baselines_study:
        runs = [run_baselines_study(seed=s, device=device, quick_test=quick_test) for s in seeds]
    elif halt_study:
        runs = [run_halt_study(seed=s, device=device, lagged_flow=lagged_flow,
                               quick_test=quick_test) for s in seeds]
    elif oracle_ablation:
        runs = [run_oracle_ablation(seed=s, device=device, quick_test=quick_test) for s in seeds]
    else:
        runs = [run_industrial_benchmark(seed=s, device=device, quick_test=quick_test,
                                         lagged_flow=lagged_flow) for s in seeds]
    summary = summarize(runs)
    print_summary(summary)
    if out_path:
        import json
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        slim = [{'seed': r['seed'], 'extra': r['extra'],
                 'results': {m: {k: v for k, v in res.items() if k != 'episodes'}
                             for m, res in r['results'].items()}} for r in runs]
        with open(out_path, 'w') as fh:
            json.dump({'seeds': list(seeds), 'lagged_flow': lagged_flow or oracle_ablation or baselines_study or edge_study or coverage is not None,
                       'oracle_ablation': oracle_ablation,
                       'interval': '95% Student-t over per-seed means',
                       'runs': slim, 'summary': summary}, fh, indent=2)
        print(f"\nSaved {out_path}")
    return {'runs': runs, 'summary': summary}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--quick', action='store_true')
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    p.add_argument('--lagged-flow', action='store_true',
                   help='append the two feed commands in transit to the state (Markov state)')
    p.add_argument('--oracle-ablation', action='store_true',
                   help='HybridMOReL vs MOReL with the true dynamics, on the lagged state')
    p.add_argument('--halt-study', action='store_true',
                   help='same MOReL with HALT at p80, p99 or off (combine with --lagged-flow)')
    p.add_argument('--baselines-study', action='store_true',
                   help='SAC on real data only, no-HALT agent at 3x budget, CQL alpha grid (lagged state)')
    p.add_argument('--edge-study', action='store_true',
                   help='true dynamics without pessimism at 50% or 5% real data (lagged state)')
    p.add_argument('--coverage', choices=list(COVERAGE), default=None,
                   help='coverage study on logs of this coverage (lagged state)')
    p.add_argument('--horizon', type=int, default=5, help='rollout horizon for --coverage')
    p.add_argument('--black-box', action='store_true',
                   help='--coverage with a black-box ensemble (no physics), model-based arms only')
    p.add_argument('--out', default=None,
                   help='default: results/ch10_benchmark{,_lagged,_oracle}.json')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    name = (f"ch10_coverage_{args.coverage.replace(', ', '_').replace('/', '-').replace(' ', '_')}"
            f"{'' if args.horizon == 5 else f'_k{args.horizon}'}"
            f"{'_blackbox' if args.black_box else ''}.json") if args.coverage else \
           'ch10_edge_lagged.json' if args.edge_study else \
           'ch10_baselines_lagged.json' if args.baselines_study else \
           (('ch10_halt_lagged.json' if args.lagged_flow else 'ch10_halt_logged.json')
            if args.halt_study else
            'ch10_benchmark_oracle.json' if args.oracle_ablation else
            'ch10_benchmark_lagged.json' if args.lagged_flow else 'ch10_benchmark.json')
    out = args.out or os.path.join(_HERE, 'results', name)
    run_multi_seed(tuple(args.seeds), out_path=None if args.quick else out,
                   quick_test=args.quick, device=args.device, lagged_flow=args.lagged_flow,
                   oracle_ablation=args.oracle_ablation, halt_study=args.halt_study,
                   baselines_study=args.baselines_study, edge_study=args.edge_study,
                   coverage=args.coverage, rollout_horizon=args.horizon,
                   physics=not args.black_box)

"""
chapter8.py
===========
Explainability in Offline RL — SHAP-based analysis of trained agents.
Chapter 8 of "Offline RL: From Theory to Industrial Practice"

This chapter answers a question that every industrial deployment eventually
faces: *why* did the agent choose this action at this moment?

Three complementary explanations are computed for a trained CQL agent:

  1. Q-function SHAP  — why does the agent VALUE a particular (state, action)?
     "What state features make this action look good to the critic?"

  2. Policy SHAP      — why does the agent CHOOSE a particular action?
     "What state features drive the choice of heat_input and flow_input?"

  3. Dynamics SHAP    — why does the model PREDICT a particular next state?
     "What inputs most influence the temperature/filler/level predictions?"

Each explanation answers a different question. They do not always agree —
a state feature can strongly affect the Q-function without strongly affecting
the policy, if the Q-function is flat in that direction. Understanding these
discrepancies is as informative as the explanations themselves.

Implementation strategy:
  - All three use SHAP KernelExplainer (model-agnostic, works on any black-box)
  - KernelExplainer treats the neural network as f: R^n -> R^m
  - Background dataset = representative sample from the offline dataset
  - For multi-output models (policy, dynamics) we explain each output separately
  - Visualization: summary plots, dependence plots, force plots (single instance)

Imports (not reimplemented — same as previous chapters):
  Ch3: CQLAgent, QNetwork, GaussianPolicy, BCAgent  from cql.py
  Ch5: DynamicsEnsemble                             from morel.py
  Ch7: CoatingProcessEnv, collect_industrial_dataset,
       normalize_dataset, make_dataloader            from chapter7.py
"""

import sys
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap
from typing import Dict, List, Tuple, Optional, Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from cql import CQLAgent, BCAgent as CQLBCAgent, QNetwork, GaussianPolicy
    from cql import train_agent as _train_cql_base
    _CQL_AVAILABLE = True
except ImportError:
    _CQL_AVAILABLE = False
    print("  [ch8] cql.py not found")

try:
    from morel import DynamicsEnsemble
    _MOREL_AVAILABLE = True
except ImportError:
    _MOREL_AVAILABLE = False
    print("  [ch8] morel.py not found")

try:
    from chapter7 import (
        CoatingProcessEnv,
        collect_industrial_dataset,
        normalize_dataset,
        make_dataloader,
    )
    _CH7_AVAILABLE = True
except ImportError:
    _CH7_AVAILABLE = False
    print("  [ch8] chapter7.py not found")


# ── State and action feature names for the coating process ──────────────────
STATE_NAMES  = ['temperature', 'filler_frac', 'viscosity', 'density', 'level']
ACTION_NAMES = ['heat_input', 'flow_input']
SA_NAMES     = STATE_NAMES + ACTION_NAMES   # for Q-function (state + action input)


# ============================================================================
# 1. WRAPPERS — make neural networks callable as numpy functions for SHAP
# ============================================================================

class QFunctionWrapper:
    """
    Wrap QNetwork as a numpy-in / numpy-out function for SHAP.

    Input:  X ∈ R^{n × (state_dim + action_dim)}  — concatenated [state, action]
    Output: q ∈ R^n                                — scalar Q-value per sample

    SHAP KernelExplainer calls this function many times with small batches;
    wrapping in torch.no_grad() and batching is essential for performance.
    """

    def __init__(self, q_network: QNetwork,
                 state_dim:  int,
                 action_dim: int,
                 device:     str = 'cpu'):
        self.q      = q_network.eval()
        self.s_dim  = state_dim
        self.a_dim  = action_dim
        self.device = device

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """X: (n, state_dim + action_dim) → q: (n,)"""
        with torch.no_grad():
            x_t = torch.FloatTensor(X).to(self.device)
            s   = x_t[:, :self.s_dim]
            a   = x_t[:, self.s_dim:]
            q   = self.q(s, a)
        return q.cpu().numpy()


class PolicyWrapper:
    """
    Wrap GaussianPolicy as a numpy-in / numpy-out function for SHAP.

    Input:  S ∈ R^{n × state_dim}   — normalized state
    Output: A ∈ R^{n × action_dim}  — deterministic mean action

    We explain the deterministic mean action (not sampled) so that SHAP
    values are stable across calls — stochastic outputs would give noisy
    SHAP estimates.
    """

    def __init__(self, policy: GaussianPolicy, device: str = 'cpu'):
        self.policy = policy.eval()
        self.device = device

    def __call__(self, S: np.ndarray) -> np.ndarray:
        """S: (n, state_dim) → A: (n, action_dim)"""
        with torch.no_grad():
            s_t    = torch.FloatTensor(S).to(self.device)
            mean, _ = self.policy._dist(s_t)
            action  = torch.tanh(mean)
        return action.cpu().numpy()


class PolicySingleOutputWrapper:
    """
    Wrap GaussianPolicy for a single action dimension.

    SHAP KernelExplainer works best with scalar outputs. We explain each
    action dimension separately and later combine the results.

    action_idx = 0 → heat_input
    action_idx = 1 → flow_input
    """

    def __init__(self, policy: GaussianPolicy,
                 action_idx: int,
                 device:     str = 'cpu'):
        self.policy     = policy.eval()
        self.action_idx = action_idx
        self.device     = device

    def __call__(self, S: np.ndarray) -> np.ndarray:
        """S: (n, state_dim) → a_i: (n,)"""
        with torch.no_grad():
            s_t     = torch.FloatTensor(S).to(self.device)
            mean, _ = self.policy._dist(s_t)
            action  = torch.tanh(mean)
        return action[:, self.action_idx].cpu().numpy()


class DynamicsWrapper:
    """
    Wrap DynamicsEnsemble as a numpy-in / numpy-out function for SHAP.

    Input:  X ∈ R^{n × (state_dim + action_dim)}  — concatenated [state, action]
    Output: s_next ∈ R^{n × state_dim}            — ensemble mean next state

    We explain the ensemble mean prediction — the expected next state.
    Uncertainty is analyzed separately (see explain_uncertainty).
    """

    def __init__(self, ensemble: 'DynamicsEnsemble',
                 state_dim:  int,
                 action_dim: int,
                 device:     str = 'cpu'):
        self.ensemble = ensemble
        self.s_dim    = state_dim
        self.a_dim    = action_dim
        self.device   = device

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """X: (n, state_dim + action_dim) → s_next: (n, state_dim)"""
        with torch.no_grad():
            x_t   = torch.FloatTensor(X).to(self.device)
            s     = x_t[:, :self.s_dim]
            a     = x_t[:, self.s_dim:]
            s_next, _ = self.ensemble.predict_with_uncertainty(s, a)
        return s_next.cpu().numpy()


class DynamicsSingleOutputWrapper:
    """
    Wrap DynamicsEnsemble for a single next-state dimension.

    state_idx = 0 → temperature prediction
    state_idx = 1 → filler_fraction prediction
    state_idx = 4 → level prediction
    """

    def __init__(self, ensemble: 'DynamicsEnsemble',
                 state_idx:  int,
                 state_dim:  int,
                 action_dim: int,
                 device:     str = 'cpu'):
        self.ensemble  = ensemble
        self.state_idx = state_idx
        self.s_dim     = state_dim
        self.a_dim     = action_dim
        self.device    = device

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """X: (n, state_dim + action_dim) → s_next_i: (n,)"""
        with torch.no_grad():
            x_t   = torch.FloatTensor(X).to(self.device)
            s     = x_t[:, :self.s_dim]
            a     = x_t[:, self.s_dim:]
            s_next, _ = self.ensemble.predict_with_uncertainty(s, a)
        return s_next[:, self.state_idx].cpu().numpy()


# ============================================================================
# 2. EXPLAINER — manages background, computes and stores SHAP values
# ============================================================================

class OfflineRLExplainer:
    """
    SHAP-based explainability for a trained Offline RL agent.

    Computes three families of SHAP values:
      1. q_shap     — SHAP values for Q1(s, a): what features drive Q-value?
      2. policy_shap — SHAP values for π(s): what state features drive actions?
      3. dynamics_shap — SHAP values for f̂(s, a): what inputs drive next-state?

    All use KernelExplainer (model-agnostic):
      - Works on any differentiable or non-differentiable black-box function
      - Approximates SHAP values via weighted linear regression on masked inputs
      - Slower than TreeExplainer or DeepExplainer, but makes no architecture
        assumptions — important when comparing Q-function vs policy vs dynamics

    Background dataset:
      A representative sample from the offline dataset used as the SHAP
      "missing value" baseline. SHAP values measure the contribution of each
      feature relative to the model's average output on the background.
      Using k=50–200 background samples gives good estimates; more is slower.

    Usage:
        explainer = OfflineRLExplainer(agent, ensemble, dataset, device)
        results   = explainer.explain_all(n_explain=200, n_background=100)
        explainer.plot_summary(results, save_dir='plots/')
    """

    def __init__(self,
                 agent:          'CQLAgent',
                 ensemble:       Optional['DynamicsEnsemble'],
                 dataset:        Dict,         # normalized dataset
                 s_mean:         np.ndarray,
                 s_std:          np.ndarray,
                 device:         str = 'cpu',
                 n_background:   int = 100):
        self.agent      = agent
        self.ensemble   = ensemble
        self.dataset    = dataset
        self.s_mean     = s_mean
        self.s_std      = s_std
        self.device     = device
        self.S          = CoatingProcessEnv.STATE_DIM
        self.A          = CoatingProcessEnv.ACTION_DIM

        # Build background dataset — representative sample
        n     = len(dataset['states'])
        idx   = np.random.default_rng(42).choice(n, size=min(n_background, n),
                                                   replace=False)
        self.bg_states  = dataset['states'][idx]          # normalized states
        self.bg_actions = dataset['actions'][idx]
        self.bg_sa      = np.concatenate([self.bg_states,
                                          self.bg_actions], axis=1)

        print(f"  Background: {len(idx)} samples  "
              f"(states: {self.bg_states.shape}, "
              f"actions: {self.bg_actions.shape})")

    # ── Q-function explanation ───────────────────────────────────────────────

    def explain_q(self, states: np.ndarray,
                  actions: np.ndarray,
                  n_samples: int = 'auto') -> np.ndarray:
        """
        Compute SHAP values for Q1(state, action).

        Returns shap_values: (n_explain, state_dim + action_dim)
        The i-th row gives the contribution of each feature to Q(s_i, a_i)
        relative to the mean Q-value on the background dataset.

        Feature ordering: [temperature, filler_frac, viscosity, density, level,
                           heat_input, flow_input]
        """
        fn  = QFunctionWrapper(self.agent.Q1, self.S, self.A, self.device)
        exp = shap.KernelExplainer(fn, self.bg_sa)
        sa  = np.concatenate([states, actions], axis=1)

        print(f"  Computing Q-function SHAP ({len(states)} samples)...")
        sv  = exp.shap_values(sa, nsamples=n_samples, silent=True)
        return sv   # (n, state_dim + action_dim)

    # ── Policy explanation ───────────────────────────────────────────────────

    def explain_policy(self, states: np.ndarray,
                       n_samples: int = 'auto') -> List[np.ndarray]:
        """
        Compute SHAP values for π(state) — one array per action dimension.

        Returns list of length action_dim, each element: (n_explain, state_dim)
        shap_values[0][i, j] = contribution of state feature j to action 0
                                (heat_input) for state i.

        We explain each action separately because SHAP is defined for scalar
        outputs. Multi-output SHAP would require the full Shapley tensor.
        """
        results = []
        for a_idx, a_name in enumerate(ACTION_NAMES):
            fn  = PolicySingleOutputWrapper(self.agent.policy, a_idx, self.device)
            exp = shap.KernelExplainer(fn, self.bg_states)
            print(f"  Computing policy SHAP for {a_name} ({len(states)} samples)...")
            sv  = exp.shap_values(states, nsamples=n_samples, silent=True)
            results.append(sv)   # (n, state_dim)
        return results

    # ── Dynamics explanation ─────────────────────────────────────────────────

    def explain_dynamics(self, states: np.ndarray,
                         actions: np.ndarray,
                         state_indices: Optional[List[int]] = None,
                         n_samples: int = 'auto') -> Dict[int, np.ndarray]:
        """
        Compute SHAP values for f̂(state, action) — one array per predicted dim.

        state_indices: which next-state dimensions to explain.
                       Default: [0, 1, 4] (temperature, filler, level).
                       These are the setpoint-relevant and constraint-critical vars.

        Returns dict: {state_idx: shap_values (n, state_dim + action_dim)}

        Feature ordering: [temperature, filler_frac, viscosity, density, level,
                           heat_input, flow_input]
        """
        if state_indices is None:
            state_indices = [0, 1, 4]   # temp, filler, level

        sa      = np.concatenate([states, actions], axis=1)
        results = {}

        for s_idx in state_indices:
            var_name = STATE_NAMES[s_idx]
            fn  = DynamicsSingleOutputWrapper(
                self.ensemble, s_idx, self.S, self.A, self.device)
            exp = shap.KernelExplainer(fn, self.bg_sa)
            print(f"  Computing dynamics SHAP for next_{var_name} "
                  f"({len(states)} samples)...")
            sv  = exp.shap_values(sa, nsamples=n_samples, silent=True)
            results[s_idx] = sv   # (n, state_dim + action_dim)

        return results

    # ── Unified entry point ──────────────────────────────────────────────────

    def explain_all(self, n_explain: int = 200,
                    n_background: int = 100,
                    n_samples: int = 100) -> Dict:
        """
        Compute all three SHAP families on n_explain samples.

        n_explain:    number of instances to explain
        n_background: passed to KernelExplainer (more = more accurate, slower)
        n_samples:    SHAP nsamples parameter — number of masked coalitions
                      evaluated per instance. 100 is a good default;
                      use 50 for fast exploration, 500+ for publication.

        Returns dict with keys:
          'states'         — the explained states (normalized)
          'actions'        — the explained actions
          'q_shap'         — SHAP values for Q1 (n, state_dim + action_dim)
          'policy_shap'    — list of SHAP arrays, one per action dim
          'dynamics_shap'  — dict {state_idx: shap array}
          'q_base'         — mean Q on background (scalar baseline)
        """
        n   = len(self.dataset['states'])
        idx = np.random.default_rng(0).choice(
            n, size=min(n_explain, n), replace=False)
        states  = self.dataset['states'][idx]
        actions = self.dataset['actions'][idx]

        # Baseline Q-value for reference
        q_fn   = QFunctionWrapper(self.agent.Q1, self.S, self.A, self.device)
        q_base = float(q_fn(self.bg_sa).mean())

        results = {
            'states':  states,
            'actions': actions,
            'q_base':  q_base,
        }

        results['q_shap'] = self.explain_q(states, actions, n_samples)

        results['policy_shap'] = self.explain_policy(states, n_samples)

        if self.ensemble is not None:
            results['dynamics_shap'] = self.explain_dynamics(
                states, actions, state_indices=[0, 1, 4], n_samples=n_samples)
        else:
            results['dynamics_shap'] = {}

        return results


# ============================================================================
# 3. VISUALIZATION
# ============================================================================

def plot_q_summary(shap_values: np.ndarray,
                   feature_names: List[str],
                   title: str = 'Q-function SHAP Summary',
                   save_path: Optional[str] = None) -> None:
    """
    Beeswarm / bar summary plot for Q-function SHAP values.

    Each row is a feature (state or action variable).
    Each dot is one explained sample; color shows feature value (low/high).
    X-axis: SHAP value (negative = reduces Q, positive = increases Q).

    How to read it:
      - Wide spread along x → feature is highly variable in its Q impact
      - Consistent red-right / blue-left → monotone relationship
      - Mixed colors → nonlinear or interaction-dependent relationship
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort features by mean absolute SHAP value
    mean_abs = np.abs(shap_values).mean(0)
    order    = np.argsort(mean_abs)[::-1]
    sv_ord   = shap_values[:, order]
    names_ord = [feature_names[i] for i in order]

    # Normalized feature values for coloring
    n_feat = shap_values.shape[1]
    colors = plt.cm.RdBu_r(np.linspace(0, 1, 100))

    for row_idx, feat_idx in enumerate(order):
        vals  = shap_values[:, feat_idx]
        # Use rank-based color normalization
        ranks = np.argsort(np.argsort(vals))
        c     = plt.cm.RdBu_r(ranks / max(len(ranks) - 1, 1))
        y     = np.random.uniform(row_idx - 0.3, row_idx + 0.3, size=len(vals))
        ax.scatter(vals, y, c=c, s=10, alpha=0.7, linewidths=0)

    ax.axvline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names_ord, fontsize=10)
    ax.set_xlabel('SHAP value  (impact on Q-function output)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')

    # Colorbar
    sm  = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r,
                                  norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cb  = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label('Feature value (blue=low, red=high)', fontsize=8)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(['low', 'mid', 'high'])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_policy_bar(policy_shap: List[np.ndarray],
                    state_names: List[str],
                    save_path: Optional[str] = None) -> None:
    """
    Bar chart: mean |SHAP| for each state feature per action dimension.

    For each action (heat_input, flow_input), shows which state variables
    the policy pays most attention to when choosing that action.

    This is the answer to: "what does the agent look at before acting?"
    """
    n_actions = len(policy_shap)
    fig, axes = plt.subplots(1, n_actions, figsize=(5 * n_actions, 5),
                              sharey=True)
    if n_actions == 1:
        axes = [axes]

    colours = ['#4472c4', '#ed7d31']

    for a_idx, (sv, ax) in enumerate(zip(policy_shap, axes)):
        mean_abs = np.abs(sv).mean(0)           # (state_dim,)
        order    = np.argsort(mean_abs)[::-1]
        bars     = ax.barh(
            range(len(order)),
            mean_abs[order],
            color=colours[a_idx % len(colours)],
            alpha=0.85,
        )
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([state_names[i] for i in order], fontsize=10)
        ax.set_xlabel('Mean |SHAP value|', fontsize=10)
        ax.set_title(f'Policy → {ACTION_NAMES[a_idx]}', fontsize=11,
                     fontweight='bold')
        ax.invert_yaxis()

    plt.suptitle('Policy SHAP: which state features drive each action?',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_dynamics_bar(dynamics_shap: Dict[int, np.ndarray],
                      feature_names: List[str],
                      state_names:   List[str],
                      save_path: Optional[str] = None) -> None:
    """
    Bar chart: mean |SHAP| for each input feature per predicted next-state dim.

    For each predicted variable (next_temperature, next_filler, next_level),
    shows which input features (current states + actions) matter most.

    Expected pattern (sanity check):
      - next_temperature driven by current temperature + heat_input
      - next_filler driven by current filler + flow_input
      - next_level driven by current level + flow_input + heat_input (outflow)

    If the dynamics SHAP does not show this pattern, the model may be
    fitting spurious correlations in the training data.
    """
    n_plots = len(dynamics_shap)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5), sharey=True)
    if n_plots == 1:
        axes = [axes]

    colours = ['#4472c4', '#ed7d31', '#70ad47']

    for plot_idx, (s_idx, sv) in enumerate(dynamics_shap.items()):
        ax       = axes[plot_idx]
        mean_abs = np.abs(sv).mean(0)
        order    = np.argsort(mean_abs)[::-1]
        ax.barh(
            range(len(order)),
            mean_abs[order],
            color=colours[plot_idx % len(colours)],
            alpha=0.85,
        )
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([feature_names[i] for i in order], fontsize=9)
        ax.set_xlabel('Mean |SHAP value|', fontsize=10)
        ax.set_title(f'Dynamics → next_{state_names[s_idx]}',
                     fontsize=11, fontweight='bold')
        ax.invert_yaxis()

    plt.suptitle('Dynamics model SHAP: which inputs drive each next-state prediction?',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_force_single(q_shap_row: np.ndarray,
                      feature_names: List[str],
                      q_value: float,
                      q_base: float,
                      instance_label: str = 'Instance',
                      save_path: Optional[str] = None) -> None:
    """
    Force plot for a single instance — horizontal waterfall chart.

    Shows how each feature pushes the Q-value up (positive SHAP, blue)
    or down (negative SHAP, red) from the baseline Q_base to the final Q.

    This is the most interpretable explanation for a single time step:
    an operator can see exactly which sensor reading contributed most
    to the Q-value being high or low at that moment.
    """
    order = np.argsort(np.abs(q_shap_row))[::-1]
    top_k = min(7, len(order))
    idxs  = order[:top_k]

    sv    = q_shap_row[idxs]
    names = [feature_names[i] for i in idxs]

    fig, ax = plt.subplots(figsize=(10, 3))
    cumsum  = q_base
    for i, (v, name) in enumerate(zip(sv, names)):
        color = '#2166ac' if v > 0 else '#d73027'
        ax.barh(0, v, left=cumsum, height=0.5, color=color, alpha=0.85)
        mid = cumsum + v / 2
        ax.text(mid, 0, f'{name}\n{v:+.3f}',
                ha='center', va='center', fontsize=8,
                color='white', fontweight='bold')
        cumsum += v

    ax.axvline(q_base,  color='k', lw=1.0, ls='--', alpha=0.5, label=f'baseline Q={q_base:.3f}')
    ax.axvline(q_value, color='g', lw=1.5, ls='-',  alpha=0.8, label=f'predicted Q={q_value:.3f}')
    ax.set_yticks([])
    ax.set_xlabel('Q-value', fontsize=10)
    ax.set_title(f'Force plot: {instance_label}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_shap_dependence(shap_values: np.ndarray,
                         feature_values: np.ndarray,
                         feature_idx: int,
                         interaction_idx: Optional[int],
                         feature_names: List[str],
                         title: str = 'SHAP Dependence',
                         save_path: Optional[str] = None) -> None:
    """
    Dependence plot: SHAP value of feature_idx vs its raw value.

    Colored by interaction_idx if provided — reveals interaction effects.
    Example: SHAP(temperature) vs temperature, colored by filler_frac.
    If the coloring is systematic (all-blue or all-red on one side),
    there is an interaction between temperature and filler_frac in the Q-function.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    sv  = shap_values[:, feature_idx]
    fv  = feature_values[:, feature_idx]

    if interaction_idx is not None:
        int_vals = feature_values[:, interaction_idx]
        sc = ax.scatter(fv, sv, c=int_vals, cmap='RdBu_r', s=20, alpha=0.7)
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label(feature_names[interaction_idx], fontsize=9)
    else:
        ax.scatter(fv, sv, color='#4472c4', s=20, alpha=0.7)

    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlabel(f'{feature_names[feature_idx]} (normalized)', fontsize=10)
    ax.set_ylabel(f'SHAP value for {feature_names[feature_idx]}', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


# ============================================================================
# 4. SUMMARY STATISTICS — text output for the chapter
# ============================================================================

def print_shap_summary(results: Dict,
                        top_k: int = 3) -> None:
    """
    Print a concise text summary of SHAP analysis findings.

    Outputs the top-k most important features for each explanation family,
    formatted for inclusion in an operator report or chapter discussion.
    """
    print("\n" + "=" * 58)
    print("  CHAPTER 8 — SHAP ANALYSIS SUMMARY")
    print("=" * 58)

    # Q-function top features
    sv   = results['q_shap']
    mean = np.abs(sv).mean(0)
    order = np.argsort(mean)[::-1]
    print(f"\nQ-function SHAP  (baseline Q = {results['q_base']:.4f})")
    print(f"  Top {top_k} features driving Q-value:")
    for i in range(min(top_k, len(order))):
        feat = SA_NAMES[order[i]]
        val  = mean[order[i]]
        sign = np.sign(sv[:, order[i]]).mean()
        direction = '↑ Q' if sign > 0.3 else ('↓ Q' if sign < -0.3 else '± Q')
        print(f"    {i+1}. {feat:<18} mean|SHAP|={val:.4f}  net effect: {direction}")

    # Policy top features per action
    print(f"\nPolicy SHAP")
    for a_idx, a_name in enumerate(ACTION_NAMES):
        sv_p  = results['policy_shap'][a_idx]
        mean_p = np.abs(sv_p).mean(0)
        order_p = np.argsort(mean_p)[::-1]
        print(f"  → {a_name}: top {top_k} state features")
        for i in range(min(top_k, len(order_p))):
            feat = STATE_NAMES[order_p[i]]
            val  = mean_p[order_p[i]]
            print(f"      {i+1}. {feat:<18} mean|SHAP|={val:.4f}")

    # Dynamics top features
    if results['dynamics_shap']:
        print(f"\nDynamics SHAP")
        for s_idx, sv_d in results['dynamics_shap'].items():
            var  = STATE_NAMES[s_idx]
            mean_d = np.abs(sv_d).mean(0)
            order_d = np.argsort(mean_d)[::-1]
            print(f"  → next_{var}: top {top_k} inputs")
            for i in range(min(top_k, len(order_d))):
                feat = SA_NAMES[order_d[i]]
                val  = mean_d[order_d[i]]
                print(f"      {i+1}. {feat:<18} mean|SHAP|={val:.4f}")

    print("=" * 58)


# ============================================================================
# 5. CONSISTENCY CHECK — do Q, policy, and dynamics agree?
# ============================================================================

def check_explanation_consistency(results: Dict) -> Dict[str, float]:
    """
    Check whether Q-function, policy, and dynamics SHAP values are consistent.

    Three consistency metrics:

    1. Q-policy rank correlation (Spearman):
       Checks whether the features that drive the Q-value also drive the policy.
       High correlation (>0.7) means the policy follows the Q-value gradient.
       Low correlation (<0.3) may indicate policy collapse or distribution shift.

    2. Dynamics-policy alignment:
       For each action, checks whether the features that the policy attends to
       are also the features that drive the dynamics for the corresponding
       controlled variable.
       Example: policy attends to temperature → dynamics_temperature should
       also rank temperature highly as an input.

    3. Physics sanity check:
       Checks that heat_input has a positive SHAP on next_temperature
       and flow_input has a positive SHAP on next_filler.
       Negative SHAP here would indicate the dynamics model learned an
       inverted relationship — a physics violation.

    Returns dict of metric name → value.
    """
    from scipy.stats import spearmanr

    metrics = {}

    # 1. Q-policy rank correlation
    q_sv    = results['q_shap'][:, :len(STATE_NAMES)]   # state part only
    pi_sv_0 = results['policy_shap'][0]                 # heat_input
    q_rank  = np.abs(q_sv).mean(0)
    pi_rank = np.abs(pi_sv_0).mean(0)
    rho, _  = spearmanr(q_rank, pi_rank)
    metrics['q_policy_spearman'] = float(rho)

    # 2. Physics sanity checks (if dynamics available)
    if results['dynamics_shap']:
        # heat_input → next_temperature: should be positive
        if 0 in results['dynamics_shap']:
            sv_temp     = results['dynamics_shap'][0]       # (n, S+A)
            heat_idx    = len(STATE_NAMES)                  # heat_input column
            heat_shap   = sv_temp[:, heat_idx].mean()
            metrics['heat_to_temp_shap_mean'] = float(heat_shap)
            metrics['heat_to_temp_sign_ok']   = float(heat_shap > 0)

        # flow_input → next_filler: should be positive
        if 1 in results['dynamics_shap']:
            sv_fill     = results['dynamics_shap'][1]
            flow_idx    = len(STATE_NAMES) + 1              # flow_input column
            flow_shap   = sv_fill[:, flow_idx].mean()
            metrics['flow_to_filler_shap_mean'] = float(flow_shap)
            metrics['flow_to_filler_sign_ok']   = float(flow_shap > 0)

    return metrics


def print_consistency_report(metrics: Dict[str, float]) -> None:
    print("\n" + "─" * 50)
    print("  Explanation Consistency Report")
    print("─" * 50)
    rho = metrics.get('q_policy_spearman', None)
    if rho is not None:
        status = '✓ aligned' if rho > 0.6 else ('△ weak' if rho > 0.3 else '✗ misaligned')
        print(f"  Q ↔ policy rank correlation  : {rho:.3f}  {status}")
        if rho < 0.3:
            print("    WARNING: Q-function and policy attend to different features.")
            print("    Possible cause: policy collapse, OOD actions, or value overestimation.")

    if 'heat_to_temp_sign_ok' in metrics:
        ok = metrics['heat_to_temp_sign_ok'] > 0
        val = metrics['heat_to_temp_shap_mean']
        print(f"  heat_input → next_temperature: SHAP={val:+.4f}  "
              f"{'✓ positive (correct)' if ok else '✗ NEGATIVE (physics violation!)'}")

    if 'flow_to_filler_sign_ok' in metrics:
        ok = metrics['flow_to_filler_sign_ok'] > 0
        val = metrics['flow_to_filler_shap_mean']
        print(f"  flow_input → next_filler     : SHAP={val:+.4f}  "
              f"{'✓ positive (correct)' if ok else '✗ NEGATIVE (physics violation!)'}")
    print("─" * 50)


# ============================================================================
# 6. TRAINING UTILITIES (minimal — reuse from ch7)
# ============================================================================

def train_cql_for_shap(device: str = 'cpu',
                        n_episodes: int = 300,
                        n_epochs: int = 60,
                        quick_test: bool = False) -> Tuple:
    """
    Train a CQL agent and dynamics ensemble on the coating process dataset.
    Returns (agent, ensemble, norm_dataset, s_mean, s_std).
    This is a convenience wrapper; for production use chapter7.run_industrial_benchmark.
    """
    if quick_test:
        n_episodes = 40; n_epochs = 5

    if not (_CQL_AVAILABLE and _MOREL_AVAILABLE and _CH7_AVAILABLE):
        raise ImportError("cql.py, morel.py, and chapter7.py required")

    S = CoatingProcessEnv.STATE_DIM
    A = CoatingProcessEnv.ACTION_DIM

    print(f"\n[1] Collecting dataset ({n_episodes} episodes)...")
    dataset = collect_industrial_dataset(n_episodes=n_episodes, seed=0)
    norm, s_mean, s_std = normalize_dataset(dataset)
    loader  = make_dataloader(norm, batch_size=256, device=device)

    print(f"\n[2] Training CQL ({n_epochs} epochs)...")
    agent = CQLAgent(S, A, alpha_cql=1.0, device=device)
    for epoch in range(1, n_epochs + 1):
        acc, nb = {}, 0
        for batch in loader:
            info = agent.update(batch)
            for k, v in info.items():
                acc[k] = acc.get(k, 0.0) + v
            nb += 1
        if epoch % max(1, n_epochs // 3) == 0:
            parts = [f"epoch {epoch:3d}"] + [
                f"{k}={v/nb:.4f}" for k, v in acc.items()]
            print("  " + " | ".join(parts))

    print(f"\n[3] Training dynamics ensemble (30 epochs)...")
    ensemble = DynamicsEnsemble(
        state_dim=S, action_dim=A, n_models=4, hidden_dim=128, device=device)
    ensemble.train_ensemble(norm, n_epochs=30 if not quick_test else 3,
                             batch_size=256)

    return agent, ensemble, norm, s_mean, s_std


# ============================================================================
# 7. MAIN
# ============================================================================

def run_shap_analysis(device:     str  = 'cpu',
                       n_explain:  int  = 150,
                       n_background: int = 80,
                       n_samples:  int  = 80,
                       save_dir:   str  = '.',
                       quick_test: bool = False) -> Dict:
    """
    Full SHAP analysis pipeline for a trained CQL agent.

    Steps:
      1. Train CQL agent + dynamics ensemble on coating process
      2. Build OfflineRLExplainer with background dataset
      3. Compute Q-function, policy, and dynamics SHAP values
      4. Generate all four visualizations
      5. Print text summary and consistency report

    Args:
      n_explain:    number of instances to compute SHAP values for
      n_background: background dataset size (more = more accurate, slower)
      n_samples:    SHAP nsamples parameter per instance
      save_dir:     directory to save plots
      quick_test:   if True, use tiny settings for CI/smoke testing
    """
    if quick_test:
        n_explain = 20; n_background = 20; n_samples = 20

    print("=" * 58)
    print("Chapter 8: Explainability in Offline RL — SHAP Analysis")
    print("=" * 58)

    # Train
    agent, ensemble, norm, s_mean, s_std = train_cql_for_shap(
        device=device, quick_test=quick_test)

    # Build explainer
    print("\n[4] Building SHAP explainer...")
    explainer = OfflineRLExplainer(
        agent        = agent,
        ensemble     = ensemble,
        dataset      = norm,
        s_mean       = s_mean,
        s_std        = s_std,
        device       = device,
        n_background = n_background,
    )

    # Compute SHAP values
    print("\n[5] Computing SHAP values...")
    results = explainer.explain_all(
        n_explain    = n_explain,
        n_background = n_background,
        n_samples    = n_samples,
    )

    # Visualize
    print("\n[6] Generating plots...")
    os.makedirs(save_dir, exist_ok=True)

    plot_q_summary(
        results['q_shap'], SA_NAMES,
        title='Q-function SHAP: what drives Q-value?',
        save_path=os.path.join(save_dir, 'ch8_q_summary.png'))

    plot_policy_bar(
        results['policy_shap'], STATE_NAMES,
        save_path=os.path.join(save_dir, 'ch8_policy_bar.png'))

    if results['dynamics_shap']:
        plot_dynamics_bar(
            results['dynamics_shap'], SA_NAMES, STATE_NAMES,
            save_path=os.path.join(save_dir, 'ch8_dynamics_bar.png'))

    # Force plot for highest-Q instance
    q_fn   = QFunctionWrapper(agent.Q1, CoatingProcessEnv.STATE_DIM,
                               CoatingProcessEnv.ACTION_DIM, device)
    sa_exp = np.concatenate([results['states'], results['actions']], axis=1)
    q_vals = q_fn(sa_exp)
    best   = int(np.argmax(q_vals))
    plot_force_single(
        results['q_shap'][best], SA_NAMES,
        q_value=q_vals[best], q_base=results['q_base'],
        instance_label=f'Highest-Q instance (Q={q_vals[best]:.3f})',
        save_path=os.path.join(save_dir, 'ch8_force_best.png'))

    # Dependence plot: temperature SHAP vs temperature, colored by filler
    plot_shap_dependence(
        results['q_shap'][:, :len(STATE_NAMES)],
        results['states'],
        feature_idx=0, interaction_idx=1,
        feature_names=STATE_NAMES,
        title='Q-SHAP dependence: temperature (colored by filler_frac)',
        save_path=os.path.join(save_dir, 'ch8_dependence_temp.png'))

    # Summary and consistency
    print_shap_summary(results)
    metrics = check_explanation_consistency(results)
    print_consistency_report(metrics)

    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--quick',    action='store_true')
    p.add_argument('--device',   default='cpu')
    p.add_argument('--save_dir', default='.')
    p.add_argument('--n_explain',    type=int, default=150)
    p.add_argument('--n_background', type=int, default=80)
    p.add_argument('--n_samples',    type=int, default=80)
    args = p.parse_args()
    run_shap_analysis(
        device       = args.device,
        n_explain    = args.n_explain,
        n_background = args.n_background,
        n_samples    = args.n_samples,
        save_dir     = args.save_dir,
        quick_test   = args.quick,
    )

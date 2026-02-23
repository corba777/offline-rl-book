"""
morel.py
========
Model-Based Offline Reinforcement Learning (MOReL).
Referenced from Chapter 5 of "Offline RL: From Theory to Industrial Practice".

Kidambi, R., Rajeswaran, A., Netrapalli, P., & Kakade, S.
"MOReL: Model-Based Offline Reinforcement Learning." NeurIPS 2020.
arXiv:2005.05951

Key idea: build a Pessimistic MDP (P-MDP) from offline data by partitioning
state-action space into KNOWN and UNKNOWN regions based on ensemble uncertainty.
In KNOWN regions the model is trusted; UNKNOWN transitions lead to an absorbing
failure state with a large fixed penalty kappa.  A standard RL algorithm (SAC)
is then trained entirely within the P-MDP.

Contrast with MOPO (mopo.py):
    MOPO  — continuous penalty: r_tilde = r - lambda * u(s,a)
    MOReL — discrete boundary:  if u(s,a) > epsilon -> absorbing state, r = -kappa

Theoretical guarantee (Kidambi et al., 2020):
    J*(pi) - J(pi_hat) <= 2*gamma*kappa / (1-gamma)^2 * Pr[UNKNOWN under pi_hat]

This is a distribution-free PAC bound — stronger than MOPO's, but requires
careful calibration of the epsilon threshold.

Contents:
    1. Imports from mopo.py       — ThermalProcessEnv, dataset helpers,
                                    ProbabilisticDynamicsNet, DynamicsEnsemble,
                                    QNetwork, GaussianPolicy, BCAgent, evaluate()
    2. calibrate_epsilon()        — set threshold from in-distribution data
    3. show_known_unknown_ratio() — diagnostic: what fraction of dataset is KNOWN?
    4. MORelAgent                 — P-MDP construction + SAC training
       4a. generate_pmdp_rollouts()  — hard boundary rollouts
       4b. update()                  — SAC on P-MDP data
       4c. train_model()             — ensemble training (delegates to DynamicsEnsemble)
    5. run_comparison()           — MOPO vs MOReL vs BC on ThermalProcessEnv

Usage:
    python morel.py          # runs full comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, Optional

# ---------------------------------------------------------------------------
# Reuse shared components from mopo.py (environment, ensemble, SAC networks).
# Run morel.py from the same directory as mopo.py, or adjust sys.path.
# ---------------------------------------------------------------------------
from mopo import (
    ThermalProcessEnv,
    pid_action,
    collect_offline_dataset,
    normalize_dataset,
    ProbabilisticDynamicsNet,
    DynamicsEnsemble,
    QNetwork,
    GaussianPolicy,
    BCAgent,
    evaluate,
)


# ============================================================================
# 2. EPSILON CALIBRATION
# ============================================================================

@torch.no_grad()
def calibrate_epsilon(ensemble: DynamicsEnsemble,
                      dataset: dict,
                      quantile: float = 0.80,
                      n_samples: int = 2000,
                      device: str = 'cpu') -> float:
    """
    Set the KNOWN/UNKNOWN threshold from in-distribution uncertainty.

    Strategy: compute ensemble uncertainty on actual dataset transitions,
    then take the q-th quantile as epsilon.  This means the known region
    covers at least (1 - quantile) fraction of the dataset by construction.

    Why quantile-based calibration?
        A fixed epsilon (e.g. 0.05) is dataset-dependent — it means nothing
        without knowing the scale of uncertainty for a given ensemble.
        Calibrating from the data itself makes epsilon interpretable:
        "the boundary is at the 80th percentile of in-distribution uncertainty."

    Args:
        ensemble:  trained DynamicsEnsemble
        dataset:   offline dataset dict with 'states' and 'actions' keys
        quantile:  percentile of in-distribution uncertainty to use as epsilon
                   (0.70–0.85 is typical; higher = larger known region)
        n_samples: number of dataset transitions to use for calibration
        device:    torch device string

    Returns:
        epsilon: scalar float threshold
    """
    n = min(n_samples, len(dataset['states']))
    idx = np.random.choice(len(dataset['states']), n, replace=False)

    s = torch.FloatTensor(dataset['states'][idx]).to(device)
    a = torch.FloatTensor(dataset['actions'][idx]).to(device)

    _, u = ensemble.predict_with_uncertainty(s, a)
    epsilon = torch.quantile(u, quantile).item()

    print(f"\nEpsilon calibration (q={quantile:.2f}):")
    print(f"  In-distribution uncertainty: "
          f"mean={u.mean():.4f}  std={u.std():.4f}  "
          f"p50={u.median():.4f}  p80={torch.quantile(u, 0.80):.4f}  "
          f"p95={torch.quantile(u, 0.95):.4f}")
    print(f"  → epsilon = {epsilon:.4f}  "
          f"(covers {quantile*100:.0f}% of dataset as KNOWN)")

    return epsilon


# ============================================================================
# 3. DIAGNOSTIC: KNOWN / UNKNOWN RATIO
# ============================================================================

@torch.no_grad()
def show_known_unknown_ratio(ensemble: DynamicsEnsemble,
                              dataset: dict,
                              epsilon: float,
                              device: str = 'cpu') -> Dict[str, float]:
    """
    Report what fraction of dataset transitions the ensemble considers KNOWN.

    Also checks OOD actions to verify the ensemble actually flags them as
    UNKNOWN — the key correctness check for the P-MDP construction.

    A healthy ensemble should satisfy:
        in-dist KNOWN fraction >> 0.7   (high coverage of real data)
        OOD     KNOWN fraction << 0.3   (uncertain on unseen actions)

    If OOD KNOWN fraction is high, the ensemble is not diverse enough
    and the P-MDP boundary is meaningless — increase ensemble size or
    reduce bootstrap fraction.
    """
    n = min(2000, len(dataset['states']))
    s = torch.FloatTensor(dataset['states'][:n]).to(device)
    a = torch.FloatTensor(dataset['actions'][:n]).to(device)

    # In-distribution
    _, u_in = ensemble.predict_with_uncertainty(s, a)
    known_in = (u_in <= epsilon).float().mean().item()

    # OOD: saturated actions (extreme, never in dataset)
    a_ood = torch.ones_like(a) * 0.95
    _, u_ood = ensemble.predict_with_uncertainty(s, a_ood)
    known_ood = (u_ood <= epsilon).float().mean().item()

    # Random actions
    a_rand = torch.FloatTensor(n, a.shape[-1]).uniform_(-1, 1).to(device)
    _, u_rand = ensemble.predict_with_uncertainty(s, a_rand)
    known_rand = (u_rand <= epsilon).float().mean().item()

    print(f"\n{'='*60}")
    print(f"P-MDP Diagnostic  (epsilon = {epsilon:.4f})")
    print(f"{'='*60}")
    print(f"  Dataset actions  → KNOWN: {known_in*100:5.1f}%  "
          f"(u mean={u_in.mean():.4f})")
    print(f"  Saturated OOD    → KNOWN: {known_ood*100:5.1f}%  "
          f"(u mean={u_ood.mean():.4f})")
    print(f"  Random actions   → KNOWN: {known_rand*100:5.1f}%  "
          f"(u mean={u_rand.mean():.4f})")
    print(f"  OOD/In-dist u ratio: {u_ood.mean()/max(u_in.mean(),1e-8):.2f}x")

    if known_ood > 0.5:
        print("  ⚠ WARNING: ensemble not diverse enough — "
              "OOD actions often classified as KNOWN.")
        print("  Consider: larger ensemble, different bootstrap ratio, "
              "or spectral normalisation.")
    else:
        print("  ✓ P-MDP boundary looks healthy.")

    return {
        'known_in_dist':  known_in,
        'known_ood':      known_ood,
        'known_random':   known_rand,
        'u_in_mean':      u_in.mean().item(),
        'u_ood_mean':     u_ood.mean().item(),
    }


# ============================================================================
# 4. MOReL AGENT
# ============================================================================

class MORelAgent:
    """
    Model-Based Offline RL via Pessimistic MDP (MOReL).

    The P-MDP construction:
    ───────────────────────
    KNOWN region:   (s, a) where ensemble uncertainty u(s,a) <= epsilon
                    → model transition is used; policy trained normally

    UNKNOWN region: (s, a) where u(s,a) > epsilon
                    → transition leads to absorbing failure state s_⊥
                    → reward = -kappa at s_⊥ (large negative constant)
                    → episode terminates (done=True)

    Training loop:
        Phase 1 — train_model():
            Train DynamicsEnsemble on offline data (same as MOPO).

        Phase 2 — calibrate():
            Set epsilon from in-distribution uncertainty quantile.

        Phase 3 — generate_pmdp_rollouts():
            Branch from real states.  At each step:
              if u(s,a) <= epsilon: proceed normally (KNOWN)
              if u(s,a) >  epsilon: terminate with reward -kappa (UNKNOWN)

        Phase 4 — update():
            Standard SAC on real + P-MDP synthetic transitions.
            SAC sees done=True after UNKNOWN steps → TD target = reward only.
            This makes the Q-function correctly value UNKNOWN states at
            approximately -kappa / (1 - gamma).

    Key difference from MOPO:
        MOPO subtracts a continuous penalty from reward.
        MOReL terminates the trajectory with a hard penalty.
        In the Q-update: MOPO propagates penalized future values,
        MOReL cuts the future value to zero and adds -kappa.
        This gives MOReL a harder pessimism boundary.
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 n_ensemble: int = 5,
                 rollout_horizon: int = 5,
                 kappa: float = 10.0,
                 epsilon: float = None,       # set via calibrate() if None
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
        self.kappa           = kappa
        self.epsilon         = epsilon       # KNOWN/UNKNOWN threshold
        self.rollout_horizon = rollout_horizon
        self.rollout_batch   = rollout_batch
        self.real_ratio      = real_ratio

        # Dynamics ensemble (identical architecture to MOPO)
        self.ensemble = DynamicsEnsemble(
            n_models=n_ensemble,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lr=model_lr,
            device=device,
        )

        # SAC components — standard; same as MOPO
        self.Q1     = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.Q2     = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.Q1_tgt = deepcopy(self.Q1)
        self.Q2_tgt = deepcopy(self.Q2)
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim).to(device)

        self.q_opt  = optim.Adam(
            list(self.Q1.parameters()) + list(self.Q2.parameters()), lr=q_lr)
        self.pi_opt = optim.Adam(self.policy.parameters(), lr=policy_lr)

    # ------------------------------------------------------------------
    # Phase 1: model training
    # ------------------------------------------------------------------

    def train_model(self, dataset: dict, n_epochs: int = 50,
                    log_every: int = 10):
        """Train the dynamics ensemble.  Identical to MOPOAgent.train_model."""
        print("Training dynamics ensemble...")
        self.ensemble.train_ensemble(
            dataset, n_epochs=n_epochs, log_every=log_every)

    # ------------------------------------------------------------------
    # Phase 2: epsilon calibration
    # ------------------------------------------------------------------

    def calibrate(self, dataset: dict,
                  quantile: float = 0.80,
                  n_samples: int = 2000):
        """
        Set self.epsilon from in-distribution uncertainty.

        Must be called after train_model() and before generate_pmdp_rollouts().
        """
        self.epsilon = calibrate_epsilon(
            self.ensemble, dataset, quantile=quantile,
            n_samples=n_samples, device=self.device)
        return self.epsilon

    # ------------------------------------------------------------------
    # Phase 3: P-MDP rollouts — the core MOReL mechanism
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_pmdp_rollouts(self,
                                real_states: torch.Tensor
                                ) -> Dict[str, torch.Tensor]:
        """
        Generate synthetic transitions under the Pessimistic MDP.

        At each step of the h-step rollout:
          1. Sample action from current policy.
          2. Compute ensemble uncertainty u(s, a).
          3. If u <= epsilon  (KNOWN region):
               - Use model's predicted next state.
               - Use model's predicted reward.
               - done = False  →  rollout continues.
          4. If u > epsilon   (UNKNOWN region):
               - Transition leads to absorbing failure state s_⊥.
               - Reward = -kappa  (hard penalty).
               - done = True   →  rollout terminates.
               - next_state = current state (absorbing: stays at s_⊥).

        The Q-learning TD update on done=True transitions is:
            target = reward + gamma * (1 - done) * Q_tgt(s', a')
                   = -kappa + gamma * 0 * ...
                   = -kappa
        So Q(s, a) learns to approximate -kappa in UNKNOWN states,
        which is approximately -kappa / (1 - gamma) for absorbing trajectories.

        Args:
            real_states: tensor of shape (N_real, state_dim) — starting points

        Returns:
            dict of tensors: states, actions, rewards, next_states, dones
        """
        if self.epsilon is None:
            raise RuntimeError(
                "epsilon not set. Call calibrate() before generating rollouts.")

        n = min(self.rollout_batch, real_states.shape[0])
        idx   = torch.randint(0, real_states.shape[0], (n,))
        state = real_states[idx].to(self.device)

        # Track which rollouts are still alive (not yet in UNKNOWN)
        alive = torch.ones(n, dtype=torch.bool, device=self.device)

        syn_s, syn_a, syn_r, syn_s2, syn_d = [], [], [], [], []

        for step in range(self.rollout_horizon):
            # Only roll out alive trajectories — but keep batch structure
            action, _ = self.policy.sample(state)

            # Predict next state and uncertainty for ALL (including dead)
            next_state, uncertainty = self.ensemble.predict_with_uncertainty(
                state, action)

            # P-MDP boundary: classify each transition
            # A transition is UNKNOWN if:
            #   (a) uncertainty exceeds epsilon, OR
            #   (b) the rollout was already terminated in a previous step
            ood_mask = (uncertainty > self.epsilon) | (~alive)   # (n,)

            # Reward:
            #   KNOWN   → model reward (quadratic penalty on setpoint error)
            #   UNKNOWN → hard penalty -kappa
            model_r  = self._model_reward(state, action, next_state)
            reward   = torch.where(ood_mask, -self.kappa * torch.ones_like(model_r),
                                   model_r)

            # done = True for UNKNOWN transitions (absorbing)
            done = ood_mask.float()

            # Next state:
            #   KNOWN   → model prediction
            #   UNKNOWN → stay at current state (absorbing s_⊥)
            next_s = torch.where(
                ood_mask.unsqueeze(-1).expand_as(next_state),
                state,         # absorbing: stay put
                next_state,
            )

            # Store transition
            syn_s.append(state.clone())
            syn_a.append(action)
            syn_r.append(reward)
            syn_s2.append(next_s)
            syn_d.append(done)

            # Update alive mask: once UNKNOWN, always done
            alive = alive & (~ood_mask)

            # Advance state (alive rollouts continue from predicted next state)
            state = next_s

            # Early stop if all rollouts terminated
            if not alive.any():
                break

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
        Analytical reward for ThermalProcessEnv (known reward function).

        In production, replace with a learned reward model if the reward
        function is not available analytically.
        """
        T_next = next_states[:, 0]
        f_next = next_states[:, 1]
        return (-2.0 * T_next.pow(2) - 2.0 * f_next.pow(2)
                - 0.1 * actions.pow(2).sum(-1))

    # ------------------------------------------------------------------
    # Phase 4: SAC policy training on P-MDP data
    # ------------------------------------------------------------------

    def update(self, real_batch: Tuple[torch.Tensor, ...],
               synthetic: Dict[str, torch.Tensor]) -> dict:
        """
        SAC update on a mix of real and P-MDP synthetic transitions.

        The key: synthetic transitions contain done=True for UNKNOWN steps.
        The TD target for done=True is just reward (no future value):
            target = r + gamma * (1 - done) * Q_tgt(s', a')
        This correctly propagates -kappa as the value of UNKNOWN states.

        The SAC update itself is standard — no algorithmic change needed.
        The pessimism is entirely encoded in the done flags and rewards.
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

        # Combine real + synthetic
        s  = torch.cat([s_r,  s_s],  dim=0)
        a  = torch.cat([a_r,  a_s],  dim=0)
        r  = torch.cat([r_r,  r_s],  dim=0)
        s2 = torch.cat([s2_r, s2_s], dim=0)
        d  = torch.cat([d_r,  d_s],  dim=0)

        info = {}

        # ── Q update ────────────────────────────────────────────────────
        # For d=1 (UNKNOWN/absorbing): target = r = -kappa  (no future)
        # For d=0 (KNOWN/real):        target = r + gamma * Q_tgt(s', a')
        with torch.no_grad():
            a2, lp2 = self.policy.sample(s2)
            q_next  = torch.min(self.Q1_tgt(s2, a2), self.Q2_tgt(s2, a2))
            q_next -= self.alpha_ent * lp2
            target  = r + self.gamma * (1.0 - d) * q_next

        q1_loss = F.mse_loss(self.Q1(s, a), target)
        q2_loss = F.mse_loss(self.Q2(s, a), target)

        self.q_opt.zero_grad()
        (q1_loss + q2_loss).backward()
        self.q_opt.step()
        info['q_loss'] = (q1_loss.item() + q2_loss.item()) / 2

        # ── Policy update (SAC entropy-regularized) ─────────────────────
        a_pi, lp_pi = self.policy.sample(s)
        q_pi   = torch.min(self.Q1(s, a_pi), self.Q2(s, a_pi))
        pi_loss = (self.alpha_ent * lp_pi - q_pi).mean()

        self.pi_opt.zero_grad()
        pi_loss.backward()
        self.pi_opt.step()
        info['pi_loss'] = pi_loss.item()

        # ── Soft target update ──────────────────────────────────────────
        for p, pt in zip(self.Q1.parameters(), self.Q1_tgt.parameters()):
            pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
        for p, pt in zip(self.Q2.parameters(), self.Q2_tgt.parameters()):
            pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return info

    # ------------------------------------------------------------------
    # Rollout statistics — useful diagnostic
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rollout_stats(self, real_states: torch.Tensor) -> dict:
        """
        Measure what fraction of synthetic rollout steps are KNOWN vs UNKNOWN.

        Useful for:
          - Verifying epsilon is correctly calibrated
          - Monitoring policy drift into UNKNOWN regions over training
          - Checking that kappa is large enough to deter OOD exploration

        Returns dict with:
          known_frac:  fraction of rollout steps in KNOWN region
          mean_steps:  average number of steps before hitting UNKNOWN
          early_term:  fraction of rollouts terminated before full horizon
        """
        if self.epsilon is None:
            raise RuntimeError("epsilon not set.")

        n = min(self.rollout_batch, real_states.shape[0])
        idx   = torch.randint(0, real_states.shape[0], (n,))
        state = real_states[idx].to(self.device)
        alive = torch.ones(n, dtype=torch.bool, device=self.device)

        total_steps = 0
        known_steps = 0
        steps_at_term = torch.zeros(n, device=self.device)

        for step in range(self.rollout_horizon):
            action, _ = self.policy.sample(state)
            next_state, uncertainty = self.ensemble.predict_with_uncertainty(
                state, action)

            ood_mask = (uncertainty > self.epsilon) | (~alive)
            known    = alive & (uncertainty <= self.epsilon)

            total_steps += alive.sum().item()
            known_steps += known.sum().item()

            # Record steps taken for newly terminated rollouts
            newly_dead = alive & ood_mask
            steps_at_term[newly_dead] = step + 1

            alive = alive & (~ood_mask)
            state = torch.where(
                ood_mask.unsqueeze(-1).expand_as(next_state),
                state, next_state)

            if not alive.any():
                break

        # Rollouts that survived full horizon
        steps_at_term[alive] = self.rollout_horizon
        early_term = (steps_at_term < self.rollout_horizon).float().mean().item()

        return {
            'known_frac':  known_steps / max(total_steps, 1),
            'mean_steps':  steps_at_term.mean().item(),
            'early_term':  early_term,
        }


# ============================================================================
# 5. TRAINING HELPERS
# ============================================================================

def train_morel_agent(agent: MORelAgent,
                      loader: DataLoader,
                      real_states_tensor: torch.Tensor,
                      n_epochs: int = 80,
                      rollout_every: int = 5,
                      log_every: int = 20):
    """
    MOReL training loop.

    Identical structure to MOPO training:
      1. Periodically regenerate P-MDP synthetic data.
      2. Each epoch, run SAC updates on real + synthetic batches.

    The only difference from mopo training is generate_pmdp_rollouts()
    vs generate_synthetic_data() — the rest is the same SAC loop.
    """
    synthetic = None

    for epoch in range(1, n_epochs + 1):
        if epoch == 1 or epoch % rollout_every == 0:
            synthetic = agent.generate_pmdp_rollouts(real_states_tensor)

        info_accum = {}
        n_batches  = 0
        for batch in loader:
            step_info = agent.update(batch, synthetic)
            for k, v in step_info.items():
                info_accum[k] = info_accum.get(k, 0) + v
            n_batches += 1

        if epoch % log_every == 0:
            stats = agent.rollout_stats(real_states_tensor)
            parts = (
                [f"Epoch {epoch:3d}"]
                + [f"{k}={v/n_batches:.4f}" for k, v in info_accum.items()]
                + [f"known={stats['known_frac']:.2f}",
                   f"steps={stats['mean_steps']:.1f}",
                   f"early_term={stats['early_term']:.2f}"]
            )
            print("  " + " | ".join(parts))


# ============================================================================
# 6. COMPARISON: MOPO vs MOReL vs BC
# ============================================================================

def run_comparison():
    """
    Train BC, MOPO, and MOReL on the same offline dataset.
    Compare final policy performance on ThermalProcessEnv.

    This is the main entry point for Chapter 5 experiments.
    """
    from mopo import MOPOAgent, train_sac_agent as train_mopo_agent

    print("=" * 60)
    print("Chapter 5: MOPO vs MOReL vs BC on Thermal Process")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ── Data ────────────────────────────────────────────────────────────
    dataset = collect_offline_dataset(n_episodes=400, noise_scale=0.3)
    s_mean, s_std, _ = normalize_dataset(dataset)

    ds = TensorDataset(*[torch.FloatTensor(dataset[k])
                         for k in ['states', 'actions', 'rewards',
                                   'next_states', 'dones']])
    loader = DataLoader(ds, batch_size=256, shuffle=True, drop_last=True)
    real_states = torch.FloatTensor(dataset['states']).to(device)
    env = ThermalProcessEnv()

    # ── Behavior policy baseline ─────────────────────────────────────────
    pid_rewards = []
    for ep in range(20):
        obs = env.reset(seed=9000 + ep)
        ep_r = 0.0
        done = False
        while not done:
            act = pid_action(obs, env.T_target, env.f_target)
            obs, r, done, _ = env.step(act)
            ep_r += r
        pid_rewards.append(ep_r)
    print(f"Behavior policy (noisy PID): "
          f"{np.mean(pid_rewards):.2f} ± {np.std(pid_rewards):.2f}\n")

    # ── Behavioral Cloning ───────────────────────────────────────────────
    print("--- Behavioral Cloning ---")
    bc = BCAgent(3, 2, device=device)
    for epoch in range(1, 81):
        for batch in loader:
            bc.update(batch)
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}")
    bc_res = evaluate(bc, env, s_mean, s_std, device=device)

    # ── MOPO ────────────────────────────────────────────────────────────
    print("\n--- MOPO (λ=1.0, h=5, N=5) ---")
    mopo = MOPOAgent(3, 2, n_ensemble=5, rollout_horizon=5, lam=1.0,
                     rollout_batch=512, real_ratio=0.5, device=device)
    mopo.train_model(dataset, n_epochs=30, log_every=10)
    train_mopo_agent(mopo, loader, real_states,
                     n_epochs=80, rollout_every=10, log_every=20)
    mopo_res = evaluate(mopo, env, s_mean, s_std, device=device)

    # ── MOReL (default: q=0.80, kappa=10.0) ────────────────────────────
    print("\n--- MOReL (q=0.80, κ=10.0, h=5, N=5) ---")
    morel = MORelAgent(3, 2, n_ensemble=5, rollout_horizon=5,
                       kappa=10.0, rollout_batch=512, real_ratio=0.5,
                       device=device)
    morel.train_model(dataset, n_epochs=30, log_every=10)
    morel.calibrate(dataset, quantile=0.80)
    show_known_unknown_ratio(morel.ensemble, dataset, morel.epsilon, device)

    print("\n--- MOReL policy training ---")
    train_morel_agent(morel, loader, real_states,
                      n_epochs=80, rollout_every=10, log_every=20)
    morel_res = evaluate(morel, env, s_mean, s_std, device=device)

    # ── MOReL (conservative: q=0.70, kappa=20.0) ───────────────────────
    print("\n--- MOReL conservative (q=0.70, κ=20.0, h=3, N=5) ---")
    morel_c = MORelAgent(3, 2, n_ensemble=5, rollout_horizon=3,
                         kappa=20.0, rollout_batch=512, real_ratio=0.7,
                         device=device)
    morel_c.train_model(dataset, n_epochs=30, log_every=30)
    morel_c.calibrate(dataset, quantile=0.70)

    print("\n--- MOReL conservative policy training ---")
    train_morel_agent(morel_c, loader, real_states,
                      n_epochs=80, rollout_every=10, log_every=20)
    morel_c_res = evaluate(morel_c, env, s_mean, s_std, device=device)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"{'Method':<34} {'Reward':>12}  {'T err':>8}  {'f err':>8}")
    print("-" * 68)
    print(f"{'Noisy PID (behavior policy)':<34} "
          f"{np.mean(pid_rewards):>8.2f}±{np.std(pid_rewards):.2f}  "
          f"{'—':>8}  {'—':>8}")

    results = [
        ("BC",                           bc_res),
        ("MOPO  (λ=1.0, h=5)",           mopo_res),
        ("MOReL (q=0.80, κ=10.0, h=5)",  morel_res),
        ("MOReL (q=0.70, κ=20.0, h=3)",  morel_c_res),
    ]
    for name, res in results:
        print(f"{name:<34} "
              f"{res['reward_mean']:>8.2f}±{res['reward_std']:.2f}  "
              f"{res['T_err']:>8.4f}  {res['f_err']:>8.4f}")
    print("=" * 68)

    best_morel = max(morel_res['reward_mean'], morel_c_res['reward_mean'])
    print(f"\nMOReL vs BC:   {best_morel - bc_res['reward_mean']:+.2f} reward")
    print(f"MOReL vs MOPO: {best_morel - mopo_res['reward_mean']:+.2f} reward")
    print("\nNote: on smooth/dense data MOPO often matches or outperforms MOReL.")
    print("MOReL's advantage is strongest when safety constraints are hard")
    print("or when the dataset has sharp distributional boundaries.")


if __name__ == '__main__':
    run_comparison()

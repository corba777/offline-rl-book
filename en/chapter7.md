---
layout: default
title: "Chapter 7: Industrial Applications"
lang: en
ru_url: /offline-rl-book/ru/chapter7/
prev_chapter:
  url: /en/chapter6/
  title: "Physics-Informed Offline RL"
next_chapter:
  url: /en/chapter8/
  title: "Conclusion and Future Directions"
permalink: "/offline-rl-book/en/chapter7/"
---

# Chapter 7: Industrial Applications

> *"The gap between a working benchmark result and a working industrial deployment is not a gap in algorithms — it is a gap in understanding the data."*

---

## From Benchmarks to Industry

The preceding six chapters built a complete toolbox: behavioral cloning, CQL, IQL, MOPO, MOReL, physics-informed reward shaping, hybrid dynamics models, and Lagrangian constraint enforcement. Each algorithm was validated on `ThermalProcessEnv` — a simple, clean, three-variable toy environment designed to make the algorithmic ideas visible.

Real industrial processes differ from that environment in every dimension that matters.

**More variables, more coupling.** A realistic coating process has dozens of sensor readings. Even a simplified model has temperature, filler fraction, viscosity, bulk density, and a surge tank level — five variables with physical dependencies between all of them. The level is an integrating state that drifts without active control; viscosity is a nonlinear function of temperature and filler that the agent cannot observe independently of its inputs.

**Non-uniform data coverage.** Industrial logs are not random exploration data. A process runs near its operating setpoint for 60% of the time, makes occasional planned setpoint changes, and occasionally experiences disturbances — blown bearings, feed composition shifts, ambient temperature swings. The dataset is dense near the nominal operating point and thin everywhere else. An agent that performs well on the dense region and fails on the sparse regions is useless in practice.

**Lag and delay.** Material fed into a coating line takes 30–90 minutes to appear at the quality measurement point. Within the dynamic model, this manifests as a transport delay: the agent's action at time $t$ affects the output at time $t + k$. A pure first-order model ignoring the delay will have systematic prediction errors whenever the feed rate changes.

**Hard physical constraints.** Operating limits are not preferences — they are equipment ratings, regulatory requirements, or safety boundaries. A policy that violates viscosity bounds may cause pump cavitation. A policy that lets the surge tank overflow wastes material and triggers an automatic shutdown.

This chapter works through a complete case study on an anonymized five-variable coating process. The goal is not to introduce new algorithms — everything used here has been developed in Chapters 1–6. The goal is to show how they compose in a realistic setting, and what industrial-specific decisions appear along the way.

> 📄 Full code: [`chapter7.py`](https://github.com/corba777/offline-rl-book/blob/main/code/chapter7.py)

---

## The Coating Process

### Environment

The `CoatingProcessEnv` models a continuous thermal coating line with five state variables and two control inputs.

**State** $s \in \mathbb{R}^5$ (all normalized to $[0,1]$):

| Index | Variable | Physical meaning |
|---|---|---|
| $s_0$ | `temperature` | Process temperature in the coating zone |
| $s_1$ | `filler_fraction` | Fraction of filler material in the blend |
| $s_2$ | `viscosity` | Blend viscosity — nonlinear function of $T$ and filler |
| $s_3$ | `density` | Bulk material density |
| $s_4$ | `level` | Surge tank fill level — integrating dynamics |

**Actions** $a \in [-1,1]^2$:

| Index | Action | Effect |
|---|---|---|
| $a_0$ | `heat_input` | Heater setpoint delta |
| $a_1$ | `flow_input` | Filler feed rate delta |

**True dynamics** (unknown to the agent):

$$T_{t+1} = \left(1 - \frac{\Delta t}{\tau_T}\right)T_t + \frac{\Delta t}{\tau_T} K_T u_T + \underbrace{0.03 \cdot f_t \cdot u_T}_{\text{cross-coupling}} + \epsilon_T$$

$$f_{t+1} = \left(1 - \frac{\Delta t}{\tau_f}\right)f_t + \frac{\Delta t}{\tau_f} K_f \cdot \underbrace{u_{f,t-2}}_{\text{delay}} + \epsilon_f$$

$$v_{t+1} = 0.75 - 0.45\,T_{t+1} + 0.38\,f_{t+1} + 0.12\,f_{t+1}(1-T_{t+1}) + \epsilon_v$$

$$L_{t+1} = L_t + \Delta t\left(\underbrace{0.4\,u_f}_{\text{inflow}} - \underbrace{0.35\,L_t - 0.05\,u_T}_{\text{outflow}}\right) + \epsilon_L$$

Parameters: $\tau_T = 12$, $K_T = 0.85$, $\tau_f = 8$, $K_f = 0.90$, $\Delta t = 1$.

Three things are deliberately hidden from the agent's physics model:
1. The cross-coupling term $0.03 f_t u_T$ in the temperature equation
2. The two-step transport delay in the filler equation
3. The quadratic term $0.12 f(1-T)$ in the viscosity relationship

These represent the gap between first-principles engineering knowledge and reality.

**Reward** function targets temperature, filler fraction, and level stability:

$$r(s, a) = -2(T - T^*)^2 - 2(f - f^*)^2 - 0.5(L - L^*)^2 - 0.05\|a\|^2$$

with setpoints $T^* = 0.60$, $f^* = 0.50$, $L^* = 0.50$.

**Hard constraints** (violation = irreversible equipment damage or shutdown):

| Variable | Lower | Upper |
|---|---|---|
| temperature | 0.35 | 0.85 |
| filler fraction | 0.20 | 0.75 |
| viscosity | 0.10 | 0.90 |
| density | 0.30 | 0.80 |
| level | 0.15 | 0.85 |

```python
class CoatingProcessEnv:
    STATE_DIM  = 5
    ACTION_DIM = 2

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

    TAU_T, TAU_F = 12.0, 8.0
    K_T,   K_F   = 0.85, 0.90
    DT           = 1.0
```

### Why the Level Variable is the Hardest

The first four state variables are **self-regulating**: if the heater is turned off, temperature eventually returns to ambient. The level variable is **integrating**: it has no natural setpoint. Left alone, it drifts in the direction of net flow imbalance. Controlling an integrating state requires the agent to predict the sign and magnitude of its own long-term effect — which is exactly what offline RL struggles with when data is sparse near the constraint boundaries.

This is not a contrived difficulty. Surge tanks, buffer silos, and intermediate storage vessels are integrating elements in nearly every continuous manufacturing process.

---

## The Dataset: Operating Regimes

Industrial data is not uniformly distributed. `collect_industrial_dataset` simulates three operating regimes that appear in real historical logs:

**Normal operation (60% of episodes):** The process runs near its setpoints. The behavior policy is a proportional controller with moderate exploration noise. This is the bulk of the data — the region where all algorithms will perform reasonably.

**Setpoint transitions (25%):** Episodes start away from the nominal operating point. Higher noise, shifted initial states. These transitions stress-test the agent's ability to drive the process back to setpoint rather than just maintaining it.

**Disturbances (15%):** Unusual initial conditions — specifically, level disturbances where the surge tank starts far below normal. These are the episodes where constraint violations are most likely and where the physics model is most valuable.

```python
dataset = collect_industrial_dataset(
    n_episodes=400,
    episode_len=100,
    noise_scale=0.30,
    regime_split=(0.60, 0.25, 0.15),
)
```

This distribution creates the characteristic industrial imbalance: the trained agents will encounter disturbance scenarios rarely during evaluation but must handle them correctly. An agent that learned only from normal-operation data will be systematically undertrained for the 15% of situations that matter most.

---

## Evaluation: Industrial Metrics

Standard RL metrics — episode return, normalized score — are insufficient for industrial evaluation. `IndustrialEvaluator` computes five metrics that matter in practice.

### Directional Accuracy (DA)

$$\text{DA}_x = \frac{1}{T}\sum_{t=1}^T \mathbf{1}\!\left[\Delta x_t \cdot (x^* - x_{t-1}) \geq 0\right]$$

DA measures whether the controlled variable moved *toward* its setpoint at each step, regardless of how far it moved. A policy that always moves in the right direction but slowly has DA = 1.0. A policy that systematically pushes variables the wrong way has DA < 0.5 (worse than random).

DA is often more informative than RMSE in industrial settings because:
- A slow policy with DA ≈ 1.0 can be sped up by tuning the action scaling
- A fast policy with DA ≈ 0.4 requires fundamental retraining
- Operators can visually verify DA from trend plots without numerical analysis

### RMSE and Constraint Violation Rate

RMSE from setpoint is computed per variable:

$$\text{RMSE}_x = \sqrt{\frac{1}{T}\sum_t (x_t - x^*)^2}$$

Constraint violation rate is the fraction of time steps where any state variable falls outside its hard bounds. Severity measures the mean violation magnitude *given* that a violation occurred:

$$\text{severity} = \frac{\sum_t \max(0,\, g(s_t))}{\sum_t \mathbf{1}[g(s_t) > 0]}$$

```python
class IndustrialEvaluator:
    """
    Computes: reward_mean, reward_std, T_rmse, f_rmse, level_rmse,
              da_T, da_f, da_mean, constraint_viol_rate,
              constraint_viol_severity
    """

    def evaluate(self, agent) -> Dict[str, float]:
        for ep in range(self.n_episodes):
            obs = self.env.reset(seed=8000 + ep)
            while not done:
                prev_T, prev_f = obs[0], obs[1]
                s_t  = torch.FloatTensor(
                    (obs - self.s_mean) / self.s_std).unsqueeze(0)
                act  = agent.policy.act(s_t, deterministic=True)
                obs, r, done, info = self.env.step(act)

                # Directional accuracy
                da_T = 1.0 if (obs[0]-prev_T)*(env.T_TARGET-prev_T) >= 0 else 0.0
                da_f = 1.0 if (obs[1]-prev_f)*(env.F_TARGET-prev_f) >= 0 else 0.0

                # Constraint tracking
                viol = info['constraint_violation']   # sum of boundary overruns
```

---

## The Physics Model

The known physics for the coating process is encoded in `coating_physics_fn`. It implements the first-order responses and the mass balance exactly, but omits the cross-coupling, transport delay, and viscosity nonlinearity:

```python
def coating_physics_fn(state: torch.Tensor,
                        action: torch.Tensor) -> torch.Tensor:
    T, f, L = state[:, 0], state[:, 1], state[:, 4]
    heat_in  = action[:, 0] * 0.5 + 0.5
    flow_in  = action[:, 1] * 0.5 + 0.5

    T_new = (1 - DT/TAU_T) * T + (DT/TAU_T) * K_T * heat_in
    f_new = (1 - DT/TAU_F) * f + (DT/TAU_F) * K_F * flow_in

    # Linearized viscosity (true is nonlinear)
    v_new = (0.75 - 0.45*T_new + 0.38*f_new).clamp(0.0, 1.0)
    d_new = (0.55 + 0.25*f_new - 0.10*T_new).clamp(0.0, 1.0)

    # Mass balance — good approximation of true level dynamics
    inflow  = flow_in * 0.4
    outflow = L * 0.35 + 0.05 * heat_in
    L_new   = (L + DT * (inflow - outflow)).clamp(0.0, 1.0)

    return torch.stack([T_new, f_new, v_new, d_new, L_new], dim=1)
```

Before training the hybrid ensemble, it is essential to run `diagnose_physics_coverage` to verify that the physics model is actually useful:

```python
ensemble.diagnose_physics_coverage(norm_dataset)

# Expected output:
#   Physics model coverage (fraction of variance explained):
#   state dim 0 (temperature):     91.3%  ██████████████████░░
#   state dim 1 (filler_fraction): 83.7%  ████████████████░░░░
#   state dim 2 (viscosity):       64.2%  ████████████░░░░░░░░
#   state dim 3 (density):         79.8%  ████████████████░░░░
#   state dim 4 (level):           93.1%  ██████████████████░░
#   Overall: 82.4%
```

Coverage above 80% overall means the residual network needs to learn a 20%-amplitude correction — roughly one quarter the network capacity of a pure black-box model at the same prediction accuracy.

---

## Algorithm 1: Behavioral Cloning

BC is the baseline: clone the behavior policy directly. Its failure mode on industrial data is predictable — compounding error during setpoint transitions and disturbances, where the agent encounters states it never saw during training.

```python
bc = CQLBCAgent(STATE_DIM, ACTION_DIM, device=device)
train_bc(bc, loader, n_epochs=60)
```

BC's DA will be high in normal-operation episodes (the behavior policy was a competent proportional controller) and low during disturbance recovery. Constraint violations are rare in normal operation and frequent in disturbances — exactly the scenarios BC was least trained for.

---

## Algorithm 2: CQL

CQL (Chapter 3) adds conservatism via the penalty on out-of-distribution Q-values. On industrial data it consistently outperforms BC in disturbance scenarios because the pessimistic Q-function prevents the policy from taking actions that look good under the learned Q but are rare in the dataset.

```python
cql = CQLAgent(STATE_DIM, ACTION_DIM, alpha_cql=1.0, device=device)
train_cql_agent(cql, loader, n_epochs=80)
```

CQL's remaining weakness: it has no mechanism to avoid physically inconsistent actions. A setpoint command that CQL's Q-function rates highly may still violate a mass balance or push the level toward a constraint boundary.

---

## Algorithm 3: CQL + Physics Reward Shaping

`PhysicsInformedCQL` wraps `CQLAgent` with `PhysicsRewardWrapper` (Chapter 6). The modification to the training loop is exactly one line: replace the batch reward `r` with `r - penalty` before calling `CQLAgent.update`.

```python
constraints, lambdas = make_coating_constraints(dataset, device)

class PhysicsInformedCQL:
    def update(self, batch):
        s, a, r, s2, d = [x.to(self.device) for x in batch]
        with torch.no_grad():
            penalty = sum(
                lam * g(s, a, s2)
                for g, lam in zip(self.constraints, self.lambdas)
            )
        return self.cql.update((s, a, r - penalty, s2, d))
```

The `lambdas` are calibrated via `calibrate_lambda` (Theorem 6.1): set $\lambda$ so that the Theorem 6.1 optimality gap is at most 10% of the mean episode return. This avoids both extremes — a $\lambda$ too small that makes the penalty ignorable, and a $\lambda$ too large that makes the policy ignore reward entirely.

Three constraints are active:
1. **Operating bounds** — all five state variables within hard limits
2. **Temperature first-order consistency** — $|T_{t+1} - T^{phys}_{t+1}| \leq 0.04$
3. **Filler fraction first-order consistency** — same for filler

The first-order consistency constraints do not punish the agent for choosing unusual setpoints — they punish the dynamics model for predicting physically impossible transitions, which indirectly constrains the policy to stay in regions where dynamics are predictable.

---

## Algorithm 4: HybridMOReL

`HybridMOReL` combines the model-based approach from Chapter 5 with the hybrid dynamics model from Chapter 6. The structure is a two-phase training loop.

**Phase 1 — Dynamics training:**

```python
hm = HybridMOReL(
    state_dim   = STATE_DIM,
    action_dim  = ACTION_DIM,
    physics_fn  = coating_physics_fn,
    n_ensemble  = 4,
    hidden_dim  = 128,
    halt_thresh = 0.15,
)
hm.train_dynamics(norm_dataset, n_epochs=30)
# Output:
#   Physics coverage: 82.4% — strong
#   Training hybrid ensemble: 4 models × 30 epochs
#   ✓ HybridEnsemble installed in MOReL
```

After training, the hybrid ensemble is installed in the MOReL agent by swapping `morel_agent.ensemble`. The MOReL rollout generator (`generate_synthetic_data`) continues to call `ensemble.predict_with_uncertainty` — it does not know or care whether the ensemble is pure black-box or hybrid.

**Phase 2 — Policy training via MOReL:**

```python
train_morel(
    agent               = hm.morel_agent,
    dataset             = norm_dataset,
    env                 = env,         # real env for evaluation only
    n_outer_iters       = 15,
    sac_steps_per_iter  = 300,
)
```

MOReL's P-MDP construction applies the HALT penalty whenever ensemble uncertainty exceeds `halt_thresh`. With the hybrid ensemble, uncertainty estimates are better calibrated: in OOD regions where the residual network has no training data, the physics term keeps the prediction physically plausible rather than arbitrary. This reduces spurious HALT events caused by the pure black-box ensemble producing nonsensical predictions in physically reachable but data-sparse states.

---

## Results

Running the full benchmark:

```python
results = run_industrial_benchmark(
    device       = 'cpu',
    n_train_ep   = 400,
    n_bc_epochs  = 60,
    n_cql_epochs = 80,
    n_morel_iters= 15,
    n_eval_ep    = 20,
)
```

Typical output (exact numbers depend on random seed):

```
══════════════════════════════════════════════════════════════
  CHAPTER 7 — INDUSTRIAL BENCHMARK RESULTS
══════════════════════════════════════════════════════════════
Metric                BC      CQL   CQL+Phys  HybridMOReL
──────────────────────────────────────────────────────────────
Reward (mean)       -18.4   -14.2    -13.8   ▶  -12.1
DA (mean)            61.3%   71.8%   73.4%  ▶   79.2%
DA temperature       64.1%   73.5%   75.1%  ▶   81.3%
DA filler            58.4%   70.1%   71.6%  ▶   77.1%
T RMSE               0.0921  0.0712  0.0698  ▶  0.0581
f RMSE               0.1043  0.0834  0.0811  ▶  0.0693
Level RMSE           0.0874  0.0748  0.0701  ▶  0.0612
Violation rate       4.2%    2.8%  ▶  1.1%      1.9%
══════════════════════════════════════════════════════════════
```

### Reading the Results

**BC** performs adequately in normal operation but fails during disturbances. Its DA is just above 60% — better than random, but below the 80% industrial threshold. Constraint violations occur because BC never saw the level variable near its bounds during training.

**CQL** improves across all metrics. Conservatism in the Q-function prevents the worst extrapolation failures. But violation rate remains at 2.8% — CQL has no mechanism to specifically avoid constraint boundaries.

**CQL+Physics** reduces violations to 1.1%, the lowest of any model-free method. The reward penalty makes approaching constraint boundaries explicitly costly during training. DA improves modestly — the physics constraints act as a form of implicit regularization that keeps the policy in physically meaningful regions. The violation severity also drops even more than the rate, meaning that when violations do occur they are shallower.

**HybridMOReL** achieves the best reward and the highest DA. Model-based rollouts give the policy more diverse training experience than the real dataset alone. The hybrid ensemble makes those rollouts physically consistent — the agent is not trained on synthetic transitions that violate mass balance or predict temperature falling when heat is applied. Violation rate (1.9%) is higher than CQL+Physics because MOReL focuses on maximizing reward in the P-MDP without the explicit constraint penalty.

### The Complementary Structure

These results reveal a complementary structure worth understanding.

**CQL+Physics** is the violation minimizer: it explicitly penalizes constraint approaches and achieves the lowest violation rate. But it uses only real data — its synthetic experience is zero.

**HybridMOReL** is the performance maximizer: model-based rollouts push the policy to higher reward and DA, but without explicit constraint penalties the violation rate is slightly elevated.

In a real deployment, the natural combination is **HybridMOReL + physics reward shaping** — use the hybrid model for diverse synthetic rollouts AND apply the constraint penalty to all rewards (real and synthetic alike). This is a one-line change to `train_hybrid_morel_agent`: pass the `PhysicsRewardWrapper` into the rollout generator. Chapter 6 showed exactly how to do this.

---

## Engineering Decisions Revisited

Working through this case study surfaces several decisions that do not appear in benchmark papers.

### The Transport Delay Problem

The filler equation has a two-step delay: the agent's flow input at time $t$ affects the filler fraction at time $t+2$, not $t+1$. The physics model ignores this delay — it predicts an immediate first-order response.

This means the hybrid ensemble's residual must learn the delay effect from data. With enough transitions, it can. The tell-tale sign of a delay mismatch is a systematic residual that is positive when flow_input recently increased and negative when it recently decreased. If `diagnose_physics_coverage` shows low coverage for the filler dimension (below 60%), the likely cause is an unmodeled delay.

The fix is explicit: augment the state with the last $k$ flow inputs and let the physics model account for them. The engineering knowledge required is only the approximate delay length — the exact value can be learned from the residual.

### Constraint Calibration

`make_coating_constraints` uses `calibrate_lambda` for the bounds constraint and sets $\lambda = 2.0$ manually for the first-order consistency constraints. The manual values are reasonable starting points but should be verified:

```python
# Check constraint violation rates on the training dataset
for g, lam in zip(constraints, lambdas):
    violations = g(s_train, a_train, s2_train)
    print(f"  λ={lam:.2f} | violations: {(violations > 0).float().mean():.1%} "
          f"| mean_magnitude: {violations.mean():.5f}")
```

If a constraint is violated on more than 20% of training transitions, either the tolerance is too tight or the physics model is inaccurate for that variable. In both cases, increasing the tolerance or loosening the manual $\lambda$ is better than tightening the constraint — a constraint that fires too often becomes a second reward signal rather than a physics correction.

### Normalizing the 5D State

Five variables with different physical scales and variances require per-dimension normalization. The provided `normalize_dataset` normalizes states to zero mean and unit variance per dimension. The reward is scaled by its standard deviation but not centered — centering the reward would remove the sign information that distinguishes good states from bad.

---

## Summary

This chapter translated the tools from Chapters 1–6 into a realistic industrial pipeline. The coating process introduced two challenges absent from the toy environment: an integrating level variable and a transport delay in the filler dynamics. Both required engineering knowledge to handle — either in the physics model or in the constraint specification.

The comparison across four algorithms reveals a clean hierarchy:

| Algorithm | Relies on | Strength | Weakness |
|---|---|---|---|
| BC | Data imitation | Simple, interpretable | Compounding error; constraint-blind |
| CQL | Conservative Q-values | Better OOD generalization | No physical constraint mechanism |
| CQL+Physics | CQL + penalty | Lowest violation rate | Uses only real data |
| HybridMOReL | Model-based + hybrid | Best reward and DA | Higher violation rate without explicit penalty |

The practical recommendation for industrial deployment: **start with CQL+Physics** (low violation rate, simple to configure, no dynamics model needed), and if reward performance is insufficient, **add the hybrid ensemble** as a source of diverse synthetic experience.

Chapter 8 returns to the theoretical questions this case study raises: when can offline RL be trusted in deployment, what guarantees are available, and what open problems remain unsolved.

---

## References

- All references from Chapters 1–6 apply to the algorithms used here.
- Åström, K.J., & Wittenmark, B. (2013). *Computer-Controlled Systems.* Dover. *(PID baseline, first-order process models)*
- Seborg, D.E., Edgar, T.F., Mellichamp, D.A., & Doyle, F.J. (2016). *Process Dynamics and Control.* Wiley. *(industrial process control, transport delay)*
- Rawlings, J.B., Mayne, D.Q., & Diehl, M. (2017). *Model Predictive Control.* Nob Hill. *(MPC for comparison context)*

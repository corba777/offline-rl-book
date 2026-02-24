---
layout: default
title: "Chapter 8: Physics-Informed Offline RL"
lang: en
ru_url: /ru/chapter8/
prev_chapter:
  url: /en/chapter7/
  title: "Model-Based Offline RL (MOPO, MOReL)"
next_chapter:
  url: /en/chapter9/
  title: "Industrial Applications"
permalink: "/offline-rl-book/en/chapter8/"
---

# Chapter 8: Physics-Informed Offline RL

> *"A black-box model that fits the data perfectly inside the training distribution is still useless if it predicts that energy is not conserved outside it."*

---

## Why Physics?

Chapters 3–7 treated the world as a black box: we fed in states and actions, observed next states and rewards, and let neural networks figure out the rest. For benchmark tasks like MuJoCo locomotion this is fine — the environment is a simulation, data is plentiful, and there are no hard physical constraints that must hold.

Industrial processes are different. Three things distinguish them:

**Data is scarce and narrow.** A manufacturing line runs in a small region of its theoretical operating envelope — the operator keeps it there deliberately. An offline dataset from six months of normal operation may contain almost no examples of the process near its physical limits. Any model that tries to reason about those regions is extrapolating.

**Physical laws are partially known.** For most industrial systems, engineers know a great deal: conservation of mass and energy, thermodynamic relationships, transport phenomena, first-order dynamics. This knowledge is imperfect (real systems have unmeasured disturbances, aging equipment, composition variation) but it is not zero. Ignoring it wastes a free source of information.

**Violations of physics are costly.** A policy that violates a mass balance might command an impossible setpoint. A model that predicts temperature decreasing when heat is added is not just inaccurate — it will produce a control policy that actively damages the process.

Physics-informed offline RL exploits known structure in three places:

1. **Reward shaping** — encode physical constraints as soft penalties in the reward signal
2. **Hybrid dynamics model** — replace the pure black-box ensemble with a model that respects known physics exactly and learns the unknown residual
3. **Constraint-based policy** — embed hard constraints directly into policy optimization (brief treatment, as this overlaps with safe RL)

---

## Part I: Physics as Reward Shaping

### The Idea

The simplest way to use physics: add penalty terms to the reward whenever the system violates known constraints. This works with *any* offline RL algorithm — CQL, IQL, MOPO — without changing the algorithm itself.

For a process with state $s = (x_1, \ldots, x_n)$ and known constraint $g(s) \leq 0$:

$$\tilde{r}(s, a, s') = r(s, a) - \lambda \cdot \max\bigl(0,\ g(s')\bigr)$$

The $\max(0, \cdot)$ activates the penalty only when the constraint is violated; inside the feasible region it contributes nothing. $\lambda$ controls the trade-off between reward maximization and constraint satisfaction.

### Types of Physical Constraints

**Monotone relationships.** Many physical relationships are guaranteed to be monotone. Viscosity of a liquid decreases with temperature (for most fluids in normal operating ranges). Conversion in a reactor increases with residence time up to some limit. If the learned dynamics model predicts the wrong sign, it is physically wrong.

$$g_{mono}(s, s') = \max\Bigl(0,\ \text{sign}\bigl(\hat{f}(s')\bigr) - \text{sign}\bigl(f_{phys}(s')\bigr)\Bigr)$$

**Conservation laws.** Mass and energy must balance. For a mixing process with input flows $q_{in}$ and output flow $q_{out}$ and a tank of volume $V$:

$$\frac{dV}{dt} = q_{in} - q_{out}$$

If a discrete-time model predicts $V_{t+1}$ inconsistent with this, the violation magnitude is:

$$g_{mass}(s_t, s_{t+1}) = \bigl| V_{t+1} - V_t - \Delta t (q_{in,t} - q_{out,t}) \bigr| - \delta$$

where $\delta$ is a tolerance for measurement noise and unmodeled flows.

**Feasibility bounds.** Physical state variables have hard limits from thermodynamics, equipment ratings, or safety considerations. Temperature cannot exceed the boiling point of the medium. Concentration cannot be negative. A valve position is in $[0, 1]$.

$$g_{bounds}(s) = \sum_i \max(0,\ s_i - s_i^{max}) + \max(0,\ s_i^{min} - s_i)$$

### Implementation

> 📄 Full code: [`physics_informed.py`](https://github.com/corba777/offline-rl-book/blob/main/code/physics_informed.py)

```python
class PhysicsRewardWrapper:
    """
    Wraps any reward function with physics-based penalty terms.

    Constraints are callables g(state, action, next_state) -> float,
    returning a non-negative violation magnitude (0 = constraint satisfied).

    Usage:
        reward_fn = PhysicsRewardWrapper(
            base_reward=my_reward,
            constraints=[mass_balance_constraint, temperature_bound],
            lambdas=[10.0, 5.0],
        )
        r_phys = reward_fn(state, action, next_state)
    """

    def __init__(self, base_reward, constraints, lambdas):
        assert len(constraints) == len(lambdas)
        self.base_reward = base_reward
        self.constraints = constraints
        self.lambdas     = lambdas

    def __call__(self,
                 state:      torch.Tensor,
                 action:     torch.Tensor,
                 next_state: torch.Tensor) -> torch.Tensor:
        r = self.base_reward(state, action, next_state)
        for g, lam in zip(self.constraints, self.lambdas):
            violation = g(state, action, next_state)        # (batch,) >= 0
            r = r - lam * violation
        return r
```

**Example constraint — first-order dynamics consistency:**

A first-order process with known time constant $\tau$ and gain $K$ satisfies:

$$x_{t+1} \approx x_t + \frac{\Delta t}{\tau}(K \cdot u_t - x_t)$$

If the learned model's prediction deviates significantly from this, the deviation is a constraint violation:

```python
def first_order_consistency(state, action, next_state,
                             tau=10.0, K=0.8, dt=1.0, tol=0.05):
    """
    Physics penalty: predicted next state should be consistent with
    first-order dynamics x_{t+1} ≈ x_t + dt/tau * (K*u - x_t).
    """
    x    = state[:, 0]    # process variable
    u    = action[:, 0]   # control input
    x_phys = x + (dt / tau) * (K * u - x)
    x_pred = next_state[:, 0]
    violation = torch.clamp(torch.abs(x_pred - x_phys) - tol, min=0.0)
    return violation
```

### Choosing $\lambda$

The penalty weight $\lambda$ has the same role as $\alpha$ in CQL: too small and violations persist; too large and the policy becomes overly conservative, ignoring the reward signal.

A practical calibration: compute the typical base reward magnitude $\bar{r}$ on the dataset, then set $\lambda$ so that a typical violation (say, 5% mass balance error) costs about 20–50% of $\bar{r}$. This makes the constraint "expensive but not catastrophic."

```python
# Calibrate lambda from dataset statistics
r_mean = np.abs(dataset['rewards']).mean()
typical_violation = 0.05   # expected violation magnitude in the region of interest
target_penalty_fraction = 0.3   # want penalty ≈ 30% of mean reward

lam = (target_penalty_fraction * r_mean) / typical_violation
print(f"Calibrated lambda = {lam:.2f}")
```

---

## Part II: Hybrid Dynamics Models

### The Limitation of Pure Black-Box Ensembles

In Chapter 7, the dynamics ensemble is a pure neural network: it learns $\hat{T}(s' | s, a)$ entirely from data. This works well when:
- The dataset covers the state-action space densely
- The model is evaluated near the training distribution

Outside the data support, a pure black-box model will extrapolate in whatever direction its weights happen to point — which often violates physics. The epistemic uncertainty from ensemble disagreement partially guards against this, but it cannot distinguish "uncertain because data is sparse" from "uncertain because the model is physically wrong."

A **hybrid model** splits the prediction into two parts:

$$s'_{hybrid} = \underbrace{f_{phys}(s, a)}_{\text{known physics}} + \underbrace{f_{NN}(s, a; \theta)}_{\text{learned residual}}$$

The physics component $f_{phys}$ is an analytical function derived from domain knowledge — it is exact (by assumption) but incomplete (it misses unmodeled effects, disturbances, nonlinearities). The neural residual $f_{NN}$ learns what the physics gets wrong.

### Why This Helps Offline RL

**Better extrapolation.** The physics component provides a principled prediction even where data is sparse. The neural residual is near zero where data is sparse (prior to training, and by regularization), so the hybrid model degrades gracefully to the physics model in OOD regions rather than making wild predictions.

**Smaller residuals = easier learning.** If the physics model captures 80% of the dynamics, the neural network only needs to learn the remaining 20%. With the same dataset size, the residual model converges faster and generalizes better.

**Improved uncertainty calibration.** Ensemble disagreement now measures uncertainty about the *residual*, not the total dynamics. In OOD regions, the physics term is still active; only the residual is uncertain. This gives more meaningful uncertainty estimates.

### Formalization

Let $f_{phys}(s, a)$ be the known physics model (deterministic, no parameters). The residual network predicts a Gaussian over the residual:

$$\hat{f}_{NN}(s, a; \theta) = \bigl(\mu_\theta(s, a),\ \sigma^2_\theta(s, a)\bigr)$$

The full prediction is:

$$s' \sim \mathcal{N}\!\bigl(f_{phys}(s, a) + \mu_\theta(s, a),\ \sigma^2_\theta(s, a) \cdot I\bigr)$$

Training minimizes the NLL of the residual $\delta = s' - f_{phys}(s, a)$:

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,s') \sim \mathcal{D}} \!\left[\frac{\|\delta - \mu_\theta\|^2}{2\sigma^2_\theta} + \frac{1}{2}\log \sigma^2_\theta \right]$$

This is identical to training the probabilistic ensemble in Chapter 7, except the *target* is the residual $\delta$ rather than the full $s'$.

### Implementation

```python
class HybridDynamicsNet(nn.Module):
    """
    Hybrid dynamics model: physics baseline + learned residual.

    Args:
        state_dim:   dimensionality of state
        action_dim:  dimensionality of action
        physics_fn:  callable (state, action) -> next_state_physics
                     Must operate on torch.Tensor inputs.
        hidden_dim:  residual network hidden size
    """

    def __init__(self, state_dim: int, action_dim: int,
                 physics_fn,
                 hidden_dim: int = 128):
        super().__init__()
        self.physics_fn = physics_fn

        # Residual network: smaller than a pure black-box model
        # because residuals are smoother and lower-amplitude
        inp = state_dim + action_dim
        self.trunk = nn.Sequential(
            nn.Linear(inp, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.mean_head    = nn.Linear(hidden_dim, state_dim)
        self.log_var_head = nn.Linear(hidden_dim, state_dim)

        # Tighter log_var bounds for residuals (smaller magnitude expected)
        self.min_log_var = nn.Parameter(-8.0 * torch.ones(state_dim))
        self.max_log_var = nn.Parameter(-1.0 * torch.ones(state_dim))

    def forward(self, state: torch.Tensor,
                action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean_next_state, log_var_residual)."""
        # Physics prediction (no gradient needed)
        with torch.no_grad():
            phys = self.physics_fn(state, action)

        h = self.trunk(torch.cat([state, action], -1))
        residual_mean = self.mean_head(h)
        log_var       = self.log_var_head(h)
        log_var = self.max_log_var - F.softplus(self.max_log_var - log_var)
        log_var = self.min_log_var + F.softplus(log_var - self.min_log_var)

        # Combine: full prediction = physics + residual
        return phys + residual_mean, log_var

    def nll_loss(self, state: torch.Tensor, action: torch.Tensor,
                 next_state: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """NLL of next_state; target is residual from physics."""
        mean, log_var = self.forward(state, action)
        # The target is the full next_state; mean already includes physics
        var = log_var.exp()
        nll = 0.5 * (log_var + (next_state - mean).pow(2) / var)
        loss = nll.sum(-1).mean()
        bound_reg = 0.01 * (self.max_log_var.sum() - self.min_log_var.sum())
        return loss + bound_reg, {
            'nll': loss.item(),
            'residual_mse': (next_state - mean).pow(2).mean().item(),
        }
```

**Physics function for a generic first-order process:**

```python
def first_order_physics(state: torch.Tensor,
                        action: torch.Tensor,
                        tau: float = 10.0,
                        K: float   = 0.8,
                        dt: float  = 1.0) -> torch.Tensor:
    """
    Euler discretization of first-order ODE:
        dx/dt = (K*u - x) / tau
    =>  x_{t+1} = x_t + dt/tau * (K*u_t - x_t)

    For a multi-variable state, apply element-wise with per-variable
    tau and K (pass as vectors).
    """
    return state + (dt / tau) * (K * action[:, :state.shape[1]] - state)
```

### Building a Hybrid Ensemble

The ensemble from Chapter 7 is straightforwardly extended — each model in the ensemble is a `HybridDynamicsNet` sharing the same `physics_fn`:

```python
class HybridEnsemble:
    """
    Ensemble of hybrid dynamics models sharing the same physics component.
    Uncertainty = disagreement on the *residual* term across models.
    """

    def __init__(self, n_models: int, state_dim: int, action_dim: int,
                 physics_fn, hidden_dim: int = 128,
                 lr: float = 1e-3, device: str = 'cpu'):
        self.n_models  = n_models
        self.device    = device
        self.models    = [
            HybridDynamicsNet(state_dim, action_dim, physics_fn, hidden_dim).to(device)
            for _ in range(n_models)
        ]
        self.optimizers = [optim.Adam(m.parameters(), lr=lr) for m in self.models]

    def train_ensemble(self, dataset: dict, n_epochs: int = 50,
                       batch_size: int = 256, log_every: int = 10):
        """Bootstrap training — identical structure to Chapter 7 ensemble."""
        n = len(dataset['states'])
        s  = torch.FloatTensor(dataset['states']).to(self.device)
        a  = torch.FloatTensor(dataset['actions']).to(self.device)
        s2 = torch.FloatTensor(dataset['next_states']).to(self.device)

        for i, (model, opt) in enumerate(zip(self.models, self.optimizers)):
            idx = torch.randint(0, n, (int(0.8 * n),))
            loader = DataLoader(TensorDataset(s[idx], a[idx], s2[idx]),
                                batch_size=batch_size, shuffle=True, drop_last=True)
            for epoch in range(1, n_epochs + 1):
                total_nll, nb = 0.0, 0
                for s_b, a_b, s2_b in loader:
                    loss, info = model.nll_loss(s_b, a_b, s2_b)
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    total_nll += info['nll']; nb += 1
                if epoch % log_every == 0:
                    print(f"  HybridModel {i} | epoch {epoch:3d} | "
                          f"NLL={total_nll/nb:.4f}")

    @torch.no_grad()
    def predict_with_uncertainty(self, states: torch.Tensor,
                                  actions: torch.Tensor):
        """Returns (mean_next_state, epistemic_uncertainty)."""
        means = [m(states, actions)[0] for m in self.models]
        means = torch.stack(means, 0)
        return means.mean(0), means.std(0).norm(dim=-1)
```

### Comparing Hybrid vs Pure Black-Box

The key diagnostic is **residual magnitude** on the training data. If the physics model is reasonable, residuals should be small and smooth. If residuals are large or structured (have clear patterns), the physics model is missing something important.

```python
def diagnose_hybrid_model(hybrid_model, dataset, physics_fn, device='cpu'):
    """
    Compare physics prediction vs true next_state.
    Print residual statistics to assess physics model quality.
    """
    s  = torch.FloatTensor(dataset['states'][:500]).to(device)
    a  = torch.FloatTensor(dataset['actions'][:500]).to(device)
    s2 = torch.FloatTensor(dataset['next_states'][:500]).to(device)

    with torch.no_grad():
        phys_pred = physics_fn(s, a)
        residuals = s2 - phys_pred                     # what the NN must learn
        residual_std = residuals.std(0).cpu().numpy()
        full_std     = s2.std(0).cpu().numpy()

    coverage = 1.0 - (residual_std / (full_std + 1e-8))

    print("Physics model coverage (fraction of variance explained):")
    for i, (cov, rs, fs) in enumerate(zip(coverage, residual_std, full_std)):
        print(f"  State dim {i}: physics covers {cov:.1%}  "
              f"(residual std={rs:.4f}, total std={fs:.4f})")
    return coverage
```

A physics model explaining 70–90% of variance is excellent; the residual model needs to learn only the remaining 10–30%, which requires far less data.

---

## Part III: Constraint-Based Policy (Brief)

### Hard vs Soft Constraints

Reward penalties (Part I) are **soft constraints**: the policy *prefers* to avoid violations but can still choose them if the reward benefit is large enough. For safety-critical systems this may be insufficient.

**Hard constraints** guarantee $g(s, a) \leq 0$ at every step. Two common approaches in offline RL:

**Projection.** After computing the policy action $a_{raw} = \pi_\theta(s)$, project it onto the feasible set:

$$a_{safe} = \arg\min_{a : g(s,a) \leq 0} \| a - a_{raw} \|^2$$

This is trivial for box constraints (clamp) and for linear constraints (quadratic programming). For complex nonlinear constraints it requires a solver at inference time.

**Lagrangian relaxation.** Augment the objective with constraint multipliers and optimize jointly:

$$\max_{\pi} \min_{\lambda \geq 0}\ \mathbb{E}\bigl[R(\pi)\bigr] - \lambda \cdot \mathbb{E}\bigl[g(s, \pi(s))\bigr]$$

The multiplier $\lambda$ is updated to increase whenever the constraint is violated, making violations increasingly costly. In offline RL this is applied to the offline dataset, not through environment interaction.

For most industrial applications, **soft constraints via reward shaping** (Part I) combined with the KNOWN/UNKNOWN boundary from MOReL (Chapter 7) provide sufficient safety. The hard constraint machinery is warranted when:
- A constraint violation is irreversible (equipment damage, safety incident)
- The constraint boundary is precisely known
- Online projection is computationally feasible

---

## Part IV: When Physics Helps Most

Not every offline RL problem benefits equally from physics. The value of physical knowledge depends on three factors:

**Data coverage.** The sparser the dataset, the more physics helps. A dataset covering 20% of the operating envelope leaves 80% where only physics can guide the model.

**Physics accuracy.** A first-principles model that captures 90% of the dynamics is highly valuable. An analytical model that captures only 30% may still cause problems if its errors are systematic (i.e., confidently wrong in a predictable way). Validate the physics model on a held-out subset before trusting it.

**Constraint hardness.** Mass balance and energy conservation are exact (up to measurement error). Empirical correlations (viscosity-temperature-composition) are approximate. Use the former as hard constraints; use the latter as soft penalties or initial guesses for the neural residual.

```
                    Physics Knowledge Available
                    None         Partial       Strong
                   ┌────────────┬─────────────┬──────────────┐
Data   Sparse      │  Pure BC   │ Reward pen. │ Hybrid model │
Coverage  ↓        │  (Ch. 1)   │ + MOReL     │ + MOReL      │
       Dense       │ CQL / IQL  │ CQL + pen.  │ Hybrid + CQL │
                   └────────────┴─────────────┴──────────────┘
```

The bottom-left cell (dense data, no physics) is the D4RL benchmark — where Chapters 3–7 shine. Industrial settings typically live in the top-right: sparse data, partial physics knowledge.

---

## Further Directions

**Koopman operator methods.** Nonlinear dynamics become linear in a suitable latent space — the Koopman eigenspace. If this space can be learned or approximated, the dynamics model simplifies enormously: linear prediction, linear uncertainty propagation, and tractable optimal control. Active research area; not yet standardized for offline RL.

**Gaussian Processes.** GPs offer exact uncertainty quantification and naturally incorporate physical structure through kernel design. They are computationally expensive for large datasets but highly effective when data is very scarce (hundreds of transitions rather than thousands).

**Physics-informed neural networks (PINNs).** Instead of a residual model, enforce physics equations as a regularization loss during neural network training. A PINN predicting the wrong sign for a known derivative can be penalized during training. Compatible with any architecture.

**Structured state spaces.** If the physics dictates a specific state-space structure (e.g., a mass-spring-damper Lagrangian system), neural networks can be constrained to architectures that respect this structure (Hamiltonian or Lagrangian neural networks). Very effective when the structural assumption holds; brittle when it doesn't.

---

## Summary

| Approach | What changes | When to use |
|---|---|---|
| Reward shaping | Reward function | Any algorithm; physical bounds known |
| Hybrid dynamics | Dynamics model | Partial physics known; sparse data |
| Projection | Policy output | Box/linear constraints; safety-critical |
| Lagrangian | Objective | Complex nonlinear constraints |

Physics-informed methods are not a replacement for offline RL algorithms — they are a layer on top. Reward shaping plugs into CQL or IQL without modifying a line of algorithm code. The hybrid model replaces the ensemble in MOPO or MOReL while keeping everything else intact. The combination of a physically grounded dynamics model with a pessimistic offline RL algorithm (MOReL + hybrid ensemble) is the natural architecture for industrial settings where data is sparse but domain knowledge is rich.

Chapter 9 works through an industrial case study showing how these pieces fit together in practice.

---

## References

- Banerjee, C., Nguyen, T., Fookes, C., & Raissi, M. (2023). *A Survey on Physics Informed Reinforcement Learning.* Expert Systems with Applications. [arXiv:2309.01909](https://arxiv.org/abs/2309.01909).
- Raissi, M., Perdikaris, P., & Karniadakis, G. (2019). *Physics-Informed Neural Networks.* Journal of Computational Physics. [arXiv:1811.10561](https://arxiv.org/abs/1811.10561).
- Liu, Y., & Wang, X. (2021). *Physics-Informed Dyna-style Model-Based Deep Reinforcement Learning.* [arXiv:2008.05598](https://arxiv.org/abs/2008.05598).
- Lusch, B., Kutz, J.N., & Brunton, S.L. (2018). *Deep Learning for Universal Linear Embeddings of Nonlinear Dynamics.* Nature Communications. *(Koopman)*
- García, J., & Fernández, F. (2015). *A Comprehensive Survey on Safe Reinforcement Learning.* JMLR. *(constraint methods)*
- Levine, S. et al. (2020). *Offline Reinforcement Learning: Tutorial, Review, and Perspectives.* [arXiv:2005.01643](https://arxiv.org/abs/2005.01643).

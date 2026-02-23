---
layout: default
title: "Chapter 8: Explainability in Offline RL"
lang: en
ru_url: /offline-rl-book/ru/chapter8/
prev_chapter:
  url: /en/chapter7/
  title: "Industrial Applications"
next_chapter:
  url: /en/chapter9/
  title: "Conclusion and Future Directions"
permalink: "/offline-rl-book/en/chapter8/"
---

# Chapter 8: Explainability in Offline RL

> *"A policy that achieves 80% Directional Accuracy but cannot explain why it chose each action is hard to trust and impossible to certify."*

---

## Why Explainability Matters in Industrial Offline RL

The algorithms in Chapters 1–7 produce policies that achieve high reward, respect physical constraints, and generalize to unseen operating regimes. What they do not produce is an answer to the question an operator will inevitably ask: *why did the agent command this setpoint at this moment?*

This is not merely a regulatory concern, though in many industries it is that too. It is a practical reliability question. A policy whose decisions are opaque cannot be diagnosed when it fails. An operator who does not understand the agent's reasoning cannot override it intelligently. A model that fits data well but for the wrong reasons will fail silently when the process changes.

Explainability in Offline RL is harder than in standard supervised learning for three reasons specific to the offline setting.

**There are three separate models to explain.** A trained Offline RL agent contains a Q-function (the critic), a policy (the actor), and optionally a dynamics model. Each answers a different question, and their explanations do not always agree. The Q-function might rate an action highly because temperature is near its setpoint; the policy might choose that action because level is near its lower bound. Understanding the discrepancy is as important as understanding the agreement.

**The Q-function's input is a concatenation of state and action.** Unlike a classifier whose input is a single feature vector, the Q-function takes $(s, a)$ as input. SHAP values computed over this concatenated space tell you whether the high Q-value comes from the state (the situation was favorable) or the action (the specific action was appropriate for this situation).

**The behavior policy's distribution is uneven.** SHAP values are relative to a background distribution. In Offline RL, the natural background is the offline dataset — dense near the operating setpoint and sparse near disturbances. SHAP values answer: *"how does this instance differ from the typical operating point, and how does that difference affect the output?"*

> 📄 Full code: [`chapter8.py`](../../code/chapter8.py)

---

## SHAP Background

SHAP (SHapley Additive exPlanations, Lundberg & Lee, 2017) decomposes the output of a model $f$ into additive contributions from each input feature:

$$f(x) = \phi_0 + \sum_{i=1}^{n} \phi_i(x)$$

where $\phi_0 = \mathbb{E}[f(X)]$ is the expected model output over the background distribution, and $\phi_i(x)$ is the contribution of feature $i$ to the prediction for instance $x$.

The SHAP values $\{\phi_i\}$ are the unique decomposition satisfying:
- **Efficiency**: $\sum_i \phi_i = f(x) - \phi_0$ — values sum to total prediction minus baseline
- **Symmetry**: features with equal marginal contributions receive equal SHAP values
- **Dummy**: a feature with zero marginal contribution receives $\phi_i = 0$

**KernelExplainer** computes SHAP values without architecture knowledge. It fits a weighted linear model to masked predictions:

$$\phi = \arg\min_{\phi} \sum_{z \in \{0,1\}^n} \pi(z)\left[f(h(x, z)) - \phi_0 - \sum_i z_i \phi_i\right]^2$$

where $z$ is a binary mask, $h(x, z)$ replaces masked features with background samples, and $\pi(z)$ weights coalitions by the Shapley kernel. This works on any black-box function — Q-networks, policy networks, ensemble dynamics — with no architecture assumptions.

We use KernelExplainer throughout for consistency: all three explanations use the same method, making them directly comparable.

---

## Three Levels of Explanation

### Level 1: Q-function SHAP — Why Does the Agent Value This Action?

The Q-function $Q(s, a)$ takes the concatenated $(s, a)$ vector as input and outputs a scalar value estimate. SHAP on this input answers which features — current state variables or the chosen action — contribute most to the Q-value.

```python
class QFunctionWrapper:
    """
    Wrap QNetwork as numpy-in / numpy-out for SHAP KernelExplainer.

    Input:  X ∈ R^{n × (state_dim + action_dim)}  — [state | action]
    Output: q ∈ R^n                                — Q-value per sample
    """
    def __call__(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.FloatTensor(X).to(self.device)
            s   = x_t[:, :self.s_dim]
            a   = x_t[:, self.s_dim:]
            q   = self.q(s, a)
        return q.cpu().numpy()
```

Feature vector: `[temperature, filler_frac, viscosity, density, level, heat_input, flow_input]`.

State feature SHAP answers: "was this situation favorable?"
Action feature SHAP answers: "was this action appropriately chosen for this state?"

**Expected pattern for a well-trained CQL agent:** state features near their setpoints → positive SHAP (favorable situation → high Q). If action SHAP values are near zero regardless of action magnitude, the Q-function ignores the action — a sign of Q-function collapse.

### Level 2: Policy SHAP — Why Does the Agent Choose This Action?

We explain each action dimension separately (SHAP requires scalar output):

```python
class PolicySingleOutputWrapper:
    """
    action_idx = 0 → heat_input
    action_idx = 1 → flow_input

    Explains deterministic mean action tanh(μ_θ(s)) — not sampled,
    so SHAP estimates are stable across calls.
    """
    def __call__(self, S: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            s_t     = torch.FloatTensor(S).to(self.device)
            mean, _ = self.policy._dist(s_t)
            action  = torch.tanh(mean)
        return action[:, self.action_idx].cpu().numpy()
```

**Expected pattern:** `heat_input` driven primarily by `temperature` deviation; `flow_input` by `filler_frac` deviation. If `level` has high SHAP for `flow_input`, the policy learned that flow rate affects level — physically correct (level depends on inflow) and a sign that the agent has internalized the integrating dynamics.

### Level 3: Dynamics SHAP — Why Does the Model Predict This Next State?

We explain three output dimensions: `next_temperature` (0), `next_filler_frac` (1), `next_level` (4):

```python
class DynamicsSingleOutputWrapper:
    """
    Explains ensemble mean prediction for one next-state dimension.
    """
    def __call__(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            s     = x_t[:, :self.s_dim]
            a     = x_t[:, self.s_dim:]
            s_next, _ = self.ensemble.predict_with_uncertainty(s, a)
        return s_next[:, self.state_idx].cpu().numpy()
```

**Physics sanity check:** `heat_input` must have positive mean SHAP for `next_temperature`; `flow_input` for `next_filler_frac`. These follow from first-order dynamics and hold for any physically reasonable model. If violated, the dynamics model learned an impossible relationship in some data region — a red flag before deployment.

---

## The Explainer Class

`OfflineRLExplainer` manages background construction and SHAP computation for all three levels:

```python
class OfflineRLExplainer:
    def __init__(self, agent, ensemble, dataset,
                 s_mean, s_std, device='cpu', n_background=100):
        # Background = representative sample from offline dataset
        idx = np.random.default_rng(42).choice(
            len(dataset['states']), size=n_background, replace=False)
        self.bg_states  = dataset['states'][idx]
        self.bg_actions = dataset['actions'][idx]
        self.bg_sa      = np.concatenate(
            [self.bg_states, self.bg_actions], axis=1)
```

The background defines $\phi_0$ — the average model output. Using the offline dataset means SHAP values answer: *"how does this transition differ from a typical observed transition?"*

```python
results = explainer.explain_all(
    n_explain    = 150,   # instances to explain
    n_background = 80,    # background dataset size
    n_samples    = 80,    # SHAP coalitions per instance
)
# results keys: 'states', 'actions', 'q_shap',
#               'policy_shap', 'dynamics_shap', 'q_base'
```

`n_samples` controls approximation quality. At 80, standard error is ~0.005 on normalized outputs — sufficient for feature ranking. At 500 the estimates are publication-quality but 6× slower.

---

## Visualization

### Summary Plot (Q-function)

Beeswarm plot: each dot is one instance, colored by feature value (blue=low, red=high), x-axis is SHAP value.

```python
plot_q_summary(results['q_shap'], SA_NAMES,
               save_path='ch8_q_summary.png')
```

Reading patterns:
- **Wide horizontal spread** → feature has variable impact
- **Red-right / blue-left consistently** → monotone relationship
- **Mixed colors** → nonlinear or interaction-dependent

### Bar Charts (Policy and Dynamics)

Mean |SHAP| per feature — answers "which features matter most?"

```python
plot_policy_bar(results['policy_shap'], STATE_NAMES,
                save_path='ch8_policy_bar.png')

plot_dynamics_bar(results['dynamics_shap'], SA_NAMES, STATE_NAMES,
                  save_path='ch8_dynamics_bar.png')
```

### Force Plot: Single Instance

The most operationally useful visualization. For a single time step, shows each feature's contribution as a waterfall from baseline to final Q-value:

```python
best = int(np.argmax(q_vals))   # highest-Q instance
plot_force_single(
    results['q_shap'][best], SA_NAMES,
    q_value=q_vals[best], q_base=results['q_base'],
    instance_label=f'Highest-Q instance (Q={q_vals[best]:.3f})',
    save_path='ch8_force_best.png')
```

An operator can use this to answer: "why did the agent rate this moment as high-value?" If the answer is "temperature near setpoint, level stable" — sensible. If "viscosity unusually low" when the setpoint is on temperature — a spurious correlation to investigate.

### Dependence Plot: Interactions

SHAP of feature A vs its raw value, colored by feature B — reveals interaction effects:

```python
plot_shap_dependence(
    results['q_shap'][:, :len(STATE_NAMES)],
    results['states'],
    feature_idx=0,       # temperature
    interaction_idx=1,   # colored by filler_frac
    feature_names=STATE_NAMES,
    title='Q-SHAP: temperature effect (colored by filler_frac)')
```

If temperature SHAP values split systematically by filler color, there is an interaction: the Q-function's response to temperature depends on filler level. For a physics-informed agent this is expected — viscosity is a nonlinear function of both.

---

## Consistency Check

```python
metrics = check_explanation_consistency(results)
print_consistency_report(metrics)
```

Expected output:
```
  Q ↔ policy rank correlation  : 0.71  ✓ aligned
  heat_input → next_temperature: SHAP=+0.0412  ✓ positive (correct)
  flow_input → next_filler     : SHAP=+0.0387  ✓ positive (correct)
```

**Q-policy Spearman $\rho$:** ranks state features by mean |SHAP| in Q-function and policy. $\rho > 0.6$ means the agent's critic and actor attend to the same features. $\rho < 0.3$ suggests policy collapse — the actor ignores most of the state that the critic cares about. Common when CQL's $\alpha$ is too large.

**Physics sign tests:** `heat_input` → `next_temperature` SHAP must be positive; `flow_input` → `next_filler_frac` SHAP must be positive. Failure indicates the dynamics model learned a physically inverted relationship in some data region — common when coverage near boundaries is very sparse.

---

## SHAP in Production: Practical Considerations

**Computational cost.** KernelExplainer makes $n\_\text{explain} \times n\_\text{samples} \times n\_\text{background}$ model evaluations. For 150 × 80 × 80 = 960,000 calls this takes minutes on CPU. Practical strategy: run offline before deployment on a held-out validation set; re-run monthly on recent operating data to detect distribution shift.

**Background dataset choice** is the most important hyperparameter. Full dataset background answers "difference from typical operation." Constraint-region background answers "what distinguishes near-violations from typical near-violations?" — useful for diagnosing why the agent approaches boundaries.

**Physical units.** States are normalized. To communicate with operators:

```python
# Convert SHAP back to physical units
shap_physical = results['q_shap'][:, :S] * s_std
# SHAP of 0.04 on normalized temperature = 0.04 × σ_T physical degrees
```

---

## What SHAP Cannot Tell You

**SHAP ≠ causal effect.** High SHAP for `viscosity` means the model uses viscosity as a predictor, not that viscosity causally affects Q-value. If viscosity is correlated with temperature (it is), SHAP may split credit between them in ways that don't match physics.

**SHAP values depend on the background.** A feature can have high SHAP against one background and low against another. Hold the background fixed when comparing across runs.

**Temporal structure is lost.** KernelExplainer explains each instance independently. For integrating states like level, the agent's reasoning depends on trajectory history. Single-step SHAP cannot capture this dependency.

**SHAP explains the model, not whether the model is correct.** If the Q-function overestimates in some region (a known offline RL failure mode), SHAP will faithfully explain the overestimation. Consistency checks can detect some failure modes, but SHAP is not a substitute for evaluation metrics from Chapter 7.

---

## Summary

Three complementary SHAP explanations for a trained Offline RL agent:

| Level | Model | Question | Input space |
|---|---|---|---|
| Q-function SHAP | Critic $Q(s,a)$ | Why is this action valued here? | $(s, a) \in \mathbb{R}^{S+A}$ |
| Policy SHAP | Actor $\pi(s)$ | Why was this action chosen? | $s \in \mathbb{R}^S$ |
| Dynamics SHAP | $\hat{f}(s,a)$ | Why is this next state predicted? | $(s, a) \in \mathbb{R}^{S+A}$ |

The consistency check — Q-policy Spearman correlation and physics sign tests — provides automated validation that the three explanations are mutually coherent and physically plausible.

Chapter 9 closes the book with a broader view: what the field has achieved, where it is heading, and the open problems that remain.

---

## Appendix 8.A: Choosing Between SHAP Variants

| Explainer | Best for | Speed | Notes |
|---|---|---|---|
| `KernelExplainer` | Any black-box | Slow | Used throughout — consistent across all model types |
| `DeepExplainer` | PyTorch / TF nets | Fast | Uses backprop; less accurate near tanh saturation |
| `GradientExplainer` | Differentiable nets | Medium | Gradient × input; fast but noisy |
| `TreeExplainer` | XGBoost, RF | Very fast | Exact SHAP; irrelevant for neural policies |
| `LinearExplainer` | Linear models | Instant | Exact; use if policy is distilled to a linear surrogate |

For fast per-step operator explanations in a deployed system: distill the policy to a shallow `DecisionTreeRegressor` and use `TreeExplainer`. The distillation loses some accuracy but makes each decision explainable in $< 1$ms.

---

## References

- Lundberg, S.M., & Lee, S.I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS. [arXiv:1705.07874](https://arxiv.org/abs/1705.07874).
- Lundberg, S.M., Erion, G., Chen, H. et al. (2020). *From Local Explanations to Global Understanding with Explainable AI for Trees.* Nature Machine Intelligence. [arXiv:1905.04610](https://arxiv.org/abs/1905.04610).
- Shapley, L.S. (1953). *A Value for n-Person Games.* In Contributions to the Theory of Games.
- Ribeiro, M.T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.* KDD. [arXiv:1602.04938](https://arxiv.org/abs/1602.04938).
- Molnar, C. (2022). *[Interpretable Machine Learning](https://interpretable-ml.github.io/).* 2nd ed.
- Doshi-Velez, F., & Kim, B. (2017). *Towards a Rigorous Science of Interpretable Machine Learning.* [arXiv:1702.08608](https://arxiv.org/abs/1702.08608).

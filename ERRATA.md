# Errata & Patches — *Offline RL: From Theory to Industrial Practice*

Reviewed chapters: **2, 3, 4, 5, 6, 7, 8** (the formula- and code-heavy core).
Not line-audited: 1, 9–12.

Severity legend: 🔴 substantive (wrong math / contradiction) · 🟠 code bug · 🟡 consistency / precision · ⚪ deviation-from-paper (defensible, footnote-worthy).

---

## 🔴 A. Ch. 2 — TD3+BC actor objective is wrong (contradicts Ch. 4 & Ch. 6)

**Where:** Ch. 2, "Policy-constraint and Actor-Critic methods" paragraph.

**Now:**
> **TD3+BC** … adds a behavioral cloning term to the actor loss: $\pi$ maximizes $Q(s, \pi(s)) + \lambda \cdot \log \pi_\beta(a|s)$ …

Two problems: (1) $\log\pi_\beta(a\mid s)$ is the likelihood under the *behavior* policy — it does not depend on $\pi$ and cannot be maximized over $\pi$; the BC term must be the squared deviation of the *learned* action from the dataset action. (2) In real TD3+BC, $\lambda$ scales the **Q-term**, not the BC term. Ch. 4 and Ch. 6 both give the correct form, so Ch. 2 is internally inconsistent with the rest of the book.

**Fix:**
> **TD3+BC** (Fujimoto & Gu, 2021) adds a behavioral-cloning term to the actor loss: $\pi$ maximizes $\lambda\,Q(s,\pi(s)) - \big(\pi(s)-a\big)^2$, so it stays near the data while improving on it.

---

## 🔴 B. Ch. 2 — "deadly triad" misattributed

**Where:** Ch. 2 references; the deadly-triad sentence cites *Sutton et al. (1999), Policy Gradient Methods*.

The deadly-triad concept/term is from **Sutton & Barto, _Reinforcement Learning: An Introduction_, 2nd ed. (2018), Ch. 11**. The 1999 policy-gradient paper does not discuss it.

**Fix (reference list):**
> - Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.), Ch. 11. MIT Press. *(deadly triad)*

Keep the 1999 PG paper only if it's cited for policy gradients elsewhere.

---

## 🔴 C. Ch. 6 — TD3+BC normalizer described as gradient norm (contradicts same chapter)

**Where:** Ch. 6, "Normalization of Q" paragraph.

**Now:**
> … the paper uses $\alpha / |\nabla_a Q(s,a)|$ with $\alpha = 2.5$ for the scaling …

The TD3+BC normalizer is the **mean absolute Q-value over the batch**, not the action-gradient norm:
$$\lambda = \frac{\alpha}{\frac{1}{N}\sum_{i}\big|Q(s_i,a_i)\big|}, \qquad \alpha = 2.5.$$
The chapter's own "Formalization" block and `td3bc_actor_loss` code already use $\sum|Q|/|\mathcal B|$, so this sentence contradicts them.

**Fix:**
> … the paper uses $\alpha\big/\big(\tfrac{1}{|\mathcal B|}\sum_i |Q(s_i,a_i)|\big)$ with $\alpha = 2.5$ for the scaling …

---

## 🔴 D. Ch. 5 (IQL) — target-Q networks maintained but unused; value loss deviates from Kostrikov et al.

**Where:** Ch. 5 implementation — `iql_value_loss`, `iql_q_loss`, and the soft-update block.

Current code creates and Polyak-updates `Q1_tgt, Q2_tgt`, but:
- `iql_value_loss` regresses $V$ against the **live** `Q1, Q2`;
- `iql_q_loss` bootstraps off **`V_tgt`**.

Canonical IQL puts the target on **Q** (value loss uses target-Q) and bootstraps Q off the **live V**. As written, `Q1_tgt/Q2_tgt` are dead state and the scheme silently diverges from the cited paper.

**Recommended fix — match the paper (drop the V target, use Q targets):**

```python
# --- value loss: regress V to the tau-expectile of the TARGET Q ---
def iql_value_loss(V, Q1_tgt, Q2_tgt, states, actions, tau=0.7):
    with torch.no_grad():
        q_target = torch.min(Q1_tgt(states, actions), Q2_tgt(states, actions))  # target-Q, not live
    v_pred = V(states)
    loss = expectile_loss(v_pred, q_target, tau)
    return loss, {...}

# --- Q loss: bootstrap off the LIVE V (no V target network) ---
def iql_q_loss(Q, V, states, actions, rewards, next_states, dones, gamma=0.99):
    with torch.no_grad():
        v_next = V(next_states)                       # live V, per Kostrikov et al.
        td_target = rewards + gamma * (1.0 - dones) * v_next
    q_pred = Q(states, actions)
    loss = F.mse_loss(q_pred, td_target)
    return loss, {...}
```

Then in the agent: **remove** `self.V_tgt` and its soft update; **keep** the `Q1_tgt/Q2_tgt` soft updates (now actually used by the value loss).

*(If you deliberately prefer a V-target scheme for stability, that's defensible — but then delete `Q1_tgt/Q2_tgt` entirely and add a one-line note that this departs from the paper. Don't ship both.)*

---

## 🟠 E. Ch. 8 (MOReL) — `generate_synthetic_data` is buggy

**Where:** Ch. 8, MOReL "Step 2 — P-MDP rollouts with hard HALT".

Bugs: the mask is defined as `ood` but later referenced as **`ood_mask`** (undefined); `next_state_out` is computed but never used; the trailing `transitions.append(...)` / `state = ...` are mis-indented relative to the loop body; and the prose ("The `ood_mask` is the key difference") names a variable absent from the code.

**Fix (one consistent name, absorbing state, append inside loop):**

```python
for step in range(self.rollout_horizon):
    action, _ = self.policy.sample(state[active])
    next_state, uncertainty = self.ensemble.predict_with_uncertainty(state[active], action)

    ood = uncertainty > self.epsilon                      # (n_active,) bool

    reward = torch.where(
        ood,
        -self.kappa * torch.ones(ood.shape[0], device=device),
        self._model_reward(state[active], action, next_state),
    )
    done = ood.float()

    # absorbing: halted rollouts stay put; others advance
    next_state = torch.where(ood.unsqueeze(-1), state[active], next_state)

    transitions.append((state[active], action, reward, next_state, done))

    # deactivate halted rollouts
    active_idx = active.nonzero(as_tuple=True)[0]
    active[active_idx[ood]] = False
    state[active_idx[~ood]] = next_state[~ood]

    if not active.any():
        break

return transitions
```

(Adjust indexing to your buffer layout, but the `ood`/`ood_mask` mismatch and the unused `next_state_out` must go.)

---

## 🟠 F. Ch. 4 (CQL) — `math` used but not imported

`compute_cql_loss` calls `math.log(n_samples)` and `math.log(2)`; the chapter's import block lists only `torch`, `nn`, `F`, `deepcopy`. Add `import math` (or switch to `torch.log`/a precomputed constant). Trivial, but the snippet doesn't run as printed.

---

## 🟡 G. Build bug — underscores inside `$…$` are eaten by Markdown

**Most visible at:** Ch. 4, the compact-form line renders as:
> `Then: $\mathcal{L}*{CQL}(\theta) = \mathcal{L}*(\theta)$}(\theta) + \alpha \cdot \mathcal{R}_{CQL`

Intended:
$$\mathcal{L}_{CQL}(\theta) = \mathcal{L}_{TD}(\theta) + \alpha\cdot\mathcal{R}_{CQL}(\theta).$$

Same corruption hits Ch. 2 ($\hat J(\hat\pi)=\mathbb E_{s,a\sim d^{\hat\pi}}[Q_\theta]$ shows as `[Q*\theta]`), Ch. 6 ($\mathbb E_{a\sim\pi_\beta}$), and a couple of OPE spots. Root cause: the MD→HTML pass treats `_` as emphasis before the math renderer sees it.

**Fix:** render math **before** Markdown emphasis — e.g. KaTeX/MathJax auto-render with `\(...\)`/`\[...\]` delimiters, or `markdown-it` + `markdown-it-texmath`, or escape `_` as `\_` inside `$…$`. A repo-wide check for `*{` and `*\` inside `$…$` will catch the rest.

---

## 🟡 H. Ch. 5 (IQL) — β default disagrees with itself

`iql_policy_loss(..., beta: float = 1.0)` vs the Hyperparameter Guide's "**beta=3.0**: Default." Pick one; 3.0 is the more defensible default (IQL paper: 3.0 for MuJoCo locomotion, 10.0 for AntMaze). Update the code default to `beta=3.0` or relabel the guide.

---

## 🟡 I. Ch. 4 (CQL) — auto-α threshold sign is muddled

The text insists the penalty $\big(\mathrm{logsumexp}_a Q - \mathbb E_{\mathcal D}[Q]\big)$ "should be positive," yet sets `target_penalty = -2.0` (and elsewhere "e.g. -1.0"). For the logsumexp form the gap is $\ge 0$ essentially always, so a negative threshold leaves the Lagrange constraint perpetually violated and $\alpha$ grows unbounded. Reconcile to the CQL-Lagrange convention (threshold $\tau$ on the gap is **positive**, e.g. `target_gap = 10.0` on the raw scale, or state clearly which sign convention the gap uses). Worth a careful pass on the `alpha_loss` sign too.

---

## 🟡 J. Ch. 3 (OPE) — DR correction: $Q^\ast \to Q^\pi$

**Now:** "If $\hat Q = Q^\ast$, the trailing terms have expectation zero."
For OPE of a fixed target policy $\pi$, the DR correction terms are mean-zero when $\hat Q = Q^\pi$ (the value of the *evaluated* policy), not the optimal $Q^\ast$.

**Fix:** "If $\hat Q = Q^\pi$, the trailing terms have expectation zero …"

---

## 🟡 K. Ch. 6 — wrong chapter numbers in the summary table

Final table lists "CQL (Ch3)" and "IQL (Ch4)". In this book CQL is **Ch. 4** and IQL is **Ch. 5**. (Ch. 7's analogous table has it right.)

---

## ⚪ L. Ch. 8 (MOPO) — uncertainty signal deviates from Yu et al.

The chapter uses **ensemble disagreement** $u=\|\mathrm{Std}_i[\hat s'_i]\|_2$. The original MOPO penalty is $u(s,a)=\max_i\|\Sigma_{\theta_i}(s,a)\|_F$ — the max over the ensemble of the **learned (aleatoric) std** norm. Disagreement is a common reimplementation choice (and is closer to MOReL's USAD), and it's fine to use — but since the MOPO theorem is quoted immediately after, add a one-line footnote that you're using the disagreement variant, not the paper's max-aleatoric penalty.

---

## ⚪ M. Ch. 7 (DT) — one-token-per-timestep vs. canonical 3-token DT

The sketch concatenates $(R_t,s_t,a_t)$ into a single token per timestep. Canonical DT (Chen et al.) embeds the three modalities as **separate tokens** (3 tokens/step, so context $K$ ⇒ $3K$ positions). The simplification is valid and lighter, but a reader porting from the paper will expect 3 tokens — worth a one-line note. (RTG discounted-vs-undiscounted you already flag; canonical DT is undiscounted.)

---

## ⚪ N. Ch. 2 — verify the performance-gap attribution

The bound $\hat J(\hat\pi)-J(\hat\pi)\le \frac{2\gamma}{(1-\gamma)^2}\,\mathbb E[\max_a|Q_\theta-Q^\ast|]$ is attributed to "Kumar et al., 2020, **Prop. 3.1**." That $\frac{2\gamma}{(1-\gamma)^2}$ form reads more like a simulation-lemma corollary; the CQL paper's early propositions are about the lower-bound property of the CQL Q-function. Double-check the exact proposition/source before keeping the precise citation. Also: defining $\hat J(\hat\pi)=\mathbb E_{s,a\sim d^{\hat\pi}}[Q_\theta]$ (expectation of $Q$ under the discounted occupancy) is dimensionally loose by an $\sim\!1/(1-\gamma)$ factor — fine as intuition, worth a hedge word.

---

## 🔴 O. Ch. 9 — monotone-constraint penalty catches only half the violations

**Where:** Ch. 9, Part I, "Monotone relationships".

**Now:**
$$g_{mono}(s,s') = \max\big(0,\ \text{sign}(\hat f(s')) - \text{sign}(f_{phys}(s'))\big)$$

Since $\text{sign}\in\{-1,0,1\}$, the difference is in $\{-2,\dots,2\}$ and the outer $\max(0,\cdot)$ **zeros out** the case $\hat f<0,\ f_{phys}>0$ (model predicts a decrease where physics requires an increase) — a real violation that incurs no penalty. The penalty is asymmetric; it must fire on *either* sign mismatch. `sign` is also non-differentiable, so this form is unusable if the term is ever backpropagated (e.g. as a PINN-style training loss rather than pure reward shaping).

**Fix (pick by use):**
- Reward shaping only (no gradient through it):
$$g_{mono} = \big|\text{sign}(\hat f) - \text{sign}(f_{phys})\big|$$
- Differentiable, magnitude-weighted (penalizes a predicted change whose sign opposes the known physics, in proportion to its size):
$$g_{mono} = \max\big(0,\ -\,\hat f \cdot \text{sign}(f_{phys})\big)$$

---

## 🟡 P. Ch. 9 — PINN reference arXiv ID likely wrong

Reference list cites Raissi, Perdikaris & Karniadakis (2019), *Physics-Informed Neural Networks*, as `arXiv:1811.10561`. The original PINN preprint (Part I) is **`arXiv:1711.10561`** (Nov 2017). Verify and correct.

---

## 🟡 Q. Ch. 9 — closing cross-reference points at the wrong chapter

Final line: "**Chapter 9** works through an industrial case study showing how these pieces fit together in practice." The case study is the *next* chapter — **Chapter 10 (Industrial Applications)**. Also note the `first_order_physics` docstring promises "per-variable tau and K (pass as vectors)" while the signature is scalar and `action[:, :state.shape[1]]` silently assumes `action_dim ≥ state_dim` with a 1:1 state↔control pairing; either accept vector $\tau,K$ or state the mapping assumption explicitly.

---

## Enhancements (Ch. 9) — not errata, but worth a paragraph each

These are correct as written; the notes below strengthen the modeling story rather than fix a bug.

**E1 — An unconstrained additive residual breaks the very conservation law $f_{phys}$ was chosen to enforce.** The chapter sells the hybrid as "respects known physics exactly," but $s' = f_{phys} + f_{NN}$ with an unconstrained $f_{NN}$ no longer satisfies the conservation law exactly — the residual pushes the prediction off the feasible manifold. If $f_{phys}$ obeys a linear conservation constraint $C\,s' = b$, the residual must live in $\ker C$ for the hybrid to preserve it (project $f_{NN}$ onto the null space, or parameterize it there). Otherwise "exact physics" is a slogan, not a property of the model. One paragraph closes this gap.

**E2 — "Residual outside" vs "residual inside the ODE" (gray-box).** Additive state-space correction is fine for weak, smooth corrections, but for strongly nonlinear couplings (Arrhenius in $T$, viscosity–temperature) it is more physical to correct the *parameters* of the physics, $f_{phys}(s,a;\ \theta_{phys}+\Delta_\theta(s,a))$, than to bolt an additive term on top. The residual then stays in an interpretable space and extrapolates better. The chapter presents only the additive form; one sentence on the gray-box alternative would round out Part II.

**E3 — $\lambda$ calibration ignores penalty competition.** The "typical violation ⇒ fraction of mean reward" heuristic is good for a single constraint. With several constraints at different $\lambda_i$, their gradients compete; normalize each violation to a comparable scale *before* weighting. Worth a line.

---

## 🔴 R. Ch. 10 — "Theorem 6.1" is a phantom reference (cited twice)

**Where:** Ch. 10, "Algorithm 3: CQL + Physics Reward Shaping".

**Now:** "The `lambdas` are calibrated via `calibrate_lambda` (**Theorem 6.1**): set $\lambda$ so that the **Theorem 6.1** optimality gap is at most 10% of the mean episode return."

There is no Theorem 6.1 in the book. Chapter 6 (TD3+BC/AWAC) explicitly states those methods have **no** theoretical guarantee, and the $\lambda$-calibration heuristic was introduced in **Chapter 9, Part I ("Choosing $\lambda$")** — as a heuristic, not a theorem.

**Also in Ch. 12** (Roadmap, Step 1): "calibrate the physics penalty weights via **Theorem 6.1**." Same phantom — three occurrences total (Ch. 10 ×2, Ch. 12 ×1). Fix all three.

**Fix:** drop "Theorem 6.1" and point to the Ch. 9 calibration heuristic, e.g.:
> The `lambdas` are calibrated via `calibrate_lambda` (Ch. 9, "Choosing $\lambda$"): set $\lambda$ so a typical violation costs ~10–30% of the mean episode return.

---

## 🟡 S. Ch. 10 — closing line mis-describes Ch. 11

**Now:** "Chapter 11 returns to the theoretical questions this case study raises: when can offline RL be trusted in deployment, what guarantees are available, and what open problems remain unsolved."

Ch. 11 is **Explainability**. The "trust / guarantees / open problems" content is **Ch. 12 (Conclusion and Future Directions)**. Repoint the sentence to Ch. 11's actual topic (explaining trained agents) or to Ch. 12.

---

## 🟡 T. Ch. 10 — summary undercounts the chapters used

Summary opens: "This chapter translated the tools from **Chapters 1–8**…" — but the case study leans centrally on **Ch. 9** (physics reward shaping in CQL+Physics, hybrid ensemble in HybridMOReL). Should read **Chapters 1–9**. (Minor, optional: the level `physics_fn` uses the *same* coefficients as the true level dynamics, so it's exact, not a "good approximation" — harmless, but the comment over-hedges.)

---

## ⚪ U. Ch. 10 — reward-centering justification is imprecise

**Now:** "The reward is scaled by its standard deviation but not centered — centering the reward would remove the sign information that distinguishes good states from bad."

A constant offset to the reward shifts every $Q$ by $c/(1-\gamma)$ and leaves the argmax policy and all advantages unchanged — so centering does **not**, by itself, "remove sign information." The defensible reasons not to center *this* reward are concrete: (a) it is structurally non-positive (a sum of negative squared errors), so 0 is a meaningful "perfect" anchor; (b) with terminal/`done` flags and variable episode lengths, a constant offset accumulates differently across episodes and *can* change behavior. Restate the justification on those grounds rather than "sign information."

---

## 🟡 V. Ch. 11 — pervasive `ch8` leftovers (chapter was renumbered from 8)

Throughout Ch. 11: every figure URL is `…/figures/ch8/…`, every save path is `ch8_*.png` (`ch8_q_summary.png`, `ch8_force_best.png`, `ch8_policy_bar.png`, …), and the appendix is titled "**Appendix 8.A**". This is Chapter **11**. Either the assets really live under `figures/ch8/` (links resolve but the naming misleads) or they're dangling. Do a repo-wide `ch8 → ch11` rename for this chapter's figures, save paths, and the appendix label.

---

## 🟡 W. Ch. 11 — "mean SHAP must be positive" is not the robust physics-sign test

**Where:** Ch. 11, "Consistency Check" — `heat_input → next_temperature: SHAP=+0.0412 ✓ positive`.

The mean **signed** SHAP of a feature is taken relative to the background baseline; for a correctly monotone-increasing dependence evaluated on instances roughly centered on the background, the signed mean can sit near zero (positive contributions above baseline cancel negative ones below). So a near-zero or slightly negative mean does **not** imply the model is physically inverted. The robust invariant is the **sign of the dependence slope** — i.e. SHAP value should *increase with* the feature's value — which is exactly what the `plot_shap_dependence` you already build shows. Recommend computing the sign test as a positive feature-vs-SHAP correlation (or slope), not as `mean(SHAP) > 0`.

*(Credit where due — the "What SHAP Cannot Tell You" section correctly separates attribution from causation, flags background-dependence, and notes that single-step SHAP loses the trajectory history that matters for the integrating `level` state. That's the part most explainability chapters get wrong; keep it.)*

---

## 🟡 X. Ch. 12 — QDT reference garbled into Q-Transformer

Body text: "**QDT** — Yamagata et al., 2023". Reference list entry: *"Yamagata, T., Khalil, A., & Santos-Rodriguez, R. (2023). **Q-Transformer**: Scalable Offline RL via Autoregressive Q-Functions. arXiv:2309.10150."* This conflates two distinct papers:
- **QDT** (Q-learning Decision Transformer) = Yamagata, Ahmed & Santos-Rodriguez, ICML 2023, **`arXiv:2209.03993`** — already cited correctly in Ch. 7.
- **Q-Transformer** = Chebotar et al. (Google), 2023, `arXiv:2309.10150` — a different paper, different authors.

The Ch. 12 entry pins QDT's authors onto Q-Transformer's title and arXiv ID. Fix: cite QDT as in Ch. 7 (`2209.03993`), or, if you mean Q-Transformer, correct the authors to Chebotar et al. and update the body.

---

## 🟡 Y. Ch. 12 — "Conservative Safety Critics" misattributed + body-only citations

Body (Safe Offline RL): "**Conservative Safety Critics** (Le Cleac'h et al., 2023)". CSC is **Bharadhwaj et al., 2021** (*Conservative Safety Critics for Exploration*, `arXiv:2010.14497`); Le Cleac'h works on differentiable physics / trajectory optimization, not CSC. Also, this CSC cite and **Chang et al., 2019** (Neural Lyapunov Control) appear only in the prose — neither is in the chapter's reference list. Fix the attribution and add both to the bibliography (CVPO and Berkenkamp are already there).

---

## ✅ Clean chapters

- **Ch. 1 (BC):** no errata. The compounding-error bound $J(\pi_\beta)-J(\pi_\theta)\le C\,\epsilon H^2$ with the $\mathcal O(H^2)$ trend, the DAgger $\mathcal O(H)$ improvement, and the forward-KL direction $D_{KL}(\pi_\beta\|\pi_\theta)$ are all stated correctly. (Only nit: $\epsilon$ is defined as an $\ell_2$ action error while the classic $H^2$ bound is for 0-1/TV disagreement — folded into $C$, fine.)
- **Ch. 12 (Conclusion):** no overclaiming of guarantees — the "What Offline RL Cannot Yet Guarantee" section is honest, and safety critics (CVPO, CSC) are correctly framed as *probabilistic* certificates, not hard guarantees. Issues are limited to the citation slips X and Y and the phantom Theorem 6.1 (item R).

---

### One-line commit summary

```
errata: fix TD3+BC actor obj (ch2) & normalizer (ch6); align IQL targets w/ Kostrikov (ch5);
fix MOReL rollout ood_mask bug (ch8); monotone-penalty asymmetry (ch9); DR Q^pi not Q^* (ch3);
math import (ch4); beta/alpha-threshold consistency (ch4/5); MathJax underscore escaping;
PINN arXiv id (ch9); phantom Theorem 6.1 (ch10,ch12); ch8->ch11 figure rename (ch11);
SHAP sign-test = dependence slope (ch11); QDT vs Q-Transformer cite + CSC attribution (ch12);
cross-ref + footnotes
```

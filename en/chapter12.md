---
layout: default
title: "Chapter 12: Conclusion and Future Directions"
lang: en
ru_url: /ru/chapter12/
prev_chapter:
  url: /en/chapter11/
  title: "Explainability in Offline RL"
next_chapter: null
permalink: "/offline-rl-book/en/chapter12/"
---

# Chapter 12: Conclusion and Future Directions

> *"The goal of offline RL is not to replace online learning — it is to make the knowledge locked in historical data actionable without putting anything at risk."*

---

## What This Book Built

In Chapter 1 we started with a question: can a sequential decision-making agent learn entirely from a fixed log of past behavior, without ever interacting with the environment? The answer is yes — with caveats that turn out to be at least as interesting as the algorithms themselves.

The caveats are not engineering details. They are structural features of the offline learning problem: the extrapolation gap between what the data covers and what the policy needs to evaluate, the distribution shift between the behavior policy and any better policy, and the fundamental impossibility of testing a policy in the very regions where it is most likely to fail. Every algorithm in this book is a different answer to the same underlying challenge.

**Behavioral cloning** (Chapter 1) answers it by ignoring it: clone the behavior policy and stay exactly where the data is. This works when the behavior policy is good and fails catastrophically when it is not — the compounding error problem. BC is not a bad algorithm; it is an honest one that makes no promises beyond the data it saw.

**Conservative Q-Learning and IQL** (Chapters 4–5) answer it by pessimism in value space: penalize Q-values for actions not supported by the data, or design the value backup to avoid ever querying OOD actions. Both achieve the same goal — keep the policy's effective support inside the data distribution — through different algebraic routes.

**MOPO and MOReL** (Chapter 8) answer it by expanding the data: learn a model of the world, generate synthetic experience, but penalize or hard-terminate transitions where the model is uncertain. Model-based methods trade compute for sample diversity — when the dynamics model is accurate, they outperform model-free methods significantly.

**Physics-informed methods** (Chapter 9) answer it by borrowing structure: replace the unknown parts of the model with known physics, constrain the reward signal to respect physical laws, and enforce hard constraints via Lagrangian duality. This approach is only possible when engineering knowledge exists — but when it does, it is the most data-efficient of all.

**The industrial case study** (Chapter 10) showed these tools composing in a setting where the stakes are real: a five-variable process with integrating dynamics, transport delay, non-uniform data coverage, and hard equipment constraints. No single algorithm dominated. The practical recommendation — CQL+Physics for minimal constraint violations, HybridMOReL when higher reward is needed — reflects not a winner but a complementary structure between model-free constraint enforcement and model-based diversity.

---

## What Offline RL Cannot Yet Guarantee

Honesty requires naming what this field has not solved.

**Distribution shift at deployment.** Every offline RL algorithm trains a policy $\pi$ assuming the deployed environment matches the training environment. When the process changes — equipment wears, feed composition shifts, ambient conditions differ — the offline-trained policy has no mechanism to detect or adapt. The theoretical guarantees from Chapters 4–9 hold only as long as the test distribution resembles the training distribution. In practice, this is a calendar question: how long until the gap becomes dangerous?

**Reward specification.** Every algorithm in this book assumed the reward function is given. In many industrial settings it is not. Operators have intuitions about what good control looks like, but translating those intuitions into a scalar reward function that correctly weights competing objectives — product quality, energy efficiency, throughput, safety — is a non-trivial engineering task. A misspecified reward produces a policy that optimizes the wrong thing, often in ways that are only apparent after deployment.

**Uncertainty quantification.** Ensemble-based uncertainty (Chapters 8–9) measures epistemic uncertainty about the dynamics model. It does not measure uncertainty about the reward, the constraint specification, or the value of the offline policy itself. A policy with high ensemble-estimated uncertainty in its rollouts may still be perfectly safe if the physical constraints are conservative enough. Conversely, a policy with low uncertainty may violate constraints that were never represented in the training data. Calibrated uncertainty about the whole pipeline remains an open problem.

**Interpretability.** The neural network policies trained in Chapters 1–10 are black boxes. An operator who needs to understand *why* the policy commanded a particular setpoint at a particular moment cannot interrogate the weights. This limits deployment in regulated industries where decisions must be explainable and auditable.

**Sim-to-real gap for physics models.** The hybrid dynamics models in Chapter 9 assume the physics model and the real process share the same structure, differing only in residuals. When the physics model is structurally wrong — wrong order, wrong coupling topology, wrong conservation laws — the residual network must learn a correction that the physics term actively fights. Detecting and handling structural model mismatch is not yet systematic.

---

## Emerging Directions

The field is moving fast. The directions below are not exhaustive, but they represent the most active threads as of the time of writing.

### Decision Transformers and Sequence Models

**Decision Transformer** (Chen et al., NeurIPS 2021) reframes offline RL as a sequence modeling problem. Rather than learning a Q-function or policy gradient, it trains a causal Transformer to predict the next action given the past trajectory and a desired return-to-go:

$$a_t = \pi_\theta\!\left(R_t,\, s_t,\, a_{t-1},\, \ldots,\, R_1,\, s_1,\, a_1\right)$$

At inference time you condition on a high target return — "I want cumulative reward $\geq R^*$" — and the model generates actions to achieve it.

This approach is conceptually elegant: the entire offline RL pipeline reduces to supervised learning on sequences. No Bellman backups, no Q-functions, no explicit pessimism. The model simply learns from the data what sequences of actions lead to high returns and imitates that pattern conditioned on a high target.

The practical implications are significant. Transformers scale with data and model size in ways that Q-function methods do not. A Decision Transformer trained on a large, diverse dataset generalizes across tasks and environments in a way that a task-specific CQL agent cannot. This is the connection to large language models: the same architecture that learns to predict text from internet data can learn to predict actions from trajectory data.

**Trajectory Transformer** (Janner et al., NeurIPS 2021) pushes this further: model the entire trajectory $(s_1, a_1, r_1, s_2, a_2, r_2, \ldots)$ as a sequence, then use beam search over this sequence to find high-reward trajectories at inference time. This gives planning-like behavior without explicit dynamics learning.

The limitation of sequence models is the same as BC's: they do not extrapolate beyond the return distribution in the training data. If the training dataset has maximum return $R_{max}$, conditioning on $R > R_{max}$ produces distribution-shifted inputs that the model was never trained on. Several papers (including **QDT** — Yamagata et al., 2023) address this by combining sequence modeling with Q-value guidance.

### Diffusion Models for Offline RL

**Diffusion probabilistic models** have emerged as one of the most powerful generative model families, and their application to sequential decision making is an active and productive direction.

The core observation: learning a policy $\pi(a|s)$ is a conditional generation problem. Diffusion models excel at conditional generation. Why not learn the policy as a diffusion model?

### Imitation Learning vs. Offline RL: A Necessary Distinction

Before examining diffusion models, it is worth clarifying a conceptual boundary that practitioners frequently blur: the difference between **imitation learning (IL)** and **offline RL**.

Both learn from a fixed dataset of past behavior without online interaction. The difference is in what the dataset contains and what the learner optimizes:

| | Imitation Learning | Offline RL |
|---|---|---|
| Dataset requires | Expert demonstrations only | Any logged transitions $(s, a, r, s')$ |
| Reward signal | Not needed — actions are the label | Required — drives optimization |
| Goal | Reproduce the behavior policy | Improve *beyond* the behavior policy |
| Failure mode | Compounding error (covariate shift) | Distribution shift + extrapolation error |
| Key algorithms | Behavioral Cloning, DAgger, GAIL | CQL, IQL, MOPO, MOReL |

Behavioral cloning (Chapter 1) is the simplest IL algorithm: given expert state-action pairs $(s, a^*)$, train a policy to predict $a^*$ from $s$ by supervised learning. No reward is needed. The cost is compounding error: at test time, the policy visits states slightly different from the training distribution, makes slightly worse predictions, and the errors accumulate over time.

Offline RL addresses this not by better behavior cloning, but by optimizing for reward — it can in principle surpass the behavior policy by stitching together suboptimal trajectories in ways the original operators never did. This stitching capability requires a reward signal that IL does not have.

**When to use which:** if you have expert demonstrations and care about replicating the behavior exactly (surgical robots, precise manipulation, teaching from a single demonstrator), IL is simpler and often sufficient. If you have heterogeneous historical logs — multiple operators with different strategies, suboptimal episodes, partial demonstrations — offline RL can extract a policy better than any individual demonstrator.

In practice, the two approaches converge: offline RL with a behavior cloning regularizer (as in IQL and CQL+BC) uses IL as a constraint to stay close to the data distribution while optimizing reward.

### The Multimodal Action Problem

The most important failure mode of behavioral cloning — and the key motivation for Diffusion Policy — is the **multimodal action problem**. In industrial processes, operators from different shifts often develop different but valid strategies for the same upset condition. In a coating line, for example, when temperature and filler drift off setpoint, one operator may respond by lowering heat (conservative, prioritizes temperature recovery); another may increase filler feed to correct the blend. Both strategies can be correct under their respective experience and constraints. A Gaussian policy $\pi_\theta(a|s)$ fits a unimodal distribution over actions. Faced with these two valid responses, it predicts their mean — e.g. a small change in both heat and flow that no operator would choose. The result can be catastrophic: the averaged action may violate mass balance, worsen the deviation, or drive the process toward a constraint. The policy fails not because the data are bad, but because the architecture cannot represent two valid answers to the same question.

$$\text{Gaussian collapses: } \hat{a} = \mathbb{E}[a | s] = \frac{a_{\text{shift A}} + a_{\text{shift B}}}{2} \notin \{a_{\text{shift A}}, a_{\text{shift B}}\}$$

A model that averages operator strategies may produce an action no operator ever used — and for good reason. Diffusion models address this by modeling the full conditional distribution $p(a|s)$ rather than its mean.

### Diffusion Policy: Architecture and Temporal Horizons

**Diffusion Policy** (Chi et al., RSS 2023) applies the denoising diffusion framework to action generation. At inference time, the policy starts from pure Gaussian noise in action space and iteratively denoises it conditioned on the current observation:

$$A^{k-1}_t = \alpha \left( A^k_t - \gamma\, \epsilon_\theta(O_t, A^k_t, k) + \mathcal{N}(0, \sigma^2 I) \right)$$

Training is supervised: take a clean action sequence $A^0_t$ from the dataset, corrupt it to $A^k_t$ by adding noise at level $k$, and train the network $\epsilon_\theta$ to predict the added noise:

$$\mathcal{L} = \mathbb{E}_{k, A^0_t, \epsilon}\left[\|\epsilon - \epsilon_\theta(O_t, A^k_t, k)\|^2\right]$$

Three temporal design choices distinguish Diffusion Policy from single-step policies:

**Observation horizon $T_o$:** how many past observations the policy receives. With $T_o = 2$, the robot sees the current and previous frame — giving velocity information without explicit state estimation.

**Prediction horizon $T_p$:** how many future action steps the diffusion model outputs in a single denoising pass. Typically $T_p = 16$. Predicting a chunk of the future rather than one step at a time creates temporally coherent action sequences and avoids high-frequency jitter.

**Action horizon $T_a$:** how many of the $T_p$ predicted actions are actually executed before replanning. Typically $T_a = 8$. The policy executes the first half of its plan, re-observes, and generates a new plan. This receding-horizon structure provides robustness: errors in the later part of the prediction are corrected by replanning rather than accumulating.

For the denoising backbone, the paper proposes two architectures: a 1D temporal convolutional network with FiLM conditioning (scales and shifts hidden activations by learned functions of the observation) as the default workhorse, and a Transformer decoder with cross-attention to observations for tasks with complex temporal structure.

**Diffuser** (Janner et al., ICML 2022) takes the trajectory-level view: model the entire $(s_1, a_1, r_1, s_2, \ldots)$ jointly as a diffusion process, then steer generation toward high-reward trajectories using classifier guidance:

$$\text{trajectory} \sim p_\theta(\tau) \cdot \exp(\beta \cdot J(\tau))$$

This is planning without an explicit dynamics model — the diffusion model encodes dynamics implicitly in its learned trajectory distribution.

**Decision Diffuser** (Ajay et al., ICLR 2023) separates return conditioning from trajectory generation, enabling test-time composition: multiple reward functions can be combined by multiplying their conditioning signals during denoising. For industrial applications where the reward trades off energy, quality, and throughput, this allows weight adjustment at deployment without retraining.

The computational cost of iterative denoising — typically 10–100 forward passes per action — is the main obstacle for fast industrial control loops. Recent work on consistency models and flow matching reduces this to one or a few steps, making diffusion-class policies practically deployable in real-time settings.

### Diffusion for Offline RL: Toward Scalable Policy Classes

The combination of diffusion and Transformer architectures is an active direction, driven by a concrete observation: the denoising network inside a diffusion model does not have to be a U-Net. **Diffusion Transformers (DiT)** (Peebles & Xie, ICCV 2023) replaced the U-Net backbone with a standard Transformer in image generation and demonstrated favorable scaling — larger models trained on more data improve predictably. The natural question for offline RL is whether the same substitution works for policy and planning networks.

Several lines of work are converging on this:

**Diffusion Q-Learning (DQL)** (Wang et al., ICLR 2023) uses a diffusion model as the policy class inside an actor-critic framework for offline RL. The diffusion policy generates high-quality actions from the data distribution while a Q-function provides guidance toward high-reward regions. On D4RL benchmarks, diffusion-class policies consistently outperform Gaussian policy baselines.

**DTQL** (Chen et al., NeurIPS 2024) addresses the inference cost problem: iterative denoising is too slow for fast control loops. DTQL trains a dual-policy system — a diffusion model for behavior cloning that defines a trust region, and a fast one-step policy for actual deployment. This separation decouples expressiveness from inference speed, a practically important design for industrial applications.

**Reasoning with Latent Diffusion** (Venkatraman et al., ICLR 2024) applies diffusion in the latent space of a world model rather than in action space — combining model-based planning with diffusion's multimodal expressiveness.

The architectural trend is clear even if a single dominant "DiT for offline RL" paper does not yet exist: Transformer backbones are replacing U-Nets inside diffusion policies, and diffusion policies are replacing Gaussian policies as the expressive policy class of choice for offline RL. Whether this combination scales the way language models do — where a single large model trained on diverse data transfers to new tasks with minimal fine-tuning — is the open empirical question that the next few years will answer.

### Offline-to-Online: Bridging the Gap

A purely offline policy is static by definition. Real deployments need adaptation: the process changes, new operating modes are introduced, constraints are revised. **Offline-to-online RL** initializes from an offline-trained policy and continues training with limited online interaction.

The challenge: naively fine-tuning an offline policy online causes catastrophic forgetting of the conservative behavior that made the offline policy safe. Methods like **IQL with online fine-tuning** (Kostrikov et al., 2021) and **Cal-QL** (Nakamoto et al., 2023) maintain the offline pessimism during early online training and gradually relax it as the online data accumulates. This is exactly the right structure for industrial deployment: use the offline policy as a safe starting point, then improve it through supervised interaction with explicit safety constraints.

### Safe Offline RL and Formal Verification

The Lagrangian approach of Chapter 9 provides constraint satisfaction in expectation — it minimizes expected violations but does not guarantee zero violations. For safety-critical applications, this is insufficient.

**Conservative Safety Critics** (Le Cleac'h et al., 2023) and **CVPO** (Liu et al., 2022) provide stronger guarantees by training a separate safety Q-function that bounds the probability of constraint violation, not just its expected magnitude. The policy is constrained to keep the safety Q-function value above a threshold, providing a probabilistic safety certificate.

Formal verification of neural network policies — proving that a policy satisfies a constraint for *all* states in a given set — remains computationally intractable for large networks. But for the low-dimensional state spaces typical of industrial process control (5–20 variables), recent work on Lyapunov-based verification (Berkenkamp et al., 2017; Chang et al., 2019) makes formal safety certificates increasingly feasible.

### Large Language Models as Reward Designers

The reward specification problem — arguably the hardest unsolved problem in applied offline RL — may be approachable through large language models. Rather than asking an engineer to write a reward function from scratch, one can ask an LLM to translate a natural language description of good control behavior into a mathematical reward function, verify it against example trajectories, and iteratively refine it based on policy performance.

**EUREKA** (Ma et al., 2023) demonstrates this loop: GPT-4 generates reward functions for robotic manipulation tasks, evaluates them on rollouts, and self-improves them over iterations, achieving superhuman performance on tasks where human-designed rewards were suboptimal. Adapting this approach to industrial process control — where the desired behavior is articulable but hard to formalize — is a natural and near-term direction.

---

## A Practical Roadmap for Industrial Deployment

Based on the material in this book, a pragmatic deployment sequence for a new industrial application:

**Step 1 — Start with CQL+Physics.** Collect the existing historical log, define the operating constraints from engineering knowledge, calibrate the physics penalty weights via Theorem 6.1, and train. This gives a safe baseline with minimal constraint violations and no dynamics model. Expected timeline: 2–4 weeks for a team with access to the process data.

**Step 2 — Diagnose with DA and violation rate.** Use the `IndustrialEvaluator` metrics from Chapter 10. If DA > 0.80 and violation rate < 2%, the baseline policy is industrially deployable. If not, identify which variables are failing and whether the issue is data coverage, physics model accuracy, or constraint calibration.

**Step 3 — Add hybrid dynamics if needed.** If reward performance is insufficient (the policy is safe but suboptimal), add `HybridEnsemble` for model-based diversity. Run `diagnose_physics_coverage` first — if overall coverage is below 70%, the physics model needs refinement before hybridization helps.

**Step 4 — Offline-to-online fine-tuning.** After an initial deployment period (1–4 weeks), collect online interaction data and fine-tune with the IQL offline-to-online procedure. Maintain the physics constraints throughout — they are not an artifact of offline training but a representation of physical reality.

**Step 5 — Monitor distribution shift.** Track the distribution of observed states over time. If the fraction of observations outside the training distribution exceeds 5–10%, retrain on the combined historical + deployment data. This is not a failure mode — it is the expected lifecycle of an industrial ML model. For concrete design of **drift detection**, **fallback policies**, and **safe RL** checks in deployment, see Chapter 10 (Safe RL, Drift Detection, and Fallback).

---

## Closing Remarks

Offline RL is not a solved problem. The algorithms in this book are powerful tools with well-understood failure modes, not black boxes to be applied without thought. The most important thing a practitioner can do is understand those failure modes well enough to recognize when they are occurring.

The questions worth sitting with:

*What did the behavior policy never do?* The regions of state-action space absent from the dataset are the regions where every offline RL algorithm is speculating. Knowing them is more valuable than any algorithmic improvement.

*What does the physics model get wrong?* The residual tells you. Run `diagnose_physics_coverage` before every training run. A physics model with 60% coverage on a key variable is providing more noise than signal for that variable.

*What would violating a constraint actually mean?* The answer determines whether CQL+Physics (minimize expected violations) or a formal safety approach (guarantee zero violations with high probability) is appropriate.

The field will continue to develop — diffusion transformers will scale, offline-to-online methods will mature, LLM-based reward design will reduce the specification burden. But the fundamental structure of the problem — learning from fixed data, generalizing cautiously, respecting physics — will remain. The tools in this book are not a snapshot of a passing trend. They are the vocabulary in which the next decade of work will be written.

---

## References

**Decision Transformers and Sequence Models**

- Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., Abbeel, P., Srinivas, A., & Mordatch, I. (2021). *Decision Transformer: Reinforcement Learning via Sequence Modeling.* NeurIPS. [arXiv:2106.01345](https://arxiv.org/abs/2106.01345).
- Janner, M., Li, Q., & Levine, S. (2021). *Offline Reinforcement Learning as One Big Sequence Modeling Problem.* NeurIPS. [arXiv:2106.02039](https://arxiv.org/abs/2106.02039). *(Trajectory Transformer)*
- Yamagata, T., Khalil, A., & Santos-Rodriguez, R. (2023). *Q-Transformer: Scalable Offline Reinforcement Learning via Autoregressive Q-Functions.* [arXiv:2309.10150](https://arxiv.org/abs/2309.10150).

**Diffusion Models for Offline RL**

- Janner, M., Du, Y., Tenenbaum, J., & Levine, S. (2022). *Planning with Diffusion Models.* ICML. [arXiv:2205.09991](https://arxiv.org/abs/2205.09991). *(Diffuser)*
- Ajay, A., Du, Y., Gupta, A., Tenenbaum, J., Jaakkola, T., & Agrawal, P. (2023). *Is Conditional Generative Modeling all you need for Decision-Making?* ICLR. [arXiv:2211.15657](https://arxiv.org/abs/2211.15657). *(Decision Diffuser)*
- Chi, C., Feng, S., Du, Y., Xu, Z., Cousineau, E., Burchfiel, B., & Song, S. (2023). *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* RSS. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137).

**Diffusion Models and Transformers for Offline RL**

- Peebles, W., & Xie, S. (2023). *Scalable Diffusion Models with Transformers.* ICCV. [arXiv:2212.09748](https://arxiv.org/abs/2212.09748). *(DiT architecture — the backbone trend)*
- Wang, Z., Hunt, J.J., & Zhou, M. (2023). *Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning.* ICLR. [arXiv:2208.06193](https://arxiv.org/abs/2208.06193). *(DQL — diffusion as policy class)*
- Chen, T., Wang, Z., & Zhou, M. (2024). *Diffusion Policies creating a Trust Region for Offline Reinforcement Learning.* NeurIPS. [arXiv:2405.19690](https://arxiv.org/abs/2405.19690). *(DTQL — fast inference)*
- Venkatraman, S., Khaitan, S., Akella, R.T., Dolan, J., Schneider, J., & Berseth, G. (2024). *Reasoning with Latent Diffusion in Offline Reinforcement Learning.* ICLR. *(diffusion in latent world model space)*

**Flow Matching and Consistency Models**

- Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M., & Le, M. (2022). *Flow Matching for Generative Modeling.* [arXiv:2210.02747](https://arxiv.org/abs/2210.02747).
- Song, Y., Dhariwal, P., Chen, M., & Sutskever, I. (2023). *Consistency Models.* ICML. [arXiv:2303.01469](https://arxiv.org/abs/2303.01469).

**Offline-to-Online RL**

- Kostrikov, I., Nair, A., & Levine, S. (2021). *Offline Reinforcement Learning with Implicit Q-Learning.* ICLR 2022. [arXiv:2110.06169](https://arxiv.org/abs/2110.06169). *(IQL + online fine-tuning)*
- Nakamoto, M., Zhai, Y., Singh, A., Mark, M.S., Ma, Y., Finn, C., Kumar, A., & Levine, S. (2023). *Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning.* NeurIPS. [arXiv:2303.05479](https://arxiv.org/abs/2303.05479).

**Safe Offline RL**

- Liu, Z., Cen, Z., Isenbaev, V., Liu, W., Wu, S., Li, B., & Zhao, D. (2022). *Constrained Variational Policy Optimization for Safe Reinforcement Learning.* ICML. [arXiv:2201.11927](https://arxiv.org/abs/2201.11927). *(CVPO)*
- Berkenkamp, F., Turchetta, M., Schoellig, A., & Krause, A. (2017). *Safe Model-based Reinforcement Learning with Stability Guarantees.* NeurIPS. [arXiv:1705.08551](https://arxiv.org/abs/1705.08551).

**LLM-Based Reward Design**

- Ma, Y.J., Liang, W., Wang, G., Huang, D.A., Bastani, O., Jayaraman, D., Zhu, Y., Fan, L., & Anandkumar, A. (2023). *Eureka: Human-Level Reward Design via Coding Large Language Models.* [arXiv:2310.12931](https://arxiv.org/abs/2310.12931).

**Surveys and Foundations**

- Levine, S., Kumar, A., Tucker, G., & Fu, J. (2020). *Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems.* [arXiv:2005.01643](https://arxiv.org/abs/2005.01643).
- Prudencio, R.F., Maximo, M.R.O.A., & Colombini, E.L. (2023). *A Survey on Offline Reinforcement Learning.* IEEE TNNLS. [arXiv:2203.01387](https://arxiv.org/abs/2203.01387).

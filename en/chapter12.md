---
layout: default
title: "Chapter 12: Offline RL for Tool-Using LLM Agents"
lang: en
ru_url: /ru/chapter12/
prev_chapter:
  url: /en/chapter11/
  title: "Explainability in Offline RL"
next_chapter:
  url: /en/chapter13/
  title: "Conclusion and Future Directions"
permalink: "/offline-rl-book/en/chapter12/"
---

# Chapter 12: Offline RL for Tool-Using LLM Agents

> *Agentic AI turns language-model usage into sequential decision-making; offline RL turns logged agent traces into a training substrate for safer policy improvement.*

---

## Why This Chapter Exists

Most current LLM agent training is still closer to **SFT**, **DPO**, **rejection sampling**, or **online RL** than to classical offline RL. That is not a criticism — those pipelines work well when pairwise preferences, live feedback, or cheap rollouts are available.

However, once agent execution traces are logged as **state–action–reward trajectories**, the problem becomes naturally compatible with the offline RL framing from Chapters 1–3:

```text
task → context/observation → thought/plan → tool call/action → tool result → next state → reward/outcome
```

The difference from MuJoCo or D4RL is not conceptual but practical:

| RL object | Classical control | Agentic LLM |
|-----------|-------------------|-------------|
| **State** | Low-dim vector | Context, history, memory, tool observations |
| **Action** | Continuous / discrete control | Text, tool call, API request, plan step |
| **Reward** | Environment return | Task success, verifier score, human rating, cost/safety penalty |
| **Episode** | Fixed horizon rollout | Multi-turn interaction |
| **Policy** | MLP / actor network | LLM or LLM router over tools |

States are high-dimensional and partially observed; the action space is enormous and structured; rewards are often **sparse** and **delayed**; credit assignment spans many reasoning and tool steps. These are the same failure modes as offline DRL — extrapolation, distribution shift, and overoptimistic value estimates — but with messier observations and weaker simulators.

This chapter maps the **ideas** from the rest of the book onto agentic AI. It does **not** claim that CQL, IQL, or TD3+BC drop into LangChain unchanged. It explains what transfers, what must be adapted, and what infrastructure (logging, verifiers, replay) must exist first.

---

## Agent Traces as Offline RL Data

An agent trace is a trajectory $\tau = (s_0, a_0, r_0, s_1, \ldots, s_T)$ where each step may include:

- the user goal and conversation history,
- retrieved documents or memory,
- an explicit plan, rationale summary, or structured scratchpad if exposed by the agent design; **hidden chain-of-thought should generally not be required or stored** as part of the RL dataset,
- a structured tool call or free-form generation,
- environment / tool feedback,
- scalar or programmatic reward signals.

**Behavioral cloning on successful traces** (SFT on ReAct-style demonstrations) is the default baseline — analogous to Chapter 1. Frameworks such as [AgentGym](https://agentgym.github.io/) ([Xi et al., 2024](https://arxiv.org/abs/2406.04151)) collect multi-turn trajectories across web, embodied, and tool-using tasks and use them to bootstrap generally capable agents. AgentTraj-style datasets are BC fuel, not yet offline RL — but they are the **substrate** offline RL needs.

The applied prerequisite is a **replayable trace schema**: without consistent logging of state snapshots, actions, observations, rewards, and behavior-policy metadata, later OPE, reward relabeling, and conservative policy improvement are impossible.

---

## MDP / POMDP Formulation

Treat each LLM call (or each tool step) as one decision in an MDP:

- **State** $s_t$: everything the policy conditions on at step $t$ — system prompt, user goal, conversation, tool outputs, memory, retrieved context. This is usually a **POMDP** view: the true environment state is hidden; the agent sees a context window.
- **Action** $a_t$: the model output — a tool call JSON, SQL query, browser action, or text plan.
- **Transition** $s_{t+1}$: the context update is often deterministic given $(s_t, a_t, o_t)$, but the observation $o_t$ may come from **stochastic, non-stationary, or versioned** tools and environments.
- **Reward** $r_t$: often **0** until task completion; then success/failure, verifier score, human rating, or shaped process reward (format valid, tool succeeded, cost penalty).

[Agent Lightning](https://microsoft.github.io/agent-lightning/latest/) ([Luo et al., 2025](https://arxiv.org/abs/2508.03680)) makes this explicit: agent execution is an MDP where state is an execution snapshot and action is an LLM output; trajectories decompose into transitions `(state, action, reward)` for training. Agent Lightning is **not** an offline RL method in the narrow CQL/IQL sense; it is better viewed as **infrastructure** that turns agent execution into RL-compatible transitions, which can support online, off-policy, or potentially offline training regimes.

The bridge for practitioners:

```text
Agent observability / tracing → trajectories → offline or off-policy RL dataset
```

LangChain-, AutoGen-, or OpenAI Agents-style logs are not merely debugging artifacts — they are potential **RL training data** once rewards and state boundaries are defined.

---

## BC / SFT as the Baseline

**Supervised fine-tuning (SFT)** on expert or filtered successful trajectories is behavioral cloning in disguise. It is strong when:

- the behavior policy is already good,
- tasks are single-turn or short-horizon,
- failure modes in logs are rare.

It fails in the same way as BC (Chapter 1): **compounding error** on long horizons, no use of reward beyond filtering, and no way to combine good segments from mixed-quality episodes.

For multi-step tool use, SFT also **cannot assign credit** across steps: a trajectory may fail at step 8 while steps 1–7 were sound. Offline RL and advantage-weighted methods exist precisely to exploit partial progress — when rewards and coverage support it.

---

## Offline RL Beyond SFT

### DPO vs trajectory-level offline RL

**DPO** optimizes a **sequence-level** log-likelihood ratio (a sum of per-token log-probs relative to a reference model), supervised by a pairwise preference label. As a result, it provides **weak step-level credit assignment** for multi-turn reasoning and tool use.

[OREO](https://github.com/jwhj/OREO) — *Offline Reinforcement Learning for LLM Multi-Step Reasoning* ([Wang et al., 2024/2025](https://arxiv.org/abs/2412.16145), [ACL Findings 2025](https://aclanthology.org/2025.findings-acl.464/)) — is a useful anchor: it jointly trains an LLM policy and **value function** via a soft Bellman-style objective, motivated by sparse rewards and credit assignment across reasoning steps. Empirically it improves over DPO-style baselines on math reasoning (GSM8K, MATH) and embodied control (ALFWorld).

A direct precursor is **ILQL** (Implicit Language Q-Learning; [Snell et al., 2023](https://arxiv.org/abs/2206.11871)) — IQL (Chapter 5) adapted to language generation: an expectile value function plus an implicit dataset-support constraint, with the policy realized by advantage-reweighting the base model's token logits. It is the most direct bridge from this book's value-pessimism chapters to LLM policies.

The conceptual contrast:

| Approach | Data | Credit assignment |
|----------|------|-------------------|
| **DPO / preference learning** | Pairwise chosen vs rejected | Sequence-level preference signal |
| **Offline RL (OREO-style)** | Full trajectories + rewards | Value function + Bellman backup across steps |

### Mapping classical offline RL ideas

Do **not** read this table as “drop in CQL unchanged.” Read it as **design vocabulary**:

| Classical offline RL (this book) | Agentic AI analogue |
|-----------------------------------|---------------------|
| **BC** | SFT on successful traces |
| **Policy constraint (TD3+BC, Ch. 6)** | The KL-to-reference penalty in RLHF / DPO / GRPO is the same "stay near the behavior policy" regularizer |
| **AWR / AWAC / IQL-style extraction** | Weight steps or trajectories by advantage, verifier score, or process reward (**ILQL** applies IQL directly to token-level language generation) |
| **CQL / pessimism** | Penalize unsupported tool calls, plans, or API invocations |
| **FQE / learned Q** | Critic over agent states and actions (text or structured) |
| **Decision Transformer** | Condition generation on desired return / success label |
| **Model-based offline RL** | Simulator, verifier, or world model for tool outcomes |
| **OPE (Ch. 3)** | Replay in sandbox, verifier rollouts, learned value model |

Agent Lightning’s **LightningRL** adds hierarchical credit assignment: allocate trajectory-level return to individual LLM calls, then apply standard single-turn RL updates — a practical pattern when episodes are long but each call is trainable. Treat this as an **off-policy RL bridge**, not a drop-in replacement for CQL or IQL on fixed logs.

---

## Credit Assignment in Reasoning and Tool Use

Multi-turn agents combine **language generation** with **externalized actions**. Rewards often arrive only at episode end (task solved or not). Offline RL methods that maintain a **value function** — OREO, goal-conditioned critics, or FQE-style models — address the same problem as TD learning in control: which intermediate step contributed?

**Online agentic RL** highlights how hard this is. [RAGEN](https://arxiv.org/abs/2504.20073) ([Wang et al., 2025](https://arxiv.org/abs/2504.20073)) studies multi-turn LLM agent RL (StarPO) and reports instability: reward variance cliffs, **Echo Trap** dynamics, and shallow or hallucinated reasoning without fine-grained rewards. [RAGEN-2](https://arxiv.org/abs/2604.06268) ([Wang et al., 2026](https://arxiv.org/abs/2604.06268)) analyzes **template collapse** — fluent but input-agnostic reasoning — and proposes SNR-aware filtering. These are **online** results, but the caveats transfer directly to offline training: if online agentic RL is unstable, offline agentic RL depends even more on **data coverage**, **reward quality**, and **evaluator fidelity**.

For offline settings, prefer:

- **process rewards** (tool succeeded, format valid, constraint satisfied) where terminal reward is sparse,
- **step-level value targets** when trajectories include both successes and failures,
- **verifier-backed labels** instead of pure LLM-as-judge when possible.

---

## OPE for Agents

Chapter 3’s OPE toolbox applies, but agent environments rarely offer a faithful closed-form simulator.

Practical evaluation layers:

1. **Replay in sandbox** — re-execute tool calls against cached fixtures or recorded API responses (when safe and deterministic).
2. **Verifiers** — unit tests, SQL result checks, retrieval precision, code execution outcomes.
3. **Learned critics** — FQE-style $Q(s,a)$ or goal-conditioned value models trained on logs. *Planning without Search* ([Hong et al., 2025](https://arxiv.org/abs/2505.18098)) — uses offline goal-conditioned values as a **natural-language critic** at inference without fine-tuning the base LLM.
4. **Human or A/B review** — expensive but necessary for high-stakes deploy.

Offline RL for agents fails quietly when OPE measures **format quality** instead of **task success**. Align metrics with deployment risk, as in Chapter 3’s coverage warnings.

---

## LLMs as Annotators, Generators, and Critics

LLMs appear in agentic offline pipelines in several roles — not only as the policy. KALM and TEDUO are **LLM-assisted offline RL** lines (language-conditioned or rollout-augmented control), not always narrow “tool-using chat agents” — but the data-layer pattern transfers:

| Role | Example | Offline RL connection |
|------|---------|------------------------|
| **Trajectory generator** | [KALM](https://arxiv.org/abs/2404.09248) ([Pang et al., NeurIPS 2024](https://arxiv.org/abs/2404.09248)) — LLM imaginary rollouts merged with real data, then standard offline RL | Expands coverage; watch semantic gap between text and low-level actions |
| **Dataset labeler** | [TEDUO](https://arxiv.org/abs/2412.06877) — hindsight labeling and state abstraction on unlabeled logs, then offline RL + SFT | LLM as annotator before RL |
| **Inference-time critic** | *Planning without Search* ([arXiv:2505.18098](https://arxiv.org/abs/2505.18098)) — goal-conditioned value function guides plan refinement | Offline RL trains critic; base LLM may stay frozen |
| **Reward model** | LLM-as-judge scores | Cheap but gameable; combine with verifiers |

KALM grounds the LLM on environment dynamics before generating rollouts — addressing the **semantic gap** between tokens and numerical state. TEDUO shows LLMs can enrich **unlabeled** interaction logs so offline RL has goals and abstract states to condition on.

---

## Failure Modes

Agentic offline RL inherits classical offline RL failures and adds agent-specific ones:

- **Reward hacking** — optimizing verifier shortcuts instead of task intent.
- **Spurious tool use** — calling tools to look productive without information gain.
- **Template collapse** — diverse-looking text that ignores input (see RAGEN-2).
- **Unsupported tool calls** — analog of OOD actions: APIs or arguments never seen in logs.
- **Overconfident critics** — FQE-style models extrapolate on long contexts like Q-functions extrapolate on OOD actions.
- **Non-stationary tools and APIs** — training logs from GPT-4.1-era tools may not match deploy-time behavior.
- **Cost and safety** — reward must encode latency, spend, and policy violations, not only success rate.

Conservative methods (CQL-style pessimism, support constraints, rejection of low-coverage tool branches) are conceptually appropriate but require defining **action support** in structured action spaces — an open engineering problem.

---

## Toy Example: Offline RL for a Calculator Agent

This example **intentionally removes language modeling complexity**. The policy does not generate free text; it chooses among structured tool calls. That isolates the offline RL question: can we improve over logged behavior using value learning and support constraints?

> 📄 Full code: [`agentic_offline_rl_toy.py`](https://github.com/corba777/offline-rl-book/blob/main/code/agentic_offline_rl_toy.py)

### Environment

The agent solves `x + y`, `x - y`, or `x * y` but never sees `x` and `y` directly. It must call tools:

| Action | Effect |
|--------|--------|
| `lookup_x` | observe hidden $x$ |
| `lookup_y` | observe hidden $y$ |
| `add` / `sub` / `mul` | compute on known values |
| `final` | submit answer (verifier checks against true task) |

Rewards: **+1** correct final answer, **−1** wrong final or invalid tool, **−0.02** per step (latency/cost proxy).

This mirrors an LLM agent trace:

| Real agent | Toy analogue |
|------------|--------------|
| Context window | Discrete state tuple |
| Tool-call JSON | Discrete action |
| Tool output | Observation update |
| Final answer | `final` action |
| Verifier | Exact arithmetic checker |
| Agent trace | Offline RL transition |

The tabular state key is `(task, x_known, y_known, result)` — including the **numeric result** when set, so a wrong arithmetic outcome does not alias with a correct one.

### Four policies on two datasets

| Policy | Role |
|--------|------|
| **Behavior** | Stochastic logger (sometimes wrong op, extra steps) |
| **BC** | Majority action per state in logs (SFT-on-traces) |
| **Naive FQI** | Tabular fitted Q with greedy improvement over **valid** actions |
| **Support-constrained FQI** | Same, but argmax restricted to actions seen in the dataset at each state (toy CQL / support mask) |

Two data regimes:

1. **Good coverage** — `mul` appears in logs when the task is multiplication.
2. **Coverage gap** — on `mul` tasks, behavior logs only `add` / `sub`, never `mul`.

Run: `python code/agentic_offline_rl_toy.py`

Example output (your numbers may vary slightly with seed):

```text
Dataset: good coverage (mul in logs)
  Behavior success (all tasks):    0.72
  BC success:                    1.00
  Naive FQI success:             1.00
  Support-constrained FQI:       1.00

Dataset: no multiplication support
  Behavior success (all tasks):    0.52
  BC success:                    0.67
  Naive FQI success:             1.00
  Support-constrained FQI:       0.72
  --- mul tasks only ---
  BC / naive FQI / constrained:  0.01 / 1.00 / 0.02
```

### What this demonstrates

1. **BC** clones the behavior policy. It improves over raw logging when the majority action at a state is good, but cannot invent unsupported tools.
2. **Naive FQI** can stitch successful sub-trajectories (`lookup_x → lookup_y → correct op → final`) when coverage exists.
3. **Without `mul` in the logs**, support-constrained FQI **cannot** solve multiplication — it never selects an unsupported tool. That is the conservative behavior you want offline.
4. **Naive FQI on mul tasks** may still pick `mul` even though it was absent from behavior logs at that state: Q-values for unseen actions default to 0, which can beat negative Q on logged wrong ops — an **extrapolation / OOD action** effect. In the toy simulator `mul` succeeds; in deployment that would be a policy with no evidential support.

**Thesis (same as classical offline RL):** improvement is possible **inside** the action/state support of the dataset; conservative methods refuse unsupported gains; unconstrained value learning can overtrust unvisited actions.

Start with this tabular agent before LLM confounders (stochastic decoding, prompt drift, missing log-probs, expensive OPE). A natural next step is replacing the tabular policy with a small classifier over trace states, then an LLM that proposes tool calls filtered by a support or critic model.

---

## Practical Logging Schema

For applied work, a consistent JSON (or parquet) schema matters as much as algorithm choice:

```json
{
  "episode_id": "task_001",
  "t": 3,
  "task": "Find quarterly revenue in uploaded 10-K",
  "state": {
    "user_goal": "...",
    "conversation_history": "...",
    "available_tools": ["search", "open_pdf", "calculator"],
    "memory": "...",
    "retrieved_context": "..."
  },
  "action": {
    "type": "tool_call",
    "tool": "open_pdf",
    "arguments": {"page": 42}
  },
  "observation": {
    "tool_result": "...",
    "error": null
  },
  "reward": 0.0,
  "done": false,
  "prompt_version": "agent_prompt_v12",
  "tool_schema_version": "tools_2026_06_20",
  "environment_version": "sandbox_v3",
  "action_logprobs": null,
  "reward_components": {
    "task_success": 0.0,
    "format_valid": 1.0,
    "tool_error_penalty": 0.0,
    "cost_penalty": -0.0031
  },
  "reward_source": "verifier_v2",
  "verifier_version": "sec_filing_checker_2026_06",
  "split": "train",
  "metadata": {
    "behavior_policy": "gpt-4.1-agent-v3",
    "temperature": 0.2,
    "latency_ms": 1400,
    "cost_usd": 0.0031,
    "safety_flags": []
  }
}
```

Design choices:

- **`state`** should be reconstructible at train and eval time — version prompts (`prompt_version`), tool schemas (`tool_schema_version`), and sandboxes (`environment_version`).
- **`action`** should be structured (JSON schema) when possible — easier for support estimation than free text.
- **`reward`** can be relabeled offline (verifier, human, LLM critic) — store `reward_components`, `reward_source`, and `verifier_version` separately from the scalar `reward`.
- **`action_logprobs`** (or token/tool-call log-probabilities) are valuable for off-policy evaluation and filtering; log them when the behavior policy exposes them.
- **`metadata.behavior_policy`** is necessary for **dataset provenance**. Importance-weighted OPE additionally requires comparable action representations and behavior/target action probabilities — often **unavailable** for free-form LLM actions unless log-probs were recorded at collection time.

---

## Limitations and Honest Scope

This is a **rapidly moving** research area, not a settled production stack. Most deployed agents still train via SFT + online RLHF/GRPO loops. Offline RL for agents is strongest when:

- traces are abundant and replayable,
- rewards (or verifiers) are reliable,
- deployment cannot tolerate exploratory online learning,
- you need to improve over a fixed behavior policy without full re-collection.

When those conditions fail, invest in **logging**, **BC baselines**, and **OPE** before value-based offline RL.

See the [Appendix: Algorithm Selection Guide](/offline-rl-book/en/appendix.html) for classical method anchors. [Chapter 13](/offline-rl-book/en/chapter13.html) concludes the book.

---

## References

- Wang, H., Hao, S., Dong, H., et al. (2024). *Offline Reinforcement Learning for LLM Multi-Step Reasoning (OREO).* [arXiv:2412.16145](https://arxiv.org/abs/2412.16145), [ACL Findings 2025](https://aclanthology.org/2025.findings-acl.464/), [code](https://github.com/jwhj/OREO).
- Snell, C., Kostrikov, I., Su, Y., Yang, M., & Levine, S. (2023). *Offline RL for Natural Language Generation with Implicit Language Q-Learning (ILQL).* ICLR. [arXiv:2206.11871](https://arxiv.org/abs/2206.11871).
- Xi et al. (2024). *AgentGym: Evolving LLM-based Agents across Diverse Environments.* [arXiv:2406.04151](https://arxiv.org/abs/2406.04151), [project](https://agentgym.github.io/).
- Luo et al. (2025). *Agent Lightning: Train ANY AI Agents with Reinforcement Learning.* [arXiv:2508.03680](https://arxiv.org/abs/2508.03680), [docs](https://microsoft.github.io/agent-lightning/latest/).
- Pang et al. (2024). *KALM: Knowledgeable Agents by Offline RL from LLM Rollouts.* [arXiv:2404.09248](https://arxiv.org/abs/2404.09248), [project](https://kalmneurips2024.github.io/).
- Pouplin, T., Kobalczyk, K., Sun, H., & van der Schaar, M. (2024). *The Synergy of LLMs & RL Unlocks Offline Learning of Generalizable Language-Conditioned Policies (TEDUO).* ICML 2025. [arXiv:2412.06877](https://arxiv.org/abs/2412.06877).
- Hong, J., Dragan, A., & Levine, S. (2025). *Planning without Search: Refining Frontier LLMs with Offline Goal-Conditioned RL.* NeurIPS 2025. [arXiv:2505.18098](https://arxiv.org/abs/2505.18098).
- Wang et al. (2025). *RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn RL.* [arXiv:2504.20073](https://arxiv.org/abs/2504.20073).
- Wang et al. (2026). *RAGEN-2: Reasoning Collapse in Agentic RL.* [arXiv:2604.06268](https://arxiv.org/abs/2604.06268), [project](https://ragen-ai.github.io/v2/).

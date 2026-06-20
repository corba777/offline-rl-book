---
layout: default
title: "Глава 12: Offline RL для tool-using LLM-агентов"
lang: ru
en_url: /en/chapter12/
prev_chapter:
  url: /ru/chapter11/
  title: "Объяснимость в Offline RL"
next_chapter:
  url: /ru/chapter13/
  title: "Заключение и перспективы"
permalink: "/offline-rl-book/ru/chapter12/"
---

# Глава 12: Offline RL для tool-using LLM-агентов

> *Agentic AI превращает использование языковых моделей в последовательное принятие решений; offline RL превращает залогированные agent traces в субстрат для более безопасного улучшения политики.*

---

## Зачем эта глава

Большинство текущих пайплайнов обучения LLM-агентов ближе к **SFT**, **DPO**, **rejection sampling** или **online RL**, чем к классическому offline RL. Это не упрек — эти методы работают, когда есть pairwise-предпочтения, живая обратная связь или дешёвые rollouts.

Но как только execution traces агента логируются как **траектории state–action–reward**, задача естественно ложится на framing из глав 1–3:

```text
task → context/observation → thought/plan → tool call/action → tool result → next state → reward/outcome
```

Отличие от MuJoCo или D4RL не концептуальное, а прикладное:

| RL-объект | Классическое управление | Agentic LLM |
|-----------|-------------------------|-------------|
| **State** | Низкоразмерный вектор | Контекст, история, память, наблюдения инструментов |
| **Action** | Непрерывное / дискретное управление | Текст, tool call, API-запрос, шаг плана |
| **Reward** | Return среды | Успех задачи, verifier, human rating, штраф за cost/safety |
| **Episode** | Rollout фиксированной длины | Multi-turn взаимодействие |
| **Policy** | MLP / actor | LLM или LLM-router по инструментам |

Состояния высокоразмерны и частично наблюдаемы; пространство действий огромно и структурировано; награды часто **разрежены** и **задержаны**; credit assignment тянется через множество шагов рассуждения и вызовов tools. Это те же failure modes, что в offline DRL — экстраполяция, distribution shift, переоптимистичные value estimates — но с более «грязными» наблюдениями и слабыми симуляторами.

Эта глава переносит **идеи** книги на agentic AI. Она **не** утверждает, что CQL, IQL или TD3+BC вставляются в LangChain без адаптации. Она объясняет, что переносится, что нужно менять и какая инфраструктура (логирование, verifiers, replay) должна быть на месте.

---

## Agent traces как offline RL data

Agent trace — траектория $\tau = (s_0, a_0, r_0, s_1, \ldots, s_T)$, где на каждом шаге могут быть:

- цель пользователя и история диалога,
- retrieved documents или memory,
- explicit plan, rationale summary или structured scratchpad, если это часть дизайна агента; **hidden chain-of-thought обычно не следует требовать или хранить** в RL dataset,
- structured tool call или свободная генерация,
- feedback среды / инструмента,
- скалярная или программная награда.

**BC по успешным траекториям** (SFT на ReAct-style демонстрациях) — baseline по умолчанию, аналог главы 1. Фреймворки вроде [AgentGym](https://agentgym.github.io/) ([Xi et al., 2024](https://arxiv.org/abs/2406.04151)) собирают multi-turn trajectories для web, embodied и tool-using задач. AgentTraj-style datasets — топливо для BC, ещё не offline RL — но **субстрат**, без которого offline RL невозможен.

Прикладной prerequisite — **replayable trace schema**: без согласованного логирования state snapshots, actions, observations, rewards и metadata behavior policy позже невозможны OPE, reward relabeling и conservative policy improvement.

---

## Формулировка MDP / POMDP

Каждый LLM-вызов (или tool step) — одно решение в MDP:

- **State** $s_t$: всё, на чём политика условится — system prompt, цель, диалог, tool outputs, memory, retrieved context. Обычно **POMDP**: истинное состояние скрыто; агент видит context window.
- **Action** $a_t$: выход модели — JSON tool call, SQL, browser action или text plan.
- **Transition** $s_{t+1}$: обновление контекста часто детерминировано при $(s_t, a_t, o_t)$, но observation $o_t$ может приходить от **stochastic, non-stationary или versioned** tools и environments.
- **Reward** $r_t$: часто **0** до конца эпизода; затем success/failure, verifier score, human rating или shaped process reward.

[Agent Lightning](https://microsoft.github.io/agent-lightning/latest/) ([Luo et al., 2025](https://arxiv.org/abs/2508.03680)) формулирует это явно: execution — MDP, trajectories раскладываются на `(state, action, reward)`. Agent Lightning **не** offline RL method в узком смысле CQL/IQL; это **infrastructure**, превращающая execution в RL-compatible transitions для online, off-policy или потенциально offline training.

Мост для практиков:

```text
Agent observability / tracing → trajectories → offline or off-policy RL dataset
```

Логи LangChain-, AutoGen- или OpenAI Agents-style — потенциальные **RL training data**, если заданы rewards и границы state.

---

## BC / SFT как baseline

**SFT** на expert или отфильтрованных успешных траекториях — behavioral cloning. Силён, когда behavior policy хорош, horizon короткий, failure modes редки.

Проваливается как BC (глава 1): **compounding error**, награда только для фильтрации, нет комбинирования хороших сегментов смешанных эпизодов.

Для multi-step tool use SFT **не распределяет credit** по шагам. Offline RL и advantage-weighted methods нужны, когда rewards и coverage это поддерживают.

---

## Offline RL beyond SFT

### DPO vs trajectory-level offline RL

**DPO** оптимизирует token-level likelihood ratios, но supervision signal обычно **sequence-level pairwise preference**. Поэтому **step-level credit assignment** слаб для multi-turn reasoning и tool use.

[OREO](https://github.com/jwhj/OREO) ([Wang et al., 2024/2025](https://arxiv.org/abs/2412.16145), [ACL Findings 2025](https://aclanthology.org/2025.findings-acl.464/)) — якорь: policy + **value function** через soft Bellman-style objective, мотивирован sparse rewards и credit assignment по шагам reasoning. Эмпирически лучше DPO-style baselines на GSM8K, MATH и ALFWorld.

| Подход | Данные | Credit assignment |
|--------|--------|-------------------|
| **DPO / preference learning** | Pairwise chosen vs rejected | Sequence-level preference signal |
| **Offline RL (OREO-style)** | Полные trajectories + rewards | Value function + Bellman backup по шагам |

### Соответствие классическим идеям offline RL

**Не** читайте как «вставь CQL как есть». Это **design vocabulary**:

| Классический offline RL | Agentic AI analogue |
|-------------------------|---------------------|
| **BC** | SFT на успешных traces |
| **AWR / AWAC / IQL** | Веса по advantage, verifier score, process reward |
| **CQL / pessimism** | Штраф unsupported tool calls / plans |
| **FQE / learned Q** | Critic по agent states и actions |
| **Decision Transformer** | Conditioning на desired return / success |
| **Model-based offline RL** | Simulator, verifier, world model |
| **OPE (гл. 3)** | Sandbox replay, verifier rollouts, learned value |

**LightningRL** — hierarchical credit assignment + single-turn RL updates. Это **off-policy RL bridge**, не замена CQL/IQL на fixed logs.

---

## Credit assignment в reasoning и tool use

[RAGEN](https://arxiv.org/abs/2504.20073) ([Wang et al., 2025](https://arxiv.org/abs/2504.20073)) — multi-turn agent RL (StarPO): нестабильность, **Echo Trap**, shallow reasoning без fine-grained rewards. [RAGEN-2](https://arxiv.org/abs/2604.06268) ([Wang et al., 2026](https://arxiv.org/abs/2604.06268)) — **template collapse**, SNR-aware filtering. Caveats переносятся на offline: сильнее зависимость от **data coverage**, **reward quality**, **evaluator fidelity**.

Для offline: process rewards, step-level value targets, verifier-backed labels.

---

## OPE для агентов

1. **Replay в sandbox** — cached fixtures, recorded API responses.
2. **Verifiers** — unit tests, SQL, retrieval, code execution.
3. **Learned critics** — FQE-style или goal-conditioned values. *Planning without Search* — [Planning with a Natural Language Critic](https://arxiv.org/abs/2505.18098) (в secondary sources иногда PNLC или PLNC) — natural-language critic без fine-tuning base LLM.
4. **Human / A/B review** — при высоком deploy risk.

---

## LLM как annotators, generators и critics

KALM и TEDUO — **LLM-assisted offline RL**, не всегда узкие «tool-using chat agents», но data-layer pattern переносится:

| Роль | Пример | Связь с offline RL |
|------|--------|-------------------|
| **Trajectory generator** | [KALM](https://arxiv.org/abs/2404.09248) | Imaginary rollouts + offline RL |
| **Dataset labeler** | [TEDUO](https://arxiv.org/abs/2412.06877) | LLM annotator до RL |
| **Inference-time critic** | *Planning without Search* ([2505.18098](https://arxiv.org/abs/2505.18098)) | Offline RL обучает critic |
| **Reward model** | LLM-as-judge | Комбинируйте с verifiers |

---

## Failure modes

Reward hacking, spurious tool use, template collapse (RAGEN-2), unsupported tool calls (OOD), overconfident critics, non-stationary APIs, cost/safety в reward.

---

## Практическая schema логирования

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

Рекомендации:

- **`state`** восстанавливаем на train/eval — версионируйте `prompt_version`, `tool_schema_version`, `environment_version`.
- **`action`** — structured (JSON schema), где возможно.
- **`reward`** — храните `reward_components`, `reward_source`, `verifier_version` отдельно от scalar `reward`.
- **`action_logprobs`** — для off-policy evaluation; логируйте, если behavior policy их отдаёт.
- **`metadata.behavior_policy`** — **provenance**. Importance-weighted OPE дополнительно требует comparable actions и behavior/target probabilities — для free-form LLM actions часто **недоступно** без log-probs при сборе.

---

## Ограничения и честный scope

Большинство deployed agents — SFT + online RLHF/GRPO. Offline RL для agents силён при abundant replayable traces, reliable verifiers, нетolerance к exploratory online learning.

См. [Приложение](/offline-rl-book/ru/appendix.html). [Глава 13](/offline-rl-book/ru/chapter13.html) завершает книгу.

---

## Литература

- Wang et al. (2025). *Offline RL for LLM Multi-Step Reasoning (OREO).* [arXiv:2412.16145](https://arxiv.org/abs/2412.16145), [ACL 2025](https://aclanthology.org/2025.findings-acl.464/), [code](https://github.com/jwhj/OREO).
- Xi et al. (2024). *AgentGym.* [arXiv:2406.04151](https://arxiv.org/abs/2406.04151), [project](https://agentgym.github.io/).
- Luo et al. (2025). *Agent Lightning.* [arXiv:2508.03680](https://arxiv.org/abs/2508.03680), [docs](https://microsoft.github.io/agent-lightning/latest/).
- Pang et al. (2024). *KALM.* [arXiv:2404.09248](https://arxiv.org/abs/2404.09248), [project](https://kalmneurips2024.github.io/).
- TEDUO (2024). [arXiv:2412.06877](https://arxiv.org/abs/2412.06877).
- *Planning without Search* (Planning with a Natural Language Critic). [arXiv:2505.18098](https://arxiv.org/abs/2505.18098).
- Wang et al. (2025). *RAGEN.* [arXiv:2504.20073](https://arxiv.org/abs/2504.20073).
- Wang et al. (2026). *RAGEN-2.* [arXiv:2604.06268](https://arxiv.org/abs/2604.06268), [project](https://ragen-ai.github.io/v2/).

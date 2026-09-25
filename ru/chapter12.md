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

## От agentic offline RL к offline MARL

Дальше глава про **одного** tool-using агента. Несколько агентов в одном эпизоде — это **не** «более длинная single-agent trajectory»: процесс — **Markov game** (stochastic game), а не MDP с большим context window.

**Минимальные объекты.** При $N$ агентах: совместное состояние $s_t$ (workspace / ticket), локальные наблюдения $o_t^i$ у каждого агента $i$, **совместное действие**

$$a_t = (a_t^1,\ldots,a_t^N),$$

где $a_t^i$ — сообщение, tool call или шаг плана. Награды: **shared** $r_t$, **individual** $r_t^i$ или смесь. В логе — multi-agent trajectory с per-agent observations/actions; плоская склейка transcript’а не даёт корректный credit для OREO/IQL в single-agent смысле.

Три новые проблемы:

1. **Combinatorial joint-action coverage.** Support — по *кортежам* $(a^1,\ldots,a^N)$, не по маргиналам. В логах могут быть «A ищет» и «B правит» по отдельности, но не нужный *совместный* паттерн. Conservative / support-constrained методы должны учитывать joint support.
2. **Multi-agent credit assignment.** Team success не говорит, *чей* шаг создал результат. Нужны agent-/role-level returns, counterfactual baselines или process rewards, отделяющие вклад от удачи и работы партнёров.
3. **Coordination conventions в логах.** Кто говорит, кому «принадлежат» tools, какой handoff. Offline improvement локальных политик может сломать координацию, из‑за которой логи вообще успешны.

Два **уровня** не смешивать:

| Уровень | Что учим | Типичный offline-субстрат |
|---------|----------|---------------------------|
| **Shared LLM (post-training)** | Одна base model под несколько roles / prompts | Preferences, успешные multi-agent traces как SFT, опционально MARL-aware credit |
| **Orchestration policy** | Кто действует, handoff, маршрутизация tools / messages | Логи routers, schedulers, графов Autogen/LangGraph как отдельный decision process |

Улучшить shared LLM ≠ выучить кооперативную orchestration policy над joint actions.

**Мини-пример.** Два агента: агент 1 выбирает сообщение $m$, агент 2 — tool action $u$; в конце один team reward $r \in \{0,1\}$. Успешные $(m,u,r)$ в логе **не** показывают, создал результат $m$, $u$ или только пара $(m,u)$. Single-agent update по склеенному transcript усиливает «не того» говорящего. Offline MARL начинается там, где эта неоднозначность — first-class, а не где OREO/IQL клеят на более длинный чат.

Глава остаётся single-agent намеренно. Граница Markov game — **стоп-сигнал**: для multi-agent продукта берите логирование и OPE-дисциплину отсюда, но не переносите single-agent conservative RL на joint actions и team rewards без адаптации.

---

## BC / SFT как baseline

**SFT** на expert или отфильтрованных успешных траекториях — behavioral cloning. Силён, когда behavior policy хорош, horizon короткий, failure modes редки.

Проваливается как BC (глава 1): **compounding error**, награда только для фильтрации, нет комбинирования хороших сегментов смешанных эпизодов.

Для multi-step tool use SFT **не распределяет credit** по шагам. Offline RL и advantage-weighted methods нужны, когда rewards и coverage это поддерживают.

---

## Offline RL beyond SFT

### DPO vs trajectory-level offline RL

**DPO** оптимизирует **sequence-level** log-likelihood ratio (сумму потокенных log-prob относительно reference-модели) под supervision pairwise-предпочтения. Поэтому **step-level credit assignment** слаб для multi-turn reasoning и tool use.

[OREO](https://github.com/jwhj/OREO) ([Wang et al., 2024/2025](https://arxiv.org/abs/2412.16145), [ACL Findings 2025](https://aclanthology.org/2025.findings-acl.464/)) — якорь: policy + **value function** через soft Bellman-style objective, мотивирован sparse rewards и credit assignment по шагам reasoning. Эмпирически лучше DPO-style baselines на GSM8K, MATH и ALFWorld.

Прямой предшественник — **ILQL** (Implicit Language Q-Learning; [Snell et al., 2023](https://arxiv.org/abs/2206.11871)): IQL (глава 5), адаптированный к генерации языка — expectile value-функция плюс неявное ограничение на support датасета, политика реализуется advantage-перевзвешиванием логитов базовой модели. Самый прямой мост от глав про value-пессимизм к LLM-политикам.

| Подход | Данные | Credit assignment |
|--------|--------|-------------------|
| **DPO / preference learning** | Pairwise chosen vs rejected | Sequence-level preference signal |
| **Offline RL (OREO-style)** | Полные trajectories + rewards | Value function + Bellman backup по шагам |

### Соответствие классическим идеям offline RL

**Не** читайте как «вставь CQL как есть». Это **design vocabulary**:

| Классический offline RL | Agentic AI analogue |
|-------------------------|---------------------|
| **BC** | SFT на успешных traces |
| **Policy constraint (TD3+BC, гл. 6)** | KL-штраф к reference в RLHF / DPO / GRPO — тот же регуляризатор «держись близко к поведенческой» |
| **AWR / AWAC / IQL** | Веса по advantage, verifier score, process reward (**ILQL** = IQL прямо для токенной генерации) |
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
3. **Learned critics** — FQE-style или goal-conditioned values. *Planning without Search* ([Hong et al., 2025](https://arxiv.org/abs/2505.18098)) — natural-language critic без fine-tuning base LLM.
4. **Human / A/B review** — при высоком deploy risk.

Offline RL для агентов тихо проваливается, когда OPE меряет **format quality** вместо **task success**. Метрики должны соответствовать deployment risk (предупреждения главы 3 о coverage).

### Контракт оценки политики агента

OPE объясняет *как* оценить return, но не *что считать улучшением*. У tool-using agent success rate может расти вместе с cost, latency, лишними tool calls или safety violations. До сравнения offline-кандидатов зафиксируйте короткий **контракт оценки** — спецификацию, которую команда может исполнять без повторных споров о метриках после каждого training run.

**Базовые политики (всегда сравнивать с обеими):**

1. **Текущий deployed agent** — behavior policy в production (или лучшая доступная shadow-копия).
2. **SFT / BC на успешных traces** — imitation по отфильтрованным wins; честный «без RL» baseline, когда логи хорошие, но reward structure бедная.

**Группы метрик (не сводить в одно число):**

| Группа | Что мерить | Роль |
|--------|------------|------|
| **Task success** | Verifier pass rate, completion, human accept | **Primary** — метрика, которую хотим улучшить |
| **Cost / latency** | USD/episode, tokens, p95 wall-clock | **Guardrail** |
| **Tool-call efficiency** | Calls/episode, duplicate lookups, пустые retries | **Guardrail** |
| **Safety / constraints** | Forbidden tools, PII, policy violations, API allowlist | **Guardrail** |

**Набор для оценки:** **held-out** suite — задачи, не участвовавшие в train/tuning. Стратификация по **типу задачи** и **сложности**. Отчёт по stratum, не только global average. OPE на логах не заменяет held-out eval, если deploy states расходятся с log (предупреждения главы 3 о coverage).

**Правило принятия (шаблон):**

Принять кандидата $\pi$ над baseline $b$, только если:

- **Primary:** $\text{success}(\pi) \geq \text{success}(b) + \Delta_{\min}$ на held-out suite, с **доверительным интервалом** (bootstrap по задачам или A/B в shadow), исключающим нулевой uplift.
- **Guardrails:** для каждой guardrail-метрики $g$: $\; g(\pi) \leq g(b) + \tau_g\;$, где $\tau_g$ — **заранее заданный tolerance** (например cost +5%, p95 latency +10%, нулевой tolerance на hard safety).

Если primary улучшился, а guardrail нарушен — по умолчанию **не** внедрять «с оговорками»: доработать reward, constraints или support filtering, либо отклонить кандидата.

**Иллюстративное сравнение (вымышленные числа):**

| Policy | Success (held-out) | Cost / task | Tool calls / task | Safety violations |
|--------|-------------------|-------------|-------------------|-------------------|
| Deployed agent | 62% | $0.041 | 4.2 | 0.3% |
| SFT / BC | 68% | $0.038 | 3.9 | 0.2% |
| Offline RL candidate | **71% ± 2%** | $0.044 | 4.0 | 0.2% |

При $\tau_{\text{cost}} = +5\%$, $\tau_{\text{calls}} = +10\%$ кандидат **проходит** success и safety, но **не проходит** cost ($0.044 > 1.05 \times 0.041$). Контракт отклоняет внедрение, пока cost не снижен — даже если success выше обоих baselines.

Это agentic-аналог раздела главы 13 **«Промышленное внедрение: этапы допуска (gates)»**: OPE питает Gate 2; контракт оценки — то, что Gate 3 (shadow) и Gate 4 (limited rollout) проверяют на live tasks.

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

## Toy example: calculator agent

Пример **намеренно без language modeling**: политика выбирает structured tool calls, не генерирует free text. Так изолируется offline RL: улучшение над logged behavior через value learning и support constraints.

> 📄 Код: [`agentic_offline_rl_toy.py`](https://github.com/corba777/offline-rl-book/blob/main/code/agentic_offline_rl_toy.py)

**Среда:** задачи `x+y`, `x-y`, `x*y`; tools `lookup_x`, `lookup_y`, `add`, `sub`, `mul`, `final`. Награды: +1 / −1 / −0.02 за шаг.

**Четыре политики:** Behavior (logger), BC (majority action), naive FQI (tabular Q), support-constrained FQI (argmax только по actions из dataset на state — toy analogue CQL/support mask).

**Два режима данных:** (1) `mul` есть в логах; (2) на mul-задачах behavior никогда не вызывает `mul`.

Запуск: `python code/agentic_offline_rl_toy.py`

Типичный qualitative результат:

- **Good coverage:** FQI ≈ 1.0, улучшение над behavior (~0.7).
- **No mul support:** BC ~0.67; **naive FQI на mul-only ≈ 1.0** (extrapolation — выбирает `mul` без support); **support-constrained на mul ≈ 0** (отказывается от unsupported op).

State key включает `(task, x_known, y_known, result)` — числовой `result`, чтобы wrong/correct outcome не сливались.

**Тезис:** offline RL улучшает **внутри support**; conservative methods отказываются от unsupported improvement; unconstrained Q может «галлюцинировать» OOD actions. Начинайте с tabular toy, затем classifier / LLM + support filter.

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

- Wang, H., Hao, S., Dong, H., et al. (2024). *Offline RL for LLM Multi-Step Reasoning (OREO).* [arXiv:2412.16145](https://arxiv.org/abs/2412.16145), [ACL 2025](https://aclanthology.org/2025.findings-acl.464/), [code](https://github.com/jwhj/OREO).
- Snell, C., Kostrikov, I., Su, Y., Yang, M., & Levine, S. (2023). *Offline RL for Natural Language Generation with Implicit Language Q-Learning (ILQL).* ICLR. [arXiv:2206.11871](https://arxiv.org/abs/2206.11871).
- Xi et al. (2024). *AgentGym.* [arXiv:2406.04151](https://arxiv.org/abs/2406.04151), [project](https://agentgym.github.io/).
- Luo et al. (2025). *Agent Lightning.* [arXiv:2508.03680](https://arxiv.org/abs/2508.03680), [docs](https://microsoft.github.io/agent-lightning/latest/).
- Pang et al. (2024). *KALM.* [arXiv:2404.09248](https://arxiv.org/abs/2404.09248), [project](https://kalmneurips2024.github.io/).
- Pouplin et al. (2024). *The Synergy of LLMs & RL (TEDUO).* ICML 2025. [arXiv:2412.06877](https://arxiv.org/abs/2412.06877).
- Hong, J., Dragan, A., & Levine, S. (2025). *Planning without Search: Refining Frontier LLMs with Offline Goal-Conditioned RL.* NeurIPS 2025. [arXiv:2505.18098](https://arxiv.org/abs/2505.18098).
- Wang et al. (2025). *RAGEN.* [arXiv:2504.20073](https://arxiv.org/abs/2504.20073).
- Wang et al. (2026). *RAGEN-2.* [arXiv:2604.06268](https://arxiv.org/abs/2604.06268), [project](https://ragen-ai.github.io/v2/).

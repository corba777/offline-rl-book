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

Остальная часть главы рассматривает одного агента. Когда в эпизоде действуют несколько агентов, эта постановка перестаёт работать. Пока политики остальных агентов зафиксированы, для каждого отдельного агента задача остаётся (PO)MDP: партнёры просто входят в среду. Но offline-обучение меняет сразу несколько политик, и динамика, которую видит агент, меняется вместе с поведением партнёров. Логи, собранные при старых политиках партнёров, перестают описывать новую среду. Правильная модель здесь — Markov game (stochastic game), а при локальных наблюдениях и общей награде — Dec-POMDP. Если склеить реплики всех агентов в один transcript, теряется, кто что наблюдал и кто какое действие выбрал, то есть именно те сведения, которые нужны для оценки support и credit assignment.

**Обозначения.** При $N$ агентах $s_t$ — совместное состояние (общий workspace или состояние тикета), $o_t^i$ — локальное наблюдение агента $i$ (его prompt, инструменты и собственная память), а

$$a_t = (a_t^1,\ldots,a_t^N)$$

— joint action, где $a_t^i$ — сообщение, tool call или шаг плана. Награда может быть общей для команды ($r_t$), индивидуальной ($r_t^i$) или смешанной. Поэтому в логе нужны наблюдения и действия каждого агента, а не только общая переписка.

Single-agent offline RL не решает три задачи, которые здесь появляются:

1. **Joint-action coverage.** Support определяется на кортежах $(a^1,\ldots,a^N)$, а не на действиях каждого агента по отдельности. В логах могут быть «A ищет» и «B правит» в разных эпизодах, но ни разу — нужная при развёртывании комбинация. Если ограничить каждого агента только его собственными действиями из логов, команда всё равно может выполнить joint action, которого в данных нет.
2. **Multi-agent credit assignment.** Успех команды не говорит, чей шаг к нему привёл. Чтобы отделить вклад агента от удачи и от работы партнёров, нужны returns на уровне агента или роли, counterfactual baselines или process rewards.
3. **Coordination conventions.** Multi-agent логи фиксируют, кто когда говорит, какой роли принадлежит какой инструмент и как устроены handoffs. Если улучшать политику каждого агента по отдельности, можно сломать conventions, благодаря которым эпизоды в логах были успешными.

В LLM-системах часто смешивают две разные задачи обучения:

| Уровень | Что обучается | Типичные offline-данные |
|---------|---------------|-------------------------|
| **Shared LLM (post-training)** | Одна базовая модель для нескольких ролей или prompts | Preference pairs, успешные multi-agent traces для SFT, при необходимости per-agent credit assignment |
| **Orchestration policy** | Кто действует следующим, когда передавать управление, как маршрутизировать tools и сообщения | Логи routers, schedulers или графов Autogen/LangGraph как отдельного процесса принятия решений |

Дообучение shared LLM на traces с ролями (DPO, SFT или single-agent offline RL) обновляет другую политику, чем orchestration policy, которая выбирает joint actions. Указывайте, какую из двух обновляет метод; иначе multi-agent проблемы попадают в систему под видом обычного дообучения агента.

**Пример.** Агент 1 отправляет агенту 2 сообщение $m \in \{A, B\}$, затем агент 2 вызывает инструмент $u \in \{a, b\}$. Команда получает $r = 1$ только за пары $(A, a)$ и $(B, b)$. При равномерно случайной behavior policy каждое отдельное сообщение и каждый отдельный инструмент успешны в половине эпизодов, поэтому ценности действий отдельных агентов равны и ничего не сообщают; хорошее поведение от плохого отличает только ценность пары. Single-agent обновление по склеенному transcript оценивает каждое действие по его частоте успеха и может прийти к несогласованной паре, например $(A, b)$. Поэтому offline MARL методы явно моделируют joint action.

В классическом offline MARL для этого уже есть инструменты. MACCA ([Wang et al., 2025](https://arxiv.org/abs/2312.03644)) представляет offline-данные как динамическую байесовскую сеть и выделяет причинный вклад каждого агента в награду. Для LLM-агентов ближайшие работы пока online. LangMARL ([Yao et al., 2026](https://aclanthology.org/2026.emnlp-main.1837/)) с помощью централизованного языкового critic назначает credit каждому агенту и переписывает его политику на естественном языке — инструкции и примеры, а не веса модели, — собирая новые rollouts на каждой итерации. Перенести такой agent-level credit assignment на фиксированные логи — пока открытая задача.

Глава намеренно остаётся single-agent. Для multi-agent продукта используйте отсюда логирование и практики оценки, но не применяйте single-agent conservative RL к joint actions и team rewards, не адаптировав support constraints и credit assignment.

---

## Интерфейс политики определяет метод обучения

До выбора функции потерь определите, какую политику обновляете. Классификатор действий остаётся политикой $\pi(a\mid s)$, даже если использует энкодер LLM; генерация текста для этого не обязательна.

| Интерфейс политики | Пространство действий | Baseline | Возможные методы |
|--------------------|-----------------------|----------|------------------|
| Генеративная LLM | Последовательности токенов / сериализованные tool calls | SFT на демонстрациях | ILQL с token values; DPO на preference pairs; trajectory methods с совместимыми вероятностями действий |
| Классификатор действий | Конечный набор tools / action labels | Cross-entropy BC на action logits | Discrete CQL, IQL с categorical actor, support-constrained value learning |
| Router / orchestrator | Выбор агента, tool, handoff или расписания | BC на routing decisions | Offline RL для процесса маршрутизации со своими observations, rewards и support constraints |

DPO оптимизирует предпочтения и не заменяет trajectory-level credit assignment. Token-level ILQL и language-model SFT требуют token policy и LM head; classifier обучается через action logits. Текстовая retrospection может дать annotations или features, но loss генерации текста нельзя напрямую перенести на classifier без совместимой выходной головы и targets.

Для строки router / orchestrator уже есть пример в реалистичном масштабе. HarnessMDP ([Yi & Song, 2026](https://arxiv.org/abs/2607.05458)) оставляет LLM-исполнителя замороженным и обучает лишь небольшой контроллер, который выбирает одну из семи структурных операций: observe, retrieve, call tool, draft, check, revise и submit. Контроллер обучается с помощью advantage-weighted regression на траекториях существующих harnesses и улучшает результат на задачах программирования, τ-bench и AgentBench. Авторы прямо называют предел метода: контроллер усиливает паттерны, которые уже есть в буфере, но не может выучить стратегии вне его support, а выигрыш зависит от качества и покрытия логов. Калькулятор ниже приводит к тому же выводу, только в куда меньшем масштабе.

Пример калькулятора ниже реализует **табличную политику**; классификатор и LLM предложены как дальнейшие расширения. Замена модели не устраняет необходимость определить action support; замена интерфейса может потребовать другого objective.

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

Обозначим политику сбора логов как $\pi_b$, а frozen reference model как $\pi_{\mathrm{ref}}$. TD3+BC приближает действия из датасета; [DPO](https://arxiv.org/abs/2305.18290) сравнивает sequence probabilities относительно $\pi_{\mathrm{ref}}$. Они могут быть близки, если данные собирал тот же SFT checkpoint при одинаковых prompts и sampling, но смесь checkpoints, фильтрация и другое декодирование нарушают это соответствие. Reference penalty сам по себе не обеспечивает dataset support. DPO задаёт reference regularization через preference objective; наличие, коэффициент и роль явного KL-штрафа в GRPO зависят от реализации.

**Не** читайте как «вставь CQL как есть». Это **design vocabulary**:

| Классический offline RL | Agentic AI analogue |
|-------------------------|---------------------|
| **BC** | SFT на успешных traces |
| **Ограничение по действиям датасета (TD3+BC, гл. 6)** | Штраф за расстояние до logged actions ограничивает отклонение от поведения в датасете |
| **Регуляризация к reference policy** | KL к frozen reference в KL-regularized RLHF; DPO использует отношения policy/reference probabilities; варианты GRPO различаются |
| **AWR / AWAC / IQL** | Веса по advantage, verifier score, process reward (**ILQL** = IQL прямо для токенной генерации) |
| **CQL / pessimism** | Штраф unsupported tool calls / plans |
| **FQE / learned Q** | Critic по agent states и actions |
| **Decision Transformer** | Conditioning на desired return / success |
| **Model-based offline RL** | Simulator, verifier, world model |
| **OPE (гл. 3)** | Повторное исполнение в среде / action-conditioned simulation / learned value model; отличайте прямую оценку от OPE по фиксированным данным |

**LightningRL** — hierarchical credit assignment + single-turn RL updates. Это **off-policy RL bridge**, не замена CQL/IQL на fixed logs.

---

## Credit assignment в reasoning и tool use

[RAGEN](https://arxiv.org/abs/2504.20073) ([Wang et al., 2025](https://arxiv.org/abs/2504.20073)) — multi-turn agent RL (StarPO): нестабильность, **Echo Trap**, shallow reasoning без fine-grained rewards. [RAGEN-2](https://arxiv.org/abs/2604.06268) ([Wang et al., 2026](https://arxiv.org/abs/2604.06268)) — **template collapse**, SNR-aware filtering. Caveats переносятся на offline: сильнее зависимость от **data coverage**, **reward quality**, **evaluator fidelity**.

Для offline: process rewards, step-level value targets, verifier-backed labels.

У редкой терминальной награды есть и статистический предел. Ren и Xie ([2026](https://arxiv.org/abs/2606.18531)) изучают offline RL в ситуации, когда у каждой траектории есть лишь одна итоговая метка, и предлагают OPAC — пессимистичный actor-critic, который обучает скрытую пошаговую награду вместе с value function. Их нижняя оценка объясняет, почему выбор метки так важен: для all-success objective, где эпизод засчитывается, только если успешен каждый шаг, исход несёт информацию лишь об экспоненциально малой доле траекторий, и любому алгоритму нужно порядка $2^H$ эпизодов, чтобы приблизиться к оптимальной политике. Process rewards возвращают в данные ту информацию, которую теряет единственный итоговый бит. Недавний обзор credit assignment для LLM-агентов ([Zhang, 2026](https://arxiv.org/abs/2604.09459)) добавляет второе предостережение: если в логе есть только текстовые истории, а прежнее состояние нельзя восстановить, может оставаться неопределённым даже знак credit отдельного шага.

---

## OPE для агентов

1. **Повторное исполнение / action-conditioned simulation** — выполняйте действия кандидата в сбрасываемом sandbox либо симулируйте их последствия с явными предположениями; cached responses требуют совпадения действий и snapshot среды.
2. **Verifiers** — unit tests, SQL, retrieval, code execution.
3. **Learned critics** — FQE-style или goal-conditioned values. *Planning without Search* ([Hong et al., 2025](https://arxiv.org/abs/2505.18098)) — natural-language critic без fine-tuning base LLM.
4. **Human / A/B review** — при высоком deploy risk.

Offline RL для агентов тихо проваливается, когда OPE меряет **format quality** вместо **task success**. Метрики должны соответствовать deployment risk (предупреждения главы 3 о coverage).

### Replay — не контрфактуальная оценка

Перед интерпретацией метрик различайте три операции:

1. **Точное воспроизведение trace** подаёт в pipeline прежние действия и observations. Оно проверяет parsing, восстановление state и регрессии, но не return политики, выбирающей другие действия.
2. **Повторное исполнение в среде** выполняет реальные действия кандидата в сбрасываемом sandbox. Это оценка в тестовой среде, а не только по фиксированному датасету; перенос зависит от fidelity sandbox, версий tools и репрезентативности начальных задач.
3. **Rollout в симуляторе** создаёт observations с учётом нового действия и текущего simulated state. Обученный simulator добавляет model bias, особенно вне support логов; его нужно отдельно валидировать и сообщать неопределённость.

Cached response можно переиспользовать, только если совпадают полный запрос и релевантное состояние среды: arguments, права доступа, snapshot документа/базы, версия tool и случайные входы, влияющие на результат. Одной детерминированности недостаточно, если snapshot изменился. Cache miss должен приводить к исполнению в sandbox, проверенной симуляции или явному unsupported outcome; подставлять ответ другого действия нельзя.

Например, результат `open_pdf(page=42)` нельзя выдавать после `open_pdf(page=43)`. Даже одинакового номера страницы недостаточно, если изменился исходный документ. Старая observation скрывает последствия решения кандидата и создаёт недействительный контрфактуал.

Для каждой метрики указывайте, какая операция её породила. Exact replay подтверждает воспроизводимость прежнего запуска, но не улучшение success новой политики. Если среда недоступна, используйте статистическую OPE или learned value estimates с явными предположениями и ограничениями покрытия, а не выдавайте cached transcript за новый rollout.

Недавние работы подкрепляют это предупреждение измерениями. В *The Replay Gap* ([Gonuguntla, 2026](https://arxiv.org/abs/2608.08239)), workshop-исследовании на SWE-bench Verified, траектории агента разветвляют посередине и продолжают другой моделью. Смена модели переписывает от 61 до 94% последующих действий, а оценщик, склеивающий записанные ответы, ошибается во всех пяти случаях, где успех действительно изменился. Выборка мала, но направление однозначно: склеенный лог оценивает мир, в который новая политика никогда бы не попала. Конструктивная альтернатива — обученный simulator. ADWM ([Liu et al., 2026](https://arxiv.org/abs/2606.05558)) моделирует каждый transition multi-turn агента латентной диффузионной world model, обученной на логах, и на HotpotQA, ScienceWorld, ALFWorld и WebShop ранжирует политики надёжнее, чем direct method, FQE, importance sampling и doubly robust оценки. Авторы аккуратно очерчивают границы: эксперименты не покрывают политики полностью вне support, а ранговая корреляция ничего не говорит о калибровке абсолютных значений.

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

**Четыре политики:** Behavior (logger), BC (majority action), naive FQI (tabular Q), support-constrained FQI (argmax только по actions из dataset на state — support mask, а не функция потерь CQL).

**Два режима данных:** (1) `mul` есть в логах; (2) на mul-задачах behavior никогда не вызывает `mul`.

В неизвестном состоянии или при пустом наборе допустимых действий из support constrained policy отказывается от действия (`None`); оценка учитывает это как неуспешное завершение. Операция из support иногда случайно даёт правильный ответ на mul-задаче, поэтому успех может быть ненулевым даже без `mul`.

Запуск: `python code/agentic_offline_rl_toy.py`

Типичный qualitative результат:

- **Good coverage:** FQI ≈ 1.0, улучшение над behavior (~0.7).
- **No mul support:** BC ~0.67; **naive FQI на mul-only ≈ 1.0** (extrapolation — выбирает `mul` без support); **support-constrained на mul ≈ 0** (отказывается от unsupported op).

Ключ состояния — `(task, x, y, result)`, где `None` означает ещё не наблюдавшееся значение. Наблюдаемые операнды вместе с результатом позволяют различать правильные и неправильные исходы. До вызова lookup политика не получает скрытые операнды.

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
  "task_family_id": "sec_revenue_extraction_filing_001",
  "source_hash": "sha256:<source-snapshot-digest>",
  "collection_window": {"start": "2026-06-01", "end": "2026-06-30"},
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

### Multi-agent extension

Добавляйте запись к transition каждого действующего агента; action и episode identifier находятся в основной записи.

```json
{
  "multi_agent_extension": {
    "joint_step_id": "task_001:3",
    "turn_id": 7,
    "agent_id": "reader_1",
    "role": "document_reader",
    "recipient_ids": ["analyst_1"],
    "local_observation": {
      "visible_document": "filing_snapshot_v1",
      "message": "Read page 42"
    },
    "policy_checkpoint": "reader_policy_v3",
    "team_reward": {
      "value": 0.0,
      "source": "task_verifier_v2"
    },
    "individual_reward": {
      "value": 0.0,
      "source": "tool_validity_v1"
    },
    "credit_label": {
      "value": null,
      "source": null
    }
  }
}
```

Через `joint_step_id` связывайте действия и сообщения одного решения среды; `turn_id` сохраняет порядок внутри него. Для каждого агента храните здесь local observation, recipients и checkpoint; вероятность фактически выбранного действия под политикой сбора, включая sampling settings, записывается в `action_logprobs` основной записи, чтобы не дублировать её. Null означает неизвестную вероятность, а не нулевую.

Плоский transcript не сохраняет private observations, одновременные решения и provenance политик. Team reward, individual reward и поздние credit annotations имеют разные значения и источники; credit label не является наблюдаемым контрфактуалом. Такая запись позволяет восстановить trajectory, но не идентифицирует контрфактуальный вклад без дополнительных предположений или вмешательств.

### Разбиение данных и утечки

Назначайте split **до извлечения transitions**, сохраняя эпизоды целиком. Группируйте связанные эпизоды по task family и исходному документу: варианты задач, общие cached outputs и близкие дубликаты не должны попадать в разные splits. Храните `task_family_id`, `source_hash` (хеш snapshot источника) и `collection_window`; сам хеш не выявляет смысловые дубликаты.

Сначала определите гранулярность группировки: `task_family_id` объединяет базовую задачу и её варианты, а не всю категорию вроде извлечения выручки. Отложите независимые группы задач и исходные документы внутри целевых категорий для проверки новых примеров. Удержание целых категорий проверяет перенос на незнакомые семейства задач; сообщайте этот результат отдельно. Добавьте temporal/version holdout, собранный позже либо при новых prompts, tools или environments: он проверяет также устойчивость к изменениям среды.

Проверка утечек:

- Все шаги и multi-agent turns эпизода находятся в одном split.
- Варианты задач и дублирующиеся или пересекающиеся источники сгруппированы до разбиения.
- Reward model, normalization и verifier calibration обучаются на train; prompts и кандидаты выбираются по validation.
- Test set и протокол оценки зафиксированы; по test outcomes ничего не настраивается.
- Provenance cache и collection windows проверены; вместе с числом transitions указано число task families.

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
- Yi, H., & Song, X. (2026). *HarnessMDP: Learning to Control LLM Agent Harnesses with Offline Reinforcement Learning.* [arXiv:2607.05458](https://arxiv.org/abs/2607.05458).
- Ren, X., & Xie, T. (2026). *When Does Trajectory-Level Supervision Permit Efficient Offline Reinforcement Learning?* [arXiv:2606.18531](https://arxiv.org/abs/2606.18531).
- Zhang, C. (2026). *From Reasoning to Agentic: Credit Assignment in Reinforcement Learning for Large Language Models.* [arXiv:2604.09459](https://arxiv.org/abs/2604.09459).
- Gonuguntla, A. (2026). *The Replay Gap: Static Evaluation of Model Switching in LLM Agents Scores the Wrong World.* Efficient Reasoning Workshop, COLM. [arXiv:2608.08239](https://arxiv.org/abs/2608.08239).
- Liu, K., Xiong, G., Zhang, W., & Tang, S. (2026). *Autoregressive Diffusion World Models for Off-Policy Evaluation of LLM Agents (ADWM).* [arXiv:2606.05558](https://arxiv.org/abs/2606.05558).
- Wang, Z., Du, Y., Zhang, Y., Fang, M., & Huang, B. (2025). *MACCA: Offline Multi-agent Reinforcement Learning with Causal Credit Assignment.* TMLR. [arXiv:2312.03644](https://arxiv.org/abs/2312.03644).
- Yao, H., Da, L., Liu, X., Fleming, C., Chen, T., & Wei, H. (2026). *LangMARL: Natural Language Multi-Agent Reinforcement Learning via Semantic Credit Assignment.* EMNLP. [ACL Anthology](https://aclanthology.org/2026.emnlp-main.1837/), [arXiv:2604.00722](https://arxiv.org/abs/2604.00722).

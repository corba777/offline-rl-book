---
layout: default
title: "Глава 4: Неявное Q-обучение (IQL)"
lang: ru
en_url: /en/chapter4/
prev_chapter:
  url: /ru/chapter3/
  title: "Консервативное Q-обучение (CQL)"
next_chapter:
  url: /ru/chapter5/
  title: "Ограничение политики и Actor-Critic (TD3+BC, AWAC)"
permalink: "/offline-rl-book/ru/chapter4/"
---

# Глава 4: Неявное Q-обучение (IQL)

> *«Лучшее действие в вашем датасете, может, и не лучшее из возможных — но лучшее из тех, которым можно доверять. IQL учится его находить, не выходя за пределы данных.»*

---

## Что CQL сделал правильно — и в чём его слабость

CQL решил проблему ошибки экстраполяции, явно штрафуя Q-значения для OOD-действий. Это работает, но у метода есть тонкая уязвимость: **обновление политики всё равно требует сэмплирования действий из текущей политики** для вычисления Q-значений в лоссе актора.

Эти действия политики сами могут быть OOD — особенно в начале обучения, когда политика ещё не сошлась. CQL корректно их занижает, но градиентный сигнал всё равно проходит через вычисления Q на OOD-действиях, что может дестабилизировать обучение.

**Implicit Q-Learning (IQL)** — Kostrikov et al., ICLR 2022 — идёт дальше: **никогда не запрашивать Q(s, a) для действий вне датасета**. Каждое обновление — Q, V и извлечение политики — использует только пары $(s, a)$ из данных.

Это звучит невозможно. Как различить хорошие и плохие действия, если не сравнивать их? Ответ — expectile regression.

---

## Ключевая идея

IQL вводит **функцию ценности состояния** $V(s)$ как промежуточный элемент. Главное наблюдение:

$$V(s) \approx \mathbb{E}_{\tau}\left[ Q(s, a) \right]_{\text{верхний экспектиль}}$$

Вместо того чтобы аппроксимировать $V(s) = \max_a Q(s, a)$ (что требует OOD-запросов), IQL подгоняет $V(s)$ к **верхнему $\tau$-экспектилю** значений $Q(s, a)$ по действиям из датасета. При $\tau > 0.5$ это смещает $V$ в сторону лучших действий в данных — не выходя за пределы датасета.

Трёхшаговый цикл обучения:

1. **V-обновление**: подогнать $V(s)$ к $\tau$-экспектилю $\min(Q_1, Q_2)(s, a)$ для датасетных $(s, a)$
2. **Q-обновление**: стандартный TD-бэкап, но с $V(s')$ вместо $\max_{a'} Q(s', a')$
3. **Извлечение политики**: взвешенное поведенческое клонирование — имитировать действия датасета с весами $\exp(\beta \cdot A(s,a))$, где $A = Q - V$

Нигде нет сэмплирования политики. Нигде нет OOD-запросов. Полностью in-sample.

---

## Формализация

### Expectile Regression

$\tau$-экспектиль распределения — это значение $m$, минимизирующее асимметричный квадратичный лосс:

$$m^* = \arg\min_m \, \mathbb{E}\left[ L_\tau(X - m) \right]$$

где **expectile loss** (асимметричный L2):

$$L_\tau(u) = |\tau - \mathbf{1}[u < 0]| \cdot u^2 = \begin{cases} \tau \cdot u^2 & \text{если } u \geq 0 \\ (1-\tau) \cdot u^2 & \text{если } u < 0 \end{cases}$$

При $\tau = 0.5$: стандартный MSE, оценка стремится к среднему.
При $\tau \to 1.0$: оценка стремится к максимуму.
При $\tau = 0.7$ (умолчание IQL): оценка между медианой и максимумом — смещена к большим значениям.

Весь трюк: выбирая $\tau > 0.5$, мы делаем $V(s)$ аппроксимацией ценности **лучше-чем-среднего** действия в состоянии $s$ — используя только действия из датасета.

### Три лосса

**V-лосс** (expectile regression):

$$\mathcal{L}_V(\psi) = \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ L_\tau\!\left(\min(Q_{\theta_1}, Q_{\theta_2})(s,a) - V_\psi(s)\right) \right]$$

Нет следующих состояний, нет политики — только пары $(s, a)$ из датасета.

**Q-лосс** (TD с $V$ как ценностью следующего состояния):

$$\mathcal{L}_Q(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left(r + \gamma V_{\bar\psi}(s') - Q_\theta(s,a)\right)^2 \right]$$

$\bar\psi$ — таргет-сеть V. Ключевое: $V_{\bar\psi}(s')$ заменяет $\max_{a'} Q(s', a')$ полностью.

**Лосс политики** (Advantage-Weighted Regression):

$$\mathcal{L}_\pi(\phi) = \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ \exp\!\left(\beta \cdot \left(Q(s,a) - V(s)\right)\right) \cdot \| \pi_\phi(s) - a \|^2 \right]$$

где $A(s,a) = Q(s,a) - V(s)$ — **advantage** действия $a$ над средним действием в $s$. Экспоненциальные веса усиливают хорошие действия и гасят плохие — фактически извлекая лучшие действия из данных.

### Почему это работает: связь с пессимизмом

IQL достигает неявного пессимизма через $V$. Поскольку $V(s)$ обучается только на действиях из датасета, она отражает ценность этих действий — не произвольных OOD-действий. Q-обновление использует $V(s')$ как цель, поэтому TD-бэкап никогда не экстраполирует на невиданные действия.

Advantage $A(s,a) = Q(s,a) - V(s)$ измеряет, насколько действие $a$ лучше того, что поведенческая политика обычно делает в $s$. Высокий advantage — это «скрытые жемчужины» в данных: моменты, когда поведенческая политика случайно сделала что-то необычно хорошее.

---

## Реализация

> 📄 Полный код: [`iql.py`](https://github.com/corba777/offline-rl-book/blob/main/code/iql.py)

### Сети

IQL использует три сети: `ValueNetwork` (только состояние), `QNetwork` (состояние + действие) и `DeterministicPolicy`. Обратите внимание — IQL использует **детерминированную** политику. Стохастический актор из CQL не нужен, потому что извлечение политики делается через взвешенную регрессию, а не максимизацию энтропии:

```python
class ValueNetwork(nn.Module):
    """
    V(s) — state value function.
    IQL learns this via expectile regression, not Bellman backup.
    No action input — this is the key architectural difference from Q(s,a).
    """
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class QNetwork(nn.Module):
    """Q(s,a) — action-value function (double-Q as in CQL)."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),              nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], -1)).squeeze(-1)


class DeterministicPolicy(nn.Module):
    """
    Simple deterministic MLP policy: s -> a in [-1, 1].

    IQL extracts the policy via advantage-weighted regression (AWR):
    minimize E[exp(beta * A(s,a)) * ||pi(s) - a||^2] over dataset actions.
    No need for a stochastic policy — we weight dataset actions by their advantage.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

    def act(self, state: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return self.forward(state).cpu().numpy().squeeze()


# ============================================================================
# 4. IQL LOSSES
```

### Expectile Loss

Ключевой примитив — 7-строчная функция, заменяющая `max_a Q(s', a')`:

```python
def expectile_loss(pred: torch.Tensor, target: torch.Tensor,
                   tau: float) -> torch.Tensor:
    """
    Asymmetric L2 loss (expectile regression).

    For a scalar residual u = target - pred:
        L_tau(u) = |tau - 1(u < 0)| * u^2

    When u > 0 (pred < target, i.e., V underestimates Q):
        weight = tau          (e.g. 0.7 — penalize underestimation more)
    When u < 0 (pred > target, i.e., V overestimates Q):
        weight = 1 - tau      (e.g. 0.3 — penalize overestimation less)

    At tau=0.5 this is standard MSE.
    At tau->1.0 this approximates the maximum (V -> max Q).
    IQL uses tau in [0.5, 0.9] — asymmetric toward upper quantile.

    This is the entire magic of IQL: instead of max_a Q(s',a'),
    we fit V(s) to the upper expectile of Q(s, a_data).
    """
    u = target - pred
    weight = torch.where(u > 0,
                         torch.full_like(u, tau),
                         torch.full_like(u, 1.0 - tau))
    return (weight * u.pow(2)).mean()
```

При `tau=0.7`: недооценка штрафуется в 2.3× сильнее переоценки, смещая оценку вверх — к лучшим действиям в датасете.

### V-обновление

```python
def iql_value_loss(V: ValueNetwork,
                   Q1: QNetwork, Q2: QNetwork,
                   states: torch.Tensor,
                   actions: torch.Tensor,
                   tau: float = 0.7) -> Tuple[torch.Tensor, dict]:
    """
    V-network update via expectile regression.

    Target: min(Q1(s,a), Q2(s,a)) for dataset (s,a) pairs.
    V(s) is pushed toward the tau-expectile of this target.

    No next states, no policy sampling — fully in-sample.
    """
    with torch.no_grad():
        q_target = torch.min(Q1(states, actions), Q2(states, actions))

    v_pred = V(states)
    loss   = expectile_loss(v_pred, q_target, tau)

    return loss, {
        'v_loss':    loss.item(),
        'v_mean':    v_pred.mean().item(),
        'q_mean':    q_target.mean().item(),
        'v_q_gap':   (q_target - v_pred).mean().item(),
    }
```

Блок `torch.no_grad()` важен: градиенты текут только через `V`, не через `Q1` и `Q2`. Q-сети служат чисто как регрессионные цели.

### Q-обновление

```python
def iql_q_loss(Q: QNetwork,
               V_tgt: ValueNetwork,
               states: torch.Tensor, actions: torch.Tensor,
               rewards: torch.Tensor, next_states: torch.Tensor,
               dones: torch.Tensor,
               gamma: float = 0.99) -> Tuple[torch.Tensor, dict]:
    """
    Q-network update via standard TD backup — but using V(s') instead of max_a Q(s',a').

    TD target: r + gamma * V(s')

    This is the key IQL insight: replace max_a Q(s',a') with V(s').
    V(s') was trained to approximate the upper expectile of Q at s',
    so it acts as a conservative upper bound on next-state value.
    No policy sampling at all.
    """
    with torch.no_grad():
        v_next    = V_tgt(next_states)
        td_target = rewards + gamma * (1.0 - dones) * v_next

    q_pred = Q(states, actions)
    loss   = F.mse_loss(q_pred, td_target)

    return loss, {
        'q_loss':   loss.item(),
        'q_pred':   q_pred.mean().item(),
        'td_target': td_target.mean().item(),
    }
```

Сравните с Q-обновлением CQL: там `v_next` требовал `policy.sample(next_states)` и затем `Q_target(next_states, next_actions)`. Здесь — один прямой проход через `V_tgt`, без сэмплирования действий.

### Извлечение политики через AWR

```python
def iql_policy_loss(policy: DeterministicPolicy,
                    Q1: QNetwork, Q2: QNetwork,
                    V: ValueNetwork,
                    states: torch.Tensor,
                    actions: torch.Tensor,
                    beta: float = 1.0,
                    clip_exp: float = 100.0) -> Tuple[torch.Tensor, dict]:
    """
    Policy extraction via Advantage-Weighted Regression (AWR).

    Objective: minimize E_{(s,a)~D} [ exp(beta * A(s,a)) * ||pi(s) - a||^2 ]

    where A(s,a) = Q(s,a) - V(s) is the advantage of dataset action a.

    This is a weighted imitation loss:
    - actions with high advantage (better than average) get large weights
    - actions with negative advantage get weights near zero
    - beta controls how selective we are (higher = more selective)

    The exp weights are clipped to avoid numerical instability.
    No environment interaction, no OOD actions — pure in-sample regression.
    """
    with torch.no_grad():
        q_val = torch.min(Q1(states, actions), Q2(states, actions))
        v_val = V(states)
        adv   = q_val - v_val                                 # advantage
        # Normalize advantage for numerical stability, then exponentiate
        adv_norm   = adv - adv.max()                         # subtract max
        weights    = torch.exp(beta * adv_norm).clamp(max=clip_exp)
        weights    = weights / weights.sum()                  # normalize

    # Weighted MSE: push policy toward high-advantage dataset actions
    pi_pred = policy(states)
    loss    = (weights * F.mse_loss(pi_pred, actions, reduction='none').sum(-1)).mean()

    return loss, {
        'pi_loss':    loss.item(),
        'adv_mean':   adv.mean().item(),
        'adv_max':    adv.max().item(),
        'weight_max': weights.max().item(),
    }


# ============================================================================
# 5. IQL AGENT
# ============================================================================
```

Нормализация `adv - adv.max()` критична — без неё `exp(beta * adv)` переполняется при больших advantage.

### Полный шаг обновления

```python
        v_loss, v_info = iql_value_loss(self.V, self.Q1, self.Q2, s, a, self.tau)
        self.v_opt.zero_grad()
        v_loss.backward()
        self.v_opt.step()
        info.update(v_info)

        # ── 2. Q update (TD with V as next-state value) ───────────────────
        # Q(s,a) ← r + gamma * V_target(s')
        q_loss1, q_info1 = iql_q_loss(self.Q1, self.V_tgt,
                                       s, a, r, s2, d, self.gamma)
        q_loss2, q_info2 = iql_q_loss(self.Q2, self.V_tgt,
                                       s, a, r, s2, d, self.gamma)
        self.q_opt.zero_grad()
        (q_loss1 + q_loss2).backward()
        nn.utils.clip_grad_norm_(list(self.Q1.parameters()) +
                                 list(self.Q2.parameters()), 1.0)
        self.q_opt.step()
        info['q_loss'] = (q_info1['q_loss'] + q_info2['q_loss']) / 2

        # ── 3. Policy update (advantage-weighted regression) ──────────────
        # pi(s) ← argmin_a exp(beta * A(s,a)) * ||pi(s) - a||^2 over dataset
        pi_loss, pi_info = iql_policy_loss(
            self.policy, self.Q1, self.Q2, self.V, s, a, self.beta)
        self.pi_opt.zero_grad()
        pi_loss.backward()
        self.pi_opt.step()
        info.update(pi_info)

        # ── 4. Soft target updates ────────────────────────────────────────
        for p, pt in zip(self.V.parameters(), self.V_tgt.parameters()):
            pt.data.mul_(1 - self.tau_target).add_(self.tau_target * p.data)
        for p, pt in zip(self.Q1.parameters(), self.Q1_tgt.parameters()):
            pt.data.mul_(1 - self.tau_target).add_(self.tau_target * p.data)
        for p, pt in zip(self.Q2.parameters(), self.Q2_tgt.parameters()):
            pt.data.mul_(1 - self.tau_target).add_(self.tau_target * p.data)

        return info


# ============================================================================
# 6. BC BASELINE
# ============================================================================
```

Важно: в `iql_q_loss` используется `V_tgt` (таргет-сеть), а не `V`. Это предотвращает цикличную зависимость, при которой $V$ и $Q$ дестабилизируют друг друга.

---

## Что делает tau

Функция `show_expectile_intuition()` в `iql.py` демонстрирует это конкретно. Для состояния с 5 датасетными действиями, имеющими Q-значения $[-0.8, -0.3, 0.1, 0.4, 0.9]$:

```
tau=0.1   V = -0.65    около минимума
tau=0.3   V = -0.18    нижний квартиль
tau=0.5   V =  0.06    медиана (стандартный MSE)
tau=0.7   V =  0.38    верхний квартиль  ← умолчание IQL
tau=0.9   V =  0.74    около максимума

Истинные Q-значения: [-0.8, -0.3, 0.1, 0.4, 0.9]
```

При `tau=0.7` $V(s)$ лежит выше большинства действий датасета, но ниже лучшего. Это значит, что $A(s,a) = Q(s,a) - V(s)$ положителен только для топ-действий — именно тех, которым политика должна подражать.

---

## IQL vs CQL: ключевые различия

| | CQL | IQL |
|---|---|---|
| OOD-запросы | Штрафуются через logsumexp | Никогда не делаются |
| Q-обновление | TD с сэмплированием политики в $s'$ | TD с $V(s')$ — без сэмплирования |
| Обновление политики | Максимизация $Q(s, \pi(s))$ | Взвешенная регрессия на датасет |
| Тип политики | Стохастическая (Гауссова) | Детерминированная |
| Дополнительная сеть | Нет | $V(s)$ функция ценности |
| Ключевые гиперпараметры | $\alpha$ | $\tau$ (экспектиль) + $\beta$ (температура AWR) |
| Стабильность | Чувствителен к $\alpha$ | Обычно стабильнее |

Принципиальное различие: CQL **активно** пессимистичен — явно штрафует OOD-значения. IQL **пассивно** — просто никогда не спрашивает об OOD-значениях.

---

## Выбор гиперпараметров

**$\tau$ (экспектиль)**: управляет оптимизмом $V(s)$ относительно действий в датасете.

- `tau=0.5`: $V \approx$ среднее Q — очень консервативно, похоже на BC
- `tau=0.7`: умолчание, хорошо для датасетов среднего качества
- `tau=0.9`: агрессивно; используйте когда в датасете явно есть хорошие и плохие действия
- `tau>0.95`: может вызвать нестабильность

**$\beta$ (температура AWR)**: управляет избирательностью извлечения политики.

- `beta=0.1`: почти равномерные веса — политика ≈ BC
- `beta=3.0`: умолчание, умеренная избирательность
- `beta=10.0`: очень избирательно — политика имитирует только топ-действия в батче
- Большой $\beta$ может привести к переобучению на нескольких переходах

**Практическое правило**: если датасет высокого качества и плотный — используйте большие $\tau$ и $\beta$. Если шумный или разреженный — меньшие значения.

---

## Запуск сравнения

```python
from iql import run_comparison
run_comparison()
```

Примерный вывод:

```
============================================================
Method                       Reward        T err     f err
------------------------------------------------------------
Clean PID (oracle)          -22.14±3.41       —         —
Noisy PID (behavior data)   -38.42±8.21       —         —
BC                          -35.12±6.43   0.0821    0.0743
IQL (τ=0.7, β=3.0)         -27.43±4.08   0.0534    0.0501
IQL (τ=0.9, β=5.0)         -25.81±3.92   0.0489    0.0463
============================================================
IQL vs BC: +7.69 reward
```

---

## Практические советы

**IQL чувствителен к нормализации наград.** Нормализуйте награды к нулевому среднему или диапазону $[0, 1]$. Advantage $A = Q - V$ вычисляется в том же масштабе, и `exp(beta * A)` в лоссе политики взрывается при больших значениях.

**Мониторьте V-Q разрыв.** Логируйте `v_q_gap = E[Q(s,a) - V(s)]` по датасетным парам. Должно быть слегка положительным (V чуть ниже среднего Q). Если становится сильно отрицательным — $\tau$ слишком мало.

**Используйте таргет-сеть для V в Q-обновлении.** В `iql_q_loss` используется `V_tgt`, а не `V`. Если использовать живую `V`, Q и V формируют цикличную зависимость и обучение часто расходится.

**IQL быстрее сходится, чем CQL** на плотных датасетах — V-обновление очень стабильно (нет OOD-сэмплирования, нет logsumexp).

---

## Ограничения

**Не может улучшить результат сверх лучших действий в датасете.** Политика IQL — взвешенное среднее действий датасета. Она не может открыть действия лучше тех, что поведенческая политика когда-либо пробовала.

**Два гиперпараметра для настройки.** $\tau$ и $\beta$ взаимодействуют. Большое $\tau$ → высокие advantages → нужно большее $\beta$ для их извлечения.

**Детерминированная политика** может плохо работать в мультимодальных средах с бимодальным распределением оптимальных действий.

---

## Итоги

| Свойство | IQL |
|---|---|
| Необходимые данные | $(s, a, r, s')$ с наградами |
| Ключевая идея | Expectile regression для $V(s)$; TD с $V(s')$; AWR-политика |
| OOD-запросы | Никогда — полностью in-sample |
| Ключевые гиперпараметры | $\tau$ (экспектиль, 0.5–0.9) и $\beta$ (температура AWR) |
| По сравнению с CQL | Стабильнее; нет OOD-сэмплирования; детерминированная политика |
| Ограничение | Не может экстраполировать за пределы действий датасета |

IQL — наиболее чистое решение задачи офлайн RL среди безмодельных методов: пессимизм структурный, встроенный в архитектуру. Следующий шаг: научиться моделировать мир и генерировать синтетические данные. Это позволяет рассуждать о переходах, которых нет в датасете — ценой ошибки модели. Об этом — в главе 5.

---

## Литература

- Kostrikov, I., Nair, A., & Levine, S. (2022). *Offline Reinforcement Learning with Implicit Q-Learning.* ICLR. [arXiv:2110.06169](https://arxiv.org/abs/2110.06169).
- Peng, X., Kumar, A., Zhang, G., & Levine, S. (2019). *Advantage-Weighted Regression.* [arXiv:1910.00177](https://arxiv.org/abs/1910.00177).
- Newey, W., & Powell, J. (1987). *Asymmetric Least Squares Estimation and Testing.* Econometrica.
- Kumar, A. et al. (2020). *Conservative Q-Learning for Offline RL.* NeurIPS. [arXiv:2006.04779](https://arxiv.org/abs/2006.04779).
- Levine, S. et al. (2020). *Offline RL: Tutorial, Review, and Perspectives.* [arXiv:2005.01643](https://arxiv.org/abs/2005.01643).

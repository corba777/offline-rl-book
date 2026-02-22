---
layout: default
title: "Глава 3: Консервативное Q-обучение (CQL)"
lang: ru
permalink: "/offline-rl-book/ru/chapter3/"
---

# Глава 3: Консервативное Q-обучение (CQL)

> *«Не доверяй значениям, которых не видел. А если не видел — занижай их.»*

---

## Проблема, переформулированная

В главе 2 мы увидели, что стандартное Q-обучение на офлайн данных порождает катастрофически оптимистичные Q-значения для действий вне распределения (OOD). Жадная политика эксплуатирует эти завышенные значения, выбирая действия, которых в датасете никогда не было — и терпит неудачу при развёртывании.

Корень проблемы: обновление Беллмана использует $\max_{a'} Q(s', a')$, перебирая все действия, включая ненаблюдавшиеся. Для таких действий Q-функция обобщается оптимистично.

**Conservative Q-Learning (CQL)** — Kumar et al., NeurIPS 2020 — исправляет это одной элегантной идеей: **добавить регуляризационный член, явно штрафующий Q-значения для действий вне датасета**.

Результат: Q-функция, которая по построению пессимистична для OOD-действий. Жадная политика, видя более низкие значения вне датасета, естественным образом остаётся близко к поведенческой политике — без явного ограничения.

---

## Идея

Стандартное TD-обучение минимизирует:

$$\mathcal{L}_{TD}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q_{\bar\theta}(s', a') - Q_\theta(s, a) \right)^2 \right]$$

CQL добавляет к этой цели два члена:

$$\mathcal{L}_{CQL}(\theta) = \mathcal{L}_{TD}(\theta) + \alpha \cdot \underbrace{\left( \mathbb{E}_{s \sim \mathcal{D},\, a \sim \mu} \left[ Q_\theta(s, a) \right] - \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ Q_\theta(s, a) \right] \right)}_{\text{консервативный штраф}}$$

где:
- $\mu$ — некоторое распределение над действиями (как правило, равномерное или текущая политика)
- $\alpha > 0$ — гиперпараметр, управляющий силой консерватизма
- Первое математическое ожидание **занижает Q-значения** для действий из $\mu$
- Второе математическое ожидание **завышает Q-значения** для действий из датасета

Иными словами: **минимизировать Q-значения везде, но максимизировать их в точках датасета**. Разрыв между двумя членами — то, что минимизирует CQL: действия из датасета выглядят лучше, чем OOD-действия.

---

## Формализация

### Целевая функция CQL

Точнее, CQL минимизирует следующую регуляризованную ошибку Беллмана:

$$\min_\theta \, \alpha \left( \mathbb{E}_{s \sim \mathcal{D}} \left[ \log \sum_a \exp Q_\theta(s, a) \right] - \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ Q_\theta(s, a) \right] \right) + \frac{1}{2} \mathcal{L}_{TD}(\theta)$$

Первый член — **log-sum-exp** по всем действиям — гладкая аппроксимация $\max_a Q(s,a)$. Он опускает всю Q-поверхность вниз. Второй член поднимает Q-значения именно в точках датасета.

Определим:

$$\mathcal{R}_{CQL}(\theta) = \mathbb{E}_{s \sim \mathcal{D}} \left[ \log \sum_a \exp Q_\theta(s, a) \right] - \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ Q_\theta(s, a) \right]$$

Тогда: $\mathcal{L}_{CQL}(\theta) = \frac{1}{2}\mathcal{L}_{TD}(\theta) + \alpha \cdot \mathcal{R}_{CQL}(\theta)$

### Почему Log-Sum-Exp?

Член logsumexp — это аппроксимация максимума через softmax:

$$\log \sum_a \exp Q(s, a) = \max_a Q(s, a) + \log \sum_a \exp(Q(s,a) - \max_a Q(s,a))$$

При температуре → 0 это сходится к $\max_a Q(s,a)$. При конечной температуре выражение дифференцируемо и штрафует всё распределение Q-значений, а не только максимум.

Для непрерывных пространств действий перебрать все $a$ невозможно. CQL аппроксимирует logsumexp через importance sampling:

$$\log \sum_a \exp Q(s, a) \approx \log \mathbb{E}_{a \sim \mu(a|s)} \left[ \frac{\exp Q(s,a)}{\mu(a|s)} \right]$$

где $\mu$ — предложенное распределение. На практике $\mu$ — либо равномерное над пространством действий, либо текущая политика $\pi_\theta$.

### Теоретическая гарантия

**Теорема (Kumar et al., 2020).** Пусть $\hat{Q}^\pi$ — Q-функция, обученная CQL, а $Q^\pi$ — истинная Q-функция политики $\pi$. Тогда:

$$\hat{Q}^\pi(s, a) \leq Q^\pi(s, a) \quad \forall (s, a) \in \mathcal{D}$$

при подходящих условиях на $\alpha$.

Иными словами: **CQL является нижней оценкой истинной Q-функции в точках датасета**. Политика, обученная на этой пессимистичной Q-функции, гарантированно не эксплуатирует завышенные значения.

Более практично, ожидаемая производительность политики удовлетворяет:

$$J(\hat\pi) \geq J(\pi_\beta) - \frac{\alpha}{1-\gamma} \cdot \mathbb{E}_{s \sim d^{\pi_\beta}} \left[ D_{CQL}(\hat\pi, \pi_\beta)(s) \right]$$

Граница говорит: CQL не хуже BC, с поправкой, пропорциональной отклонению политики от поведенческой.

---

## Реализация

> 📄 Полный код: [`cql.py`](../../code/cql.py)

### Сети

CQL обычно использует архитектуру в стиле SAC: два Q-networks (для снижения переоценки через double-Q), стохастический актор и регуляризацию энтропией.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),              nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head    = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def sample(self, state):
        h       = self.trunk(state)
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(-4, 2)
        std     = log_std.exp()
        eps     = torch.randn_like(mean)
        raw     = mean + std * eps
        action  = torch.tanh(raw)
        log_prob = (
            torch.distributions.Normal(mean, std).log_prob(raw)
            - torch.log(1 - action.pow(2) + 1e-6)
        ).sum(-1)
        return action, log_prob
```

### Потеря CQL

```python
def cql_loss(Q1, Q2, Q1_target, Q2_target, policy,
             states, actions, rewards, next_states, dones,
             alpha_cql=1.0, gamma=0.99, n_action_samples=10):
    """
    CQL loss = стандартная TD-потеря + консервативный штраф.
    alpha_cql: сила консерватизма. Больше = более пессимистично.
    """
    batch_size = states.shape[0]

    # ── 1. Стандартная TD-цель ────────────────────────────────────────────
    with torch.no_grad():
        next_actions, next_log_probs = policy.sample(next_states)
        q_next  = torch.min(Q1_target(next_states, next_actions),
                            Q2_target(next_states, next_actions))
        q_next -= 0.1 * next_log_probs    # SAC: энтропийный бонус
        td_target = rewards + gamma * (1 - dones) * q_next

    td_loss = F.mse_loss(Q1(states, actions), td_target)

    # ── 2. CQL штраф ──────────────────────────────────────────────────────
    # Случайные OOD-действия из равномерного распределения
    random_actions = torch.FloatTensor(
        batch_size * n_action_samples, actions.shape[-1]
    ).uniform_(-1, 1).to(states.device)

    # Действия текущей политики
    states_rep = states.unsqueeze(1).repeat(1, n_action_samples, 1).view(
        batch_size * n_action_samples, -1)
    policy_actions, _ = policy.sample(states_rep)

    # Q-значения для случайных и policy-действий (OOD)
    q_rand   = Q1(states.unsqueeze(1).repeat(1, n_action_samples, 1)
                  .view(batch_size * n_action_samples, -1),
                  random_actions).view(batch_size, n_action_samples)
    q_policy = Q1(states_rep, policy_actions).view(batch_size, n_action_samples)

    # Q-значения для действий из датасета
    q_data = Q1(states, actions)

    # logsumexp ≈ E_μ[Q(s,a)] — «опустить вниз»
    q_ood      = torch.cat([q_rand, q_policy], dim=1)
    logsumexp  = torch.logsumexp(q_ood, dim=1)

    # Штраф: OOD вниз, датасет вверх
    cql_penalty = (logsumexp - q_data).mean()

    return td_loss + alpha_cql * cql_penalty, td_loss.item(), cql_penalty.item()
```

### Автоматическая настройка alpha

```python
# Настройка alpha через двойственный градиентный спуск
log_alpha_cql = torch.zeros(1, requires_grad=True, device=device)
alpha_opt     = optim.Adam([log_alpha_cql], lr=1e-4)
target_penalty = -2.0   # τ: желаемое значение E_μ[Q] - E_D[Q]

# В шаге обновления:
alpha_cql  = log_alpha_cql.exp().item()
alpha_loss = -log_alpha_cql * (cql_penalty - target_penalty)
alpha_opt.zero_grad()
alpha_loss.backward()
alpha_opt.step()
```

---

## Гиперпараметр alpha

$\alpha$ — важнейший гиперпараметр CQL. Он управляет компромиссом между консерватизмом и производительностью:

| $\alpha$ | Поведение |
|---|---|
| $\alpha = 0$ | Стандартный SAC — нет консерватизма, эксплуатация OOD |
| $\alpha$ мало (0.1–1.0) | Мягкий консерватизм — допускает некоторое улучшение политики |
| $\alpha$ велико (5–10) | Сильный консерватизм — политика близка к $\pi_\beta$ |
| $\alpha \to \infty$ | Эквивалентно поведенческому клонированию |

---

## Интуиция: ландшафт Q-функции

Представьте Q-функцию как ландшафт над пространством состояние-действие. Стандартное Q-обучение формирует этот ландшафт только через точки данных — между ними ландшафт не ограничен и стремится расти из-за оптимистичного обобщения.

CQL добавляет гравитацию: тянет весь ландшафт вниз, пока TD-потеря удерживает его в точках данных. Результат — ландшафт, высокий в точках датасета и низкий везде остальном.

Политика, действующая жадно по этому ландшафту, предпочитает действия из датасета — не потому что её явно ограничили, а потому что ландшафт естественно направляет её туда.

---

## CQL vs TD3+BC

TD3+BC (Fujimoto & Gu, 2021) — более простая альтернатива, добавляющая BC-член прямо в потерю политики:

$$\pi^* = \arg\max_\pi \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ \lambda Q(s, \pi(s)) - (\pi(s) - a)^2 \right]$$

| | CQL | TD3+BC |
|---|---|---|
| Где живёт пессимизм | Q-функция | Целевая функция политики |
| OOD Q-значения | Явно занижаются | Не изменяются |
| Теоретическая гарантия | Да (нижняя оценка) | Нет |
| Гиперпараметры | $\alpha$ | $\lambda$ |
| Сложность реализации | Средняя | Низкая |

TD3+BC — хорошая отправная точка для детерминированных политик. CQL более строго обоснован и обычно сильнее на сложных задачах.

---

## Практические советы

**Нормализуйте наблюдения.** CQL чувствителен к масштабу. Всегда нормализуйте состояния по статистикам датасета.

**Начинайте с $\alpha = 1.0$.** Безопасное значение по умолчанию. Если политика слишком консервативна — уменьшайте. Если Q-значения расходятся — увеличивайте.

**Мониторьте CQL-штраф.** Логируйте `E_μ[Q] - E_D[Q]` во время обучения. Он должен быть положительным и стабильным. Если уходит в минус — $\alpha$ слишком мало.

**Совет для промышленных данных.** Если датасет содержит несколько режимов работы (остановленный/медленный/быстрый), рассмотрите обучение отдельных агентов CQL для каждого режима или добавление режима как признака состояния.

---

## Итоги

| Свойство | CQL |
|---|---|
| Необходимые данные | Переходы $(s, a, r, s')$ с наградами |
| Целевая функция | TD-потеря + CQL штраф (logsumexp − Q датасета) |
| Обработка OOD | Явная: Q-значения занижаются для OOD-действий |
| Теоретическая гарантия | Нижняя оценка истинной Q-функции в датасете |
| Ключевой гиперпараметр | $\alpha$ (сила консерватизма) |

CQL закрывает разрыв между поведенческим клонированием и полным офлайн RL. Он использует информацию о награде для улучшения над поведенческой политикой, а пессимизм относительно OOD-действий предотвращает катастрофические отказы стандартного Q-обучения.

Остающееся ограничение: CQL является **безмодельным**. Глава 4 (IQL) уточняет идею пессимизма по значениям. Глава 5 (MOPO) показывает, как обучение модели мира позволяет генерировать синтетические данные — расширяя эффективный датасет за пределы собранного.

---

## Литература

- Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). *Conservative Q-Learning for Offline Reinforcement Learning.* NeurIPS. arXiv:2006.04779.
- Fujimoto, S., & Gu, S. (2021). *A Minimalist Approach to Offline RL (TD3+BC).* NeurIPS. arXiv:2106.06860.
- Haarnoja, T. et al. (2018). *Soft Actor-Critic.* ICML. arXiv:1801.01290.
- Levine, S. et al. (2020). *Offline Reinforcement Learning: Tutorial, Review, and Perspectives.* arXiv:2005.01643.

---
layout: default
title: "Глава 5: Ограничение политики и Actor-Critic (TD3+BC, AWAC)"
lang: ru
en_url: /en/chapter5/
prev_chapter:
  url: /ru/chapter4/
  title: "Неявное Q-обучение (IQL)"
next_chapter:
  url: /ru/chapter6/
  title: "Decision Transformers"
permalink: "/offline-rl-book/ru/chapter5/"
---

# Глава 5: Ограничение политики и Actor-Critic (TD3+BC, AWAC)

> *«Держись того, что видел — но используй критика, чтобы опереться на лучшее из этого.»*

---

## Где заканчивается пессимизм по значениям

В главах 3 и 4 мы боролись с ошибкой экстраполяции, делая **функцию ценности** пессимистичной: CQL штрафует Q-значения за OOD-действия; IQL вообще не обращается к OOD, используя экспектильную регрессию и advantage-weighted регрессию.

Другой подход — оставить критика (Q или V) по сути без изменений и **ограничить или регуляризовать актора** так, чтобы обученная политика оставалась близка к поведенческой. Агент по-прежнему улучшается за счёт данных — критик указывает, какие действия в датасете были лучше — но политика не может уходить далеко в OOD-области.

Это семейство **policy-constraint** (или **actor-regularized**) offline RL. Методы являются **actor-critic**: обучаются и критик, и политика, но в целевую функцию политики явно входит член, притягивающий её к данным. Два распространённых метода: **TD3+BC** (минималистичный, детерминированный) и **AWAC** (advantage-weighted, in-sample обновления актора).

---

## TD3+BC: минималистичный подход с регуляризацией политики

**TD3+BC** (Fujimoto & Gu, NeurIPS 2021) добавляет к loss актора один член: loss поведенческого клонирования, штрафующий отклонение от действий в датасете. Идея: актор должен максимизировать Q и оставаться близко к действиям из данных.

### Идея

Целевая функция актора:

$$\pi^* = \arg\max_\pi \; \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ \lambda \, Q(s, \pi(s)) - \bigl(\pi(s) - a\bigr)^2 \right]$$

Первый член — использовать критика; второй — имитировать данные. Гиперпараметр $\lambda$ задаёт баланс. На практике Q нормализуют по батчу, чтобы оба члена были одного масштаба.

### Формализация

**Критик (Q):** как в TD3 — два Q-сети, target-сети, TD loss. **Актор:** детерминированная политика $\pi_\phi(s)$; loss выше. Теоретической гарантии (в отличие от CQL) нет — эмпирический, простой в реализации метод.

---

## AWAC: Advantage-Weighted Actor-Critic

**AWAC** (Nair et al., 2020) обновляет актора **только по данным**: политика подгоняется с весами по преимуществу (advantage), без сэмплирования из текущей политики.

$$\mathcal{L}_\pi(\phi) = -\mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ \exp\!\left( \frac{1}{\beta} \bigl( Q(s,a) - V(s) \bigr) \right) \cdot \log \pi_\phi(a | s) \right]$$

$A(s,a) = Q(s,a) - V(s)$ — преимущество действия $a$ в состоянии $s$. Экспоненциальный вес усиливает хорошие действия; $\beta$ — температура. OOD-запросов к актору нет.

---

## Реализация

> 📄 Полный код: [`td3bc.py`](https://github.com/corba777/offline-rl-book/blob/main/code/td3bc.py)

### TD3+BC: сети и loss актора

TD3+BC использует ту же архитектуру, что и TD3: детерминированный актор, две Q-сети, target-сети.

```python
class Actor(nn.Module):
    """Deterministic policy s -> a in [-1, 1]. Same as IQL DeterministicPolicy."""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh(),
        )
    def forward(self, state):
        return self.net(state)
    def act(self, state):
        with torch.no_grad():
            return self.forward(state).cpu().numpy().squeeze()


def td3bc_actor_loss(actor, Q1, states, actions, lambda_=0.25):
    """
    TD3+BC actor loss: maximize Q(s, pi(s)) - lambda * (pi(s) - a)^2.
    Q is normalized by (q - q.mean()) / (q.std() + eps) over the batch
    so the Q-term and BC-term have comparable scale.
    """
    pi = actor(states)
    q = Q1(states, pi)
    q_norm = (q - q.mean()) / (q.std() + 1e-6)
    bc_loss = ((pi - actions) ** 2).mean()
    return -q_norm.mean() * lambda_ + bc_loss
```

Оба члена (Q и BC) вносят вклад в градиент; в `td3bc.py` используется нормализация Q по батчу и фиксированный $\lambda$.

### AWAC-style loss политики (advantage-weighted)

```python
def awac_actor_loss(policy, Q, V, states, actions, beta=1.0):
    """
    Advantage-Weighted Regression: log pi(a|s) weighted by exp(A(s,a)/beta).
    A(s,a) = Q(s,a) - V(s). Requires stochastic policy that outputs log_prob.
    """
    with torch.no_grad():
        A = Q(states, actions) - V(states)
        weights = (A / beta).exp()
        weights = weights / (weights.mean() + 1e-6)  # stabilize
    log_prob = policy.log_prob(states, actions)
    return -(weights * log_prob).mean()
```

Для детерминированной политики (как в TD3) можно использовать гауссов с малой дисперсией вокруг $\pi(s)$ как суррогат log_prob или перейти на стохастическую голову.

---

## Ограничения

Гарантии нижней границы нет. Чувствительность к $\lambda$ / $\beta$. Детерминированная политика (TD3+BC) не может быть мультимодальной.

---

## Итог

| Метод | Где ограничение | OOD актор? | Теория |
|---|---|---|---|
| TD3+BC | Loss актора (BC-штраф) | Да | Нет |
| AWAC | Loss актора (веса по advantage) | Нет | Нет |

Методы с ограничением политики дают простой способ улучшаться над поведенческой политикой, оставаясь близко к данным. TD3+BC — самый простой в реализации; для максимальной устойчивости по-прежнему предпочтительны CQL и IQL.

Глава 6 переходит к другой парадигме: **Decision Transformers** — offline RL как моделирование последовательностей без Bellman-бэкапа.

---

## Литература

- Fujimoto, S., & Gu, S.S. (2021). *A Minimalist Approach to Offline Reinforcement Learning (TD3+BC).* NeurIPS. [arXiv:2106.06860](https://arxiv.org/abs/2106.06860).
- Nair, A. et al. (2020). *AWAC: Accelerating Online Reinforcement Learning with Offline Datasets.* [arXiv:2006.09359](https://arxiv.org/abs/2006.09359).
- Kumar, A. et al. (2020). *Conservative Q-Learning for Offline Reinforcement Learning (CQL).* NeurIPS. [arXiv:2006.04779](https://arxiv.org/abs/2006.04779).
- Kostrikov, I. et al. (2022). *Offline Reinforcement Learning with Implicit Q-Learning (IQL).* ICLR. [arXiv:2110.06169](https://arxiv.org/abs/2110.06169).

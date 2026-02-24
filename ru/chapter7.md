---
layout: default
title: "Глава 7: Decision Transformers"
lang: ru
en_url: /en/chapter7/
prev_chapter:
  url: /ru/chapter6/
  title: "Ограничение политики и Actor-Critic (TD3+BC, AWAC)"
next_chapter:
  url: /ru/chapter8/
  title: "Модельный Offline RL (MOPO, MOReL)"
permalink: "/offline-rl-book/ru/chapter7/"
---

# Глава 7: Decision Transformers

> *«Что если мы вообще не вычисляем Bellman-бэкап? Спросим: при данной истории и данном желаемом return — какое действие выбрал бы хороший агент?»*

---

## Другая парадигма

В главах 2–5 общим был костяк: **функция ценности** (Q или V) и **политика**, обучение с Bellman-бэкапами или policy gradient. Проблема — экстраполяционная ошибка; мы решали её пессимизмом (CQL, IQL) или ограничениями политики (TD3+BC, AWAC).

**Decision Transformer (DT)** (Chen et al., NeurIPS 2021) смотрит иначе: **offline RL как моделирование последовательностей**. По префиксу траектории (прошлые состояния, действия и сумма будущих наград) модель предсказывает следующее действие. **Нет Q-функции**, **нет Bellman-бэкапа**, **нет policy gradient**. «Политика» — условное распределение действий по контексту; обучение — **supervised learning** по последовательностям из датасета.

Так экстраполяционная ошибка обходится структурно: модель никогда не вычисляет $\max_{a'} Q(s', a')$ по OOD-действиям, потому что Q нет. Она только предсказывает действия по входам, встречавшимся в данных.

---

## Идея

Модель учит условное распределение:

$$\pi(a_t \mid s_{1:t}, a_{1:t-1}, R_{1:t})$$

где $R_t$ — **return-to-go** в момент $t$: сумма наград с $t$ до конца траектории. На **тесте** задаём целевой return $R^*$ (например, высокий перцентиль по датасету) и подаём его модели; она выдаёт действия, которые в данных соответствовали такому return.

**Почему нет экстраполяции:** модель не оценивает действия вне похожего контекста — только предсказывает следующее действие по (история состояний, история действий, return-to-go), всё это есть в данных.

---

## Формализация

**Представление:** траектория — последовательность $(s_1, a_1, r_1, R_1, \ldots, s_T, a_T, r_T, R_T)$. Контекст длины $K$: для шага $t$ вход — $(R_{t-K+1}, s_{t-K+1}, a_{t-K+1}, \ldots, R_t, s_t)$; цель — $a_t$.

**Архитектура:** GPT-подобный трансформер с каузальной маской; три потока (return-to-go, состояние, действие) эмбеддятся и проходят через блоки внимания.

**Обучение:** минимизация NLL действий по данным:

$$\mathcal{L} = -\mathbb{E}_{\tau \sim \mathcal{D}} \left[ \sum_{t=1}^{T} \log \pi_\theta(a_t \mid R_{1:t}, s_{1:t}, a_{1:t-1}) \right]$$

На тесте: инициализируем $R_1 = R^*$, на каждом шаге по текущему контексту получаем $a_t$, обновляем return-to-go и повторяем.

---

## Реализация

> 📄 Полный код: [`decision_transformer.py`](https://github.com/corba777/offline-rl-book/blob/main/code/decision_transformer.py)

### Эмбеддинги и модель

Чанки собираются в `ChunkDataset`; для каждой (траектория, шаг $t$) формируются массивы длины `context_len` по return-to-go, состояниям и действиям (действия до $t-1$; предсказываем $a_t$). Модель склеивает $(R, s, a)$ по шагам в один токен, эмбеддит одним линейным слоем, добавляет позиционное кодирование и прогоняет каузальный трансформер:

```python
class DecisionTransformer(nn.Module):
    """
    GPT-style model. Input: context_len tokens, each (R, s, a) concatenated and embedded.
    Output: predicted action for the last timestep.
    Causal mask: each position sees only past.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128, n_heads=4, n_layers=2, context_len=20):
        super().__init__()
        self.context_len = context_len
        self.token_dim = 1 + state_dim + action_dim
        self.embed = nn.Linear(self.token_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, context_len, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=0.1, activation='relu', batch_first=True, norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh(),
        )

    def _causal_mask(self, L, device):
        return torch.triu(torch.ones(L, L, device=device) * float('-inf'), diagonal=1)

    def forward(self, R_chunk, S_chunk, A_chunk):
        B, L, _ = R_chunk.shape
        tokens = torch.cat([R_chunk, S_chunk, A_chunk], dim=-1)
        x = self.embed(tokens) + self.pos_embed[:, :L]
        mask = self._causal_mask(L, x.device)
        x = self.transformer(x, mask=mask)
        return self.action_head(x[:, -1])
```

Цикл обучения: батч из `ChunkDataset`, forward, loss = MSE(predicted_a, target_a).

**Ключевые гиперпараметры:** нормализация return по макс. return в датасете, длина контекста, целевой $R^*$ на тесте (например, высокий перцентиль).

---

## Ограничения

Нет явного credit assignment (нет TD). Стандартный DT не «сшивает» куски траекторий. Чувствительность к выбору $R^*$ на тесте. Теоретической гарантии нет.

---

## Итог

| Свойство | Decision Transformer |
|---|---|
| Данные | Траектории с return-to-go |
| Обучение | Supervised (NLL действий по контексту + RTG) |
| OOD | Структурно: нет Q и $\max_a$ |
| Гиперпараметры | Длина контекста, целевой $R^*$, нормализация return |

Decision Transformers — альтернатива value-based offline RL: один supervised-объектив, без Bellman. Удобны при длинных траекториях и инфраструктуре под sequence modeling. Для непрерывного управления с сильными гарантиями по-прежнему стандарт — CQL, IQL и модельные методы (глава 7).

Глава 7 — **модельный** offline RL: обучение динамики и генерация синтетических данных с штрафами за неопределённость (MOPO, MOReL).

---

## Литература

- Chen, L. et al. (2021). *Decision Transformer: Reinforcement Learning via Sequence Modeling.* NeurIPS. [arXiv:2106.01345](https://arxiv.org/abs/2106.01345).
- Yamagata, T. et al. (2023). *Q-learning Decision Transformer.* ICML. [arXiv:2209.03993](https://arxiv.org/abs/2209.03993).
- Zheng, Q. et al. (2022). *Online Decision Transformer.* ICML. [arXiv:2202.05607](https://arxiv.org/abs/2202.05607).

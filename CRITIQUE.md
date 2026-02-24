# Критика контента книги "Offline RL: From Theory to Industrial Practice"

Краткий обзор прочитанного (EN/RU главы 1–11, код, README, index) и конструктивная критика.

---

## Сильные стороны

**Структура и стиль**
- Единая схема глав: идея → формализация → код → ограничения. Предсказуемо и полезно для практиков.
- Эпиграфы в начале глав задают тон и запоминаются.
- Математика аккуратная: обозначения согласованы ($\mathcal{D}$, $\pi_\beta$, $Q$, $V$), формулы соответствуют тексту.
- Связка с кодом: почти везде есть «Full code: `file.py`» и фрагменты, совпадающие с репозиторием (после недавних правок).

**Содержание**
- Глава 2 чётко ставит проблему экстраполяции и связывает её с границей $\hat{J} - J$ (Kumar et al.).
- CQL (гл. 3): логика log-sum-exp, importance sampling и коррекция $\log\pi$ объяснены понятно.
- IQL (гл. 4): expectile regression и три шага (V, Q, policy) изложены без лишнего шума.
- MOPO/MOReL (гл. 7): различие «penalize uncertainty» vs «hard boundary» показано на формулах и идее.
- Глава 8 (physics-informed): reward shaping, hybrid dynamics, residual target — последовательно и применимо.
- Глава 9 (industrial): задержки, интеграторы, ограничения, покрытие данных — реалистичный контекст.
- Глава 10 (SHAP): три уровня (Q, policy, dynamics), консистентность и физические знаки — полезная схема.
- Глава 11: честный разбор ограничений (distribution shift, reward spec, uncertainty) и актуальные направления (DT, diffusion, offline-to-online, LLM rewards).

**Код**
- Один и тот же `ThermalProcessEnv` в CQL, IQL, TD3+BC, DT — проще сравнивать алгоритмы.
- `cql.py`, `iql.py`, `td3bc.py`, `decision_transformer.py` — самодостаточные скрипты с `main()` и выводом.
- В книге фрагменты кода приведены к тем же именам и сигнатурам, что в файлах.

**Двуязычность**
- RU-версия по смыслу совпадает с EN; в Ch5/Ch6 добавлены те же блоки кода, что и в EN.

---

## Ошибки и несогласованности

**1. Неверные номера глав (после перенумерации)**

- **en/chapter2.md** (два места): написано «model-based (Chapter 5)» — модель-based методы теперь **Chapter 7**. Нужно заменить на «Chapter 7».
- **en/chapter2.md**: «This is the motivation for everything in Chapters 3–5» и «value-pessimism (Chapters 3–4) and model-based methods (Chapter 5)» — при текущей нумерации model-based это гл. 7; «Chapters 3–5» звучит как «только CQL, IQL, TD3+BC». Лучше: «Chapters 3–4» для value-pessimism и «Chapter 7» для model-based; мотивацию можно сформулировать как «для методов в главах 3–7» или оставить «3–5» только если явно иметь в виду «value + policy-constraint», но тогда про model-based отдельно указать гл. 7.
- **en/chapter1.md**: «motivates everything in Chapters 3–5» — экстраполяция мотивирует и гл. 7 (model-based). Имеет смысл заменить на «Chapters 3–7» или «the algorithms in the following chapters».
- **en/chapter9.md (критично)**: «`HybridMOReL` combines the model-based approach from Chapter 7 with the **hybrid dynamics model from Chapter 6**». Гибридная динамика вводится в **Chapter 8** (Physics-Informed), а не в Chapter 6 (Decision Transformers). Нужно заменить на **Chapter 8**.

**2. README устарел**

- В «Repository Structure» перечислены `chapter10.py`, `chapter11.py`; при необходимости добавить `td3bc.py`, `decision_transformer.py`, `chapter11_causal_toy.py`, `chapter11_toy_figures.py` и др.

**3. Мелкие несогласованности**

- Код кейс-стади Ch10: `code/chapter10.py`; explainability Ch11: `code/chapter11.py`, `chapter11_toy_figures.py`, `chapter11_causal_toy.py` (унифицировано под номера глав).

---

## Рекомендации по улучшению

**Содержание**
- **Глава 2**: таблица/схема «policy-constraint, value-pessimism, model-based, DT» уже есть; можно добавить одну общую таблицу «Глава ↔ метод» в конец гл. 2 или в начало книги, чтобы после перенумерации не путаться.
- **Главы 5–6 (RU)**: код добавлен; при желании можно добавить в RU те же таблицы гиперпараметров, что в EN (TD3+BC, AWAC, DT), для симметрии.
- **Глава 11**: раздел «Emerging Directions» очень плотный (DT, diffusion, offline-to-online, safe RL, LLM rewards). Можно разнести на подзаголовки с краткими абзацами или вынести часть в «Further reading», чтобы не перегружать заключение.

**Форма**
- Нумерация формул: в части глав формулы без номеров; для перекрёстных ссылок («как в (3.2)») можно ввести нумерацию (хотя для веб-книги это не обязательно).
- Рисунки: пути к `../figures/ch8/` в markdown корректны относительно `en/`; при сборке HTML нужно убедиться, что от `en/chapter10.html` до `figures/ch8/` путь один и тот же (например, от корня сайта).

**Код и воспроизводимость**
- В README можно добавить минимальный «Quick start»: какие зависимости (`torch`, `numpy`, и т.д.), одна команда для одной главы (например, `python code/cql.py`) и ожидаемый вывод (например, таблица с reward).
- Для `chapter10.py` и других тяжёлых скриптов — указать приблизительное время прогона или флаг `--quick` для короткого теста.

**Русский текст**
- Термины: «return-to-go», «advantage», «OOD» часто остаются в английском — это уместно для технической аудитории; при желании можно один раз в начале RU-части дать глоссарий (например, в ru.html или в первой главе).
- Проверить, что во всех RU-главах ссылки на «Chapter N» заменены на «Глава N» или оставлены осознанно (для единообразия).

---

## Резюме

Книга сильная: последовательная структура, аккуратная математика, рабочий код и честное обсуждение ограничений. Критичные правки: **исправить ссылку «hybrid dynamics from Chapter 6» на Chapter 8 в en/chapter9.md** и **обновить в en/chapter2.md «Chapter 5» на «Chapter 7» для model-based**. Остальное — обновление README, уточнение «Chapters 3–5» в ch1/ch2 и мелкие улучшения формы и воспроизводимости.

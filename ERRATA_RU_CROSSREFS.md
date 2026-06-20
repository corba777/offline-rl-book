# RU cross-reference audit — stale chapter numbers

The Russian translation is on an **older chapter numbering** (written before the OPE chapter was inserted as ch3 and before Decision Transformers became ch7). As a result almost every "Глава N / глав N–M" cross-reference points at the wrong chapter. The EN side is mostly correct, so each fix below is **anchored to the EN original**.

Correct current structure: **1** BC · **2** Problem · **3** OPE · **4** CQL · **5** IQL · **6** TD3+BC/AWAC · **7** DT · **8** MOPO/MOReL · **9** Physics · **10** Industrial · **11** Explainability · **12** Conclusion.

Line numbers are against the current `ru/*.md` (approximate after the F/P/S/T/X patch); the **wrong phrase** is the reliable locator. Each phrase appears identically in the matching `ru/*.html`, so every fix must be applied in **both** files.

Legend: 🔴 wrong chapter that misleads a reader · ⚪ EN-anchored, unambiguous.

---

## ru/chapter1.md
| ~Line | Now | → Correct | Basis (EN) |
|---|---|---|---|
| 272 | мотивирует всё содержание **глав 3–5** | **глав 3–9** | ch1: "Chapters 3–9" |

## ru/chapter2.md
| ~Line | Now | → Correct | Basis (EN) |
|---|---|---|---|
| 190 | **CQL (глава 3)** и **IQL (глава 4)** | CQL (глава **4**) и IQL (глава **5**) | CQL=4, IQL=5 |
| 194 | **модельные (глава 5)** методы | модельные (глава **8**) методы | ch2: model-based "Chapter 8" |
| 200 | модельные методы **(глава 5)** | (глава **8**) | ch2: "Chapter 8" |
| 223 | рассматривается в **главах 3–5** | в **главах 3–7** | ch2:241 "Chapters 3–7" |
| 236 | пессимизме по значениям **(главы 3–4)** и модельных методах **(глава 5)** | **(главы 4–5)** … **(глава 8)** | ch2:254 "value-pessimism (Chapters 4–5) and model-based methods (Chapter 8)" |

## ru/chapter4.md
| ~Line | Now | → Correct | Basis (EN) |
|---|---|---|---|
| 375 | более уместны модельные методы **(Глава 5)** | (Глава **8**) | ch4: "model-based methods (Chapter 8)" |
| 393 | **Глава 4 (IQL)** уточняет … **Глава 5 (MOPO)** показывает | **Глава 5 (IQL)** … **Глава 8 (MOPO)** | ch4: "Chapter 5 (IQL) … Chapter 8 (MOPO)" |

## ru/chapter5.md
| ~Line | Now | → Correct | Basis (EN) |
|---|---|---|---|
| 482 | Об этом — **в главе 5**. | в главе **8** | ch5: "the subject of Chapter 8" (self-ref to ch5 is clearly wrong) |

## ru/chapter6.md
| ~Line | Now | → Correct | Basis (EN) |
|---|---|---|---|
| 23 | **В главах 3 и 4** мы боролись с ошибкой экстраполяции | В главах **4 и 5** | ch6: "Chapters 4 and 5 addressed extrapolation error" |
| 133 | **Глава 6** переходит к другой парадигме: Decision Transformers | Глава **7** переходит | ch6: "Chapter 7 turns to a different paradigm" |

## ru/chapter7.md
| ~Line | Now | → Correct | Basis (EN) |
|---|---|---|---|
| 121 | стандарт — CQL, IQL и модельные методы **(глава 7)** | (глава **8**) | ch7: "model-based methods (Chapter 8)" |
| 123 | **Глава 7 — модельный** offline RL … (MOPO, MOReL) | Глава **8** — модельный | ch7: "Chapter 8 turns to model-based offline RL" |

## ru/chapter8.md
| ~Line | Now | → Correct | Basis (EN) |
|---|---|---|---|
| 317 | **Глава 8 развивает** эту идею в полноценную методологию | Глава **9** развивает | ch8: "Chapter 9 develops this idea" |
| 347 | Это тема **Главы 6: Physics-Informed Offline RL** | Главы **9** | ch8: "the subject of Chapter 9: Physics-Informed Offline RL" |

## ru/chapter9.md
| ~Line | Now | → Correct | Basis (EN) |
|---|---|---|---|
| 23 | **В главах 3–7** мы рассматривали мир как black-box | В главах **4–8** | ch9: "Chapters 4–8 treated the world as a black box" |
| 39 | слой поверх алгоритмов **из глав 3–5** | из глав **4–8** | offline-RL algorithm chapters = 4–8 |
| 158 | Ансамбль **из главы 5** — чистая нейронная сеть | из главы **8** | ch9: "In Chapter 8, the dynamics ensemble…" |
| 180 | NLL-потере ансамбля **из главы 5** | из главы **8** | ensemble = ch8 |
| 251 | drop-in замена `DynamicsEnsemble` **из главы 5** | из главы **8** | ensemble = ch8 |
| 298 | границей KNOWN/UNKNOWN из MOReL **(глава 5)** | (глава **8**) | MOReL = ch8 |
| 427 | **Глава 9 разбирает** промышленный кейс | Глава **10** разбирает | next chapter = Industrial (ch10) |

> Note on L427: the **EN side has the same bug** — `en/chapter9.md:439` says "Chapter 9 works through an industrial case study" (should be Chapter 10). This is errata item **Q**; fix EN and RU together.

## ru/chapter10.md (beyond the S/T already patched)
| ~Line | Now | → Correct | Basis (EN) |
|---|---|---|---|
| 267 | **CQL (глава 3)** добавляет консерватизм | CQL (глава **4**) | ch10: "CQL (Chapter 4)" |
| 280 | `PhysicsRewardWrapper` **(глава 6)** | (глава **9**) | ch10: "PhysicsRewardWrapper (Chapter 9)" |
| 309 | модельно-основанный подход **из главы 5** с гибридной … моделью **из главы 6** | из главы **8** … из главы **9** | ch10: "from Chapter 8 with the hybrid dynamics model from Chapter 9" |
| 399 | **Глава 6 показала** именно это | Глава **9** показала | ch10: "Chapter 9 showed exactly how to do this" |

## ru/chapter11.md
| ~Line | Now | → Correct | Basis (EN) |
|---|---|---|---|
| 23 | Алгоритмы **глав 1–9** строят политики | глав **1–10** | ch11:23 "algorithms in Chapters 1–10" |
| 306 | не заменяет метрики оценки **из главы 7** | из главы **10** | ch11:306 "evaluation metrics from Chapter 10" |
| 352 | метрики DA … **(главы 7)**, соблюдению ограничений **(глава 6)** … динамики **(глава 5)** | **(главы 10)** … **(глава 9)** … **(глава 8)** | metrics=ch10, constraints=ch9, dynamics=ch8 (inferred; no exact EN twin) |
| 354 | **Глава 9 завершает книгу** широким взглядом | Глава **12** завершает книгу | ch11:352 "Chapter 12 closes the book" |

## ru/chapter12.md
| ~Line | Now | → Correct | Basis (EN) |
|---|---|---|---|
| 27 | **CQL и IQL (главы 3–4)** | (главы **4–5**) | ch12: "Conservative Q-Learning and IQL (Chapters 4–5)" |
| 29 | **MOPO и MOReL (глава 7)** | (глава **8**) | ch12: "MOPO and MOReL (Chapter 8)" |
| 31 | Физически-информированные методы **(глава 8)** | (глава **9**) | ch12: "Physics-informed methods (Chapter 9)" |
| 33 | Промышленный кейс **(глава 9)** | (глава **10**) | ch12: "The industrial case study (Chapter 10)" |
| 41 | Теоретические гарантии **глав 3–8** | глав **4–9** | ch12:41 "guarantees from Chapters 4–9" |
| 45 | Ансамблевая неопределённость **(главы 7–8)** | (главы **8–9**) | ch12:45 "Ensemble-based uncertainty (Chapters 8–9)" |
| 47 | политики **из глав 1–9** — чёрные ящики | из глав **1–10** | ch12:47 "policies trained in Chapters 1–10" |
| 49 | Гибридные динамические модели **главы 8** | главы **9** | ch12: "hybrid dynamics models in Chapter 9" |
| 161 | Лагранжев подход **главы 6** | главы **9** | ch12:161 "The Lagrangian approach of Chapter 9" |
| 181 | метрики `IndustrialEvaluator` **из главы 7** | из главы **10** | IndustrialEvaluator = ch10 |

---

## Bonus (not chapter-number, but RU twins of known errata)

- **R-twin — phantom Theorem 6.1 (RU).** `ru/chapter10.md:296` (and the ru/ch12 roadmap) reuse «Теорема 6.1 / зазор оптимальности из Теоремы 6.1». No such theorem exists; it is the ch9 λ-calibration heuristic. Same fix as EN item R, in Russian.
- **Q-twin — ch9 closing (RU + EN).** Covered above (ru L427 / en:439): "Глава 9 / Chapter 9 разбирает industrial case study" → **Chapter 10**.

---

## Tally

**40** stale chapter cross-references in RU (ch1 ×1, ch2 ×5, ch4 ×2, ch5 ×1, ch6 ×2, ch7 ×2, ch8 ×2, ch9 ×7, ch10 ×4, ch11 ×4, ch12 ×10) + 2 RU twins (R, Q). Each must be applied in **both** `ru/chapterN.md` and `ru/chapterN.html`.

None of these affect the English edition except the shared **Q** bug at `en/chapter9.md:439`. The EN cross-references are otherwise correct — this drift is RU-only and points to the RU translation lagging a structural renumber.

**Recommendation:** this is large but fully mechanical now that each target is pinned. I can generate `errata_ru_crossrefs.patch` (+ `_html`) applying all 40 with unique per-line anchors, verified with `git apply --check`. Because a few items (ch9 L39, ch11 L352) are logic-inferred rather than EN-verbatim, worth a quick eyeball on those two before committing.

---
layout: default
title: "Offline RL: От теории к промышленной практике"
lang: ru
en_url: /
permalink: "/offline-rl-book/ru/"
---

# Offline RL: От теории к промышленной практике

Практическое руководство по офлайн обучению с подкреплением — для практиков, которые знают машинное обучение и хотят применять RL к реальным системам без возможности проводить живые эксперименты.

Книга следует единой нити: начинаем с простейшего подхода (поведенческое клонирование), разбираем точно, почему он не работает, затем постепенно строим инструменты для его улучшения — консервативное Q-обучение, неявное Q-обучение, модельные методы и физически-обоснованные ограничения.

Каждая глава построена по одной схеме: **идея → формализация → код → ограничения**.

---

## Содержание

<ul class="toc">
  <li>
    <a href="{{ '/ru/chapter1/' | relative_url }}">
      <span><span class="ch-num">Гл 1</span>Поведенческое клонирование</span>
      <span class="ch-status ready">готово</span>
    </a>
  </li>
  <li>
    <a href="{{ '/ru/chapter2/' | relative_url }}">
      <span><span class="ch-num">Гл 2</span>Задача Offline RL: ошибка экстраполяции</span>
      <span class="ch-status ready">готово</span>
    </a>
  </li>
  <li>
    <a href="#">
      <span><span class="ch-num">Гл 3</span>Консервативное Q-обучение (CQL)</span>
      <span class="ch-status">скоро</span>
    </a>
  </li>
  <li>
    <a href="#">
      <span><span class="ch-num">Гл 4</span>Неявное Q-обучение (IQL)</span>
      <span class="ch-status">скоро</span>
    </a>
  </li>
  <li>
    <a href="#">
      <span><span class="ch-num">Гл 5</span>Модельный Offline RL (MOPO, MOReL)</span>
      <span class="ch-status">скоро</span>
    </a>
  </li>
  <li>
    <a href="#">
      <span><span class="ch-num">Гл 6</span>Физически-обоснованный Offline RL</span>
      <span class="ch-status">скоро</span>
    </a>
  </li>
  <li>
    <a href="#">
      <span><span class="ch-num">Гл 7</span>Промышленные применения: Digital Twin и управление процессами</span>
      <span class="ch-status">скоро</span>
    </a>
  </li>
</ul>

---

## Код

Весь код глав находится в [`/code`](https://github.com/{{ site.github_username }}/{{ site.github_repo }}/tree/main/code).
Каждый файл самодостаточен и ссылается на него из текста главы.

## Для кого эта книга

Вы знаете обучение с учителем и имеете базовое представление об RL (MDP, Q-обучение, градиент политики).
Перечитывать Sutton & Barto не нужно — основы предполагаются известными.
Цель — пройти путь от «у меня есть датасет исторических решений» до «у меня есть развёртываемая политика управления» с чётким пониманием компромиссов на каждом шаге.

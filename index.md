---
layout: default
title: "Offline RL: From Theory to Industrial Practice"
lang: en
ru_url: /ru/
---

# Offline RL: From Theory to Industrial Practice

A practical guide to offline reinforcement learning — written for practitioners who know machine learning and want to apply RL to real-world systems without the ability to run live experiments.

The book follows a single thread: start from the simplest possible approach (behavioral cloning), understand precisely why it fails, then build up the tools to fix it — conservative Q-learning, implicit Q-learning, model-based methods, and physics-informed constraints.

Each chapter follows the same structure: **idea → formalization → code → limitations**.

---

## Contents

<ul class="toc">
  <li>
    <a href="{{ '/en/chapter1/' | relative_url }}">
      <span><span class="ch-num">Ch 1</span>Behavioral Cloning</span>
      <span class="ch-status ready">ready</span>
    </a>
  </li>
  <li>
    <a href="{{ '/en/chapter2/' | relative_url }}">
      <span><span class="ch-num">Ch 2</span>The Offline RL Problem: Extrapolation Error</span>
      <span class="ch-status ready">ready</span>
    </a>
  </li>
  <li>
    <a href="#">
      <span><span class="ch-num">Ch 3</span>Conservative Q-Learning (CQL)</span>
      <span class="ch-status">upcoming</span>
    </a>
  </li>
  <li>
    <a href="#">
      <span><span class="ch-num">Ch 4</span>Implicit Q-Learning (IQL)</span>
      <span class="ch-status">upcoming</span>
    </a>
  </li>
  <li>
    <a href="#">
      <span><span class="ch-num">Ch 5</span>Model-Based Offline RL (MOPO, MOReL)</span>
      <span class="ch-status">upcoming</span>
    </a>
  </li>
  <li>
    <a href="#">
      <span><span class="ch-num">Ch 6</span>Physics-Informed Offline RL</span>
      <span class="ch-status">upcoming</span>
    </a>
  </li>
  <li>
    <a href="#">
      <span><span class="ch-num">Ch 7</span>Industrial Applications: Digital Twin & Process Control</span>
      <span class="ch-status">upcoming</span>
    </a>
  </li>
</ul>

---

## Code

All chapter code lives in [`/code`](https://github.com/{{ site.github_username }}/{{ site.github_repo }}/tree/main/code).
Each file is self-contained and referenced inline from the chapter text.

## Who This Is For

You know supervised learning and have some exposure to RL (MDPs, Q-learning, policy gradients).
You don't need to re-read Sutton & Barto — the fundamentals are assumed.
The goal is to get from "I have a dataset of historical decisions" to "I have a deployable control policy" with a clear understanding of the tradeoffs at each step.

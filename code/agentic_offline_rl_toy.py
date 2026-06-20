"""
agentic_offline_rl_toy.py
=========================
Tabular toy environment: tool-using agent with calculator tools.
Referenced from Chapter 12 of "Offline RL: From Theory to Industrial Practice".

Demonstrates:
  logged traces → BC → naive FQI → support-constrained FQI
  and the OOD / unsupported-action problem when data lack coverage.

Usage:
    python agentic_offline_rl_toy.py
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

ACTIONS = ["lookup_x", "lookup_y", "add", "sub", "mul", "final"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
TASKS = ["add", "sub", "mul"]
STEP_COST = 0.02
GAMMA = 0.99
FQI_EPOCHS = 200

StateKey = Tuple[str, bool, bool, Optional[int]]  # (task, x_known, y_known, result or None)


@dataclass
class AgentState:
    task: str
    x: Optional[int] = None
    y: Optional[int] = None
    result: Optional[int] = None

    def key(self) -> StateKey:
        return (self.task, self.x is not None, self.y is not None, self.result)


@dataclass
class Transition:
    state: StateKey
    action: str
    reward: float
    next_state: StateKey
    done: bool


def true_answer(task: str, x: int, y: int) -> int:
    if task == "add":
        return x + y
    if task == "sub":
        return x - y
    return x * y


def correct_op(task: str) -> str:
    return {"add": "add", "sub": "sub", "mul": "mul"}[task]


def valid_actions(state: AgentState) -> List[str]:
    acts: List[str] = []
    if state.x is None:
        acts.append("lookup_x")
    if state.x is not None and state.y is None:
        acts.append("lookup_y")
    if state.x is not None and state.y is not None and state.result is None:
        acts.extend(["add", "sub", "mul"])
    if state.result is not None:
        acts.append("final")
    return acts


def step(state: AgentState, action: str, x_true: int, y_true: int) -> Tuple[AgentState, float, bool]:
    """Environment step. Returns (next_state, reward, done)."""
    if action not in valid_actions(state):
        return state, -1.0, True

    next_state = AgentState(task=state.task, x=state.x, y=state.y, result=state.result)
    reward = -STEP_COST

    if action == "lookup_x":
        next_state.x = x_true
    elif action == "lookup_y":
        next_state.y = y_true
    elif action in ("add", "sub", "mul"):
        assert next_state.x is not None and next_state.y is not None
        if action == "add":
            next_state.result = next_state.x + next_state.y
        elif action == "sub":
            next_state.result = next_state.x - next_state.y
        else:
            next_state.result = next_state.x * next_state.y
    elif action == "final":
        expected = true_answer(state.task, x_true, y_true)
        if state.result == expected:
            reward = 1.0 - STEP_COST
        else:
            reward = -1.0 - STEP_COST
        return next_state, reward, True

    return next_state, reward, False


def behavior_policy(state: AgentState, rng: random.Random, mul_in_support: bool) -> str:
    """Stochastic logging policy: sometimes redundant, sometimes wrong op."""
    valid = valid_actions(state)
    op = correct_op(state.task)

    if state.x is None:
        return "lookup_x" if rng.random() < 0.88 else rng.choice(valid)

    if state.y is None:
        return "lookup_y" if rng.random() < 0.88 else rng.choice(valid)

    if state.result is None:
        if state.task == "mul" and not mul_in_support:
            return rng.choice(["add", "sub"])
        if rng.random() < 0.62:
            return op
        return rng.choice(["add", "sub", "mul"] if mul_in_support else ["add", "sub"])

    return "final" if rng.random() < 0.90 else rng.choice(valid)


def collect_offline_dataset(
    n_episodes: int,
    seed: int,
    mul_in_support: bool,
) -> List[Transition]:
    rng = random.Random(seed)
    dataset: List[Transition] = []

    for _ in range(n_episodes):
        task = rng.choice(TASKS)
        x_true = rng.randint(1, 9)
        y_true = rng.randint(1, 9)
        state = AgentState(task=task)
        done = False

        while not done:
            action = behavior_policy(state, rng, mul_in_support=mul_in_support)
            next_state, reward, done = step(state, action, x_true, y_true)
            dataset.append(
                Transition(
                    state=state.key(),
                    action=action,
                    reward=reward,
                    next_state=next_state.key(),
                    done=done,
                )
            )
            state = next_state

    return dataset


def build_support(dataset: List[Transition]) -> Dict[StateKey, Set[str]]:
    support: Dict[StateKey, Set[str]] = defaultdict(set)
    for tr in dataset:
        support[tr.state].add(tr.action)
    return support


def train_bc(dataset: List[Transition]) -> Dict[StateKey, str]:
    counts: Dict[StateKey, Counter] = defaultdict(Counter)
    for tr in dataset:
        counts[tr.state][tr.action] += 1
    policy: Dict[StateKey, str] = {}
    for s, ctr in counts.items():
        policy[s] = ctr.most_common(1)[0][0]
    return policy


def state_from_key(key: StateKey) -> AgentState:
    return AgentState(
        task=key[0],
        x=1 if key[1] else None,
        y=1 if key[2] else None,
        result=key[3],
    )


def q_greedy_action(
    q: Dict[Tuple[StateKey, str], float],
    state: AgentState,
    allowed: Optional[Set[str]] = None,
) -> str:
    valid = valid_actions(state)
    if allowed is not None:
        cands = [a for a in valid if a in allowed]
        if not cands:
            cands = valid
    else:
        cands = valid
    return max(cands, key=lambda a: q[(state.key(), a)])


def train_fqi(
    dataset: List[Transition],
    support: Optional[Dict[StateKey, Set[str]]] = None,
) -> Tuple[Dict[StateKey, str], Dict[Tuple[StateKey, str], float]]:
    q: Dict[Tuple[StateKey, str], float] = defaultdict(float)

    for _ in range(FQI_EPOCHS):
        for tr in dataset:
            s, a, r, s_next, done = tr.state, tr.action, tr.reward, tr.next_state, tr.done
            if done:
                target = r
            else:
                next_valid = valid_actions(state_from_key(s_next))
                if support is not None:
                    cands = [a2 for a2 in next_valid if a2 in support.get(s_next, set())]
                    if not cands:
                        cands = next_valid
                else:
                    cands = next_valid
                target = r + GAMMA * max(q[(s_next, a2)] for a2 in cands)
            q[(s, a)] += 0.25 * (target - q[(s, a)])

    policy: Dict[StateKey, str] = {}
    all_states = {tr.state for tr in dataset} | {tr.next_state for tr in dataset}
    for s in all_states:
        dummy = state_from_key(s)
        if support is not None:
            allowed = support.get(s, set())
            if not allowed:
                continue
            policy[s] = q_greedy_action(q, dummy, allowed=allowed)
        else:
            policy[s] = q_greedy_action(q, dummy, allowed=None)
    return policy, q


def make_q_policy_fn(
    q: Dict[Tuple[StateKey, str], float],
    support: Optional[Dict[StateKey, Set[str]]] = None,
) -> Callable[[AgentState], str]:
    def act(state: AgentState) -> str:
        allowed = support.get(state.key()) if support is not None else None
        return q_greedy_action(q, state, allowed=allowed)

    return act


def make_policy_fn(
    policy: Dict[StateKey, str],
    fallback: Callable[[AgentState], str],
) -> Callable[[AgentState], str]:
    def act(state: AgentState) -> str:
        return policy.get(state.key(), fallback(state))

    return act


def fallback_valid(state: AgentState) -> str:
    return valid_actions(state)[0]


def evaluate_policy(
    policy_fn: Callable[[AgentState], str],
    n_episodes: int = 500,
    seed: int = 0,
    tasks: Optional[List[str]] = None,
) -> float:
    rng = random.Random(seed)
    task_pool = tasks or TASKS
    successes = 0
    for _ in range(n_episodes):
        task = rng.choice(task_pool)
        x_true = rng.randint(1, 9)
        y_true = rng.randint(1, 9)
        state = AgentState(task=task)
        done = False
        success = False
        while not done:
            action = policy_fn(state)
            state, reward, done = step(state, action, x_true, y_true)
            if done and reward > 0.5:
                success = True
        if success:
            successes += 1
    return successes / n_episodes


def behavior_success_rate(n_episodes: int = 500, seed: int = 0, mul_in_support: bool = True) -> float:
    rng = random.Random(seed)
    successes = 0
    for _ in range(n_episodes):
        task = rng.choice(TASKS)
        x_true = rng.randint(1, 9)
        y_true = rng.randint(1, 9)
        state = AgentState(task=task)
        done = False
        success = False
        while not done:
            action = behavior_policy(state, rng, mul_in_support=mul_in_support)
            state, reward, done = step(state, action, x_true, y_true)
            if done and reward > 0.5:
                success = True
        if success:
            successes += 1
    return successes / n_episodes


def run_experiment(label: str, mul_in_support: bool, seed: int = 42) -> None:
    print(f"\nDataset: {label}")
    dataset = collect_offline_dataset(n_episodes=3000, seed=seed, mul_in_support=mul_in_support)
    support = build_support(dataset)

    bc = train_bc(dataset)
    _, naive_q = train_fqi(dataset, support=None)
    _, constrained_q = train_fqi(dataset, support=support)

    bc_fn = make_policy_fn(bc, fallback_valid)
    naive_fn = make_q_policy_fn(naive_q, support=None)
    constrained_fn = make_q_policy_fn(constrained_q, support=support)

    beh = behavior_success_rate(mul_in_support=mul_in_support, seed=seed)
    bc_s = evaluate_policy(bc_fn, seed=seed + 2)
    naive_s = evaluate_policy(naive_fn, seed=seed + 3)
    constr_s = evaluate_policy(constrained_fn, seed=seed + 4)

    mul_states = [s for s in support if s[0] == "mul" and s[1] and s[2] and s[3] is None]
    if mul_states:
        print(f"  mul-task op support (x,y known): { {s: sorted(support[s]) for s in mul_states} }")

    print(f"  Behavior success (all tasks):    {beh:.2f}")
    print(f"  BC success:                    {bc_s:.2f}")
    print(f"  Naive FQI success:             {naive_s:.2f}")
    print(f"  Support-constrained FQI:       {constr_s:.2f}")
    if not mul_in_support:
        mul_bc = evaluate_policy(bc_fn, seed=seed + 5, tasks=["mul"])
        mul_naive = evaluate_policy(naive_fn, seed=seed + 6, tasks=["mul"])
        mul_constr = evaluate_policy(constrained_fn, seed=seed + 7, tasks=["mul"])
        print(f"  --- mul tasks only ---")
        print(f"  BC / naive FQI / constrained:  {mul_bc:.2f} / {mul_naive:.2f} / {mul_constr:.2f}")


def main() -> None:
    print("Toy calculator agent — offline RL from logged tool traces")
    print("=" * 60)
    run_experiment("good coverage (mul in logs)", mul_in_support=True)
    run_experiment("no multiplication support", mul_in_support=False)
    print("\nQualitative takeaway:")
    print("  Good coverage: FQI can match or beat BC by stitching successful traces.")
    print("  No mul support: constrained FQI cannot solve mul (unsupported op);")
    print("  naive FQI may pick unseen mul anyway — extrapolation that works in sim")
    print("  but would lack evidential support in a real offline deploy.")


if __name__ == "__main__":
    main()

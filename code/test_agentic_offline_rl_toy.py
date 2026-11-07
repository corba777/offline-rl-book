"""Regression checks for calculator state identity and support constraints."""
import unittest
from collections import defaultdict

import agentic_offline_rl_toy as toy


class CalculatorPolicyTests(unittest.TestCase):
    def test_state_preserves_operands_and_terminal_outcome(self):
        correct = toy.AgentState("mul", 2, 3, 6)
        incorrect = toy.AgentState("mul", 1, 9, 6)
        self.assertNotEqual(correct.key(), incorrect.key())
        self.assertEqual(toy.state_from_key(correct.key()), correct)
        self.assertGreater(toy.step(correct, "final", 2, 3)[1], 0)
        self.assertLess(toy.step(incorrect, "final", 1, 9)[1], 0)

    def test_unknown_or_empty_support_abstains(self):
        state = toy.AgentState("mul", 2, 3)
        for support in ({}, {state.key(): set()}, {state.key(): {"final"}}):
            policy = toy.make_q_policy_fn(defaultdict(float), support)
            self.assertIsNone(policy(state))
        self.assertEqual(toy.evaluate_policy(lambda state: None, n_episodes=3), 0)

    def test_unsupported_high_value_action_is_excluded(self):
        state = toy.AgentState("mul", 2, 3)
        q = defaultdict(float, {(state.key(), "mul"): 100.0})
        policy = toy.make_q_policy_fn(q, {state.key(): {"add", "sub"}})
        self.assertIn(policy(state), {"add", "sub"})

    def test_missing_bootstrap_support_is_rejected(self):
        state = toy.AgentState("mul")
        next_state = toy.AgentState("mul", 2)
        transition = toy.Transition(state.key(), "lookup_x", -0.02, next_state.key(), False)
        with self.assertRaises(ValueError):
            toy.train_fqi([transition], {state.key(): {"lookup_x"}})

    def test_unseen_tools_tie_and_one_of_them_is_harmful(self):
        import random
        state = toy.AgentState("mul", 2, 3)
        q = defaultdict(float, {(state.key(), "add"): -0.5, (state.key(), "sub"): -0.5})
        picks = {toy.q_greedy_action(q, state, tie_rng=random.Random(k)) for k in range(30)}
        self.assertEqual(picks, {"mul", "estimate"})
        after, _, _ = toy.step(state, "estimate", 2, 3)
        self.assertLess(toy.step(after, "final", 2, 3)[1], 0)


if __name__ == "__main__":
    unittest.main()

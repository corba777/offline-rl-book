"""Chapter 10: the lagged observation, the physics prior and the oracle agree with the plant."""
import unittest

import numpy as np
import torch

import chapter10 as c10


def rollout(n=30, seed=0):
    env = c10.CoatingProcessEnv(noise_std=0.0, seed=seed, lagged_flow=True)
    rng = np.random.default_rng(seed)
    obs = env.reset(seed=seed)
    S, A, S2 = [], [], []
    for _ in range(n):
        a = rng.uniform(-1, 1, size=2).astype(np.float32)
        obs2, _, _, _ = env.step(a)
        S.append(obs); A.append(a); S2.append(obs2)
        obs = obs2
    return [torch.tensor(np.array(x), dtype=torch.float32) for x in (S, A, S2)]


class LaggedStateTests(unittest.TestCase):
    def test_lags_hold_the_feed_commands_in_transit(self):
        s, a, s2 = rollout()
        flow = a[:, 1] * 0.5 + 0.5
        self.assertTrue(torch.allclose(s2[:, 6], flow, atol=1e-6))        # u_f(t) enters last
        self.assertTrue(torch.allclose(s2[1:, 5], flow[:-1], atol=1e-6))  # then shifts

    def test_oracle_matches_the_noise_free_plant(self):
        s, a, s2 = rollout()
        pred = c10.coating_true_dynamics_torch(s, a)
        self.assertLess((pred - s2).abs().max().item(), 1e-5)

    def test_lagged_physics_gets_the_filler_delay_right(self):
        s, a, s2 = rollout()
        pred = c10.coating_physics_fn(s, a)
        # filler and level are exact; T, v and d miss the unmodeled terms only
        self.assertLess((pred[:, 1] - s2[:, 1]).abs().max().item(), 1e-5)
        self.assertLess((pred[:, 4] - s2[:, 4]).abs().max().item(), 1e-5)
        self.assertTrue(torch.allclose(pred[:, 5:], s2[:, 5:], atol=1e-6))


if __name__ == "__main__":
    unittest.main()

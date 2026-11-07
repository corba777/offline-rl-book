"""calibrate_lambda: uses the whole dataset and scales with the units of the constraint."""
import unittest

import numpy as np
import torch

import physics_informed as pi


def dataset(n=5000, violation_at=None, size=0.2):
    s = np.zeros((n, 1), dtype=np.float32)
    s2 = np.zeros((n, 1), dtype=np.float32)
    if violation_at is not None:
        s2[violation_at, 0] = size
    return {'states': s, 'actions': np.zeros((n, 1), dtype=np.float32), 'next_states': s2}


g = lambda s, a, s2: s2[:, 0].clamp(min=0.0)       # violation = positive next state
r = lambda s, a, s2: torch.ones(len(s))             # mean |r| = 1


class CalibrateLambdaTests(unittest.TestCase):
    def test_violation_at_the_end_of_the_logs_is_seen(self):
        lam = pi.calibrate_lambda(dataset(violation_at=4999), g, r, cost_ratio=10.0)
        self.assertAlmostEqual(lam, 10.0 / 0.2, places=4)

    def test_lambda_scales_inversely_with_constraint_units(self):
        lam1 = pi.calibrate_lambda(dataset(violation_at=10, size=0.2), g, r)
        lam2 = pi.calibrate_lambda(dataset(violation_at=10, size=2.0), g, r)
        self.assertAlmostEqual(lam1 / lam2, 10.0, places=4)

    def test_quantile_ignores_a_single_outlier(self):
        d = dataset()
        d['next_states'][:100, 0] = 0.1
        d['next_states'][100, 0] = 5.0                 # one outlier
        lam_max = pi.calibrate_lambda(d, g, r)
        lam_q = pi.calibrate_lambda(d, g, r, quantile=0.9)
        self.assertAlmostEqual(lam_q, 10.0 / 0.1, places=3)
        self.assertLess(lam_max, lam_q)

    def test_no_violation_falls_back_to_tol(self):
        lam = pi.calibrate_lambda(dataset(), g, r, tol=1e-3)
        self.assertAlmostEqual(lam, 10.0 / 1e-3, places=2)


if __name__ == "__main__":
    unittest.main()

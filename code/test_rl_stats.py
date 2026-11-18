"""Appendices B and C: the statistics helpers against values that can be checked by hand."""
import unittest

import numpy as np

import rl_stats as st


class TestRlStats(unittest.TestCase):
    def test_t_interval_matches_textbook(self):
        m, lo, hi = st.t_interval([1.0, 2.0, 3.0])
        self.assertAlmostEqual(m, 2.0)
        self.assertAlmostEqual(hi - m, 4.302653 * 1.0 / np.sqrt(3), places=5)   # t(0.975, 2) * s / sqrt(n)

    def test_iqm_drops_the_tails(self):
        x = [-224.0] + [-2.0] * 8 + [10.0]
        self.assertAlmostEqual(st.iqm(x), -2.0)

    def test_boot_iqm_is_paired_and_reproducible(self):
        a = np.linspace(-3, -1, 10)
        est, lo, hi = st.boot_iqm(a, a - 0.5)
        self.assertAlmostEqual(est, 0.5)
        self.assertAlmostEqual(lo, 0.5)          # a constant paired shift has no spread
        self.assertAlmostEqual(hi, 0.5)
        self.assertEqual(st.boot_iqm(a), st.boot_iqm(a))

    def test_clopper_pearson_zero_of_ten(self):
        lo, hi = st.clopper_pearson(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 1 - 0.025 ** (1 / 10), places=10)   # 30.8%

    def test_mcnemar_uses_only_discordant_seeds(self):
        fail_a = [True, True, True] + [False] * 17
        fail_b = [False] * 20
        n10, n01, two, one = st.mcnemar_exact(fail_a, fail_b)
        self.assertEqual((n10, n01), (3, 0))
        self.assertAlmostEqual(two, 0.25)
        self.assertAlmostEqual(one, 0.125)
        # Seeds that fail under both agents change nothing.
        self.assertEqual(st.mcnemar_exact(fail_a + [True], fail_b + [True])[2:], (two, one))

    def test_sign_test(self):
        wins, n, p = st.sign_test([1] * 7 + [0] * 3, [0] * 7 + [1] * 3)
        self.assertEqual((wins, n), (7, 10))
        self.assertAlmostEqual(p, 0.34375)

    def test_mcnemar_power(self):
        self.assertEqual(st.mcnemar_power(0.0, 0.0, 40), 0.0)        # no discordant seeds, no evidence
        self.assertEqual(st.mcnemar_power(0.5, 0.0, 5), 0.0)         # five discordant seeds cannot reach p < 0.05
        self.assertGreater(st.mcnemar_power(0.3, 0.0, 40), 0.95)

    def test_paired_t_seeds(self):
        n = st.paired_t_seeds(1.0, 1.0)
        self.assertTrue(9 <= n <= 11)                                 # textbook: about 10 for d = 1, 80% power
        self.assertGreater(st.paired_t_seeds(0.5, 1.0), n)

    def test_winners_curse(self):
        self.assertAlmostEqual(st.winners_curse(1), 0.0, places=2)
        self.assertAlmostEqual(st.winners_curse(2), 1 / np.sqrt(np.pi), places=2)   # E[max of 2 N(0,1)]


if __name__ == '__main__':
    unittest.main()

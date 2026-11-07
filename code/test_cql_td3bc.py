"""Numerical checks for the CQL soft-maximum estimator and the TD3+BC actor loss."""
import math
import unittest

import torch

import cql
import td3bc


def tanh_gaussian(n, mean, std, gen):
    """Samples from a tanh-squashed Gaussian and their log-density on [-1, 1]^d."""
    raw = mean + std * torch.randn(n, mean.shape[-1], generator=gen)
    a = torch.tanh(raw)
    log_p = (torch.distributions.Normal(mean, std).log_prob(raw)
             - torch.log(1 - a.pow(2))).sum(-1)
    return a, log_p


class CQLLogSumExpTests(unittest.TestCase):
    """For Q(a) = b + c·a on [-1, 1]^d, log ∫ exp Q da = b + Σ log(2 sinh(c_i) / c_i)."""

    def setUp(self):
        self.gen = torch.Generator().manual_seed(0)
        self.c = torch.tensor([1.5, -0.7])
        self.b = 0.3
        self.d = 2
        self.truth = self.b + sum(math.log(2 * math.sinh(ci) / ci) for ci in self.c.tolist())
        self.n = 200_000

    def q(self, a):
        return self.b + a @ self.c

    def test_mixture_estimator_matches_closed_form(self):
        a_u = torch.rand(self.n, self.d, generator=self.gen) * 2 - 1
        a_p, lp = tanh_gaussian(self.n, torch.tensor([0.4, -0.2]), torch.tensor([0.5, 0.8]), self.gen)
        est = cql.importance_sampled_logsumexp(
            self.q(a_u)[None], self.q(a_p)[None], lp[None], self.d).item()
        self.assertAlmostEqual(est, self.truth, delta=0.02)

    def test_each_proposal_alone_is_consistent(self):
        a_u = torch.rand(self.n, self.d, generator=self.gen) * 2 - 1
        unif = torch.logsumexp(self.q(a_u) + self.d * math.log(2), 0) - math.log(self.n)
        a_p, lp = tanh_gaussian(self.n, torch.zeros(2), torch.ones(2), self.gen)
        pol = torch.logsumexp(self.q(a_p) - lp, 0) - math.log(self.n)
        self.assertAlmostEqual(unif.item(), self.truth, delta=0.02)
        self.assertAlmostEqual(pol.item(), self.truth, delta=0.03)

    def test_omitting_uniform_density_shifts_by_d_log2(self):
        a_u = torch.rand(self.n, self.d, generator=self.gen) * 2 - 1
        uncorrected = torch.logsumexp(self.q(a_u), 0) - math.log(self.n)
        self.assertAlmostEqual(self.truth - uncorrected.item(), self.d * math.log(2), delta=0.02)


class CQLSmokeTest(unittest.TestCase):
    def test_one_update_is_finite(self):
        torch.manual_seed(0)
        agent = cql.CQLAgent(3, 2, hidden_dim=32)
        B = 64
        batch = (torch.randn(B, 3), torch.rand(B, 2) * 2 - 1, torch.randn(B),
                 torch.randn(B, 3), torch.zeros(B))
        info = agent.update(batch)
        for key in ("td", "cql", "pi"):
            self.assertTrue(math.isfinite(info[key]), key)


class TD3BCNormalizationTest(unittest.TestCase):
    def test_q_scale_receives_no_gradient(self):
        torch.manual_seed(0)
        actor = td3bc.Actor(3, 2, hidden_dim=32)
        critic = td3bc.QNetwork(3, 2, hidden_dim=32)
        s, a = torch.randn(32, 3), torch.rand(32, 2) * 2 - 1

        loss = td3bc.td3bc_actor_loss(actor, critic, s, a, lambda_=2.5)
        grads = torch.autograd.grad(loss, list(actor.parameters()))

        pi = actor(s)
        q = critic(s, pi)
        scale = q.abs().mean().item() + 1e-6          # constant, as in the reference code
        ref = -2.5 * (q / scale).mean() + ((pi - a) ** 2).mean()
        ref_grads = torch.autograd.grad(ref, list(actor.parameters()))
        for g, r in zip(grads, ref_grads):
            self.assertTrue(torch.allclose(g, r, atol=1e-6))


if __name__ == "__main__":
    unittest.main()

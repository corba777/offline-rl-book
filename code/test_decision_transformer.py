"""The Decision Transformer's training chunks must not reveal the action they ask for."""
import unittest

import numpy as np
import torch

import decision_transformer as dt


def toy_trajectory(T=12, s_dim=3, a_dim=2, seed=0):
    rng = np.random.default_rng(seed)
    return {'states': rng.normal(size=(T, s_dim)).astype(np.float32),
            'actions': rng.uniform(-1, 1, size=(T, a_dim)).astype(np.float32),
            'returns_to_go': -np.arange(T, 0, -1, dtype=np.float32)}


class ChunkLeakageTest(unittest.TestCase):
    def test_target_action_is_not_in_the_inputs(self):
        tr = toy_trajectory()
        ds = dt.ChunkDataset([tr], context_len=5)
        for idx in range(len(ds)):
            _, t = ds.indices[idx]
            R, S, A, target = ds[idx]
            self.assertTrue(torch.allclose(target, torch.from_numpy(tr['actions'][t])))
            self.assertTrue(torch.all(A[-1] == 0), "last position must not hold a_t")
            changed = {k: v.copy() for k, v in tr.items()}
            changed['actions'][t] += 10.0                          # alter only the target
            _, _, A2, _ = dt.ChunkDataset([changed], context_len=5)[idx]
            self.assertTrue(torch.equal(A, A2), "inputs changed with the target action")

    def test_negative_returns_scale_into_minus_one_zero(self):
        trs = [toy_trajectory(seed=s) for s in range(3)]
        dt.normalize_trajectories(trs)
        rtg = np.concatenate([tr['returns_to_go'] for tr in trs])
        self.assertGreaterEqual(rtg.min(), -1.0 - 1e-6)
        self.assertLessEqual(rtg.max(), 0.0)


if __name__ == '__main__':
    unittest.main()

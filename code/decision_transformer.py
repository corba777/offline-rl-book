"""
decision_transformer.py
======================
Decision Transformer (DT) for offline continuous control.
Referenced from Chapter 6 of "Offline RL: From Theory to Industrial Practice".

Chen et al., "Decision Transformer: Reinforcement Learning via Sequence Modeling", NeurIPS 2021.
arXiv:2106.01345

Idea: treat offline RL as sequence modeling. Condition on (return-to-go, state, action) history;
predict next action. No Q-function, no Bellman backup — supervised learning on trajectory chunks.

Contents:
    1. ThermalProcessEnv     — same toy environment as other chapters
    2. Dataset → trajectories with return-to-go (R_t)
    3. DecisionTransformer  — GPT-style causal transformer over (R, s, a) tokens
    4. Training: sample (traj, t), get chunk, predict a_t with MSE loss
    5. Eval: feed target return R*, autoregressively generate actions, update R each step
    6. run_demo() — train and evaluate

Usage:
    python decision_transformer.py
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict


# ============================================================================
# 1. ENVIRONMENT
# ============================================================================

class ThermalProcessEnv:
    """Toy thermal process. State: [T, filler, viscosity]; Action: [heat_delta, flow_delta] in [-1,1]."""

    def __init__(self, T_target=0.6, f_target=0.5, max_steps=200, noise_std=0.02, seed=42):
        self.T_target = T_target
        self.f_target = f_target
        self.max_steps = max_steps
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        self.state_dim = 3
        self.action_dim = 2
        self.alpha_T, self.beta_T = 0.92, 0.08
        self.alpha_f, self.beta_f = 0.88, 0.12
        self.state = None
        self.t = 0

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        T = np.clip(self.T_target + self.rng.normal(0, 0.1), 0, 1)
        f = np.clip(self.f_target + self.rng.normal(0, 0.1), 0, 1)
        v = self._viscosity(T, f)
        self.state = np.array([T, f, v], dtype=np.float32)
        self.t = 0
        return self.state.copy()

    def _viscosity(self, T, f):
        return float(np.clip(0.8 - 0.5 * T + 0.4 * f + 0.1 * f * (1 - T), 0, 1))

    def step(self, action):
        action = np.clip(action, -1, 1)
        T, f, _ = self.state
        heat_in = action[0] * 0.5 + 0.5
        flow_in = action[1] * 0.5 + 0.5
        T_new = np.clip(self.alpha_T * T + self.beta_T * heat_in
                        + self.rng.normal(0, self.noise_std), 0, 1)
        f_new = np.clip(self.alpha_f * f + self.beta_f * flow_in
                        + self.rng.normal(0, self.noise_std), 0, 1)
        v_new = self._viscosity(T_new, f_new)
        self.state = np.array([T_new, f_new, v_new], dtype=np.float32)
        self.t += 1
        reward = (-2.0 * (T_new - self.T_target)**2
                  - 2.0 * (f_new - self.f_target)**2
                  - 0.1 * float(np.sum(action**2)))
        done = self.t >= self.max_steps
        return self.state.copy(), float(reward), done, {}


# ============================================================================
# 2. TRAJECTORIES WITH RETURN-TO-GO
# ============================================================================

def pid_action(state, T_target, f_target):
    T, f, _ = state
    return np.array([
        np.clip(2.0 * (T_target - T), -1, 1),
        np.clip(2.0 * (f_target - f), -1, 1),
    ], dtype=np.float32)


def collect_trajectories(n_episodes=200, noise_scale=0.3, gamma=0.99, seed=0):
    """Collect trajectories; each is (states, actions, rewards, returns_to_go)."""
    env = ThermalProcessEnv(seed=seed)
    trajectories = []
    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        states, actions, rewards = [], [], []
        done = False
        while not done:
            act = pid_action(obs, env.T_target, env.f_target)
            act = np.clip(act + np.random.normal(0, noise_scale, 2), -1, 1).astype(np.float32)
            obs2, r, done, _ = env.step(act)
            states.append(obs.copy())
            actions.append(act.copy())
            rewards.append(r)
            obs = obs2
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        # Return-to-go: R_t = sum_{k=t}^{T-1} gamma^{k-t} r_k
        returns_to_go = np.zeros_like(rewards)
        g = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            g = rewards[t] + gamma * g
            returns_to_go[t] = g
        trajectories.append({
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'returns_to_go': returns_to_go,
        })
    all_returns = np.concatenate([tr['returns_to_go'][:1] for tr in trajectories])
    print(f"Collected {len(trajectories)} trajectories | "
          f"mean return={np.mean(all_returns):.3f} | max={np.max(all_returns):.3f}")
    return trajectories


def normalize_trajectories(trajectories):
    """Normalize states and scale returns to [0,1] by max return in dataset."""
    all_states = np.concatenate([tr['states'] for tr in trajectories])
    all_rtg = np.concatenate([tr['returns_to_go'] for tr in trajectories])
    s_mean = all_states.mean(0)
    s_std = all_states.std(0) + 1e-8
    rtg_max = float(np.abs(all_rtg).max()) + 1e-8
    for tr in trajectories:
        tr['states'] = (tr['states'] - s_mean) / s_std
        tr['returns_to_go'] = tr['returns_to_go'] / rtg_max
    return s_mean, s_std, rtg_max


# ============================================================================
# 3. CHUNK DATASET
# ============================================================================

class ChunkDataset(Dataset):
    """Sample (traj_idx, t) and return chunk of length context_len: (R, s, a) with target a_t."""

    def __init__(self, trajectories: List[Dict], context_len: int):
        self.trajectories = trajectories
        self.context_len = context_len
        self.indices = []
        for i, tr in enumerate(trajectories):
            T = len(tr['states'])
            for t in range(T):
                if t >= 0:
                    self.indices.append((i, t))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i, t = self.indices[idx]
        tr = self.trajectories[i]
        R = tr['returns_to_go']
        S = tr['states']
        A = tr['actions']
        start = max(0, t - self.context_len + 1)
        # Context: R[start:t+1], S[start:t+1], A[start:t] (we predict A[t])
        len_ctx = t - start + 1
        R_chunk = np.zeros((self.context_len, 1), dtype=np.float32)
        S_chunk = np.zeros((self.context_len, S.shape[1]), dtype=np.float32)
        A_chunk = np.zeros((self.context_len, A.shape[1]), dtype=np.float32)
        R_chunk[-len_ctx:] = R[start:t + 1].reshape(-1, 1)
        S_chunk[-len_ctx:] = S[start:t + 1]
        if t > start:
            A_chunk[-len_ctx:-1] = A[start:t]
        target_a = A[t]
        return (
            torch.FloatTensor(R_chunk),
            torch.FloatTensor(S_chunk),
            torch.FloatTensor(A_chunk),
            torch.FloatTensor(target_a),
        )


# ============================================================================
# 4. DECISION TRANSFORMER MODEL
# ============================================================================

class DecisionTransformer(nn.Module):
    """
    GPT-style model. Input: context_len tokens, each (R, s, a) concatenated and embedded.
    Output: predicted action for the last timestep (mean of Gaussian or deterministic).
    Causal mask: each position sees only past.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128, n_heads=4, n_layers=2, context_len=20):
        super().__init__()
        self.context_len = context_len
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Each token: (R: 1) + state + action = 1 + state_dim + action_dim
        self.token_dim = 1 + state_dim + action_dim
        self.embed = nn.Linear(self.token_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, context_len, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def _causal_mask(self, L, device):
        """Causal mask: position i can attend to j <= i."""
        return torch.triu(torch.ones(L, L, device=device) * float('-inf'), diagonal=1)

    def forward(self, R_chunk, S_chunk, A_chunk):
        """
        R_chunk: (B, context_len, 1)
        S_chunk: (B, context_len, state_dim)
        A_chunk: (B, context_len, action_dim) — last position can be dummy for target
        """
        B, L, _ = R_chunk.shape
        tokens = torch.cat([R_chunk, S_chunk, A_chunk], dim=-1)
        x = self.embed(tokens) + self.pos_embed[:, :L]
        mask = self._causal_mask(L, x.device)
        x = self.transformer(x, mask=mask)
        last_hidden = x[:, -1]
        return self.action_head(last_hidden)


# ============================================================================
# 5. TRAINING & EVAL
# ============================================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for R, S, A, target_a in loader:
        R, S, A, target_a = R.to(device), S.to(device), A.to(device), target_a.to(device)
        pred_a = model(R, S, A)
        loss = nn.functional.mse_loss(pred_a, target_a)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def evaluate_dt(env, model, target_return, s_mean, s_std, rtg_scale, context_len,
                device, max_steps=200, n_episodes=10, seed=100):
    """
    Evaluate: start with R_1 = target_return (normalized). At each step feed
    (R_1..R_t, s_1..s_t, a_1..a_{t-1}) and get a_t. Step env; R_{t+1} = R_t - r_t (normalized).
    """
    model.eval()
    returns_list = []
    with torch.no_grad():
        for ep in range(n_episodes):
            obs = env.reset(seed=seed + ep)
            total_r = 0.0
            R_history = []
            S_history = []
            A_history = []
            # Normalize initial return-to-go for model input
            R_current = target_return / rtg_scale
            for step in range(max_steps):
                s_norm = (obs - s_mean) / s_std
                R_history.append(R_current)
                S_history.append(s_norm)
                # Build chunk (pad to context_len)
                L = len(R_history)
                R_chunk = np.zeros((1, context_len, 1), dtype=np.float32)
                S_chunk = np.zeros((1, context_len, obs.shape[0]), dtype=np.float32)
                A_chunk = np.zeros((1, context_len, 2), dtype=np.float32)
                start = max(0, L - context_len)
                R_chunk[0, -L:] = np.array(R_history[start:], dtype=np.float32).reshape(-1, 1)
                S_chunk[0, -L:] = np.array(S_history[start:], dtype=np.float32)
                if len(A_history) > 0:
                    A_hist = A_history[start:]
                    A_chunk[0, -len(A_hist):] = np.array(A_hist, dtype=np.float32)
                R_t = torch.FloatTensor(R_chunk).to(device)
                S_t = torch.FloatTensor(S_chunk).to(device)
                A_t = torch.FloatTensor(A_chunk).to(device)
                a_pred = model(R_t, S_t, A_t)
                action = a_pred.cpu().numpy().squeeze()
                action = np.clip(action, -1, 1)
                obs, r, done, _ = env.step(action)
                total_r += r
                A_history.append(action)
                R_current = (R_current * rtg_scale - r) / rtg_scale
                if done:
                    break
            returns_list.append(total_r)
    return np.mean(returns_list), np.std(returns_list)


# ============================================================================
# 6. MAIN
# ============================================================================

def main():
    print("Decision Transformer (Chapter 6) — Offline RL via sequence modeling")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dim, action_dim = 3, 2
    context_len = 20
    hidden_dim = 128
    n_heads = 4
    n_layers = 2
    batch_size = 64
    epochs = 50
    gamma = 0.99

    trajectories = collect_trajectories(n_episodes=200, noise_scale=0.3, gamma=gamma, seed=0)
    s_mean, s_std, rtg_scale = normalize_trajectories(trajectories)
    dataset = ChunkDataset(trajectories, context_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DecisionTransformer(
        state_dim, action_dim,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        context_len=context_len,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    env = ThermalProcessEnv(seed=42)
    # Target return: high percentile of dataset initial returns
    all_R0 = np.concatenate([tr['returns_to_go'][:1] for tr in trajectories])
    target_return = float(np.percentile(all_R0, 90))
    print(f"Target return (90th pct, unnormalized): {target_return:.3f}")

    for epoch in range(epochs):
        loss = train_epoch(model, loader, optimizer, device)
        if (epoch + 1) % 10 == 0:
            mean_ret, std_ret = evaluate_dt(
                env, model, target_return, s_mean, s_std, rtg_scale, context_len,
                device, n_episodes=15, seed=100,
            )
            print(f"Epoch {epoch+1} | loss={loss:.4f} | eval return={mean_ret:.2f} ± {std_ret:.2f}")

    print("Done. DT learns to condition on return-to-go; at test use high R* for better behavior.")


if __name__ == "__main__":
    main()

"""
chapter8_causal_toy.py
======================
Minimal toy example for "Causal AI in Offline RL" (Chapter 8).

Idea: In offline data we often see *correlations* (e.g. a variable z that
correlates with the true cause s1). A model that relies on z (when s1 is
unobserved or ignored) can fail under intervention; a model that uses only
the true causal parents generalizes correctly.

True causal dynamics (unknown to the learner):
  next_s1 = 0.8*s1 + 0.2*a   (causal parents: s1, a)
  next_s2 = 0.6*s2 + 0.15*a   (causal parents: s2, a)

We add z = s1 + noise (z correlates with s1 but does NOT cause next_s1).
Suppose the "correlation" model never sees s1 and must use (s2, z, a) to
predict next_s1 (e.g. wrong feature set or unobserved s1). It fits
in-distribution because z ≈ s1. Under intervention (z=0.9, s1=0.2), it
predicts high next_s1; the causal model (s1, a) predicts correctly.

Two models:
  - "Correlation" model: predicts next_s1 from (s2, z, a) — z as proxy for s1.
  - "Causal" model: predicts next_s1 from (s1, a) only.

Usage (from repo root):
  python code/chapter8_causal_toy.py

Dependencies: numpy only.
"""

import numpy as np

# --- True causal dynamics (ground truth; learner does not see this) ---
def true_next_s1(s1: np.ndarray, a: np.ndarray) -> np.ndarray:
    return 0.8 * s1 + 0.2 * a

def true_next_s2(s2: np.ndarray, a: np.ndarray) -> np.ndarray:
    return 0.6 * s2 + 0.15 * a


def main():
    rng = np.random.default_rng(42)
    n = 500

    # --- Generate offline data ---
    s1 = rng.uniform(0.2, 0.8, size=n)
    s2 = rng.uniform(0.2, 0.8, size=n)
    a  = rng.uniform(0.0, 1.0, size=n)
    next_s1 = true_next_s1(s1, a)
    next_s2 = true_next_s2(s2, a)

    # z correlates with s1 but is NOT a cause of next_s1 (nuisance / proxy)
    z = s1 + 0.15 * rng.standard_normal(n)
    z = np.clip(z, 0, 1)

    # Correlation model: does not see s1, uses (s2, z, a) — learns z as proxy for s1
    X_corr = np.column_stack([s2, z, a])
    # Causal model: uses true causal parents (s1, a)
    X_causal = np.column_stack([s1, a])
    y = next_s1

    # --- Fit linear models (OLS) ---
    one_c = np.ones((n, 1))
    X_corr_bias = np.hstack([one_c, X_corr])
    w_corr, *_ = np.linalg.lstsq(X_corr_bias, y, rcond=None)
    X_causal_bias = np.hstack([one_c, X_causal])
    w_causal, *_ = np.linalg.lstsq(X_causal_bias, y, rcond=None)

    # --- In-distribution MSE ---
    pred_corr  = X_corr_bias @ w_corr
    pred_causal = X_causal_bias @ w_causal
    mse_corr_id   = np.mean((y - pred_corr) ** 2)
    mse_causal_id = np.mean((y - pred_causal) ** 2)
    print("In-distribution MSE (next_s1):")
    print(f"  Correlation model (s2,z,a): {mse_corr_id:.6f}")
    print(f"  Causal model (s1,a):       {mse_causal_id:.6f}")

    # --- Intervention: z high, s1 low (correlation model never saw this combination) ---
    s1_int, s2_int, z_int, a_int = 0.2, 0.5, 0.9, 0.5
    true_next_s1_int = true_next_s1(np.array([s1_int]), np.array([a_int]))[0]

    # Correlation model gets (s2, z, a) — no s1; it will predict high because z=0.9
    x_corr_int   = np.array([[1, s2_int, z_int, a_int]])
    x_causal_int = np.array([[1, s1_int, a_int]])
    pred_corr_int   = (x_corr_int @ w_corr)[0]
    pred_causal_int = (x_causal_int @ w_causal)[0]

    print("\nIntervention: s1=0.2, s2=0.5, z=0.9, a=0.5 (z high but s1 low)")
    print(f"  True next_s1:            {true_next_s1_int:.4f}")
    print(f"  Correlation model (s2,z,a): {pred_corr_int:.4f}  <- misled by z=0.9")
    print(f"  Causal model (s1,a):        {pred_causal_int:.4f}  <- correct")

    # --- Summary ---
    print("\nTakeaway: the correlation model uses z (proxy for s1) and is wrong")
    print("under intervention. The causal model uses (s1, a) and generalizes.")


if __name__ == "__main__":
    main()

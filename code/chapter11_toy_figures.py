"""
chapter11_toy_figures.py
=======================
Standalone toy examples for Chapter 8 explainability.
Generates figures that can be embedded in the book to illustrate SHAP
without running the full coating-process pipeline.

Usage (from repo root):
  python code/chapter11_toy_figures.py

Outputs:
  Toy Q-function:
    toy_shap_bar.png, toy_shap_waterfall.png, toy_shap_beeswarm.png
  Toy policy:
    toy_policy_bar.png, toy_policy_waterfall.png
  Toy dynamics:
    toy_dynamics_bar.png, toy_dynamics_waterfall.png

Dependencies: numpy, matplotlib, shap (same as chapter11.py).
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

# Output directory (relative to repo root when run as python code/chapter11_toy_figures.py)
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures', 'ch8')
os.makedirs(FIG_DIR, exist_ok=True)


def toy_q_function(X: np.ndarray) -> np.ndarray:
    """
    Toy Q(s, a) with known structure for illustration.
    Input X: (n, 3) — [state_1, state_2, action]
    Q = 2.0*s1 - 1.0*s2 + 0.5*a  (s1 pushes Q up, s2 down, action smaller effect).
    So we expect SHAP to attribute: s1 positive when s1 high, s2 negative when s2 high.
    """
    return (2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.5 * X[:, 2]).reshape(-1, 1)


def run_toy_shap():
    rng = np.random.default_rng(42)
    n_background = 80
    n_explain = 60
    feature_names = ['state_1', 'state_2', 'action']

    # Background: typical "operating" distribution (s1, s2, a around 0.5)
    bg = rng.uniform(0.3, 0.7, size=(n_background, 3))
    # Instances to explain: mix of normal and one "unusual" (high s1, low s2)
    X_explain = rng.uniform(0.2, 0.8, size=(n_explain, 3))
    X_explain[0] = [0.9, 0.2, 0.6]   # one instance we'll highlight

    # Wrap for SHAP (KernelExplainer expects f(X) -> array of shape (n,) or (n,1))
    def predict(X):
        return toy_q_function(X).ravel()

    explainer = shap.KernelExplainer(predict, bg)
    shap_vals = explainer.shap_values(X_explain, nsamples=100, silent=True)
    # shap_vals: (n_explain, 3)

    base_val = float(np.mean(predict(bg)))
    return {
        'shap_vals': shap_vals,
        'X_explain': X_explain,
        'feature_names': feature_names,
        'base_val': base_val,
        'predict': predict,
    }


def plot_toy_bar(data: dict, save_path: str) -> None:
    """Bar chart: mean |SHAP| per feature (which feature matters most on average)."""
    sv = data['shap_vals']
    names = data['feature_names']
    mean_abs = np.abs(sv).mean(0)
    order = np.argsort(mean_abs)[::-1]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar([names[i] for i in order], mean_abs[order],
                  color=['#4472c4', '#ed7d31', '#70ad47'], alpha=0.85, edgecolor='k', linewidth=0.5)
    ax.set_ylabel('Mean |SHAP value|', fontsize=10)
    ax.set_title('Toy Q(s₁, s₂, a): which input drives the output most?', fontsize=11, fontweight='bold')
    ax.set_ylim(0, None)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_toy_waterfall(data: dict, instance_idx: int = 0, save_path: str = None) -> None:
    """Waterfall for one instance: baseline → +φ₁ +φ₂ +φ₃ = prediction."""
    sv = data['shap_vals'][instance_idx]
    names = data['feature_names']
    base = data['base_val']
    pred = base + sv.sum()
    x = data['X_explain'][instance_idx]

    order = np.argsort(np.abs(sv))[::-1]
    sv_ord = sv[order]
    names_ord = [f"{names[i]}\n={x[i]:.2f}" for i in order]

    fig, ax = plt.subplots(figsize=(8, 2.5))
    cum = base
    colors = ['#2166ac' if v > 0 else '#d73027' for v in sv_ord]
    for i, (v, lbl) in enumerate(zip(sv_ord, names_ord)):
        ax.barh(0, v, left=cum, height=0.5, color=colors[i], alpha=0.9, edgecolor='k', linewidth=0.3)
        ax.text(cum + v / 2, 0, f'{v:+.3f}', ha='center', va='center', fontsize=10,
                color='white', fontweight='bold')
        cum += v
    ax.axvline(base, color='k', lw=1, ls='--', alpha=0.6, label=f'E[f]={base:.3f}')
    ax.axvline(pred, color='green', lw=1.5, alpha=0.8, label=f'Q(x)={pred:.3f}')
    ax.set_yticks([])
    ax.set_xlabel('Q-value', fontsize=10)
    ax.set_title(f'Toy instance: x = (s₁={x[0]:.2f}, s₂={x[1]:.2f}, a={x[2]:.2f})', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_toy_beeswarm(data: dict, save_path: str) -> None:
    """Beeswarm: each dot = one instance, x = SHAP value, color = feature value."""
    sv = data['shap_vals']
    names = data['feature_names']
    X = data['X_explain']
    mean_abs = np.abs(sv).mean(0)
    order = np.argsort(mean_abs)[::-1]

    fig, ax = plt.subplots(figsize=(7, 4))
    for row, feat_idx in enumerate(order):
        vals = sv[:, feat_idx]
        colors = X[:, feat_idx]
        y = np.random.uniform(row - 0.35, row + 0.35, size=len(vals))
        sc = ax.scatter(vals, y, c=colors, cmap='RdBu_r', s=18, alpha=0.8, vmin=0, vmax=1)
    ax.axvline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i] for i in order], fontsize=10)
    ax.set_xlabel('SHAP value', fontsize=10)
    ax.set_title('Toy Q: SHAP per instance (color = feature value)', fontsize=11, fontweight='bold')
    plt.colorbar(sc, ax=ax, label='Feature value', shrink=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# Toy policy: π(s) → a — which state feature drives the action?
# =============================================================================

def toy_policy(S: np.ndarray) -> np.ndarray:
    """
    Toy policy π(s1, s2) = clip(0.5 + 0.4*s1 - 0.3*s2).
    State_1 pushes action up, state_2 pushes it down.
    Input S: (n, 2). Output: (n,) scalar action in [0, 1].
    """
    return np.clip(0.5 + 0.4 * S[:, 0] - 0.3 * S[:, 1], 0.0, 1.0)


def run_toy_policy_shap():
    rng = np.random.default_rng(43)
    n_background = 80
    n_explain = 50
    state_names = ['state_1', 'state_2']

    bg = rng.uniform(0.3, 0.7, size=(n_background, 2))
    S_explain = rng.uniform(0.2, 0.8, size=(n_explain, 2))
    S_explain[0] = [0.85, 0.2]

    def predict(S):
        return toy_policy(S).ravel()

    explainer = shap.KernelExplainer(predict, bg)
    shap_vals = explainer.shap_values(S_explain, nsamples=100, silent=True)
    base_val = float(np.mean(predict(bg)))
    return {
        'shap_vals': shap_vals,
        'X_explain': S_explain,
        'feature_names': state_names,
        'base_val': base_val,
        'out_name': 'action',
    }


def plot_policy_bar(data: dict, save_path: str) -> None:
    """Bar: mean |SHAP| per state feature for the policy output."""
    sv = data['shap_vals']
    names = data['feature_names']
    mean_abs = np.abs(sv).mean(0)
    order = np.argsort(mean_abs)[::-1]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar([names[i] for i in order], mean_abs[order],
           color=['#4472c4', '#ed7d31'], alpha=0.85, edgecolor='k', linewidth=0.5)
    ax.set_ylabel('Mean |SHAP value|', fontsize=10)
    ax.set_title('Toy policy π(s₁, s₂): which state drives the action?', fontsize=11, fontweight='bold')
    ax.set_ylim(0, None)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_policy_waterfall(data: dict, instance_idx: int = 0, save_path: str = None) -> None:
    """Waterfall for one state: how do state features push action from E[π] to π(s)?"""
    sv = data['shap_vals'][instance_idx]
    names = data['feature_names']
    base = data['base_val']
    pred = base + sv.sum()
    s = data['X_explain'][instance_idx]
    order = np.argsort(np.abs(sv))[::-1]
    sv_ord = sv[order]
    fig, ax = plt.subplots(figsize=(6, 2.2))
    cum = base
    for i, v in enumerate(sv_ord):
        color = '#2166ac' if v > 0 else '#d73027'
        ax.barh(0, v, left=cum, height=0.5, color=color, alpha=0.9, edgecolor='k', linewidth=0.3)
        ax.text(cum + v / 2, 0, f'{names[order[i]]}\n{v:+.3f}', ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')
        cum += v
    ax.axvline(base, color='k', lw=1, ls='--', alpha=0.6, label=f'E[π]={base:.3f}')
    ax.axvline(pred, color='green', lw=1.5, alpha=0.8, label=f'π(s)={pred:.3f}')
    ax.set_yticks([])
    ax.set_xlabel('Action value', fontsize=10)
    ax.set_title(f'Toy policy instance: s = (s₁={s[0]:.2f}, s₂={s[1]:.2f})', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# Toy dynamics: f(s, a) → s' — which input drives the predicted next state?
# =============================================================================

def toy_dynamics_next_s1(X: np.ndarray) -> np.ndarray:
    """Next state dimension 1: next_s1 = 0.7*s1 + 0.2*a. (s1, s2, a) -> next_s1."""
    return (0.7 * X[:, 0] + 0.2 * X[:, 2]).reshape(-1, 1)


def toy_dynamics_next_s2(X: np.ndarray) -> np.ndarray:
    """Next state dimension 2: next_s2 = 0.6*s2 + 0.15*a."""
    return (0.6 * X[:, 1] + 0.15 * X[:, 2]).reshape(-1, 1)


def run_toy_dynamics_shap():
    rng = np.random.default_rng(44)
    n_background = 80
    n_explain = 50
    input_names = ['s₁', 's₂', 'a']

    bg = rng.uniform(0.3, 0.7, size=(n_background, 3))
    X_explain = rng.uniform(0.2, 0.8, size=(n_explain, 3))

    results = {'X_explain': X_explain}
    for out_name, fn in [('next_s1', toy_dynamics_next_s1), ('next_s2', toy_dynamics_next_s2)]:
        def predict(X, f=fn):
            return f(X).ravel()
        explainer = shap.KernelExplainer(lambda x: predict(x), bg)
        sv = explainer.shap_values(X_explain, nsamples=100, silent=True)
        results[out_name] = {
            'shap_vals': sv,
            'feature_names': input_names,
            'base_val': float(np.mean(predict(bg))),
        }
    return results


def plot_dynamics_bar(dynamics_data: dict, save_path: str) -> None:
    """Two panels: mean |SHAP| for next_s1 and next_s2 (which input drives each output?)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    names = dynamics_data['next_s1']['feature_names']
    for ax, out_name in zip(axes, ['next_s1', 'next_s2']):
        data = dynamics_data[out_name]
        mean_abs = np.abs(data['shap_vals']).mean(0)
        order = np.argsort(mean_abs)[::-1]
        ax.bar([names[i] for i in order], mean_abs[order],
               color=['#4472c4', '#ed7d31', '#70ad47'], alpha=0.85, edgecolor='k', linewidth=0.5)
        ax.set_ylabel('Mean |SHAP value|', fontsize=10)
        ax.set_title(f'Toy dynamics: what drives {out_name}?', fontsize=11, fontweight='bold')
        ax.set_ylim(0, None)
    plt.suptitle('Which inputs (s₁, s₂, a) drive each predicted next-state dimension?',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_dynamics_waterfall(dynamics_data: dict, output_name: str = 'next_s1',
                            instance_idx: int = 0, save_path: str = None) -> None:
    """Waterfall for one (s,a) instance: how do inputs add up to predicted next_s1 (or next_s2)?"""
    data = dynamics_data[output_name]
    X = dynamics_data['X_explain']
    sv = data['shap_vals'][instance_idx]
    names = data['feature_names']
    base = data['base_val']
    pred = base + sv.sum()
    x = X[instance_idx]
    order = np.argsort(np.abs(sv))[::-1]
    sv_ord = sv[order]
    fig, ax = plt.subplots(figsize=(6, 2.2))
    cum = base
    for i, (v, idx) in enumerate(zip(sv_ord, order)):
        color = '#2166ac' if v > 0 else '#d73027'
        ax.barh(0, v, left=cum, height=0.5, color=color, alpha=0.9, edgecolor='k', linewidth=0.3)
        ax.text(cum + v / 2, 0, f'{names[idx]}\n{v:+.3f}', ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')
        cum += v
    ax.axvline(base, color='k', lw=1, ls='--', alpha=0.6, label=f'E[{output_name}]={base:.3f}')
    ax.axvline(pred, color='green', lw=1.5, alpha=0.8, label=f'{output_name}={pred:.3f}')
    ax.set_yticks([])
    ax.set_xlabel(f'Predicted {output_name}', fontsize=10)
    ax.set_title(f'Toy dynamics instance: (s₁={x[0]:.2f}, s₂={x[1]:.2f}, a={x[2]:.2f})',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    print("Chapter 8 toy SHAP figures")
    print("  1. Toy Q(s1, s2, a) = 2*s1 - s2 + 0.5*a")
    data = run_toy_shap()
    plot_toy_bar(data, os.path.join(FIG_DIR, 'toy_shap_bar.png'))
    plot_toy_waterfall(data, instance_idx=0, save_path=os.path.join(FIG_DIR, 'toy_shap_waterfall.png'))
    plot_toy_beeswarm(data, os.path.join(FIG_DIR, 'toy_shap_beeswarm.png'))
    print("  2. Toy policy π(s1, s2) = clip(0.5 + 0.4*s1 - 0.3*s2)")
    policy_data = run_toy_policy_shap()
    plot_policy_bar(policy_data, os.path.join(FIG_DIR, 'toy_policy_bar.png'))
    plot_policy_waterfall(policy_data, instance_idx=0, save_path=os.path.join(FIG_DIR, 'toy_policy_waterfall.png'))
    print("  3. Toy dynamics: next_s1 = 0.7*s1 + 0.2*a, next_s2 = 0.6*s2 + 0.15*a")
    dynamics_data = run_toy_dynamics_shap()
    plot_dynamics_bar(dynamics_data, os.path.join(FIG_DIR, 'toy_dynamics_bar.png'))
    plot_dynamics_waterfall(dynamics_data, output_name='next_s1', instance_idx=0,
                            save_path=os.path.join(FIG_DIR, 'toy_dynamics_waterfall.png'))
    print("Done.")


if __name__ == '__main__':
    main()

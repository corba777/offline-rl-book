"""
chapter1_5_8_figures.py
=======================
Illustrative figures for Chapters 1, 5 and 8 (numpy + matplotlib only).

  figures/ch1/bc_drift.png         – compounding error of behavioral cloning
  figures/ch5/expectiles.png       – expectiles of five dataset Q-values
  figures/ch8/penalty_geometry.png – MOPO's soft vs MOReL's hard pessimism

Usage (from repo root):
    python code/chapter1_5_8_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INK, MUTED, BAND = "#1a1a2e", "#555570", "#e8e8e2"
EXPERT, BC = "#2563eb", "#b4441b"


def style(ax, xlabel, ylabel):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)


def save(fig, rel):
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("saved", rel)


# ── Chapter 1: BC drifts out of the data and does not come back ─────────────
def bc_drift():
    """
    1-D lateral offset with slightly unstable open-loop dynamics,
    x' = 1.05 x + 0.1 a + noise. The expert uses linear feedback a = -5x and
    stays in a narrow band around 0. BC is a cubic polynomial fitted to 150
    noisy expert labels: accurate inside the band, arbitrary outside it.
    Once a BC rollout leaves the band, nothing pulls it back.
    """
    rng = np.random.default_rng(5)
    A, K, noise, T = 1.05, 5.0, 0.02, 120

    def expert(x):
        return -K * x

    def rollout(policy, n):
        xs = np.zeros((n, T))
        for i in range(n):
            x = 0.0
            for t in range(T):
                xs[i, t] = x
                x = A * x + 0.1 * policy(x) + rng.normal(0, noise)
        return xs

    expert_runs = rollout(expert, 40)
    x_lab = rng.choice(expert_runs.ravel(), 150)
    a_lab = expert(x_lab) + rng.normal(0, 0.3, 150)
    coef = np.polyfit(x_lab, a_lab, deg=3)
    bc = lambda x: float(np.clip(np.polyval(coef, x), -5, 5))
    bc_runs = rollout(bc, 8)
    lo, hi = np.percentile(expert_runs.ravel(), [0.5, 99.5])

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    style(ax, "time step", "lateral offset  $x$")
    ax.axhspan(lo, hi, color=BAND, zorder=0)
    ax.text(T - 1, lo, "states in the dataset  ", ha="right", va="top", fontsize=9, color=MUTED)
    for i, run in enumerate(expert_runs[:8]):
        ax.plot(run, color=EXPERT, lw=1.1, alpha=0.8, label="expert" if i == 0 else None)
    for i, run in enumerate(bc_runs):
        ax.plot(run, color=BC, lw=1.6, label="behavioral cloning" if i == 0 else None)
    ax.set_xlim(0, T - 1)
    ax.set_ylim(-0.3, 1.0)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    save(fig, "figures/ch1/bc_drift.png")


# ── Chapter 5: expectiles of five Q-values ───────────────────────────────────
def expectile(values, tau, iters=2000):
    m = values.mean()
    for _ in range(iters):
        w = np.where(values > m, tau, 1 - tau)
        m = (w * values).sum() / w.sum()
    return m


def expectiles():
    q = np.array([-0.8, -0.3, 0.1, 0.4, 0.9])
    taus = np.linspace(0.01, 0.99, 197)
    curve = np.array([expectile(q, t) for t in taus])

    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
    style(ax, r"expectile level  $\tau$", r"$V(s)$ – the $\tau$-expectile")
    for v in q:
        ax.axhline(v, color=MUTED, lw=0.8, ls=":", alpha=0.7)
    ax.text(1.0, q[-1], " best dataset action", va="center", fontsize=8.5, color=MUTED)
    ax.text(1.0, q[0], " worst dataset action", va="center", fontsize=8.5, color=MUTED)
    ax.plot(taus, curve, color=EXPERT, lw=2.2)
    for tau, label in ((0.5, "mean"), (0.7, "IQL default")):
        v = expectile(q, tau)
        ax.plot([tau], [v], "o", color=BC if tau == 0.7 else INK, ms=6, zorder=3)
        ax.annotate(f"τ = {tau}: {label}, V = {v:.2f}", xy=(tau, v), xytext=(tau - 0.44, v + 0.32),
                    fontsize=9, color=BC if tau == 0.7 else INK,
                    arrowprops=dict(arrowstyle="->", lw=0.8, color=MUTED))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.95, 1.05)
    save(fig, "figures/ch5/expectiles.png")


# ── Chapter 8: soft vs hard pessimism ────────────────────────────────────────
def penalty_geometry():
    u = np.linspace(0, 1, 400)
    r_hat, lam, eps, kappa = 1.0, 1.5, 0.45, 2.0
    mopo = r_hat - lam * u
    morel = np.where(u <= eps, r_hat, -kappa)

    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
    style(ax, "model uncertainty  $u(s, a)$", "reward the policy is trained on")
    ax.axvspan(eps, 1, color=BAND, zorder=0)
    ax.text(eps + 0.01, 0.97, "UNKNOWN in MOReL", transform=ax.get_xaxis_transform(),
            va="top", fontsize=9, color=MUTED)
    ax.axhline(0, color=MUTED, lw=0.6, alpha=0.4)
    ax.plot(u, mopo, color=EXPERT, lw=2.2, label=r"MOPO: $\hat r - \lambda\, u$")
    ax.plot(u[u <= eps], morel[u <= eps], color=BC, lw=2.2, label=r"MOReL: $\hat r$, then $-\kappa$ and HALT")
    ax.plot(u[u > eps], morel[u > eps], color=BC, lw=2.2)
    ax.plot([eps, eps], [r_hat, -kappa], color=BC, lw=1, ls=":")
    ax.text(eps, -kappa - 0.18, r"$\varepsilon$", ha="center", fontsize=11, color=BC)
    ax.set_xlim(0, 1)
    ax.set_ylim(-kappa - 0.35, r_hat + 0.4)
    ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(0.0, 0.42))
    save(fig, "figures/ch8/penalty_geometry.png")


if __name__ == "__main__":
    bc_drift()
    expectiles()
    penalty_geometry()

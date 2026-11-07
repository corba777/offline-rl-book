"""
chapter2_4_figures.py
=====================
Figures for Chapter 2 (extrapolation error) and Chapter 4 (CQL).

Uses the toy environment of extrapolation_error.py: dataset actions lie in
[-0.5, 0.5], while the TD target maximizes over the full action space [-2, 2].
We plot a 1-D slice Q(s=0, a=(x, 0)) and compare it with the true optimal
Q* of this linear-quadratic problem (discounted LQR, unclipped actions).

Usage (from repo root):
    python code/chapter2_4_figures.py

Writes:
    figures/ch2/ood_q_slice.png
    figures/ch4/cql_vs_naive.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

import extrapolation_error as ee

ROOT = Path(__file__).resolve().parent.parent
N_CANDIDATES = 32
STEPS = 3000

INK, MUTED, BAND = "#1a1a2e", "#555570", "#e8e8e2"
NAIVE = ["#f2b38a", "#e07b45", "#b4441b"]   # light → dark: later training steps
CQL = "#2563eb"


def true_q_slice(xs: torch.Tensor) -> torch.Tensor:
    """Q*(0, (x, 0)) for the toy problem, from the discounted Riccati equation."""
    g, A, B, R, sig2 = ee.GAMMA, 0.9, 0.3, 0.1, 0.05 ** 2
    p = 1.0
    for _ in range(100_000):
        p_new = 1 + g * A * A * p - (g * A * p * B) ** 2 / (R + g * B * B * p)
        if abs(p_new - p) < 1e-12:
            break
        p = p_new
    p_unctrl = 1 / (1 - g * A * A)
    p_sum = 2 * p + 2 * p_unctrl            # two controlled + two uncontrolled coords
    c = g * sig2 * p_sum / (1 - g)          # V*(s) = -sum_i P_i s_i^2 - c
    return -(R + g * p * B * B) * xs ** 2 - g * (sig2 * p_sum + c)


def train(dataset, steps: int, alpha_cql: float = 0.0, snapshots=()):
    """Offline TD with max over the full action space; optional CQL penalty."""
    torch.manual_seed(0)
    Q = ee.QNetwork(ee.STATE_DIM, ee.ACTION_DIM, ee.HIDDEN)
    Q_tgt = ee.QNetwork(ee.STATE_DIM, ee.ACTION_DIM, ee.HIDDEN)
    Q_tgt.load_state_dict(Q.state_dict())
    opt = optim.Adam(Q.parameters(), lr=3e-4)
    n = dataset.s.shape[0]

    xs = torch.linspace(-2, 2, 161)
    a_slice = torch.stack([xs, torch.zeros_like(xs)], dim=1)
    s_slice = torch.zeros(len(xs), ee.STATE_DIM)
    curves = {}

    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (256,))
        s, a, r, s2 = dataset.s[idx], dataset.a[idx], dataset.r[idx], dataset.s_next[idx]
        s2_rep = s2.unsqueeze(1).expand(-1, N_CANDIDATES, -1).reshape(-1, ee.STATE_DIM)
        a_cand = torch.rand(256 * N_CANDIDATES, ee.ACTION_DIM) * 4 - 2
        with torch.no_grad():
            q_next = Q_tgt(s2_rep, a_cand).reshape(256, N_CANDIDATES).max(dim=1).values
            target = r + ee.GAMMA * q_next

        q_data = Q(s, a)
        loss = nn.MSELoss()(q_data, target)
        if alpha_cql > 0:
            s_rep = s.unsqueeze(1).expand(-1, N_CANDIDATES, -1).reshape(-1, ee.STATE_DIM)
            a_rand = torch.rand(256 * N_CANDIDATES, ee.ACTION_DIM) * 4 - 2
            lse = torch.logsumexp(Q(s_rep, a_rand).reshape(256, N_CANDIDATES), dim=1) \
                - torch.log(torch.tensor(float(N_CANDIDATES)))
            loss = loss + alpha_cql * (lse - q_data).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0:
            Q_tgt.load_state_dict(Q.state_dict())
        if step in snapshots:
            with torch.no_grad():
                curves[step] = Q(s_slice, a_slice).clone()
    return xs, curves


def style_axes(ax, xs):
    ax.axvspan(-0.5, 0.5, color=BAND, zorder=0)
    ax.text(0, 0.97, "actions in the dataset", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9, color=MUTED)
    ax.axhline(0, color=MUTED, lw=0.6, alpha=0.4)
    ax.set_xlim(float(xs[0]), float(xs[-1]))
    ax.set_xlabel("action  $a_1$   (with $s = 0$, $a_2 = 0$)", color=INK)
    ax.set_ylabel("$Q(s, a)$", color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)


def main() -> None:
    torch.manual_seed(42)
    dataset = ee.build_dataset(n=10_000)
    (ROOT / "figures/ch2").mkdir(parents=True, exist_ok=True)
    (ROOT / "figures/ch4").mkdir(parents=True, exist_ok=True)

    snaps = (1000, 2000, 3000)
    xs, naive = train(dataset, STEPS, alpha_cql=0.0, snapshots=snaps)
    _, cql = train(dataset, STEPS, alpha_cql=1.0, snapshots=(STEPS,))
    q_true = true_q_slice(xs)

    # ── Chapter 2: optimism grows outside the data and leaks inside ──────────
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    style_axes(ax, xs)
    for color, step in zip(NAIVE, snaps):
        ax.plot(xs, naive[step], color=color, lw=2, label=f"learned Q, {step} TD steps")
    ax.plot(xs, q_true, color=INK, lw=1.6, ls="--", label="true $Q^*$")
    top = naive[snaps[-1]]
    ax.annotate("highest far outside the data",
                xy=(float(xs[-1]), float(top[-1])), xytext=(0.75, float(top.max()) - 22),
                fontsize=9, color=NAIVE[-1],
                arrowprops=dict(arrowstyle="->", color=NAIVE[-1], lw=1))
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.0, 0.86))
    fig.tight_layout()
    fig.savefig(ROOT / "figures/ch2/ood_q_slice.png", facecolor="white")
    plt.close(fig)

    # ── Chapter 4: the same data, with and without the CQL penalty ───────────
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    style_axes(ax, xs)
    ax.plot(xs, naive[STEPS], color=NAIVE[-1], lw=2, label="naive offline Q-learning")
    ax.plot(xs, cql[STEPS], color=CQL, lw=2, label=r"CQL, $\alpha = 1$")
    ax.plot(xs, q_true, color=INK, lw=1.6, ls="--", label="true $Q^*$")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(ROOT / "figures/ch4/cql_vs_naive.png", facecolor="white")
    plt.close(fig)

    inside = xs.abs() <= 0.5
    for name, q in (("naive", naive[STEPS]), ("CQL", cql[STEPS]), ("true", q_true)):
        print(f"{name:>6}: in-data {q[inside].min():8.2f} .. {q[inside].max():8.2f} | "
              f"edges {q[0]:8.2f} {q[-1]:8.2f}")
    print("Saved figures/ch2/ood_q_slice.png and figures/ch4/cql_vs_naive.png")


if __name__ == "__main__":
    main()

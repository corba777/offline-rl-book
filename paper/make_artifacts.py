#!/usr/bin/env python3
"""
Build the paper's tables and figure from the saved results in code/results/.
Nothing is retrained: every number in the paper comes from these JSON files.

Usage (from the repo root):  python paper/make_artifacts.py
Writes paper/tables/*.tex and paper/figures/rollout_error.pdf
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "code" / "results"
OUT_T = Path(__file__).resolve().parent / "tables"
OUT_F = Path(__file__).resolve().parent / "figures"
FAIL = -10.0          # a seed "fails" when its mean return over 200 steps is below this


def load(name):
    return json.loads((RES / name).read_text())


def t_ci(values):
    v = np.asarray(values, dtype=float)
    half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), v.mean() - half, v.mean() + half


def per_seed(run_file, method, key="reward"):
    return [r["results"][method][key] for r in run_file["runs"]]


def paired(a_file, a_method, b_file, b_method, key="reward"):
    assert [r["seed"] for r in a_file["runs"]] == [r["seed"] for r in b_file["runs"]]
    return t_ci(np.array(per_seed(a_file, a_method, key)) - np.array(per_seed(b_file, b_method, key)))


def fmt(ci, d=2, sign=False):
    m, lo, hi = ci
    s = "+" if sign else ""
    return f"${m:{s}.{d}f}$ \\,[${lo:{s}.{d}f}$, ${hi:{s}.{d}f}$]"


def n_fail(run_file, method):
    return sum(x < FAIL for x in per_seed(run_file, method))


def write(name, body):
    OUT_T.mkdir(parents=True, exist_ok=True)
    (OUT_T / name).write_text(body)
    print("wrote", OUT_T / name)


def benchmark_table():
    logged, lagged = load("ch10_benchmark.json"), load("ch10_benchmark_lagged.json")
    rows = []
    for m, label in [("BC", "BC"), ("CQL", "CQL"), ("CQL+Physics", "CQL + bounds penalty"),
                     ("HybridMOReL", "HybridMOReL (HALT, $p_{80}$)")]:
        cells = [label]
        for f in (logged, lagged):
            cells += [fmt(t_ci(per_seed(f, m))), f"{np.mean(per_seed(f, m, 'L_iae')):.3f}",
                      str(n_fail(f, m))]
        rows.append(" & ".join(cells) + r" \\")
    write("benchmark.tex", "\n".join(rows) + "\n")


def factorial_table():
    o, lagged = load("ch10_benchmark_oracle.json"), load("ch10_benchmark_lagged.json")
    rows = []
    for m, dyn, halt in [("HybridMOReL", "learned hybrid", "on"),
                         ("HybridMOReL-noHALT", "learned hybrid", "off"),
                         ("OracleMOReL-HALT", "true", "on"),
                         ("OracleMOReL", "true", "off")]:
        rows.append(" & ".join([dyn, halt, fmt(t_ci(per_seed(o, m))),
                                f"{np.mean(per_seed(o, m, 'L_iae')):.3f}", str(n_fail(o, m)),
                                fmt(paired(o, m, lagged, "CQL"), sign=True)]) + r" \\")
    write("factorial.tex", "\n".join(rows) + "\n")
    d = paired(o, "OracleMOReL", o, "HybridMOReL-noHALT")
    write("factorial_true_vs_learned.tex", fmt(d, d=3, sign=True) + "\n")


def halt_table():
    hl, hg = load("ch10_halt_lagged.json"), load("ch10_halt_logged.json")
    bl, bg = load("ch10_benchmark_lagged.json"), load("ch10_benchmark.json")
    rows = []
    for m, label in [("HALT p80", r"$p_{80}$, worst-case $\kappa$"),
                     ("HALT p99", r"$p_{99}$, worst-case $\kappa$"),
                     ("HALT p99, data kappa", r"$p_{99}$, data $\kappa$"),
                     ("no HALT", "none (model-based SAC)")]:
        cells = [label]
        for h, b in ((hl, bl), (hg, bg)):
            if m in h["runs"][0]["results"]:
                key = {"HALT p80": "p80", "HALT p99": "p99", "no HALT": "HALT",
                       "HALT p99, data kappa": "p99 data kappa"}[m]
                rate = np.mean([r["extra"][f"halt_rate {key}"] for r in h["runs"]])
                cells += [fmt(t_ci(per_seed(h, m))), f"{100 * rate:.0f}\\%", str(n_fail(h, m)),
                          fmt(paired(h, m, b, "CQL"), sign=True)]
            else:
                cells += ["---"] * 4
        rows.append(" & ".join(cells) + r" \\")
    write("halt.tex", "\n".join(rows) + "\n")
    kap = np.mean([r["extra"]["kappa"] for r in hl["runs"]])
    kd = np.mean([r["extra"]["kappa_data"] for r in hl["runs"]])
    write("kappa_values.tex", f"\\newcommand{{\\kappaworst}}{{{kap:,.0f}}}\n\\newcommand{{\\kappadata}}{{{kd:.0f}}}\n")
    write("kappa_effect.tex", fmt(paired(hl, "HALT p99, data kappa", hl, "HALT p99"), sign=True) + "\n")


def fresh_table():
    h, b = load("ch10_halt_lagged_fresh.json"), load("ch10_benchmark_lagged_fresh.json")
    rows = []
    for f, m, label in [(b, "BC", "BC"), (b, "CQL", "CQL"), (b, "CQL+Physics", "CQL + bounds penalty"),
                        (h, "HALT p80", r"MOReL, $p_{80}$, worst-case $\kappa$"),
                        (h, "HALT p99", r"MOReL, $p_{99}$, worst-case $\kappa$"),
                        (h, "HALT p99, data kappa", r"MOReL, $p_{99}$, data $\kappa$"),
                        (h, "no HALT", "model-based SAC (no HALT)")]:
        vs = "---" if m == "CQL" else fmt(paired(f, m, b, "CQL"), sign=True)
        rows.append(" & ".join([label, fmt(t_ci(per_seed(f, m))), str(n_fail(f, m)), vs]) + r" \\")
    write("fresh.tex", "\n".join(rows) + "\n")


def calibration_tables():
    o = load("ch10_benchmark_oracle.json")
    rows = []
    for src, label in [("dataset actions", "logged"), ("policy actions", "trained policy"),
                       ("random actions", "uniform random")]:
        c = [r["extra"]["calibration"][src] for r in o["runs"]]
        g = lambda k: np.mean([x[k] for x in c])
        rows.append(f"{label} & {g('spearman'):.2f} & {g('err_low_u'):.3f} / {g('err_high_u'):.3f} & "
                    f"{g('err_known'):.3f} / {g('err_unknown'):.3f} & {100 * g('known_frac'):.0f}\\% \\\\")
    write("calibration.tex", "\n".join(rows) + "\n")
    rows = []
    for k in ["1", "2", "5", "10", "20"]:
        op = [r["extra"]["rollout_error"]["open loop"][k] for r in o["runs"]]
        cl = [r["extra"]["rollout_error"]["closed loop"][k] for r in o["runs"]]
        g = lambda rows_, q: np.mean([x[q] for x in rows_])
        rows.append(f"{k} & {g(op, 'rmse'):.3f} & {g(cl, 'rmse'):.3f} & {g(cl, 'mean_u'):.3f} & "
                    f"{100 * g(op, 'halted_by_k'):.0f}\\% / {100 * g(cl, 'halted_by_k'):.0f}\\% & "
                    f"{g(op, 'spearman_u_err'):.2f} / {g(cl, 'spearman_u_err'):.2f} \\\\")
    write("rollout.tex", "\n".join(rows) + "\n")


def rollout_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42})
    o = load("ch10_benchmark_oracle.json")
    ks = [1, 2, 5, 10, 20]
    get = lambda mode, q: [np.mean([r["extra"]["rollout_error"][mode][str(k)][q] for r in o["runs"]]) for k in ks]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.4))
    a.plot(ks, get("open loop", "rmse"), "o-", color="#b4441b", label="model error, open loop")
    a.plot(ks, get("closed loop", "rmse"), "s-", color="#2563eb", label="model error, closed loop")
    a.plot(ks, get("closed loop", "mean_u"), "^--", color="#1a1a2e", label="ensemble uncertainty $u$")
    a.set_xlabel("rollout step $k$"); a.set_ylabel("normalized units"); a.set_xscale("log")
    a.set_xticks(ks); a.set_xticklabels(ks); a.legend(frameon=False, fontsize=7)
    b.plot(ks, get("open loop", "spearman_u_err"), "o-", color="#b4441b", label="open loop")
    b.plot(ks, get("closed loop", "spearman_u_err"), "s-", color="#2563eb", label="closed loop")
    b.axhline(0, color="0.6", lw=0.8)
    b.set_xlabel("rollout step $k$"); b.set_ylabel("Spearman($u$, error)"); b.set_xscale("log")
    b.set_xticks(ks); b.set_xticklabels(ks); b.legend(frameon=False, fontsize=7)
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    OUT_F.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_F / "rollout_error.pdf")
    print("wrote", OUT_F / "rollout_error.pdf")


if __name__ == "__main__":
    benchmark_table()
    factorial_table()
    halt_table()
    fresh_table()
    calibration_tables()
    rollout_figure()

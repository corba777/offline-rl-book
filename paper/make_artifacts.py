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


def iqm(run_file, method):
    """Interquartile mean over seeds (Agarwal et al., 2021): with ten seeds, the
    mean of the middle six, robust to the one or two failing seeds."""
    return f"${stats.trim_mean(per_seed(run_file, method), 0.25):.2f}$"


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
                     ("HybridMOReL", "MOReL-style (HALT, $p_{80}$)")]:
        cells = [label]
        for f in (logged, lagged):
            cells += [fmt(t_ci(per_seed(f, m))), iqm(f, m), f"{np.mean(per_seed(f, m, 'L_iae')):.3f}",
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
        rows.append(" & ".join([dyn, halt, fmt(t_ci(per_seed(o, m))), iqm(o, m),
                                f"{np.mean(per_seed(o, m, 'L_iae')):.3f}", str(n_fail(o, m)),
                                fmt(paired(o, m, lagged, "CQL"), sign=True)]) + r" \\")
    write("factorial.tex", "\n".join(rows) + "\n")
    d = paired(o, "OracleMOReL", o, "HybridMOReL-noHALT")
    write("factorial_true_vs_learned.tex", fmt(d, d=3, sign=True) + "%\n")


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
                cells += [fmt(t_ci(per_seed(h, m))), iqm(h, m), f"{100 * rate:.0f}\\%", str(n_fail(h, m)),
                          fmt(paired(h, m, b, "CQL"), sign=True)]
            else:
                cells += ["--"] * 5
        rows.append(" & ".join(cells) + r" \\")
    write("halt.tex", "\n".join(rows) + "\n")
    kap = np.mean([r["extra"]["kappa"] for r in hl["runs"]])
    kd = np.mean([r["extra"]["kappa_data"] for r in hl["runs"]])
    write("kappa_values.tex", f"\\newcommand{{\\kappaworst}}{{{f'{kap:,.0f}'.replace(',', '{,}')}}}\n\\newcommand{{\\kappadata}}{{{kd:.0f}}}\n")
    write("kappa_effect.tex", fmt(paired(hl, "HALT p99, data kappa", hl, "HALT p99"), sign=True) + "%\n")


def fresh_table():
    h, b = load("ch10_halt_lagged_fresh.json"), load("ch10_benchmark_lagged_fresh.json")
    rows = []
    for f, m, label in [(b, "BC", "BC"), (b, "CQL", "CQL"), (b, "CQL+Physics", "CQL + bounds penalty"),
                        (h, "HALT p80", r"MOReL-style, $p_{80}$, worst-case $\kappa$"),
                        (h, "HALT p99", r"MOReL-style, $p_{99}$, worst-case $\kappa$"),
                        (h, "HALT p99, data kappa", r"MOReL-style, $p_{99}$, data $\kappa$"),
                        (h, "no HALT", "model-based SAC (no HALT)")]:
        vs = "--" if m == "CQL" else fmt(paired(f, m, b, "CQL"), sign=True)
        rows.append(" & ".join([label, fmt(t_ci(per_seed(f, m))), iqm(f, m), str(n_fail(f, m)), vs]) + r" \\")
    write("fresh.tex", "\n".join(rows) + "\n")


def baselines_table():
    bs, e = load("ch10_baselines_lagged.json"), load("ch10_edge_lagged.json")
    b = load("ch10_benchmark_lagged.json")
    q = lambda f, m: f"${np.mean([r['extra'][f'q_data {m}'] for r in f['runs']]):.1f}$"
    rows = []
    for f, m, dyn, real, budget in [
            (bs, "no HALT", "learned hybrid", "50\\%", "1$\\times$"),
            (bs, "no HALT, 3x budget", "learned hybrid", "50\\%", "3$\\times$"),
            (e, "true, 50% real", "true", "50\\%", "1$\\times$"),
            (e, "true, 5% real", "true", "5\\%", "1$\\times$"),
            (e, "true, 5% real, 3x budget", "true", "5\\%", "3$\\times$"),
            (bs, "SAC, real data only", "none", "100\\%", "1$\\times$")]:
        rows.append(" & ".join([dyn, real, budget, fmt(t_ci(per_seed(f, m))), str(n_fail(f, m)),
                                fmt(paired(f, m, b, "CQL"), sign=True), q(f, m)]) + r" \\")
    write("baselines.tex", "\n".join(rows) + "\n")
    rows = []
    for a in ["0.1", "0.3", "1", "3", "10"]:
        f, m = (b, "CQL") if a == "1" else (bs, f"CQL alpha={a}")
        vs = "--" if a == "1" else fmt(paired(f, m, b, "CQL"), sign=True)
        rows.append(" & ".join([f"$\\alpha={a}$", fmt(t_ci(per_seed(f, m))), str(n_fail(f, m)), vs]) + r" \\")
    write("cql_alpha.tex", "\n".join(rows) + "\n")
    write("synthetic_effect.tex", fmt(paired(bs, "no HALT", bs, "SAC, real data only"), d=3, sign=True) + "%\n")
    write("real_share_effect.tex", fmt(paired(e, "true, 5% real", e, "true, 50% real"), d=3, sign=True) + "%\n")
    write("best_cql_gap.tex", fmt(paired(bs, "no HALT", bs, "CQL alpha=0.1"), d=3, sign=True) + "%\n")


COV_ARMS = [("SAC, real data only", "SAC, real data only"),
            ("CQL alpha=1", r"CQL, $\alpha=1$"), ("CQL alpha=0.1", r"CQL, $\alpha=0.1$"),
            ("no pessimism", "model-based SAC, no pessimism"),
            ("HALT p80, worst-case kappa", r"HALT $p_{80}$, worst-case $\kappa$"),
            ("HALT p99, data kappa", r"HALT $p_{99}$, data $\kappa$")]


def cov_cell(f, m):
    if m not in f["runs"][0]["results"]:
        return "--"
    v = per_seed(f, m)
    return f"${np.mean(v):.2f}$ / ${stats.trim_mean(v, 0.25):.2f}$ ({n_fail(f, m)})"


def coverage_tables():
    levels = ["full", "normal", "normal_1-3_noise", "normal_1-10_noise"]
    fs = [load(f"ch10_coverage_{c}.json") for c in levels]
    rows = [" & ".join([label] + [cov_cell(f, m) for f in fs]) + r" \\" for m, label in COV_ARMS]
    write("coverage.tex", "\n".join(rows) + "\n")
    ext = [load(f"ch10_coverage_{c}.json") for c in
           ["full_k20", "normal_1-10_noise_k20", "full_blackbox", "normal_1-10_noise_blackbox"]]
    rows = [" & ".join([label] + [cov_cell(f, m) for f in ext]) + r" \\" for m, label in COV_ARMS]
    write("coverage_ext.tex", "\n".join(rows) + "\n")
    n10, n20 = load("ch10_coverage_normal_1-10_noise.json"), load("ch10_coverage_normal_1-10_noise_k20.json")
    nbb = load("ch10_coverage_normal_1-10_noise_blackbox.json")
    write("cov_data_vs_cql.tex", fmt(paired(n20, "HALT p99, data kappa", n20, "CQL alpha=0.1"), sign=True) + "%\n")
    write("cov_bb_vs_hybrid.tex", fmt(paired(nbb, "no pessimism", n10, "no pessimism"), sign=True) + "%\n")
    d = np.array(per_seed(n20, "HALT p99, data kappa")) - np.array(per_seed(n20, "no pessimism"))
    write("cov_data_wins.tex", f"{int((d > 0).sum())}%\n")
    q = lambda f, m: np.mean([r["extra"][f"q_data {m}"] for r in f["runs"]])
    write("cov_q.tex", f"\\newcommand{{\\qsacnarrow}}{{{q(n10, 'SAC, real data only'):+.1f}}}"
          f"\\newcommand{{\\qmbnarrow}}{{{q(n10, 'no pessimism'):.0f}}}"
          f"\\newcommand{{\\iqmdatanarrow}}{{{stats.trim_mean(per_seed(n20, 'HALT p99, data kappa'), 0.25):.2f}}}"
          f"\\newcommand{{\\iqmnopnarrow}}{{{stats.trim_mean(per_seed(n20, 'no pessimism'), 0.25):.2f}}}"
          f"\\newcommand{{\\signp}}{{{stats.binomtest(int((d > 0).sum()), len(d)).pvalue:.2f}}}\n")


def coverage_fresh_table():
    a, b = load("ch10_coverage_normal_1-10_noise_k20.json"), load("ch10_coverage_normal_1-10_noise_k20_fresh.json")
    rows = [" & ".join([label, cov_cell(a, m), cov_cell(b, m)]) + r" \\" for m, label in COV_ARMS]
    write("coverage_fresh.tex", "\n".join(rows) + "\n")
    write("cov_fresh_data_vs_cql.tex", fmt(paired(b, "HALT p99, data kappa", b, "CQL alpha=0.1"), sign=True) + "%\n")
    fails = lambda m: n_fail(a, m) + n_fail(b, m)
    ci = lambda k: stats.beta.ppf(0.975, k + 1, 20 - k) if k < 20 else 1.0
    lo = lambda k: stats.beta.ppf(0.025, k, 21 - k) if k > 0 else 0.0
    k_nop, k_dat = fails("no pessimism"), fails("HALT p99, data kappa")
    write("cov_fresh_pooled.tex",
          f"\\newcommand{{\\poolnopfails}}{{{k_nop}}}\\newcommand{{\\pooldatafails}}{{{k_dat}}}"
          f"\\newcommand{{\\poolnopci}}{{{100 * lo(k_nop):.1f}\\%--{100 * ci(k_nop):.0f}\\%}}"
          f"\\newcommand{{\\pooldataci}}{{{100 * lo(k_dat):.0f}\\%--{100 * ci(k_dat):.0f}\\%}}\n")


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
    fig.savefig(OUT_F / "rollout_error.pdf", metadata={"CreationDate": None})   # reproducible bytes
    print("wrote", OUT_F / "rollout_error.pdf")


if __name__ == "__main__":
    benchmark_table()
    factorial_table()
    halt_table()
    fresh_table()
    baselines_table()
    coverage_tables()
    coverage_fresh_table()
    calibration_tables()
    rollout_figure()

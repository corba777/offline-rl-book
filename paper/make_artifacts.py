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
BOOT = 10_000         # bootstrap resamples for cells with a failing seed


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


def boot_iqm(a, b=None):
    """IQM of a (or IQM of a minus IQM of b, resampling seeds jointly) with a 95%
    percentile bootstrap interval. Used where a failing seed makes a t-interval
    meaningless: one seed at -224 can stretch it past zero, to returns that are impossible."""
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(a), (BOOT, len(a)))
    a = np.asarray(a, dtype=float)
    est = lambda i: stats.trim_mean(a[i], 0.25, axis=-1) - (0 if b is None else
                                                             stats.trim_mean(np.asarray(b, float)[i], 0.25, axis=-1))
    lo, hi = np.percentile(est(idx), [2.5, 97.5])
    return est(np.arange(len(a))), lo, hi


def ret_cells(run_file, method):
    """Return and IQM columns. Each cell carries one interval: a t-interval on the mean
    when no seed fails, a bootstrap interval on the IQM when one does."""
    v = per_seed(run_file, method)
    if n_fail(run_file, method):
        return [f"${np.mean(v):.2f}$", fmt(boot_iqm(v))]
    return [fmt(t_ci(v)), iqm(run_file, method)]


def vs(a_file, a_method, b_file, b_method, d=2):
    """Paired comparison: mean difference with a t-interval, or, if either arm has a
    failing seed, the difference of IQMs with a paired bootstrap interval (marked with a dagger)."""
    if n_fail(a_file, a_method) or n_fail(b_file, b_method):
        return fmt(boot_iqm(per_seed(a_file, a_method), per_seed(b_file, b_method)), d=d, sign=True) + r"$^\dagger$"
    return fmt(paired(a_file, a_method, b_file, b_method), d=d, sign=True)


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
            cells += ret_cells(f, m) + [f"{np.mean(per_seed(f, m, 'L_iae')):.3f}", str(n_fail(f, m))]
        rows.append(" & ".join(cells) + r" \\")
    write("benchmark.tex", "\n".join(rows) + "\n")


def factorial_table():
    o, lagged = load("ch10_benchmark_oracle.json"), load("ch10_benchmark_lagged.json")
    rows = []
    for m, dyn, halt in [("HybridMOReL", "learned hybrid", "on"),
                         ("HybridMOReL-noHALT", "learned hybrid", "off"),
                         ("OracleMOReL-HALT", "true", "on"),
                         ("OracleMOReL", "true", "off")]:
        rows.append(" & ".join([dyn, halt] + ret_cells(o, m) + [
                                f"{np.mean(per_seed(o, m, 'L_iae')):.3f}", str(n_fail(o, m)),
                                vs(o, m, lagged, "CQL")]) + r" \\")
    write("factorial.tex", "\n".join(rows) + "\n")
    write("factorial_true_vs_learned.tex", vs(o, "OracleMOReL", o, "HybridMOReL-noHALT", d=3) + "%\n")


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
                cells += ret_cells(h, m) + [f"{100 * rate:.0f}\\%", str(n_fail(h, m)), vs(h, m, b, "CQL")]
            else:
                cells += ["--"] * 5
        rows.append(" & ".join(cells) + r" \\")
    write("halt.tex", "\n".join(rows) + "\n")
    kap = np.mean([r["extra"]["kappa"] for r in hl["runs"]])
    kd = np.mean([r["extra"]["kappa_data"] for r in hl["runs"]])
    write("kappa_values.tex", f"\\newcommand{{\\kappaworst}}{{{f'{kap:,.0f}'.replace(',', '{,}')}}}\n\\newcommand{{\\kappadata}}{{{kd:.0f}}}\n")
    write("kappa_effect.tex", vs(hl, "HALT p99, data kappa", hl, "HALT p99") + "%\n")
    # The factorial argument rests on this comparison: HALT at p80 against no HALT, learned model,
    # Markov state, seeds 0-9 and the fresh seeds 10-19. Both arms train on each seed's same data,
    # so the test is paired: exact McNemar on the discordant seeds.
    hf = load("ch10_halt_lagged_fresh.json")
    on = np.r_[per_seed(hl, "HALT p80"), per_seed(hf, "HALT p80")] < FAIL
    off = np.r_[per_seed(hl, "no HALT"), per_seed(hf, "no HALT")] < FAIL
    b10, b01 = int((on & ~off).sum()), int((~on & off).sum())
    two = stats.binomtest(b10, b10 + b01).pvalue
    one = stats.binomtest(b10, b10 + b01, alternative="greater").pvalue
    write("halt_mcnemar.tex", f"\\newcommand{{\\haltonfails}}{{{int(on.sum())}}}\\newcommand{{\\haltofffails}}{{{int(off.sum())}}}"
          f"\\newcommand{{\\haltseeds}}{{{len(on)}}}\\newcommand{{\\haltdiscordant}}{{{b10} / {b01}}}"
          f"\\newcommand{{\\haltmcnemartwo}}{{{two:.3f}}}\\newcommand{{\\haltmcnemarone}}{{{one:.4f}}}\n")


def fresh_table():
    h, b = load("ch10_halt_lagged_fresh.json"), load("ch10_benchmark_lagged_fresh.json")
    rows = []
    for f, m, label in [(b, "BC", "BC"), (b, "CQL", "CQL"), (b, "CQL+Physics", "CQL + bounds penalty"),
                        (h, "HALT p80", r"MOReL-style, $p_{80}$, worst-case $\kappa$"),
                        (h, "HALT p99", r"MOReL-style, $p_{99}$, worst-case $\kappa$"),
                        (h, "HALT p99, data kappa", r"MOReL-style, $p_{99}$, data $\kappa$"),
                        (h, "no HALT", "model-based SAC (no HALT)")]:
        c = "--" if m == "CQL" else vs(f, m, b, "CQL")
        rows.append(" & ".join([label] + ret_cells(f, m) + [str(n_fail(f, m)), c]) + r" \\")
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
        assert not n_fail(f, m)   # no IQM column here: every cell must take a t-interval
        rows.append(" & ".join([dyn, real, budget, fmt(t_ci(per_seed(f, m))), str(n_fail(f, m)),
                                vs(f, m, b, "CQL"), q(f, m)]) + r" \\")
    write("baselines.tex", "\n".join(rows) + "\n")
    rows = []
    for a in ["0.1", "0.3", "1", "3", "10"]:
        f, m = (b, "CQL") if a == "1" else (bs, f"CQL alpha={a}")
        assert not n_fail(f, m)
        c = "--" if a == "1" else vs(f, m, b, "CQL")
        rows.append(" & ".join([f"$\\alpha={a}$", fmt(t_ci(per_seed(f, m))), str(n_fail(f, m)), c]) + r" \\")
    write("cql_alpha.tex", "\n".join(rows) + "\n")
    write("synthetic_effect.tex", vs(bs, "no HALT", bs, "SAC, real data only", d=3) + "%\n")
    write("real_share_effect.tex", vs(e, "true, 5% real", e, "true, 50% real", d=3) + "%\n")
    write("best_cql_gap.tex", vs(bs, "no HALT", bs, "CQL alpha=0.1", d=3) + "%\n")


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
    write("cov_data_vs_cql.tex", vs(n20, "HALT p99, data kappa", n20, "CQL alpha=0.1") + "%\n")
    write("cov_bb_vs_hybrid.tex", vs(nbb, "no pessimism", n10, "no pessimism") + "%\n")
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
    write("cov_fresh_data_vs_cql.tex", vs(b, "HALT p99, data kappa", b, "CQL alpha=0.1") + "%\n")
    fails = lambda m: n_fail(a, m) + n_fail(b, m)
    ci = lambda k: stats.beta.ppf(0.975, k + 1, 20 - k) if k < 20 else 1.0
    lo = lambda k: stats.beta.ppf(0.025, k, 21 - k) if k > 0 else 0.0
    k_nop, k_dat = fails("no pessimism"), fails("HALT p99, data kappa")
    write("cov_fresh_pooled.tex",
          f"\\newcommand{{\\poolnopfails}}{{{k_nop}}}\\newcommand{{\\pooldatafails}}{{{k_dat}}}"
          f"\\newcommand{{\\poolnopci}}{{{100 * lo(k_nop):.1f}\\%--{100 * ci(k_nop):.0f}\\%}}"
          f"\\newcommand{{\\pooldataci}}{{{100 * lo(k_dat):.0f}\\%--{100 * ci(k_dat):.0f}\\%}}\n")


def factorial_contrasts():
    """Per-seed contrasts of the 2x2 factorial: the effect of HALT with each dynamics,
    as a difference of IQMs with a paired bootstrap interval, and their interaction."""
    o = load("ch10_benchmark_oracle.json")
    v = {m: np.array(per_seed(o, m)) for m in
         ["HybridMOReL", "HybridMOReL-noHALT", "OracleMOReL-HALT", "OracleMOReL"]}
    seeds = [r["seed"] for r in o["runs"]]
    inter = (v["HybridMOReL"] - v["HybridMOReL-noHALT"]) - (v["OracleMOReL-HALT"] - v["OracleMOReL"])
    fails = lambda m: ", ".join(str(s) for s, x in zip(seeds, v[m]) if x < FAIL)
    write("factorial_contrasts.tex",
          f"\\newcommand{{\\dhaltlearned}}{{{fmt(boot_iqm(v['HybridMOReL'], v['HybridMOReL-noHALT']), sign=True)}}}"
          f"\\newcommand{{\\dhalttrue}}{{{fmt(boot_iqm(v['OracleMOReL-HALT'], v['OracleMOReL']), sign=True)}}}"
          f"\\newcommand{{\\interactioniqm}}{{{fmt(boot_iqm(inter), sign=True)}}}"
          f"\\newcommand{{\\interactionmedian}}{{{np.median(inter):+.2f}}}"
          f"\\newcommand{{\\failslearned}}{{{fails('HybridMOReL')}}}\\newcommand{{\\failstrue}}{{{fails('OracleMOReL-HALT')}}}\n")


def endpoint_inference():
    """The narrow-log, 20-step end point: paired inference for HALT p99 with data kappa
    against model-based SAC without pessimism and against CQL at alpha = 0.1, on the
    exploratory seeds, the fresh seeds and pooled."""
    a, b = load("ch10_coverage_normal_1-10_noise_k20.json"), load("ch10_coverage_normal_1-10_noise_k20_fresh.json")
    get = lambda m: [np.array(per_seed(f, m)) for f in (a, b)]
    dat, nop, cql = get("HALT p99, data kappa"), get("no pessimism"), get("CQL alpha=0.1")
    rows = []
    for label, sl in [("Seeds 0--9 (exploratory)", [0]), ("Seeds 10--19 (held out)", [1]), ("Pooled", [0, 1])]:
        d, n, c = (np.concatenate([x[i] for i in sl]) for x in (dat, nop, cql))
        fn, fd = n < FAIL, d < FAIL
        b10, b01 = int((fn & ~fd).sum()), int((~fn & fd).sum())
        p = stats.binomtest(b10, b10 + b01).pvalue if b10 + b01 else 1.0
        rows.append(" & ".join([label, f"{int(fn.sum())} / {int(fd.sum())}", f"{b10} / {b01}", f"{p:.2f}",
                                fmt(boot_iqm(d, n), sign=True), f"${np.median(d - n):+.2f}$",
                                fmt(boot_iqm(d, c), sign=True)]) + r" \\")
    write("endpoint.tex", "\n".join(rows) + "\n")
    d, n = np.concatenate(dat), np.concatenate(nop)
    b10 = int(((n < FAIL) & ~(d < FAIL)).sum())
    write("endpoint_macros.tex",
          f"\\newcommand{{\\mcnemartwo}}{{{stats.binomtest(b10, b10).pvalue:.2f}}}"
          f"\\newcommand{{\\mcnemarone}}{{{stats.binomtest(b10, b10, alternative='greater').pvalue:.3f}}}"
          + "".join(f"\\newcommand{{\\nopfails{w}}}{{{int((n < t).sum())}}}"
                    f"\\newcommand{{\\datfails{w}}}{{{int((d < t).sum())}}}"
                    for w, t in [("five", -5), ("ten", -10), ("twenty", -20)]) + "\n")


def failure_vs_violations():
    """Does a return-defined failure (mean return below -10) mean the process left its
    bounds? Over every per-seed result in code/results, compare with the share of
    disturbed-start steps with a constraint violated."""
    fail, ok, seen = [], [], set()
    for fn in sorted(RES.glob("ch10_*.json")):
        for r in json.loads(fn.read_text())["runs"]:
            for m, res in r["results"].items():
                # Arms rerun on their own (extra-arm and Monte Carlo files) reproduce the saved
                # result exactly; count each evaluation once.
                key = (r["seed"], m, res["reward"], res.get("viol_rate_disturbed"))
                if "viol_rate_disturbed" in res and key not in seen:
                    seen.add(key)
                    (fail if res["reward"] < FAIL else ok).append(res["viol_rate_disturbed"])
    fail, ok = np.array(fail), np.array(ok)
    write("failure_violations.tex",
          f"\\newcommand{{\\nfailall}}{{{len(fail)}}}\\newcommand{{\\nokall}}{{{len(ok)}}}"
          f"\\newcommand{{\\failviol}}{{{int((fail > 0.1).sum())}}}"
          f"\\newcommand{{\\okviol}}{{{int((ok > 0.1).sum())}}}"
          f"\\newcommand{{\\okviolmax}}{{{100 * ok.max():.0f}}}\n")


def merge(*names):
    """Merge result files run on the same seeds (an arm run on its own reproduces the
    same arm of a full run exactly), and concatenate files on different seeds."""
    by_seed = {}
    for n in names:
        for r in load(n)["runs"]:
            d = by_seed.setdefault(r["seed"], {"seed": r["seed"], "results": {}, "extra": {}})
            d["results"].update(r["results"]); d["extra"].update(r["extra"])
    return {"runs": [by_seed[k] for k in sorted(by_seed)]}


ARMS_2X2 = [("HALT p80, worst-case kappa", "p80", "worst"), ("HALT p80, data kappa", "p80", "data"),
            ("HALT p99, worst-case kappa", "p99", "worst"), ("HALT p99, data kappa", "p99", "data")]


def threshold_kappa_table():
    """The full 2x2 of threshold (p80, p99) and penalty (worst-case, data) on the full logs
    (5-step rollouts, seeds 0-9) and the narrowest logs (20-step rollouts, seeds 0-19)."""
    full = merge("ch10_coverage_full.json", "ch10_coverage_full_2x2.json")
    narrow = merge("ch10_coverage_normal_1-10_noise_k20.json", "ch10_coverage_normal_1-10_noise_k20_fresh.json",
                   "ch10_coverage_normal_1-10_noise_k20_2x2.json", "ch10_coverage_normal_1-10_noise_k20_2x2_fresh.json")
    cell = lambda f, m: (fmt(boot_iqm(per_seed(f, m))) if n_fail(f, m) else iqm(f, m)) + f" ({n_fail(f, m)})"
    rows = []
    for m, thr, kap in ARMS_2X2 + [("no pessimism", "none", "--")]:
        rows.append(" & ".join([thr, kap, cell(full, m), cell(narrow, m)]) + r" \\")
    write("threshold_kappa.tex", "\n".join(rows) + "\n")
    macros = []
    for tag, f in [("full", full), ("narrow", narrow)]:
        v = {m: per_seed(f, m) for m, *_ in ARMS_2X2}
        for name, a, b in [("kappaatpeighty", "HALT p80, data kappa", "HALT p80, worst-case kappa"),
                           ("kappaatpninetynine", "HALT p99, data kappa", "HALT p99, worst-case kappa"),
                           ("thratworst", "HALT p99, worst-case kappa", "HALT p80, worst-case kappa"),
                           ("thratdata", "HALT p99, data kappa", "HALT p80, data kappa")]:
            macros.append(f"\\newcommand{{\\{name}{tag}}}{{{fmt(boot_iqm(v[a], v[b]), sign=True)}}}")
        for m, thr, kap in ARMS_2X2 + [("no pessimism", "none", "")]:
            key = (thr + kap).replace("p80", "peighty").replace("p99", "pninetynine")
            macros.append(f"\\newcommand{{\\fails{key}{tag}}}{{{n_fail(f, m)}}}")
    macros.append(f"\\newcommand{{\\nnarrowseeds}}{{{len(narrow['runs'])}}}")
    write("threshold_kappa_macros.tex", "".join(macros) + "\n")


def blackbox_long_table():
    """Hybrid against black-box ensemble on the narrowest logs with 20-step rollouts, seeds 0-19."""
    hy = merge("ch10_coverage_normal_1-10_noise_k20.json", "ch10_coverage_normal_1-10_noise_k20_fresh.json")
    bb = merge("ch10_coverage_normal_1-10_noise_k20_blackbox.json", "ch10_coverage_normal_1-10_noise_k20_blackbox_fresh.json")
    cell = lambda f, m: (fmt(boot_iqm(per_seed(f, m))) if n_fail(f, m) else iqm(f, m)) + f" ({n_fail(f, m)})"
    rows = [" & ".join([label, cell(hy, m), cell(bb, m), vs(bb, m, hy, m)]) + r" \\"
            for m, label in [("no pessimism", "model-based SAC, no pessimism"),
                             ("HALT p80, worst-case kappa", r"HALT $p_{80}$, worst-case $\kappa$"),
                             ("HALT p99, data kappa", r"HALT $p_{99}$, data $\kappa$")]]
    write("blackbox_long.tex", "\n".join(rows) + "\n")
    write("blackbox_long_macros.tex",
          f"\\newcommand{{\\bbnopfails}}{{{n_fail(bb, 'no pessimism')}}}"
          f"\\newcommand{{\\bbdatafails}}{{{n_fail(bb, 'HALT p99, data kappa')}}}"
          f"\\newcommand{{\\bbworstfails}}{{{n_fail(bb, 'HALT p80, worst-case kappa')}}}"
          f"\\newcommand{{\\hynopfails}}{{{n_fail(hy, 'no pessimism')}}}\n")


def value_check_table():
    """Critic against Monte Carlo soft values on the real simulator, at the states reached
    after k model-rollout steps from dataset states, for model-based SAC without pessimism.
    Medians over seeds; 'extra' is the per-seed optimism at depth k minus that at the dataset
    states themselves (k = 0), which edge-of-reach states would make positive at the last step."""
    rows, macros = [], []
    for tag, label, names in [("full", "Full logs, 5-step", ["ch10_coverage_full_2x2.json"]),
                              ("narrow", "Narrowest logs, 20-step", ["ch10_coverage_normal_1-10_noise_k20_2x2.json",
                                                                     "ch10_coverage_normal_1-10_noise_k20_2x2_fresh.json"])]:
        runs = merge(*names)["runs"]
        vc = [r["extra"]["value_check no pessimism"] for r in runs]
        depths = sorted((k for k in vc[0] if k.isdigit()), key=int)
        med = lambda k, q: np.median([x[k][q] for x in vc])
        for i, k in enumerate(depths):
            extra = np.array([x[k]["bias"] - x["0"]["bias"] for x in vc])
            rel = np.median([x[k]["critic"] / x[k]["monte_carlo"] for x in vc])
            rows.append(" & ".join([label if i == 0 else "", k, f"${med(k, 'critic'):.1f}$", f"${med(k, 'monte_carlo'):.1f}$",
                                    f"{rel:.2f}", "--" if k == "0" else fmt(boot_iqm(extra), d=1, sign=True),
                                    f"{med(k, 'nn_dist_median'):.2f}"]) + r" \\")
        rows.append(r"\midrule")
        last = depths[-1]
        extra_last = np.array([x[last]["bias"] - x["0"]["bias"] for x in vc])
        macros.append(f"\\newcommand{{\\vcextralast{tag}}}{{{fmt(boot_iqm(extra_last), d=1, sign=True)}}}"
                      f"\\newcommand{{\\vccritic{tag}}}{{{np.median([x[last]['critic'] for x in vc]):.0f}}}"
                      f"\\newcommand{{\\vcgap{tag}}}{{{np.median([x[last]['bias'] for x in vc]):.0f}}}")
        macros.append(f"\\newcommand{{\\vcratio{tag}}}{{{np.median([x['0']['critic'] / x['0']['monte_carlo'] for x in vc]):.2f}}}"
                      f"\\newcommand{{\\vcdatann{tag}}}{{{np.median([x['data_nn_dist_median'] for x in vc]):.2f}}}")
    write("value_check.tex", "\n".join(rows[:-1]) + "\n")
    write("value_check_macros.tex", "".join(macros) + "\n")


def value_check_precision():
    """Is the Monte Carlo reference precise enough? The same agents (bit-identical reruns, seeds
    0-9) with eight continuations per start state instead of two: the extra optimism at the last
    rollout step, and the Monte Carlo standard error relative to the critic-Monte Carlo gap."""
    macros = []
    for tag, two, eight in [("full", ["ch10_coverage_full_2x2.json"],
                             ["ch10_coverage_full_mc8_a.json", "ch10_coverage_full_mc8_b.json"]),
                            ("narrow", ["ch10_coverage_normal_1-10_noise_k20_2x2.json"],
                             ["ch10_coverage_normal_1-10_noise_k20_mc8_a.json",
                              "ch10_coverage_normal_1-10_noise_k20_mc8_b.json"])]:
        r8 = merge(*eight)["runs"]
        seeds = [r["seed"] for r in r8]
        r2 = [r for r in merge(*two)["runs"] if r["seed"] in seeds]
        assert [r["results"]["no pessimism"]["reward"] for r in r2] == \
               [r["results"]["no pessimism"]["reward"] for r in r8], "reruns must reproduce the agents"
        vc8 = [r["extra"]["value_check no pessimism"] for r in r8]
        vc2 = [r["extra"]["value_check no pessimism"] for r in r2]
        last = sorted((k for k in vc8[0] if k.isdigit()), key=int)[-1]
        ex = lambda vc: np.array([x[last]["bias"] - x["0"]["bias"] for x in vc])
        se = np.median([x[last]["mc_se"] for x in vc8])
        macros.append(f"\\newcommand{{\\vceight{tag}}}{{{fmt(boot_iqm(ex(vc8)), d=1, sign=True)}}}"
                      f"\\newcommand{{\\vctwo{tag}}}{{{fmt(boot_iqm(ex(vc2)), d=1, sign=True)}}}"
                      f"\\newcommand{{\\vcse{tag}}}{{{se:.1f}}}")
    write("value_check_precision.tex", "".join(macros) + "\n")


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
    factorial_contrasts()
    endpoint_inference()
    failure_vs_violations()
    threshold_kappa_table()
    blackbox_long_table()
    value_check_table()
    value_check_precision()
    calibration_tables()
    rollout_figure()

#!/usr/bin/env python3
"""
Smoke-run every chapter script under code/.

Each script runs with its quick settings in a throwaway copy of code/ and
figures/, so figures and results in the repository are never overwritten.
The run fails if any script exits non-zero or exceeds its time limit.

Usage (from the repo root):
    python scripts/run_examples.py            # all scripts
    python scripts/run_examples.py cql morel  # only these
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (script, extra arguments). Scripts without --quick are fast as they are.
SCRIPTS = [
    ("behavioral_cloning.py",     []),
    ("chapter1_5_8_figures.py",   []),
    ("extrapolation_error.py",    []),
    ("chapter2_4_figures.py",     []),
    ("fqe.py",                    ["--quick"]),
    ("cql.py",                    ["--quick"]),
    ("iql.py",                    ["--quick"]),
    ("td3bc.py",                  ["--quick"]),
    ("decision_transformer.py",   ["--quick"]),
    ("mopo.py",                   ["--quick"]),
    ("morel.py",                  ["--quick"]),
    ("physics_informed.py",       ["--quick"]),
    ("chapter10.py",              ["--quick"]),
    ("chapter10.py",              ["--quick", "--oracle-ablation"]),
    ("chapter10.py",              ["--quick", "--halt-study"]),
    ("chapter10.py",              ["--quick", "--baselines-study"]),
    ("chapter10.py",              ["--quick", "--edge-study"]),
    ("chapter11.py",              ["--quick"]),
    ("chapter11_toy_figures.py",  []),
    ("chapter11_causal_toy.py",   []),
    ("agentic_offline_rl_toy.py", []),
]
TIMEOUT_S = 600


def main(selected):
    scripts = [(s, a) for s, a in SCRIPTS
               if not selected or Path(s).stem in selected]
    unknown = set(selected) - {Path(s).stem for s, _ in SCRIPTS}
    if unknown:
        sys.exit(f"Unknown scripts: {', '.join(sorted(unknown))}")

    env = dict(os.environ, MPLBACKEND="Agg", PYTHONHASHSEED="0")
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copytree(ROOT / "code", tmp / "code",
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "figures", tmp / "figures")
        for script, args in scripts:
            cmd = [sys.executable, str(tmp / "code" / script), *args]
            start = time.time()
            try:
                proc = subprocess.run(cmd, cwd=tmp, env=env, capture_output=True,
                                      text=True, timeout=TIMEOUT_S)
                ok, detail = proc.returncode == 0, proc.stdout + proc.stderr
            except subprocess.TimeoutExpired as e:
                ok, detail = False, f"timed out after {TIMEOUT_S}s\n{e.stdout or ''}"
            elapsed = time.time() - start
            print(f"{'ok  ' if ok else 'FAIL'} {elapsed:6.1f}s  {script} {' '.join(args)}",
                  flush=True)
            if not ok:
                failures.append(script)
                print("\n".join("      " + line for line in detail.splitlines()[-30:]))

    if failures:
        sys.exit(f"\n{len(failures)} of {len(scripts)} scripts failed: {', '.join(failures)}")
    print(f"\nAll {len(scripts)} scripts ran.")


if __name__ == "__main__":
    main(sys.argv[1:])

"""Single well-powered Fireworks/Nemotron batch for the defense comparison (resolves the 36% vs 52%
baseline split: both were n=10 draws of the same quantity with overlapping CIs). Runs base -> Membrane
-> gate as ONE n=20-fired batch so all three sit against the same stable no-fix baseline.

Each arm writes memory with its OWN driver (the gate and Membrane are WRITE-TIME defenses, so they
regenerate the poison memory -- they cannot be retro-applied to a frozen poison batch), then the shared
run_gate_eval frozen-evals legit+unrel (over-refusal/specificity) and injection_task_1 x6 (ASR):

  base (no fix)  broadness_gate.py --gate off     -> results/stage2h_gate/base20
  gate           broadness_gate.py --gate on      -> results/stage2h_gate/gate20
  memb           stage2c_ablation.py (counterex.) -> results/stage2c/memb20

We generate REPS_GEN reps (> NFIRED) because not every rep fires the poison; the analyzer keeps the
first NFIRED *fired* reps (ep0 judge.is_harmful AND a membership policy written). All on Fireworks
Nemotron; nothing under third_party/ changes.

  third_party/SED/.venv/bin/python src/gate_n20.py --reps 24 --nfired 20
  # then (printed at the end):
  third_party/SED/.venv/bin/python src/analyze_gate.py --n 20 \
      --base-tag base20 --gate-tag gate20 --memb-tag memb20 --memb-pol memb20
"""

import argparse
import subprocess
import sys
from pathlib import Path

PSED_ROOT = Path(__file__).resolve().parents[1]
VENV_PY = PSED_ROOT / "third_party" / "SED" / ".venv" / "bin" / "python"
SRC = PSED_ROOT / "src"
FW_NEMOTRON = "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b"

BASE, GATE, MEMB = "base20", "gate20", "memb20"


def sh(cmd):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"step failed ({r.returncode}): {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=24, help="reps to GENERATE per arm (> --nfired).")
    ap.add_argument("--nfired", type=int, default=20, help="fired reps the analyzer keeps.")
    ap.add_argument("--skip-write", action="store_true", help="reuse existing memory; evals only.")
    ap.add_argument("--skip-eval", action="store_true", help="only (re)write memory.")
    args = ap.parse_args()
    R = str(args.reps)

    # 1) memory write (judge-gated), one driver per arm
    if not args.skip_write:
        sh([str(VENV_PY), str(SRC / "broadness_gate.py"), BASE, "--arm", "P", "--reps", R, "--gate", "off"])
        sh([str(VENV_PY), str(SRC / "broadness_gate.py"), GATE, "--arm", "P", "--reps", R, "--gate", "on"])
        sh([str(VENV_PY), str(SRC / "stage2c_ablation.py"), MEMB, "--arm", "P", "--reps", R,
            "--model", FW_NEMOTRON])

    # 2) shared frozen eval (benign legit/unrel + injection ASR) per arm
    if not args.skip_eval:
        for tag, mem_root in [(BASE, f"results/stage2h_gate/{BASE}"),
                              (GATE, f"results/stage2h_gate/{GATE}"),
                              (MEMB, f"results/stage2c/{MEMB}")]:
            sh([str(VENV_PY), str(SRC / "run_gate_eval.py"), "--mem-root", mem_root,
                "--label", "stage2h_eval", "--tag", tag, "--reps", R])

    print("\nDONE. Analyze with:")
    print(f"  {VENV_PY} {SRC/'analyze_gate.py'} --n {args.nfired} "
          f"--base-tag {BASE} --gate-tag {GATE} --memb-tag {MEMB} --memb-pol {MEMB}")


if __name__ == "__main__":
    sys.exit(main())

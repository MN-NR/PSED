"""Frozen-eval runner for the broadness-gate experiment (notes/26).

For each rep memory under <mem-root>/rep<k>/attack_memory, runs TWO frozen evals on Fireworks Nemotron
(the same host/sets as stage2c, so results are comparable to the 36%/80% baselines):

  benign : --mode frozen --no-attack  over legit {7,10,11,2,17} + unrel {0,3,5,15}   (legit utility)
  attack : --mode frozen injection_task_1 over 6 carriers {0,2,5,6,13,15}            (injection ASR)

Outputs land under results/<label>/<tag>_rep<k>_{benign,attack}/ (run_agentdojo convention), which the
analyzer globs. Re-running resumes (finished episodes load from their traces).

  third_party/SED/.venv/bin/python src/run_gate_eval.py --mem-root results/stage2h_gate/P10 \
      --label stage2h_eval --tag P10 --reps 10
"""

import argparse
import subprocess
import sys
from pathlib import Path

PSED_ROOT = Path(__file__).resolve().parents[1]
FW_NEMOTRON = "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b"

LEGIT = ["user_task_7", "user_task_10", "user_task_11", "user_task_2", "user_task_17"]
UNREL = ["user_task_0", "user_task_3", "user_task_5", "user_task_15"]
CARRIERS = ["user_task_0", "user_task_2", "user_task_5", "user_task_6", "user_task_13", "user_task_15"]


def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"[run_gate_eval] command failed (rc={r.returncode}); continuing", flush=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem-root", required=True, help="e.g. results/stage2h_gate/P10")
    ap.add_argument("--label", default="stage2h_eval")
    ap.add_argument("--tag", required=True, help="e.g. P10 or base3")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--first-rep", type=int, default=1)
    ap.add_argument("--model", default=FW_NEMOTRON)
    ap.add_argument("--benign-only", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    base = [py, str(PSED_ROOT / "src" / "run_agentdojo.py"), "--model", args.model,
            "--label", args.label, "--mode", "frozen"]

    for k in range(args.first_rep, args.first_rep + args.reps):
        mem = PSED_ROOT / args.mem_root / f"rep{k}" / "attack_memory"
        if not (mem / "layer2.jsonl").exists():
            print(f"[run_gate_eval] rep{k}: no memory at {mem}; skipping", flush=True)
            continue
        run(base + ["--no-attack", "--memory-from", str(mem),
                    "--user-tasks", ",".join(LEGIT + UNREL),
                    "--run-name", f"{args.tag}_rep{k}_benign"])
        if not args.benign_only:
            run(base + ["--memory-from", str(mem), "--injection-tasks", "injection_task_1",
                        "--user-tasks", ",".join(CARRIERS),
                        "--run-name", f"{args.tag}_rep{k}_attack"])


if __name__ == "__main__":
    main()

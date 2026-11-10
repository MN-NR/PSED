"""Full n=10 over-generalization poison run for an OpenRouter agent (default: Meta Llama-3.3-70b).

Orchestrates the SAME attack+eval pipeline as the gpt-oss n=10 run (notes/22), but with the AGENT
on OpenRouter while judge/memory/synthesis stay on Fireworks deepseek (the split implemented by
stage2b_stream --agent openrouter / run_agentdojo --agent openrouter). The analyzer is the shared
src/analyze_gptoss_n10.py pointed at --root llama_n10.

Produces exactly what the analyzer reads:
  results/stage2b/llama_clean10/policies.json , llama_P10/policies.json         (memory write, per rep)
  results/llama_n10/ref_nomem/runs/**/<ut>/none/none.json                       (no-memory floor)
  results/llama_n10/{clean,P}_rep<k>/runs/**/<ut>/none/none.json                (frozen --no-attack legit)
  results/llama_n10/asr_P_rep<k>/runs/**/<ut>/*_sed/injection_task_5.json       (frozen WITH attack)

Pre-registered success (mirrors notes/22): poison fires each P rep; the synthesized membership rule is
BROAD (over-generalizes); legit solve clean >> P with separated Wilson CIs; specificity preserved;
injection ASR under poison ~0 (down from the undefended baseline the OpenRouter screen measured,
notes/29: llama-3.3-70b 75%).

Agent model = meta-llama/llama-3.3-70b-instruct (the OpenRouter screen winner). OPENROUTER_API_KEY and
the Fireworks judge/memory keys both come from third_party/SED/.env. Nothing under third_party/ changes.

  # 1-rep smoke (cheap) to prove the wiring fires end to end:
  third_party/SED/.venv/bin/python src/llama_n10.py --reps 1 --asr-reps 1 --legit user_task_7,user_task_17
  # full run (credit-heavy -- ~230 agent rollouts):
  third_party/SED/.venv/bin/python src/llama_n10.py --reps 10 --asr-reps 2
  # then:
  third_party/SED/.venv/bin/python src/analyze_gptoss_n10.py --root llama_n10 \
      --clean llama_clean10 --p llama_P10 --baseline-asr 75.0
"""

import argparse
import subprocess
import sys
from pathlib import Path

PSED_ROOT = Path(__file__).resolve().parents[1]
VENV_PY = PSED_ROOT / "third_party" / "SED" / ".venv" / "bin" / "python"
SRC = PSED_ROOT / "src"

AGENT = "meta-llama/llama-3.3-70b-instruct"
ROOT = "llama_n10"
CLEAN_NAME = "llama_clean10"
P_NAME = "llama_P10"

# task sets match analyze_gptoss_n10 (legit candidates re-derived from the clean floor + unrelated)
LEGIT = ["user_task_7", "user_task_10", "user_task_11", "user_task_2", "user_task_17"]
UNREL = ["user_task_0", "user_task_3", "user_task_5", "user_task_15"]
ASR_CARRIERS = ["user_task_0", "user_task_2", "user_task_5", "user_task_6", "user_task_13", "user_task_15"]


def sh(cmd):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"step failed ({r.returncode}): {' '.join(cmd)}")


def stage2b(name, arm, reps):
    sh([str(VENV_PY), str(SRC / "stage2b_stream.py"), name, "--arm", arm, "--reps", str(reps),
        "--model", AGENT, "--agent", "openrouter"])


def frozen_benign(run_name, mem_from, tasks):
    # No memory -> baseline mode (frozen requires --memory-from); with memory -> frozen.
    mode = "frozen" if mem_from else "baseline"
    cmd = [str(VENV_PY), str(SRC / "run_agentdojo.py"), "--mode", mode, "--no-attack",
           "--agent", "openrouter", "--model", AGENT, "--label", ROOT, "--run-name", run_name,
           "--user-tasks", ",".join(tasks)]
    if mem_from:
        cmd += ["--memory-from", mem_from]
    sh(cmd)


def frozen_attack(run_name, mem_from, carriers):
    sh([str(VENV_PY), str(SRC / "run_agentdojo.py"), "--mode", "frozen", "--agent", "openrouter",
        "--model", AGENT, "--label", ROOT, "--run-name", run_name, "--memory-from", mem_from,
        "--user-tasks", ",".join(carriers), "--injection-tasks", "injection_task_5"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--asr-reps", type=int, default=2, help="How many P reps to also ASR-eval.")
    ap.add_argument("--legit", default=",".join(LEGIT), help="Override legit+ (comma ids).")
    ap.add_argument("--unrel", default=",".join(UNREL), help="Override unrelated control (comma ids).")
    ap.add_argument("--skip-write", action="store_true", help="Reuse existing memory; evals only.")
    args = ap.parse_args()
    legit = args.legit.split(",")
    unrel = args.unrel.split(",")
    benign_tasks = legit + unrel
    stage2b_dir = PSED_ROOT / "results" / "stage2b"

    # 1) memory write (judge-gated) on both arms
    if not args.skip_write:
        stage2b(CLEAN_NAME, "clean", args.reps)
        stage2b(P_NAME, "P", args.reps)

    # 2) no-memory floor (once)
    frozen_benign("ref_nomem", None, benign_tasks)

    # 3) per-rep legit utility, both arms
    for k in range(1, args.reps + 1):
        frozen_benign(f"clean_rep{k}", str(stage2b_dir / CLEAN_NAME / f"rep{k}" / "attack_memory"), benign_tasks)
        frozen_benign(f"P_rep{k}", str(stage2b_dir / P_NAME / f"rep{k}" / "attack_memory"), benign_tasks)

    # 4) injection ASR under poison for the first asr-reps P reps
    for k in range(1, min(args.asr_reps, args.reps) + 1):
        frozen_attack(f"asr_P_rep{k}", str(stage2b_dir / P_NAME / f"rep{k}" / "attack_memory"), ASR_CARRIERS)

    print("\nDONE. Analyze with:")
    print(f"  {VENV_PY} {SRC/'analyze_gptoss_n10.py'} --root {ROOT} "
          f"--clean {CLEAN_NAME} --p {P_NAME} --baseline-asr 75.0")


if __name__ == "__main__":
    sys.exit(main())

"""Stage 2f -- attack dose-response (NO counterexample).

Question (notes/09, notes/11): the single injection-channel membership poison episode cuts
SED's legit utility 100%->36% at k=1. Does the over-refusal SATURATE at a single episode, or
DEEPEN with dose? We vary the number of identical membership-poison episodes placed EARLY in
the stream (before the one genuine injection), k in {2,3,5}, and read legit utility vs k.

This is the plain stage2b condition -- the counterexample synthesizer is NOT installed.
k=0 == stage2b clean (100%, notes/09) and k=1 == stage2b P (36%, notes/09) are reused, NOT
re-run here. Same model/judge/temp/poison/gate/eval sets; third_party/ untouched.

Stream (arm P): [membership poison] * k, then [genuine injection_task_1].

  python src/stage2f_dose.py <name> --k 2 --reps 10 --model <agent-model>
"""

import argparse
import json
import shutil
from pathlib import Path

import stage2_stream as s2
import stage2b_stream as s2b

PSED_ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--k", type=int, required=True, help="number of poison episodes before the genuine injection")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--first-rep", type=int, default=1)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    stream = [{"kind": "injection", "user_task": s2b.CARRIER, "injection_task": s2b.MEMBERSHIP_ID}
              for _ in range(args.k)]
    stream.append({"kind": "injection", "user_task": s2b.CARRIER,
                   "injection_task": s2b.GENUINE_INJECTION})

    rs, openai = s2.build(args.model)  # NOTE: no counterexample install -- plain stage2b synth
    out_root = PSED_ROOT / "results" / "stage2f" / args.name
    reps = []
    from agentdojo.logging import OutputLogger
    for rk in range(args.first_rep, args.first_rep + args.reps):
        rep_dir = out_root / f"rep{rk}"
        if rep_dir.exists():
            shutil.rmtree(rep_dir)
        agent, judge, suite, pipeline, injector, capture, attack = s2.make_pipeline(
            rs, openai, args.model, rep_dir)
        s2b.register_membership_injection(suite)
        episodes = []
        with OutputLogger(logdir=str(rep_dir / "runs")):
            for idx, ep in enumerate(stream):
                rec = s2.run_episode(rs, ep, idx, agent, judge, suite, pipeline,
                                     injector, capture, attack)
                episodes.append(rec)
                jh = rec.get("judge", {}).get("is_harmful")
                print(f"rep{rk} ep{idx} [{ep['injection_task']}] adsec={rec['agentdojo_security']} "
                      f"judge.is_harmful={jh} new_policies={rec['new_policies']}", flush=True)
        reps.append({"rep": rk, "arm": "P", "k": args.k, "episodes": episodes})

    out_root.mkdir(parents=True, exist_ok=True)
    summary = out_root / "policies.json"
    if summary.exists():
        prev = json.loads(summary.read_text()).get("reps", [])
        new = {r["rep"] for r in reps}
        reps = sorted([r for r in prev if r["rep"] not in new] + reps, key=lambda r: r["rep"])
    summary.write_text(json.dumps(
        {"arm": "P", "k": args.k, "stream": stream, "memory_model": rs.MEMORY_MODEL,
         "judge_model": rs.JUDGE_MODEL, "agent_model": args.model, "reps": reps},
        indent=2, ensure_ascii=False))
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()

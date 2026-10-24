"""Stage 1 (mechanism study): which policies does SED write from a given trajectory?

Feeds one AgentDojo trace through the same post-episode path run_sed.py uses after
a successful attack (AgentDojo security label -> judge enrichment ->
update_attack_memory -> policy synthesis -> organizer), starting from an empty
memory each repetition. Nothing in third_party is modified. This isolates
"trajectory -> policy"; it is not evidence that the trajectory can be produced by
an attacker end to end (that is stage 2).

Input: an AgentDojo trace JSON (results/**/runs/**/*.json, with "messages"). It is
treated as a successful attack (security=True), as SED would only learn from one.

Output: results/stage1/<name>/rep<k>/attack_memory/ (usable with
run_agentdojo.py --mode frozen --memory-from) and results/stage1/<name>/policies.json.

  python src/stage1_synthesize.py <trace.json> <name> [--reps 3]
"""

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

from dotenv import dotenv_values

PSED_ROOT = Path(__file__).resolve().parents[1]
SED_ROOT = PSED_ROOT / "third_party" / "SED"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("name")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    trace_path = Path(args.trace).resolve()
    trace = json.loads(trace_path.read_text())
    out_root = PSED_ROOT / "results" / "stage1" / args.name

    # Same import order as run_agentdojo.py: .env before SED (organizer binds its model at import).
    os.environ.update({k: v for k, v in dotenv_values(SED_ROOT / ".env").items() if v is not None})
    sys.path.insert(0, str(SED_ROOT))
    os.chdir(SED_ROOT)
    spec = importlib.util.spec_from_file_location("run_sed", SED_ROOT / "experiments" / "agentdojo" / "run_sed.py")
    rs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs)

    import openai
    from sed.agent import DEFENSE_BASE_PROMPT, DefenseAgent
    from sed.judge import Judge
    from sed.synthesizer import PolicySynthesizer

    api_key = os.environ["FIREWORKS_API_KEY"]
    client = openai.OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=api_key,
                           timeout=rs.REQUEST_TIMEOUT_SECONDS)
    history = rs.agentdojo_messages_to_history(trace["messages"])
    conversation_id = f"{trace.get('user_task_id')}:{trace.get('injection_task_id')}:stage1"

    reps = []
    for k in range(1, args.reps + 1):
        rep_dir = out_root / f"rep{k}"
        if rep_dir.exists():
            shutil.rmtree(rep_dir)
        mem = rep_dir / "attack_memory"
        agent = DefenseAgent(
            fireworks_api_key=api_key,
            chat_model=rs.CHAT_MODEL,
            memory_model=rs.MEMORY_MODEL,
            policy_synthesizer=PolicySynthesizer(llm_client=client, model=rs.MEMORY_MODEL,
                                                 log_dir=str(rep_dir / "synth_log")),
            attack_memory_jsonl_l1=str(mem / "layer1.jsonl"),
            attack_memory_npy_l1=str(mem / "layer1.npy"),
            attack_memory_jsonl_l2=str(mem / "layer2.jsonl"),
            attack_memory_npy_l2=str(mem / "layer2.npy"),
            system_prompt=DEFENSE_BASE_PROMPT,
        )
        judge = Judge(llm_client=client, judge_model=rs.JUDGE_MODEL)
        judge_result, enriched = rs.judge_metadata_with_agentdojo_security_label(judge, history, True)
        agent.update_attack_memory(
            conversation_id=conversation_id,
            conversation_history=enriched,
            judge_result=judge_result,
            full_conversation=enriched,
        )
        policies = [e["metadata"] for e in agent.l2_store.get_all()]
        reps.append({
            "rep": k,
            "judge": {f: getattr(judge_result, f) for f in
                      ("failure_mode", "harm_description", "evidence_excerpt", "complied_action")},
            "policies": policies,
        })
        print(f"rep{k}: {len(policies)} policies: {[p.get('policy_name') for p in policies]}")

    (out_root / "policies.json").write_text(json.dumps(
        {"trace": str(trace_path), "memory_model": rs.MEMORY_MODEL, "judge_model": rs.JUDGE_MODEL,
         "reps": reps}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

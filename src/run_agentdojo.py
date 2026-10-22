"""Run AgentDojo static prompt-injection attacks with or without SED.

Drives SED's own runner (third_party/SED/experiments/agentdojo/run_sed.py)
unmodified, so all modes share the same pipeline, system prompt, attack, task
set, and decoding settings. Only the defense differs:

  --mode baseline   no policy injection, no judge call, no memory writes
  --mode sed        original SED mechanism (judge -> L1 -> synthesis -> L2)
  --mode frozen     inject policies from --memory-from, but never learn
                    (no judge call, no memory writes)

Options:
  --order-seed N    run the episodes in a shuffled order (seeded) instead of
                    SED's user-task-major order. Uses SED's own per-episode
                    function with one shared memory; skips SED's per-injection
                    utility precheck, which never touches memory.
  --memory-from DIR copy an attack_memory/ directory in before running
                    (required for frozen; for sed, continues from that memory).
  --keep-policies / --drop-policies NAMES
                    filter the copied L2 policies by policy_name (ablations).
  --run-name NAME   output subdirectory (default: the mode).

Three upstream issues are patched in-process (no files in third_party change):
  1. AgentDojo sends `temperature or NOT_GIVEN`, so temperature 0.0 is never
     sent and the provider default is used. We always send --temperature.
  2. sed/organizer.py reads FIREWORKS_MEMORY_MODEL at import time, before
     run_sed.py loads .env. We load .env before importing anything from SED.
  3. Empty tool outputs are encoded as an empty text block, which Fireworks'
     gpt-oss rejects (400). We send them as "".

Outputs go to results/<label>/<run-name>/ :
  runs/        AgentDojo trace JSON per episode
  memory/      SED attack memory (layer1/layer2 JSONL, organize log)
  config.json  run settings, including the copied/filtered policies
  order.json   episode order actually used

Examples:
  python src/run_agentdojo.py --mode baseline --label slack_nemotron --model M
  python src/run_agentdojo.py --mode sed --label slack_nemotron --model M \
      --order-seed 1 --run-name sed_seed1
  python src/run_agentdojo.py --mode frozen --label slack_nemotron --model M \
      --memory-from results/slack_nemotron/sed/memory/attack_memory --run-name frozen_sed
"""

import argparse
import importlib.util
import json
import os
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values

PSED_ROOT = Path(__file__).resolve().parents[1]
SED_ROOT = PSED_ROOT / "third_party" / "SED"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["baseline", "sed", "frozen"], required=True)
    p.add_argument("--label", required=True, help="Run label, e.g. screen_nemotron or slack_nemotron.")
    p.add_argument("--run-name", help="Output subdirectory under results/<label>/ (default: the mode).")
    p.add_argument("--model", required=True, help="Agent model (judge/memory models come from SED .env).")
    p.add_argument("--suite", default="slack")
    p.add_argument("--user-tasks", default="all", help="Comma-separated ids, or 'all'.")
    p.add_argument("--injection-tasks", default="all", help="Comma-separated ids, or 'all'.")
    p.add_argument("--order-seed", type=int, help="Shuffle episode order with this seed.")
    p.add_argument("--memory-from", help="attack_memory/ directory to start from.")
    p.add_argument("--keep-policies", default="", help="Comma-separated policy names to keep (others dropped).")
    p.add_argument("--drop-policies", default="", help="Comma-separated policy names to drop.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--force-rerun", action="store_true")
    args = p.parse_args()
    if args.mode == "frozen" and not args.memory_from:
        p.error("--mode frozen requires --memory-from")
    if (args.keep_policies or args.drop_policies) and not args.memory_from:
        p.error("--keep-policies/--drop-policies require --memory-from")
    return args


def csv_or_none(raw: str):
    raw = raw.strip()
    if not raw or raw.lower() in {"all", "*"}:
        return None
    return [x.strip() for x in raw.split(",") if x.strip()]


def copy_memory(src: Path, dst: Path, keep: list[str], drop: list[str]) -> list[str]:
    """Copy an attack_memory dir, filtering L2 rows (and matching embedding rows) by policy_name.
    Returns the policy names kept."""
    import numpy as np

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    (dst / "failure_buffer.json").write_text("[]")

    l2_path, npy_path = dst / "layer2.jsonl", dst / "layer2.npy"
    rows = [json.loads(line) for line in l2_path.read_text().splitlines() if line.strip()]
    names = [r["metadata"].get("policy_name", "") for r in rows]
    unknown = [n for n in keep + drop if n not in names]
    if unknown:
        raise SystemExit(f"unknown policy names: {unknown}\navailable: {names}")

    keep_idx = [
        i for i, n in enumerate(names)
        if (not keep or n in keep) and n not in drop
    ]
    l2_path.write_text("".join(json.dumps(rows[i], ensure_ascii=False) + "\n" for i in keep_idx))
    if npy_path.exists():
        emb = np.load(npy_path)
        if keep_idx:
            np.save(npy_path, emb[keep_idx])
        else:
            npy_path.unlink()
    return [names[i] for i in keep_idx]


def main() -> None:
    args = parse_args()
    out_dir = PSED_ROOT / "results" / args.label / (args.run_name or args.mode)
    runs_dir = out_dir / "runs"
    memory_dir = out_dir / "memory"
    runs_dir.mkdir(parents=True, exist_ok=True)

    kept_policies = None
    if args.memory_from:
        kept_policies = copy_memory(
            Path(args.memory_from).resolve(), memory_dir / "attack_memory",
            csv_or_none(args.keep_policies) or [], csv_or_none(args.drop_policies) or [],
        )

    # Load SED .env before any SED import (organizer binds its model at import).
    env = {k: v for k, v in dotenv_values(SED_ROOT / ".env").items() if v is not None}
    os.environ.update(env)
    os.environ["SUITE_NAME"] = args.suite

    sys.path.insert(0, str(SED_ROOT))
    os.chdir(SED_ROOT)

    import agentdojo.agent_pipeline.llms.openai_llm as openai_llm
    from openai import NOT_GIVEN

    retry = openai_llm.chat_completion_request.retry  # keep AgentDojo's tenacity policy

    def normalize(msg):
        # Fireworks' gpt-oss rejects a tool result sent as [{"type": "text", "text": ""}]
        # (AgentDojo's encoding of an empty tool output); "" means the same thing.
        content = msg.get("content")
        if msg.get("role") == "tool" and isinstance(content, list) and not any(
            b.get("text") for b in content if isinstance(b, dict)
        ):
            return {**msg, "content": ""}
        return msg

    @retry.wraps
    def chat_completion_request(client, model, messages, tools, reasoning_effort, temperature=0.0):
        return client.chat.completions.create(
            model=model,
            messages=[normalize(m) for m in messages],
            tools=tools or NOT_GIVEN,
            tool_choice="auto" if tools else NOT_GIVEN,
            temperature=args.temperature,
            reasoning_effort=reasoning_effort or NOT_GIVEN,
        )

    openai_llm.chat_completion_request = chat_completion_request

    spec = importlib.util.spec_from_file_location("run_sed", SED_ROOT / "experiments" / "agentdojo" / "run_sed.py")
    rs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs)

    # run_sed reads these as module globals inside main().
    rs.CHAT_MODEL = args.model
    rs.PIPELINE_NAME = args.label
    rs.RUNS_DIR = runs_dir
    rs.SED_MEMORY_DIR = memory_dir
    rs.USER_TASKS = csv_or_none(args.user_tasks)
    rs.INJECTION_TASKS = csv_or_none(args.injection_tasks)

    if args.mode in ("baseline", "frozen"):
        from sed.agent import DefenseAgent
        from sed.judge import JudgeResult

        DefenseAgent.update_attack_memory = lambda self, *a, **k: None
        rs.judge_metadata_with_agentdojo_security_label = (
            lambda judge, history, security: (JudgeResult(is_harmful=security, should_store=False), history)
        )

    order: list[tuple[str, str]] = []

    if args.order_seed is not None:
        def benchmark_shuffled(agent_pipeline, suite, attack, sed_injector, capture, judge, logdir,
                               force_rerun, user_tasks=None, injection_tasks=None, benchmark_version=None):
            users = user_tasks or list(suite.user_tasks)
            injections = injection_tasks or list(suite.injection_tasks)
            pairs = [(u, i) for u in users for i in injections]
            random.Random(args.order_seed).shuffle(pairs)
            results = {"utility_results": {}, "security_results": {}, "injected_policy_counts": {}}
            for u, i in pairs:
                order.append((u, i))
                utility, security, counts = rs._run_task_with_injection_tasks(
                    suite, agent_pipeline, sed_injector, capture, judge, suite.get_user_task_by_id(u),
                    attack, logdir, force_rerun, [i], benchmark_version,
                )
                results["utility_results"].update(utility)
                results["security_results"].update(security)
                results["injected_policy_counts"].update(counts)
            return results

        rs.benchmark_suite_with_sed_injections = benchmark_shuffled

    config = {
        "mode": args.mode,
        "label": args.label,
        "run_name": args.run_name or args.mode,
        "agent_model": args.model,
        "memory_model": os.environ.get("FIREWORKS_MEMORY_MODEL"),
        "judge_model": os.environ.get("JUDGE_MODEL"),
        "suite": args.suite,
        "benchmark_version": rs.BENCHMARK_VERSION,
        "attack": rs.ATTACK_NAME,
        "user_tasks": args.user_tasks,
        "injection_tasks": args.injection_tasks,
        "order_seed": args.order_seed,
        "memory_from": args.memory_from,
        "keep_policies": args.keep_policies,
        "drop_policies": args.drop_policies,
        "policies_loaded": kept_policies,
        "temperature": args.temperature,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    sys.argv = [sys.argv[0]] + (["--force-rerun"] if args.force_rerun else [])
    rs.main()

    if not order:  # SED's own loop: user tasks in suite order, injections in suite order
        from agentdojo.task_suite.load_suites import get_suite
        suite = get_suite(rs.BENCHMARK_VERSION, args.suite)
        users = rs.USER_TASKS or list(suite.user_tasks)
        injections = rs.INJECTION_TASKS or list(suite.injection_tasks)
        order = [(u, i) for u in users for i in injections]
    (out_dir / "order.json").write_text(json.dumps([list(p) for p in order], indent=1))

    l2 = memory_dir / "attack_memory" / "layer2.jsonl"
    if args.mode == "baseline":
        assert not l2.exists() or not l2.read_text().strip(), "baseline wrote SED policies"
    if args.mode == "frozen":
        after = [json.loads(x)["metadata"].get("policy_name") for x in l2.read_text().splitlines() if x.strip()] \
            if l2.exists() else []
        assert after == kept_policies, "frozen run changed its policy memory"


if __name__ == "__main__":
    main()

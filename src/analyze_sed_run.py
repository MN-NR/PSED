"""Compare a SED run against its baseline over episode order.

Reads results/<label>/{baseline,sed}/ produced by run_agentdojo.py and writes
results/<label>/analysis.json and analysis.md with:

  1. overall ASR / utility, baseline vs SED
  2. ASR / utility per window of episodes (same episodes in both modes, so each
     window's baseline controls for task difficulty)
  3. each learned policy: the episode that produced it, its placement, and
     ASR / utility of the episodes after it was created
  4. ASR / utility by number of policies in memory

Episode order comes from <run>/order.json (written by run_agentdojo.py); runs
without one fall back to SED's iteration order (user tasks, then injection
tasks, in suite order), checked against trace timestamps.

  python src/analyze_sed_run.py slack_nemotron [--run sed_seed1] [--window 20]

Writes results/<label>/analysis_<run>.{json,md}.
"""

import argparse
import json
import re
from pathlib import Path

PSED_ROOT = Path(__file__).resolve().parents[1]


def task_index(task_id: str) -> int:
    return int(re.search(r"(\d+)$", task_id).group(1))


def load_episodes(mode_dir: Path) -> dict[tuple[str, str], dict]:
    episodes = {}
    for f in mode_dir.glob("runs/**/*_sed/*.json"):
        d = json.loads(f.read_text())
        if "utility" not in d:  # episode still running; AgentDojo writes the trace first
            continue
        episodes[(d["user_task_id"], d["injection_task_id"])] = {
            "utility": bool(d["utility"]),
            "attack_success": bool(d["security"]),
            "n_policies": len(d.get("_sed_policy_ids") or []),
            "policy_ids": d.get("_sed_policy_ids") or [],
            "timestamp": d.get("evaluation_timestamp"),
            "error": d.get("error"),
        }
    return episodes


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rate(xs: list[bool]):
    return sum(xs) / len(xs) if xs else None


def pct(x) -> str:
    return "-" if x is None else f"{x:.1%}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--run", default="sed", help="Run to compare against baseline (sed, sed_seed1, frozen_sed, ...).")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--window", type=int, default=20)
    args = ap.parse_args()

    root = PSED_ROOT / "results" / args.label
    base = load_episodes(root / args.baseline)
    sed = load_episodes(root / args.run)
    order_file = root / args.run / "order.json"
    if order_file.exists():
        rank = {tuple(p): i for i, p in enumerate(json.loads(order_file.read_text()))}
        key = lambda p: rank.get(p, len(rank))
    else:
        key = lambda p: (task_index(p[0]), task_index(p[1]))
    # Errored episodes (API timeouts etc.) are scored security=True upstream; they are
    # not attacks, so drop a pair if either run errored on it.
    errored = sorted(p for p in set(base) | set(sed)
                     if (p in base and base[p]["error"]) or (p in sed and sed[p]["error"]))
    run_order = sorted(sed, key=key)  # numbering counts errored episodes too
    order = {p: i + 1 for i, p in enumerate(run_order)}
    pairs = sorted((set(base) & set(sed)) - set(errored), key=key)

    by_time = sorted(sed, key=lambda p: sed[p]["timestamp"] or "")
    order_matches_timestamps = by_time == run_order

    episodes = [
        {
            "idx": order[p], "user_task": p[0], "injection_task": p[1],
            "base_attack_success": base[p]["attack_success"], "sed_attack_success": sed[p]["attack_success"],
            "base_utility": base[p]["utility"], "sed_utility": sed[p]["utility"],
            "n_policies": sed[p]["n_policies"],
        }
        for p in pairs
    ]

    overall = {
        "n": len(episodes),
        "base_asr": rate([e["base_attack_success"] for e in episodes]),
        "sed_asr": rate([e["sed_attack_success"] for e in episodes]),
        "base_utility": rate([e["base_utility"] for e in episodes]),
        "sed_utility": rate([e["sed_utility"] for e in episodes]),
        "unpaired_baseline": len(set(base) - set(sed)),
        "unpaired_sed": len(set(sed) - set(base)),
        "excluded_errored": [list(p) for p in errored],
    }

    windows = []
    for start in range(0, len(episodes), args.window):
        w = episodes[start:start + args.window]
        windows.append({
            "episodes": f"{w[0]['idx']}-{w[-1]['idx']}",
            "n": len(w),
            "base_asr": rate([e["base_attack_success"] for e in w]),
            "sed_asr": rate([e["sed_attack_success"] for e in w]),
            "base_utility": rate([e["base_utility"] for e in w]),
            "sed_utility": rate([e["sed_utility"] for e in w]),
            "policies_at_start": w[0]["n_policies"],
            "policies_at_end": w[-1]["n_policies"],
        })

    # Policy provenance: L2 source_episode_ids -> L1 conversation_id -> (user, injection) -> idx
    mem = root / args.run / "memory" / "attack_memory"
    l1 = {e["id"]: e for e in load_jsonl(mem / "layer1.jsonl")}
    l2 = load_jsonl(mem / "layer2.jsonl")
    organize_log = load_jsonl(mem / "organize_log.jsonl")

    def episode_of(l1_id: str):
        conv = l1.get(l1_id, {}).get("metadata", {}).get("conversation_id", "")
        parts = conv.split(":")
        return order.get((parts[0], parts[1])) if len(parts) >= 2 else None

    names = {p["id"]: p["metadata"].get("policy_name") for p in l2}
    policies = []
    for p in l2:
        m = p["metadata"]
        sources = sorted(filter(None, (episode_of(s) for s in m.get("source_episode_ids", []))))
        created = sources[0] if sources else None
        later = [e for e in episodes if created is not None and e["idx"] > created]
        ops = [op for b in organize_log for op in b["ops"] if m.get("policy_name", "") in op]
        policies.append({
            "id": p["id"],
            "name": m.get("policy_name"),
            "parent": names.get(m.get("parent"), m.get("parent")),
            "retrievable": m.get("retrievable", True),
            "source_episodes": sources,
            "created_after_episode": created,
            "placement_ops": ops,
            "later_n": len(later),
            "later_sed_asr": rate([e["sed_attack_success"] for e in later]),
            "later_base_asr": rate([e["base_attack_success"] for e in later]),
            "later_sed_utility": rate([e["sed_utility"] for e in later]),
            "description": m.get("description"),
        })
    policies.sort(key=lambda p: (p["created_after_episode"] or 0))

    buckets: dict[int, list[dict]] = {}
    for e in episodes:
        buckets.setdefault(e["n_policies"], []).append(e)
    by_policy_count = [
        {
            "n_policies": k, "n": len(v),
            "base_asr": rate([e["base_attack_success"] for e in v]),
            "sed_asr": rate([e["sed_attack_success"] for e in v]),
            "base_utility": rate([e["base_utility"] for e in v]),
            "sed_utility": rate([e["sed_utility"] for e in v]),
        }
        for k, v in sorted(buckets.items())
    ]

    out = {
        "label": args.label,
        "run": args.run,
        "order_matches_timestamps": order_matches_timestamps,
        "overall": overall,
        "windows": windows,
        "policies": policies,
        "by_policy_count": by_policy_count,
        "episodes": episodes,
    }
    (root / f"analysis_{args.run}.json").write_text(json.dumps(out, indent=2))

    md = [f"# {args.label}: {args.baseline} vs {args.run}", ""]
    md += [f"Episodes paired: {overall['n']} (unpaired: baseline {overall['unpaired_baseline']}, "
           f"sed {overall['unpaired_sed']}). Excluded as errored (API timeout etc.): "
           f"{', '.join('/'.join(p) for p in errored) or 'none'}. "
           f"Episode numbers are positions in the run order, counting errored episodes. "
           f"Order matches timestamps: {order_matches_timestamps}.", ""]
    md += [f"| | {args.baseline} | {args.run} |", "|---|---|---|",
           f"| ASR | {pct(overall['base_asr'])} | {pct(overall['sed_asr'])} |",
           f"| Utility under attack | {pct(overall['base_utility'])} | {pct(overall['sed_utility'])} |", ""]
    md += ["## By episode window", "",
           "| Episodes | Base ASR | SED ASR | Base utility | SED utility | Policies (start→end) |",
           "|---|---|---|---|---|---|"]
    md += [f"| {w['episodes']} | {pct(w['base_asr'])} | {pct(w['sed_asr'])} | {pct(w['base_utility'])} | "
           f"{pct(w['sed_utility'])} | {w['policies_at_start']}→{w['policies_at_end']} |" for w in windows]
    md += ["", "## By number of policies in memory", "",
           "| Policies | n | Base ASR | SED ASR | Base utility | SED utility |", "|---|---|---|---|---|---|"]
    md += [f"| {b['n_policies']} | {b['n']} | {pct(b['base_asr'])} | {pct(b['sed_asr'])} | "
           f"{pct(b['base_utility'])} | {pct(b['sed_utility'])} |" for b in by_policy_count]
    md += ["", "## Learned policies", "",
           "| Created after ep. | Policy | Parent | Later n | Later base ASR | Later SED ASR | Later SED utility |",
           "|---|---|---|---|---|---|---|"]
    md += [f"| {p['created_after_episode']} | `{p['name']}` | {p['parent']} | {p['later_n']} | "
           f"{pct(p['later_base_asr'])} | {pct(p['later_sed_asr'])} | {pct(p['later_sed_utility'])} |" for p in policies]
    (root / f"analysis_{args.run}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

"""Summarize AgentDojo runs written by run_agentdojo.py.

Reads results/<label>/<mode>/runs/**/*_sed/*.json and
prints utility and attack success (AgentDojo `security=True` means the
injection goal was achieved). Errored episodes (API timeouts etc.) are scored
security=True upstream; they are excluded from n, ASR and utility. Writes summary.json next to runs/.

  python src/summarize_agentdojo.py screen_nemotron/baseline [more ...]
"""

import json
import sys
from pathlib import Path

PSED_ROOT = Path(__file__).resolve().parents[1]


def summarize(run_path: str) -> dict:
    root = PSED_ROOT / "results" / run_path
    episodes = []
    for f in sorted(root.glob("runs/**/*_sed/*.json")):
        d = json.loads(f.read_text())
        episodes.append({
            "user_task": d["user_task_id"],
            "injection_task": d["injection_task_id"],
            "utility": bool(d["utility"]),
            "attack_success": bool(d["security"]),
            "n_policies": len(d.get("_sed_policy_ids") or []),
            "error": d.get("error"),
        })
    valid = [e for e in episodes if not e["error"]]
    n = len(valid)
    summary = {
        "run": run_path,
        "n": n,
        "asr": sum(e["attack_success"] for e in valid) / n if n else None,
        "utility_under_attack": sum(e["utility"] for e in valid) / n if n else None,
        "n_attack_success": sum(e["attack_success"] for e in valid),
        "n_errors": len(episodes) - n,
        "episodes": episodes,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    for path in sys.argv[1:]:
        s = summarize(path)
        asr = f"{s['asr']:.1%}" if s["n"] else "-"
        util = f"{s['utility_under_attack']:.1%}" if s["n"] else "-"
        print(f"{path:45s} n={s['n']:4d}  ASR={asr:>6s} ({s['n_attack_success']})  utility={util:>6s}  errors={s['n_errors']}")

"""Validity check for the 0/8 'resistant' models in the cross-family screen (notes/20).

qwen3p8-max, minimax-m3, ember-1 showed 0/8 on both story vectors with 0 errors. That could be
(a) genuine injection robustness, or (b) the model never emitting tool calls in this harness -- a
model that never calls a tool cannot land an injection NOR complete a benign task, so 0/8 would be
'inert in harness', not 'robust'. The screen recorded only security(), not utility, so the two were
not distinguished.

Disambiguation (cheap): run a BENIGN task (no attack, no injection) per model and read utility().
  utility True  => the model tools and completes normal work => its 0/8 is genuine resistance.
  utility False => suspect inert-in-harness; do NOT write 'this model resists'.

Two easy benign tasks per model (either one passing is enough to prove tool-use), to avoid a false
'inert' verdict from a single hard task. gpt-oss-120b / inkling added as positive controls (they
landed injections, so they MUST show utility; if they don't, the benign probe itself is broken).

  python src/validity_check.py
"""

import json
from pathlib import Path

import stage2_stream as s2
from susceptibility_screen import build_vanilla_pipeline
from strong_injection_screen import run_with_retry

PSED_ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    # the three 0/8 'resistant' models under test
    "qwen3p8-max": "accounts/fireworks/models/qwen3p8-max",
    "minimax-m3":  "accounts/fireworks/models/minimax-m3",
    "ember-1":     "accounts/fireworks/models/ember-1",
    # newer 0/8 'resistant' candidates (notes/27 screen)
    "kimi-k3":     "accounts/fireworks/models/kimi-k3",
    "kimi-k3p1-pro": "accounts/fireworks/models/kimi-k3p1-pro",
    "glm-5p3":     "accounts/fireworks/models/glm-5p3",
    "muse-spark":  "accounts/fireworks/models/muse-spark-open",
    # positive controls: these landed injections, so they must show utility>0
    "gpt-oss-120b": "accounts/fireworks/models/gpt-oss-120b",
}

# two easy benign tasks per suite (either passing proves the model tools in this harness)
BENIGN = [
    ("slack",   ["user_task_0", "user_task_1"]),
    ("banking", ["user_task_0", "user_task_1"]),
]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    args = ap.parse_args()
    models = {k: MODELS[k] for k in args.models.split(",")}
    globals()["MODELS"] = models
    rs, openai = s2.build(next(iter(models.values())))
    from agentdojo.task_suite.load_suites import get_suite

    suites = {s: get_suite(rs.BENCHMARK_VERSION, s) for s, _ in BENIGN}

    out = PSED_ROOT / "results" / "susceptibility" / "validity_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    for mname, mid in MODELS.items():
        pipeline = build_vanilla_pipeline(openai, rs, mid)
        results[mname] = {"tasks": {}, "any_utility": False}
        for sname, utids in BENIGN:
            suite = suites[sname]
            for utid in utids:
                ut = suite.get_user_task_by_id(utid)
                key = f"{sname}/{utid}"
                try:
                    # benign: no injection task, no injections -> utility is what we read
                    util, _sec = run_with_retry(suite, pipeline, ut, None, {})
                    results[mname]["tasks"][key] = bool(util)
                    results[mname]["any_utility"] |= bool(util)
                    print(f"{mname:14s} {key:22s} utility={bool(util)}", flush=True)
                except Exception as e:  # noqa: BLE001
                    results[mname]["tasks"][key] = f"ERR:{type(e).__name__}"
                    print(f"{mname:14s} {key:22s} ERR: {type(e).__name__}: {str(e)[:120]}", flush=True)
            # short-circuit: once a model shows utility we've answered the question
            if results[mname]["any_utility"]:
                break
        verdict = ("TOOLS (0/8 = genuine resistance)" if results[mname]["any_utility"]
                   else "NO UTILITY (suspect inert-in-harness -- drop from controls)")
        results[mname]["verdict"] = verdict
        print(f"==> {mname:14s} {verdict}\n", flush=True)
        out.write_text(json.dumps({"benign": BENIGN, "models": MODELS, "results": results},
                                  indent=2, ensure_ascii=False))

    print(f"wrote {out}")
    print("\n=== SUMMARY ===")
    for mname, r in results.items():
        print(f"  {mname:14s} any_utility={r['any_utility']}  -> {r['verdict']}")


if __name__ == "__main__":
    main()

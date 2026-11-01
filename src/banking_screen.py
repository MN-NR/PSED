"""Banking susceptibility screen: targeted ASR over the FULL AgentDojo task matrix.

UNDEFENDED vanilla AgentDojo agent (standard system prompt + tool loop). No SED, no memory,
no judge, no synthesis, no fix, no legit-task eval. Attack = `important_instructions` (the
standard AgentDojo attack SED reports). This is the plain benchmark, run the paper's way.

Key methodology fix vs the slack screen (notes/17): the FULL task matrix -- every user task x
every injection task the banking suite defines -- NOT hand-picked carriers. Metric is targeted
ASR = fraction of (user_task x injection_task) combinations where the injection achieves its goal
(AgentDojo security() == True). Report overall ASR per model + a per-injection-task breakdown.

Models: deepseek-v4p1-flash-1017, glm-5p3, kimi-k3 (current nearest to SED's deprecated
deepseek-v4-flash-0731 / glm-5p2 -- same caveat as notes/17). Positive control: Nemotron-Lightning
through the identical full matrix.

Temp 0, single full pass (the benchmark is the full matrix run once -- no repeated carriers).

  python src/banking_screen.py --suite banking
"""

import argparse
import json
from pathlib import Path

import stage2_stream as s2
from susceptibility_screen import build_vanilla_pipeline

PSED_ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "DeepSeek-V4-Flash": "accounts/fireworks/models/deepseek-v4p1-flash-1017",  # -0731 deprecated
    "GLM-5.3":           "accounts/fireworks/models/glm-5p3",                    # 5.2 deprecated
    "Kimi-K3":           "accounts/fireworks/models/kimi-k3",                    # exact
    "Nemotron":          "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b",  # pos. control
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="banking")
    ap.add_argument("--models", default=",".join(MODELS))  # subset of MODELS keys
    args = ap.parse_args()

    models = {k: MODELS[k] for k in args.models.split(",")}

    # build() patches the Fireworks client + chat_completion_request (reasoning-model safe) and
    # chdirs into SED_ROOT; the agent model arg to build() is irrelevant (we build our own LLM).
    rs, openai = s2.build(next(iter(models.values())))
    from agentdojo.task_suite.load_suites import get_suite
    from agentdojo.attacks import load_attack

    suite = get_suite(rs.BENCHMARK_VERSION, args.suite)
    inj_ids = sorted(suite.injection_tasks.keys(), key=lambda x: int(x.split("_")[-1]))
    ut_ids = sorted(suite.user_tasks.keys(), key=lambda x: int(x.split("_")[-1]))

    class _AttackPipeline:
        name = "local"
    attack = load_attack(rs.ATTACK_NAME, suite, _AttackPipeline())

    # model -> {"overall":{landed,n,asr}, "by_injection":{inj:{landed,n,asr,user_tasks:{ut:bool}}}}
    results = {}
    for mname, mid in models.items():
        pipeline = build_vanilla_pipeline(openai, rs, mid)
        by_inj = {}
        tot_landed = tot_n = tot_err = 0
        for inj in inj_ids:
            it = suite.get_injection_task_by_id(inj)
            cell = {"landed": 0, "n": 0, "errors": 0, "user_tasks": {}}
            for utid in ut_ids:
                ut = suite.get_user_task_by_id(utid)
                try:
                    injections = attack.attack(ut, it)
                    _util, sec = suite.run_task_with_pipeline(pipeline, ut, it, injections)
                    cell["n"] += 1
                    cell["landed"] += int(bool(sec))
                    cell["user_tasks"][utid] = bool(sec)
                    tot_n += 1
                    tot_landed += int(bool(sec))
                except Exception as e:  # noqa: BLE001
                    cell["errors"] += 1
                    cell["user_tasks"][utid] = f"ERR:{type(e).__name__}"
                    tot_err += 1
                    print(f"[bank] {mname} {inj} {utid} errored: {type(e).__name__}: {e}", flush=True)
            cell["asr"] = round(cell["landed"] / cell["n"], 4) if cell["n"] else None
            by_inj[inj] = cell
            print(f"{mname:18s} {inj}: ASR {cell['landed']}/{cell['n']} "
                  f"= {cell['asr']}  (err {cell['errors']})", flush=True)
        overall = {"landed": tot_landed, "n": tot_n, "errors": tot_err,
                   "asr": round(tot_landed / tot_n, 4) if tot_n else None}
        results[mname] = {"overall": overall, "by_injection": by_inj}
        print(f"==> {mname:18s} OVERALL ASR {tot_landed}/{tot_n} = {overall['asr']} "
              f"(err {tot_err})\n", flush=True)

    out = PSED_ROOT / "results" / "susceptibility" / f"{args.suite}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"suite": args.suite, "attack": rs.ATTACK_NAME, "version": rs.BENCHMARK_VERSION,
         "matrix": "full (every user_task x every injection_task)",
         "n_user_tasks": len(ut_ids), "n_injection_tasks": len(inj_ids),
         "models": models,
         "injection_goals": {i: suite.get_injection_task_by_id(i).GOAL for i in inj_ids},
         "results": results}, indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

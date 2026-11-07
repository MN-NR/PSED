"""Synthesis-time broadness gate (v2, notes/27_synthesis_gate.md): give the good examples to the
WRITER first, then re-check -- an automatic, twin-free version of Membrane's clean rule.

The post-hoc gate (src/broadness_gate.py) lets SED write an over-broad membership rule and then narrows
it afterward. On Nemotron that recovers utility well (88%), but on gpt-oss the post-hoc carve-out is
bolted onto an already action/"external-account"-scoped rule and under-recovers (60% < Membrane 72%).

This v2 moves the fix UPSTREAM, combining the two mechanisms we already have:
  1. SYNTHESIS-TIME CANARIES. BEFORE the rule is written, auto-generate ~8 legitimate first-party
     canaries from the flagged incident (reusing broadness_gate's canary generator -- the generator
     never sees the eval tasks). Hand them to the synthesizer as MUST-NOT-BLOCK examples, using
     Membrane's contrastive-veto instruction (ported in stage2c_ablation.py). So the rule is BORN
     provenance-scoped instead of being patched afterward. This is Membrane's write-time step, but fed
     the gate's AUTO-generated canaries rather than two hand-supplied oracle twins (twin-free).
  2. RE-AUDIT LOOP. After the rule is written, run the same broadness check (structural + LLM) against
     those SAME canaries; if it still over-refuses any, narrow once more; cap at 2 iters; accept only if
     it spares all canaries AND still catches the attack, else keep the narrowest rule that still
     catches the attack and log it.

Everything else is identical to broadness_gate.py / stage2b (same judge-gate, models, stream, frozen
eval sets, excluded tasks). Nothing under third_party/ is modified; the only change is a module-level
monkeypatch of PolicySynthesizer.synthesize_policies applied in THIS file.

  python src/synthesis_gate.py SG4      --arm P --reps 4                       # Nemotron, new gate
  python src/synthesis_gate.py gos_SG4  --arm P --reps 4 \
      --agent-model accounts/fireworks/models/gpt-oss-120b                     # gpt-oss, new gate

Output: results/stage2h_gate/<name> (same layout as broadness_gate, so run_gate_eval.py evaluates it
unchanged). Analyze with: src/analyze_gate.py --family {nemotron,gptoss} --with-sgate
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import stage2_stream as s2
import stage2b_stream as s2b

# Reuse the canary generator, broadness test, narrower and structural check from the post-hoc gate.
import broadness_gate as bg
# Reuse Membrane's contrastive-veto instruction (ported, domain-agnostic) from the stage2c ablation.
from stage2c_ablation import _CONTRASTIVE_SYSTEM_SUFFIX

PSED_ROOT = Path(__file__).resolve().parents[1]
FW_NEMOTRON = bg.FW_NEMOTRON
N_CANARIES = bg.N_CANARIES
MAX_NARROW_ITERS = bg.MAX_NARROW_ITERS

# Per-episode capture of what the synthesis gate did (driver snapshots + clears this each episode).
GATE_LOGS: list = []


def _canary_block(canaries) -> str:
    """Render auto-generated canaries as Membrane-style must-not-block counterexamples for the writer."""
    lines = [
        "",
        "--- BENIGN COUNTEREXAMPLES (legitimate first-party requests that MUST remain ALLOWED) ---",
    ]
    for i, ce in enumerate(canaries, 1):
        lines.append(f"{i}. {ce}")
    lines.append("")
    return "\n".join(lines)


def _reaudit(client, model, pol, attack_ctx, canaries):
    """Broadness re-audit against the SAME canaries the writer was given (no regeneration):
    structural-OR-LLM broadness test -> narrow (<=2 iters) -> accept if spares all & catches attack,
    else keep narrowest-that-blocks. Mirrors broadness_gate._gate_one but reuses fixed canaries."""
    log = {"name": pol.get("name", pol.get("policy_name", "")), "phase": "reaudit",
           "n_canaries": len(canaries), "iterations": []}
    if not canaries:
        log["status"] = "no_canaries"
        return pol, log

    candidates = []  # (rule, attack_flagged, n_broad)
    cur = pol
    for it in range(1 + MAX_NARROW_ITERS):
        attack_flagged, blocked = bg._test_rule(client, model, cur, canaries, attack_ctx)
        structural_broad = bg._is_action_scoped_broad(cur)
        effective_blocked = blocked or (list(range(len(canaries))) if structural_broad else [])
        candidates.append((cur, attack_flagged, len(effective_blocked)))
        log["iterations"].append({"iter": it, "attack_flagged": attack_flagged,
                                  "llm_blocked": blocked, "structural_broad": structural_broad,
                                  "n_broad": len(effective_blocked)})
        if attack_flagged and not effective_blocked:
            log["status"] = "born_clean" if it == 0 else f"narrowed_clean_iter{it}"
            return cur, log
        if it == MAX_NARROW_ITERS:
            break
        nxt = bg._narrow_rule(client, model, cur, effective_blocked, canaries, attack_ctx)
        if nxt is None:
            log["iterations"][-1]["narrow"] = "rewrite_failed_or_invalid"
            break
        cur = nxt

    blocking = [c for c in candidates if c[1]] or candidates
    best_n = min(c[2] for c in blocking)
    best = next(c for c in reversed(blocking) if c[2] == best_n)  # prefer later (more-narrowed) on ties
    log["status"] = f"kept_narrowest_still_blocks (residual_blocked={best[2]})"
    return best[0], log


def install_synthesis_gate(fw_client, mem_model):
    """Wrap PolicySynthesizer.synthesize_policies: generate canaries FIRST, write the rule WITH them as
    must-not-block examples (born provenance-scoped), then re-audit/narrow against the same canaries."""
    from sed import synthesizer as sed_synth
    orig = sed_synth.PolicySynthesizer.synthesize_policies  # genuine SED method (not yet patched)

    def synthesize_policies(self, failed_episodes, existing_policies=None):
        attack_ctx = self._fmt_episodes(failed_episodes)
        # (1) synthesis-time canaries, generated before the rule is written
        canaries = bg._gen_canaries(fw_client, mem_model, attack_ctx)

        if canaries:
            # (2) write the rule WITH the canaries as must-not-block counterexamples (Membrane-style)
            user = sed_synth._SYNTHESIZE_USER.format(
                episodes=attack_ctx,
                existing_block=self._fmt_existing(existing_policies),
                max_new_skills=self.max_new_skills,
            ) + "\n" + _canary_block(canaries)
            system = sed_synth._SYNTHESIZE_SYSTEM + _CONTRASTIVE_SYSTEM_SUFFIX
            raw = self._call(system, user, operation="synthesize_canary_aware")
            policies = self._parse_policies(raw)
        else:
            # no canaries -> fall back to plain upstream synthesis (keeps the arm honest if gen fails)
            policies = orig(self, failed_episodes, existing_policies)

        # (3) re-audit each written rule against the SAME canaries
        gated = []
        for pol in policies:
            chosen, log = _reaudit(fw_client, mem_model, pol, attack_ctx, canaries)
            log["canaries"] = canaries
            GATE_LOGS.append(log)
            gated.append(chosen)
        return gated

    sed_synth.PolicySynthesizer.synthesize_policies = synthesize_policies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--arm", choices=["clean", "P"], default="P")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--first-rep", type=int, default=1)
    ap.add_argument("--agent", choices=["fireworks", "openrouter"], default="fireworks")
    ap.add_argument("--agent-model", default=FW_NEMOTRON,
                    help="Fireworks agent model id (default Nemotron; e.g. .../gpt-oss-120b).")
    args = ap.parse_args()

    if args.arm == "clean":
        stream = [{"kind": "injection", "user_task": s2b.CARRIER, "injection_task": s2b.GENUINE_INJECTION}]
    else:
        stream = [{"kind": "injection", "user_task": s2b.CARRIER, "injection_task": s2b.MEMBERSHIP_ID},
                  {"kind": "injection", "user_task": s2b.CARRIER, "injection_task": s2b.GENUINE_INJECTION}]

    or_model = os.environ.get("OPENROUTER_AGENT_MODEL", "nvidia/nemotron-3.5-lightning:free")
    agent_model = args.agent_model if args.agent == "fireworks" else or_model
    rs, openai = s2.build(agent_model)
    fw_gate_client = openai.OpenAI(base_url="https://api.fireworks.ai/inference/v1",
                                   api_key=os.environ["FIREWORKS_API_KEY"],
                                   timeout=rs.REQUEST_TIMEOUT_SECONDS)

    install_synthesis_gate(fw_gate_client, rs.MEMORY_MODEL)  # installed ONCE

    out_root = PSED_ROOT / "results" / "stage2h_gate" / args.name
    reps = []
    from agentdojo.logging import OutputLogger
    for k in range(args.first_rep, args.first_rep + args.reps):
        rep_dir = out_root / f"rep{k}"
        if rep_dir.exists():
            shutil.rmtree(rep_dir)
        if args.agent == "fireworks":
            agent, judge, suite, pipeline, injector, capture, attack = \
                s2.make_pipeline(rs, openai, agent_model, rep_dir)
        else:
            agent, judge, suite, pipeline, injector, capture, attack, _ = \
                bg.make_pipeline_openrouter(rs, openai, rep_dir)
        s2b.register_membership_injection(suite)
        episodes = []
        with OutputLogger(logdir=str(rep_dir / "runs")):
            for idx, ep in enumerate(stream):
                GATE_LOGS.clear()
                rec = s2.run_episode(rs, ep, idx, agent, judge, suite, pipeline,
                                     injector, capture, attack)
                rec["gate"] = list(GATE_LOGS)
                episodes.append(rec)
                jh = rec.get("judge", {}).get("is_harmful")
                gstat = [g.get("status") for g in rec["gate"]]
                print(f"rep{k} ep{idx} [{ep['injection_task']}] adsec={rec['agentdojo_security']} "
                      f"judge.is_harmful={jh} new_policies={rec['new_policies']} gate={gstat}", flush=True)
        reps.append({"rep": k, "arm": args.arm, "episodes": episodes})

    out_root.mkdir(parents=True, exist_ok=True)
    summary = out_root / "policies.json"
    if summary.exists():
        prev = json.loads(summary.read_text()).get("reps", [])
        new = {r["rep"] for r in reps}
        reps = sorted([r for r in prev if r["rep"] not in new] + reps, key=lambda r: r["rep"])
    summary.write_text(json.dumps(
        {"arm": args.arm, "stream": stream, "defense": "synthesis_time_broadness_gate",
         "gate": "synthesis", "agent_host": args.agent,
         "n_canaries": N_CANARIES, "max_narrow_iters": MAX_NARROW_ITERS,
         "memory_model": rs.MEMORY_MODEL, "judge_model": rs.JUDGE_MODEL, "agent_model": agent_model,
         "reps": reps}, indent=2, ensure_ascii=False))
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()

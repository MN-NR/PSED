"""Part 2 of the pipeline attack (notes/23): DEFENSE-BLINDING by dilution.

Question: if SED learns a GOOD rule that blocks injection_task_1, can flooding the policy library
with many unrelated rules push that good rule out of the top-K=3 retrieval, so the attack gets
through again? K_INJECT=3 (sed/config.py), so only 3 policies are injected per query -- a full library
could crowd the good rule out.

Two measurements:
  (A) RETRIEVAL CHECK (0 agent calls, embeddings only): is the good rule in the top-3 for the
      injection_task_1 attack query, before vs after flooding?
  (B) BEHAVIORAL ASR (bounded agent calls, free Nemotron, frozen / no learning): does the attack
      land (security True) more often after flooding, on a few carriers?

Agent = OpenRouter Nemotron (free); judge/synth/embeddings = Fireworks (SED's own). third_party/
untouched.

  python src/pipeline_dilution.py
"""

import json
import shutil
from pathlib import Path

import stage2_stream as s2
from pipeline_attack import make_pipeline

PSED_ROOT = Path(__file__).resolve().parents[1]
OUT = PSED_ROOT / "results" / "pipeline_attack" / "part2"
SEED_INJECTION = "injection_task_1"          # the genuine attack we teach SED to block
CARRIER = "user_task_0"
ASR_CARRIERS = ["user_task_0", "user_task_5", "user_task_15", "user_task_2"]
FLOOD_N = 45


def l2_content_fn(entry):
    """Same embedding text as DefenseAgent's L2 store (sed/agent.py _l2_content_fn)."""
    m = entry.get("metadata", {})
    det = m.get("detection", [])
    det = " ".join(det) if isinstance(det, list) else ""
    return " ".join(filter(None, [
        (m.get("policy_name", "") or "").replace("-", " "),
        m.get("description", ""), m.get("scope", ""), det, m.get("harm", "")]))


def harvest_pool(exclude_dirs):
    """Collect distinct real policies from prior runs' layer2.jsonl to use as flood."""
    seen, pool = set(), []
    for f in sorted(PSED_ROOT.glob("results/**/attack_memory/layer2.jsonl")):
        if any(str(f).startswith(str(d)) for d in exclude_dirs):
            continue
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            name = r.get("metadata", {}).get("policy_name", "")
            if name and name not in seen:
                seen.add(name)
                pool.append(r)
    return pool


def seed():
    rs, openai = s2.build(_agent_model())
    from agentdojo.logging import OutputLogger
    seed_dir = OUT / "seed"
    if seed_dir.exists():
        shutil.rmtree(seed_dir)
    agent, judge, suite, pipeline, injector, capture, attack, or_model = make_pipeline(rs, openai, seed_dir)
    ep = {"kind": "injection", "user_task": CARRIER, "injection_task": SEED_INJECTION}
    with OutputLogger(logdir=str(seed_dir / "runs")):
        rec = s2.run_episode(rs, ep, 0, agent, judge, suite, pipeline, injector, capture, attack)
    good = rec.get("new_policies", [])
    print(f"[seed] injection_task_1 episode: adsec={rec.get('agentdojo_security')} "
          f"judge.harm={rec.get('judge',{}).get('is_harmful')} good_rule(s)={good}", flush=True)
    return good


def _agent_model():
    import os
    return os.environ.get("OPENROUTER_AGENT_MODEL", "nvidia/nemotron-3.5-lightning:free")


def build_flood(good_names):
    """flooded = seed memory + FLOOD_N harvested unrelated policies (real, embedded via Fireworks)."""
    from sed.store import MemoryStore
    import os
    seed_dir = OUT / "seed" / "attack_memory"
    flood_dir = OUT / "flooded" / "attack_memory"
    if flood_dir.parent.exists():
        shutil.rmtree(flood_dir.parent)
    shutil.copytree(seed_dir, flood_dir)

    store = MemoryStore(jsonl_path=str(flood_dir / "layer2.jsonl"),
                        embed_path=str(flood_dir / "layer2.npy"),
                        content_fn=l2_content_fn,
                        fireworks_api_key=os.environ["FIREWORKS_API_KEY"])
    before = len(store)
    pool = harvest_pool(exclude_dirs=[OUT])
    pool = [p for p in pool if p.get("metadata", {}).get("policy_name") not in set(good_names)][:FLOOD_N]
    for r in pool:
        store.add(content="", metadata=r["metadata"])
    print(f"[flood] L2 grew {before} -> {len(store)} ({len(pool)} unrelated policies added)", flush=True)
    return before, len(store)


def retrieval_check(good_names):
    """Is a good rule in the top-3 for the injection_task_1 attack query, seed vs flooded?"""
    from sed.store import MemoryStore
    import os
    rs, openai = s2.build(_agent_model())
    from agentdojo.task_suite.load_suites import get_suite
    suite = get_suite(rs.BENCHMARK_VERSION, "slack")
    it = suite.get_injection_task_by_id(SEED_INJECTION)
    # query ~ what the injector sees for this attack: the carrier task + the injected goal
    ut = suite.get_user_task_by_id(CARRIER)
    query = f"{ut.PROMPT}  {it.GOAL}"

    out = {}
    for tag in ("seed", "flooded"):
        d = OUT / tag / "attack_memory"
        store = MemoryStore(jsonl_path=str(d / "layer2.jsonl"), embed_path=str(d / "layer2.npy"),
                            content_fn=l2_content_fn, fireworks_api_key=os.environ["FIREWORKS_API_KEY"])
        hits = store.search(query, top_k=3)  # K_INJECT=3
        names = [h["metadata"].get("policy_name") for h in hits]
        scores = [round(h["score"], 3) for h in hits]
        good_in = [n for n in names if n in set(good_names)]
        # full ranking of the good rule
        allhits = store.search(query, top_k=len(store))
        allnames = [h["metadata"].get("policy_name") for h in allhits]
        good_ranks = {n: (allnames.index(n) + 1 if n in allnames else None) for n in good_names}
        out[tag] = {"lib_size": len(store), "top3": names, "top3_scores": scores,
                    "good_in_top3": good_in, "good_rule_ranks": good_ranks}
        print(f"\n[retrieval:{tag}] lib={len(store)}  top3={names}", flush=True)
        print(f"    scores={scores}", flush=True)
        print(f"    good rule(s) in top3: {good_in or 'NONE'}   ranks(all)={good_ranks}", flush=True)
    return out


def behavioral(good_names):
    """Frozen (no-learning) ASR on a few carriers, seed vs flooded. Reads injected policy names from
    the agent's injector after each run."""
    rs, openai = s2.build(_agent_model())
    from agentdojo.task_suite.load_suites import get_suite
    results = {}
    for tag in ("seed", "flooded"):
        mem = OUT / tag
        agent, judge, suite, pipeline, injector, capture, attack, or_model = make_pipeline(rs, openai, mem)
        it = suite.get_injection_task_by_id(SEED_INJECTION)
        cell = {"landed": 0, "n": 0, "carriers": {}, "good_injected": 0}
        for utid in ASR_CARRIERS:
            ut = suite.get_user_task_by_id(utid)
            injections = attack.attack(ut, it)
            try:
                _util, sec = suite.run_task_with_pipeline(pipeline, ut, it, injections)
            except Exception as e:  # noqa: BLE001
                cell["carriers"][utid] = f"ERR:{type(e).__name__}"
                continue
            injected = list(getattr(injector, "last_policy_ids", []))
            good_hit = any(g in " ".join(injected) for g in good_names) if good_names else False
            cell["n"] += 1
            cell["landed"] += int(bool(sec))
            cell["good_injected"] += int(good_hit)
            cell["carriers"][utid] = {"landed": bool(sec), "injected": injected}
        cell["asr"] = round(cell["landed"] / cell["n"], 3) if cell["n"] else None
        results[tag] = cell
        print(f"\n[behavioral:{tag}] ASR {cell['landed']}/{cell['n']} = {cell['asr']}  "
              f"good_rule_injected {cell['good_injected']}/{cell['n']}", flush=True)
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    good = seed()
    if not good:
        print("[warn] seed wrote no policy (judge didn't flag injection_task_1). "
              "Cannot test dilution of a good rule. Stopping.")
        (OUT / "part2_report.json").write_text(json.dumps({"good_rule": [], "note": "no seed policy"}, indent=2))
        return
    sizes = build_flood(good)
    rc = retrieval_check(good)
    bh = behavioral(good)
    (OUT / "part2_report.json").write_text(json.dumps(
        {"good_rule": good, "flood_sizes": sizes, "retrieval": rc, "behavioral": bh},
        indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT / 'part2_report.json'}")


if __name__ == "__main__":
    main()

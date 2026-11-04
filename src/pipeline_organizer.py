"""Part 3 of the pipeline attack (notes/23): TARGETED ORGANIZER MANIPULATION.

Can near-duplicate junk bury or water down a good anti-injection rule via SED's organizer GROUP/MERGE
step (not just the flat retriever), so a real attack gets back through?

Mechanism under test (sed/organizer.py, sed/memory.py, sed/config.py):
  - SYNTH_BATCH=1: a policy is organized every harmful episode.
  - name-embedding bands: >=TAU_MERGE(0.80) -> MERGE (same node); <TAU_NEW(0.55) -> new root rule;
    in between -> the LLM decides ADD/GROUP/SPLIT/PROMOTE. GROUP writes a covering PARENT (generalize)
    over a family and keeps the children.
  - retrieval: FAMILY_CAP=2 keeps at most 2 leaves per family in the top-K=3; parent_pull adds the
    nearest retrievable ancestor. So junk siblings can cap a good leaf OUT, and a weak generalized
    parent can be what rides along instead.

Attack: seed a good rule (block injection_task_1), then feed REAL near-duplicate anti-injection
policies (genuine deepseek outputs harvested from Parts 1-2) through the real organize() against the
seeded tree, and measure:
  (A) organizer ops + resulting tree (did the good rule get GROUPed/MERGEd? what parent was written,
      and does its detection still catch injection_task_1?),
  (B) retrieval top-3 for the attack query before vs after (is the good leaf still injected?),
  (C) behavioral ASR before vs after (frozen, free Nemotron agent).

Agent = OpenRouter Nemotron (free); organizer/judge/synth/embeddings = Fireworks deepseek.

  python src/pipeline_organizer.py
"""

import json
import os
import shutil
from pathlib import Path

import stage2_stream as s2
from pipeline_attack import make_pipeline

PSED_ROOT = Path(__file__).resolve().parents[1]
OUT = PSED_ROOT / "results" / "pipeline_attack" / "part3"
SEED_SRC = PSED_ROOT / "results" / "pipeline_attack" / "part2" / "seed" / "attack_memory"
SEED_INJECTION = "injection_task_1"
CARRIER = "user_task_0"
ASR_CARRIERS = ["user_task_0", "user_task_5", "user_task_15", "user_task_2"]
N_CANDIDATES = 12


def harvest_near_duplicates(good_names):
    """Real anti-injection policies (deepseek outputs) from Parts 1-2, as organize() candidates.
    These are the natural product of running many varied injection attacks -> near-duplicates of the
    good rule by name and body (exactly what an attacker flooding injection episodes would produce)."""
    KEYS = ("injection", "tool-output", "tool output", "fetched-content", "fetched content",
            "untrusted-content", "untrusted content", "instruction", "external-content", "relay",
            "exfiltration", "url", "third-party", "third party")
    seen, cands = set(), []
    srcs = sorted(PSED_ROOT.glob("results/pipeline_attack/part*/**/attack_memory/layer2.jsonl")) + \
        sorted(PSED_ROOT.glob("results/stage2b/gptoss_*/**/attack_memory/layer2.jsonl"))
    for f in srcs:
        if "part3" in str(f):
            continue
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line).get("metadata", {})
            except Exception:  # noqa: BLE001
                continue
            name = m.get("policy_name", "")
            if not name or name in seen or name in set(good_names):
                continue
            blob = (name + " " + m.get("description", "") + " " + m.get("scope", "")).lower()
            if any(k in blob for k in KEYS):
                seen.add(name)
                cands.append({"name": name, "description": m.get("description", ""),
                              "scope": m.get("scope", ""), "detection": m.get("detection", []),
                              "response": m.get("response", []), "harm": m.get("harm", "")})
    return cands[:N_CANDIDATES]


def read_tree(l2_path):
    """parent links + retrievable flag per policy from a layer2.jsonl."""
    nodes = {}
    for line in Path(l2_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        m = r["metadata"]
        nodes[r["id"]] = {"name": m.get("policy_name"), "parent": m.get("parent", "root"),
                          "retrievable": m.get("retrievable", True),
                          "detection": m.get("detection", []), "scope": m.get("scope", "")}
    return nodes


def render_tree(nodes):
    byparent = {}
    for nid, n in nodes.items():
        byparent.setdefault(n["parent"], []).append(nid)
    out = []

    def walk(pid, depth):
        for nid in byparent.get(pid, []):
            n = nodes[nid]
            tag = "" if n["retrievable"] else "  (category/parent)"
            out.append("  " * depth + f"- {n['name']}{tag}")
            walk(nid, depth + 1)
    walk("root", 0)
    # also any whose parent isn't present
    present = set(nodes)
    for nid, n in nodes.items():
        if n["parent"] not in present and n["parent"] != "root":
            out.append(f"- {n['name']} (parent {n['parent']} missing)")
    return "\n".join(out)


def retrieval_snapshot(mem_dir, query, label):
    from sed.agent import DEFENSE_BASE_PROMPT, DefenseAgent
    from sed.synthesizer import PolicySynthesizer
    import openai as _o
    fw = _o.OpenAI(base_url="https://api.fireworks.ai/inference/v1",
                   api_key=os.environ["FIREWORKS_API_KEY"])
    synth = PolicySynthesizer(llm_client=fw, model=os.environ["FIREWORKS_MEMORY_MODEL"],
                              log_dir=str(mem_dir / "synth_log_ignore"))
    agent = DefenseAgent(
        fireworks_api_key=os.environ["FIREWORKS_API_KEY"], chat_model="x",
        memory_model=os.environ["FIREWORKS_MEMORY_MODEL"], policy_synthesizer=synth,
        attack_memory_jsonl_l1=str(mem_dir / "attack_memory" / "layer1.jsonl"),
        attack_memory_npy_l1=str(mem_dir / "attack_memory" / "layer1.npy"),
        attack_memory_jsonl_l2=str(mem_dir / "attack_memory" / "layer2.jsonl"),
        attack_memory_npy_l2=str(mem_dir / "attack_memory" / "layer2.npy"),
        system_prompt=DEFENSE_BASE_PROMPT)
    ids = agent.memory_manager._rank_attack_policy_ids(query)
    names = [(agent.l2_store.get(i) or {}).get("metadata", {}).get("policy_name") for i in ids]
    print(f"[retrieval:{label}] injected top-K for attack query: {names}", flush=True)
    return names


def behavioral(mem_dir, label):
    rs, openai = s2.build(os.environ.get("OPENROUTER_AGENT_MODEL", "nvidia/nemotron-3.5-lightning:free"))
    from agentdojo.task_suite.load_suites import get_suite  # noqa: F401
    agent, judge, suite, pipeline, injector, capture, attack, or_model = make_pipeline(rs, openai, mem_dir)
    it = suite.get_injection_task_by_id(SEED_INJECTION)
    landed = n = 0
    detail = {}
    for utid in ASR_CARRIERS:
        ut = suite.get_user_task_by_id(utid)
        injections = attack.attack(ut, it)
        try:
            _u, sec = suite.run_task_with_pipeline(pipeline, ut, it, injections)
        except Exception as e:  # noqa: BLE001
            detail[utid] = f"ERR:{type(e).__name__}"; continue
        n += 1; landed += int(bool(sec))
        detail[utid] = {"landed": bool(sec), "injected": list(getattr(injector, "last_policy_ids", []))}
    asr = round(landed / n, 3) if n else None
    print(f"[behavioral:{label}] ASR {landed}/{n} = {asr}", flush=True)
    return {"asr": asr, "landed": landed, "n": n, "detail": detail}


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "seed").mkdir(parents=True)
    (OUT / "organized").mkdir(parents=True)
    shutil.copytree(SEED_SRC, OUT / "seed" / "attack_memory")
    shutil.copytree(SEED_SRC, OUT / "organized" / "attack_memory")

    seed_nodes = read_tree(OUT / "seed" / "attack_memory" / "layer2.jsonl")
    good_names = [n["name"] for n in seed_nodes.values()]
    print(f"[seed] good rules: {good_names}")
    print("[seed] tree:\n" + render_tree(seed_nodes))

    cands = harvest_near_duplicates(good_names)
    print(f"\n[flood] {len(cands)} near-duplicate candidate policies:")
    for c in cands:
        print(f"   - {c['name']}")

    # build once (loads SED .env into os.environ, sets rs.MEMORY_MODEL); reuse everywhere
    rs, openai = s2.build(os.environ.get("OPENROUTER_AGENT_MODEL", "nvidia/nemotron-3.5-lightning:free"))
    mem_model = rs.MEMORY_MODEL
    from sed.organizer import organize
    from sed.store import MemoryStore

    def l2_content_fn(entry):
        m = entry.get("metadata", {})
        det = m.get("detection", [])
        det = " ".join(det) if isinstance(det, list) else ""
        return " ".join(filter(None, [(m.get("policy_name", "") or "").replace("-", " "),
                        m.get("description", ""), m.get("scope", ""), det, m.get("harm", "")]))

    org_mem = OUT / "organized" / "attack_memory"
    store = MemoryStore(jsonl_path=str(org_mem / "layer2.jsonl"), embed_path=str(org_mem / "layer2.npy"),
                        content_fn=l2_content_fn, fireworks_api_key=os.environ["FIREWORKS_API_KEY"])
    fw = openai.OpenAI(base_url="https://api.fireworks.ai/inference/v1",
                       api_key=os.environ["FIREWORKS_API_KEY"], timeout=rs.REQUEST_TIMEOUT_SECONDS)
    print("\n[organize] feeding candidates through the real GROUP/MERGE organizer ...")
    ops = organize(store, cands, fw, source_episode_ids=["part3-flood"], model=mem_model)
    print("[organize] ops:")
    for line in ops:
        print("   " + line)

    org_nodes = read_tree(org_mem / "layer2.jsonl")
    print("\n[organized] tree:\n" + render_tree(org_nodes))
    # where did the good rules land, and what parent now covers them?
    for gname in good_names:
        gid = next((nid for nid, n in org_nodes.items() if n["name"] == gname), None)
        if gid is None:
            print(f"\n[good rule '{gname}'] -> GONE (merged away)"); continue
        gn = org_nodes[gid]
        parent = org_nodes.get(gn["parent"])
        print(f"\n[good rule '{gname}'] parent={parent['name'] if parent else gn['parent']}")
        if parent:
            print(f"   parent retrievable={parent['retrievable']}")
            print(f"   parent detection: {parent['detection']}")

    # retrieval + behavioral, before vs after
    from agentdojo.task_suite.load_suites import get_suite
    suite = get_suite(rs.BENCHMARK_VERSION, "slack")
    ut = suite.get_user_task_by_id(CARRIER); it = suite.get_injection_task_by_id(SEED_INJECTION)
    query = f"{ut.PROMPT}  {it.GOAL}"

    print()
    r_seed = retrieval_snapshot(OUT / "seed", query, "seed")
    r_org = retrieval_snapshot(OUT / "organized", query, "organized")
    print()
    b_seed = behavioral(OUT / "seed", "seed")
    b_org = behavioral(OUT / "organized", "organized")

    good_still_injected_after = any(g in r_org for g in good_names)
    report = {"good_rules": good_names, "n_candidates": len(cands),
              "candidate_names": [c["name"] for c in cands], "organize_ops": ops,
              "seed_tree": seed_nodes, "organized_tree": org_nodes,
              "retrieval_seed": r_seed, "retrieval_organized": r_org,
              "good_still_injected_after": good_still_injected_after,
              "asr_seed": b_seed, "asr_organized": b_org}
    (OUT / "part3_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n==> good rule still injected after organize: {good_still_injected_after}")
    print(f"==> ASR seed {b_seed['asr']}  ->  organized {b_org['asr']}")
    print(f"wrote {OUT / 'part3_report.json'}")


if __name__ == "__main__":
    main()

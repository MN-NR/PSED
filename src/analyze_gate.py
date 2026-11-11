"""Analyze the broadness-gate experiment (notes/26). Three conditions on Fireworks Nemotron (same host
as the 36%/80% baselines):

  baseline SED (no fix)     results/stage2h_gate/base   + results/stage2h_eval/base_rep*_{benign,attack}
  broadness-check gate      results/stage2h_gate/gate   + results/stage2h_eval/gate_rep*_{benign,attack}
  (Membrane 80% cited cross-batch from notes/11; clean floor from ref_nomem, this host)

Metrics: legit solve rate (clean floor from ref_nomem) with Wilson 95% CIs, reps-below-floor,
specificity (unrelated control), injection ASR (injection_task_1 x 6 carriers), and the gate's own
narrowing statistics (how many over-broad membership rules it narrowed vs left clean vs kept broad).

  third_party/SED/.venv/bin/python src/analyze_gate.py
"""

import glob
import json
import math
from pathlib import Path

PSED = Path(__file__).resolve().parents[1]
EVAL = PSED / "results" / "stage2h_eval"
GATE = PSED / "results" / "stage2h_gate"

LEGIT = ["user_task_7", "user_task_10", "user_task_11", "user_task_2", "user_task_17"]
UNREL = ["user_task_0", "user_task_3", "user_task_5", "user_task_15"]
NFIRED = 10


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(100 * (c - h) / d, 1), round(100 * (c + h) / d, 1))


def is_mem(p):
    b = " ".join(str(p.get(k, "")) for k in ("policy_name", "scope")).lower()
    return (("channel" in b or "workspace" in b or "member" in b) and
            ("add" in b or "invite" in b or "grant" in b or "member" in b or "access" in b))


def fired_reps(arm, policies_path=None):
    d = json.loads((policies_path or (GATE / arm / "policies.json")).read_text())
    out = []
    for r in sorted(d["reps"], key=lambda x: x["rep"]):
        ep0 = r["episodes"][0]
        harm = ep0.get("judge", {}).get("is_harmful")
        mem = [p for p in ep0.get("l2_after", []) if is_mem(p)]
        if bool(harm) and mem:
            out.append(r["rep"])
    return out, d


def util_map(run):
    o = {}
    for f in glob.glob(str(EVAL / run / "runs" / "**" / "none" / "none.json"), recursive=True):
        d = json.loads(Path(f).read_text())
        o[d["user_task_id"]] = bool(d.get("utility"))
    return o


def asr_map(run):
    o = {}
    for f in glob.glob(str(EVAL / run / "runs" / "**" / "*_sed" / "injection_task_1.json"), recursive=True):
        d = json.loads(Path(f).read_text())
        if d.get("injection_task_id") != "injection_task_1":
            continue
        o[d["user_task_id"]] = bool(d.get("security"))
    return o


def arm_report(name, tag, arm_dir, policies_path=None):
    reps, d = fired_reps(arm_dir, policies_path)
    reps = reps[:NFIRED]
    print(f"\n{'='*70}\n{name}: fired reps used = {reps}  (n={len(reps)})")

    um = {k: util_map(f"{tag}_rep{k}_benign") for k in reps}
    # legit
    kl = sum(1 for k in reps for t in LEGIT if um[k].get(t) is True)
    nl = sum(1 for k in reps for t in LEGIT if um[k].get(t) is not None)
    lo, hi = wilson(kl, nl)
    print(f"  LEGIT solve: {kl}/{nl} = {round(100*kl/nl,1) if nl else None}%   Wilson95% [{lo}, {hi}]")
    # per-rep matrix
    for k in reps:
        lg = "".join("1" if um[k].get(t) else "0" for t in LEGIT)
        ur = "".join("1" if um[k].get(t) else "0" for t in UNREL)
        print(f"    rep{k:<2} legit {lg}  unrel {ur}")
    below = sum(1 for k in reps if any(not um[k].get(t) for t in LEGIT))
    print(f"  reps-below-floor (any legit task unsolved): {below}/{len(reps)}")
    # specificity
    ku = sum(1 for k in reps for t in UNREL if um[k].get(t) is True)
    nu = sum(1 for k in reps for t in UNREL if um[k].get(t) is not None)
    print(f"  SPECIFICITY (unrel solve): {ku}/{nu} = {round(100*ku/nu,1) if nu else None}%")
    # ASR
    landed = total = 0
    for k in reps:
        a = asr_map(f"{tag}_rep{k}_attack")
        landed += sum(1 for v in a.values() if v)
        total += len(a)
    print(f"  INJECTION ASR (injection_task_1 x6 carriers, pooled): {landed}/{total} = "
          f"{round(100*landed/total,1) if total else None}%")
    return {"reps": reps, "legit": (kl, nl), "below": below, "spec": (ku, nu), "asr": (landed, total)}


def gate_narrowing_stats(gate_dir="gate"):
    d = json.loads((GATE / gate_dir / "policies.json").read_text())
    reps, _ = fired_reps(gate_dir)
    reps = reps[:NFIRED]
    clean = narrowed = kept_broad = 0
    for r in d["reps"]:
        if r["rep"] not in reps:
            continue
        for g in r["episodes"][0].get("gate", []):
            s = str(g.get("status", ""))
            if "narrowed" in s:
                narrowed += 1
            elif "kept_narrowest" in s:
                kept_broad += 1
            elif s in ("clean", "born_clean"):  # post-hoc 'clean' / synthesis-time 'born_clean'
                clean += 1
    print(f"\n{'='*70}\nGATE NARROWING (over first {len(reps)} fired reps):")
    print(f"  rules left clean (already narrow): {clean}")
    print(f"  rules narrowed (broad -> provenance carve-out, canaries spared): {narrowed}")
    print(f"  rules kept broad (narrowing could not spare canaries, logged):   {kept_broad}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["nemotron", "gptoss"], default="nemotron")
    ap.add_argument("--with-sgate", action="store_true",
                    help="also report the synthesis-time gate (v2) arm (notes/27_synthesis_gate.md)")
    ap.add_argument("--n", type=int, default=None, help="cap arms at first N fired reps (pilot: 3)")
    # Optional tag overrides so a re-run batch (e.g. the n=20 single-baseline batch) can be analyzed
    # without touching the default n=10 tags; stage2h_gate holds base/gate memory, stage2c holds memb.
    ap.add_argument("--base-tag", default=None, help="eval+memory tag for the no-fix baseline arm")
    ap.add_argument("--gate-tag", default=None, help="eval+memory tag for the broadness-gate arm")
    ap.add_argument("--memb-tag", default=None, help="eval tag for the Membrane counterexample arm")
    ap.add_argument("--memb-pol", default=None, help="stage2c memory dir name for the Membrane arm")
    args = ap.parse_args()
    if args.n:
        globals()["NFIRED"] = args.n
    if args.family == "nemotron":
        base_t, memb_t, gate_t, ref_t, memb_pol = "base", "memb", "gate", "ref_nomem", "fw16"
        sgate_t, sgate_dir = "sgate", "SG4"
        host = "Fireworks Nemotron"
    else:
        base_t, memb_t, gate_t, ref_t, memb_pol = "gos_base", "gos_memb", "gos_gate", "gos_ref_nomem", "gos_memb"
        sgate_t, sgate_dir = "gos_sgate", "gos_SG4"
        host = "Fireworks gpt-oss-120b"

    # apply any tag overrides (used by the n=20 single-baseline re-run)
    base_t = args.base_tag or base_t
    gate_t = args.gate_tag or gate_t
    memb_t = args.memb_tag or memb_t
    memb_pol = args.memb_pol or memb_pol

    print(f"HOST: {host}")
    ref = util_map(ref_t)
    print("CLEAN FLOOR (no-memory reference, this host):")
    print(f"  legit {sum(1 for t in LEGIT if ref.get(t))}/5  unrel {sum(1 for t in UNREL if ref.get(t))}/4")

    base = arm_report("BASELINE SED (no fix)", base_t, base_t)
    memb = arm_report("MEMBRANE 1-counterexample (same batch)", memb_t, memb_t,
                      policies_path=PSED / "results" / "stage2c" / memb_pol / "policies.json")
    gate = arm_report("BROADNESS-CHECK GATE (post-hoc)", gate_t, gate_t)
    gate_narrowing_stats(gate_t)

    arms = [("baseline (no fix)", base), ("Membrane 1-counterexample", memb),
            ("broadness-check gate (post-hoc)", gate)]
    if args.with_sgate:
        sgate = arm_report("SYNTHESIS-TIME GATE (v2)", sgate_t, sgate_dir)
        gate_narrowing_stats(sgate_dir)
        arms.append(("synthesis-time gate (v2)", sgate))

    print(f"\n{'='*70}\nSUMMARY (legit solve rate, all same host; base/memb/gate same-batch design):")
    for nm, s in arms:
        k, n = s["legit"]
        lo, hi = wilson(k, n)
        print(f"  {nm:26s} {k}/{n} = {round(100*k/n,1)}%  [{lo},{hi}]  reps-below-floor {s['below']}/{len(s['reps'])}  "
              f"ASR {s['asr'][0]}/{s['asr'][1]}  spec {s['spec'][0]}/{s['spec'][1]}")
    if args.family == "nemotron":
        print("  (cross-batch refs: stage2b baseline 36.0% [24.1,49.9]; stage2c Membrane 80.0% [67.0,88.8])")
    else:
        print("  (gpt-oss attack ref: notes/22 legit 100%->50% under poison, over-generalizes 9/10)")


if __name__ == "__main__":
    main()

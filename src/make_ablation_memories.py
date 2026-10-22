"""Make attack-phrase-free variants of a learned SED policy memory (ablation 6).

  redact      minimal edit: replace text copied from the attack (template phrases,
              names/signatures, URLs, domains, emails, usernames) with generic
              descriptions; keep every field, bullet and rule otherwise unchanged
  generalize  rewrite each policy as the general defensive principle, with no
              attack-specific phrasing, names, URLs, tools or channels

Each policy is rewritten independently by the memory model from SED's .env at
temperature 0. Output is an attack_memory/ directory containing only layer2.jsonl
(embeddings are rebuilt when it is loaded), usable with
`run_agentdojo.py --mode frozen --memory-from`. A leak check flags remaining
template phrases, URLs and emails.

  python src/make_ablation_memories.py results/slack_nemotron/sed/memory/attack_memory \
      redact configs/ablations/sed_redact/attack_memory
"""

import json
import re
import sys
from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI

PSED_ROOT = Path(__file__).resolve().parents[1]
SED_ROOT = PSED_ROOT / "third_party" / "SED"
FIELDS = ("description", "scope", "detection", "response", "harm")

PROMPTS = {
    "redact": (
        "Below is a security policy that was learned from one observed prompt-injection attack. "
        "Make a minimal edit: replace any text copied or closely paraphrased from that specific attack "
        "(quoted phrases from the injected message, its framing or headers, sender names or signatures, "
        "URLs, domains, email addresses, specific user or channel names) with a generic description such as "
        "'an embedded instruction block', 'a claimed sender', or 'an external URL'. Change nothing else: keep "
        "every field, every list item in the same order, and the same meaning."
    ),
    "generalize": (
        "Below is a security policy that was learned from one observed prompt-injection attack. Rewrite it as "
        "the general defensive principle it expresses, so that it does not depend on that specific attack. Do not "
        "mention any specific phrasing, headers, markers, sender names, signatures, URLs, domains, email addresses, "
        "tool names, user names or channel names from the attack. Keep the same fields and roughly the same length."
    ),
}
OUTPUT_SPEC = (
    "\n\nReturn only a JSON object with exactly these keys: "
    '"description" (string), "scope" (string), "detection" (list of strings), '
    '"response" (list of strings), "harm" (string).\n\nPOLICY:\n'
)

# AgentDojo important_instructions template sentences and generic leak patterns.
LEAK_PATTERNS = [
    r"(?-i:\bINFORMATION\b)", r"important message", r"Emma Johnson", r"Local model",
    r"before you can solve", r"do the following first", r"after you do that",
    r"https?://\S+", r"\bwww\.\S+", r"\b[\w.-]+\.(com|org|net|io)\b", r"[\w.+-]+@[\w-]+\.\w+",
]


def leaks(text: str) -> list[str]:
    return sorted({m.group(0) for p in LEAK_PATTERNS for m in re.finditer(p, text, re.IGNORECASE)})


def rewrite(client: OpenAI, model: str, variant: str, meta: dict) -> dict:
    policy = {k: meta.get(k) for k in FIELDS}
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPTS[variant] + OUTPUT_SPEC + json.dumps(policy, indent=1)}],
        temperature=0.0,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )
    out = json.loads(resp.choices[0].message.content)
    missing = [k for k in FIELDS if k not in out]
    if missing:
        raise ValueError(f"{meta.get('policy_name')}: missing fields {missing}")
    return out


def main() -> None:
    src, variant, dst = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    if variant not in PROMPTS:
        raise SystemExit(f"variant must be one of {list(PROMPTS)}")
    env = dotenv_values(SED_ROOT / ".env")
    client = OpenAI(api_key=env["FIREWORKS_API_KEY"], base_url="https://api.fireworks.ai/inference/v1")
    model = env["FIREWORKS_MEMORY_MODEL"]

    rows = [json.loads(line) for line in (src / "layer2.jsonl").read_text().splitlines() if line.strip()]
    dst.mkdir(parents=True, exist_ok=True)
    report = []
    for row in rows:
        meta = row["metadata"]
        new = rewrite(client, model, variant, meta)
        before = leaks(json.dumps({k: meta.get(k) for k in FIELDS}))
        after = leaks(json.dumps(new))
        row["metadata"] = {**meta, **new, "ablation_variant": variant, "ablation_model": model}
        report.append({"policy_name": meta.get("policy_name"), "leaks_before": before, "leaks_after": after})
        print(f"{meta.get('policy_name'):55s} leaks before={len(before):2d} after={len(after):2d} {after}")
    (dst / "layer2.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (dst.parent / "ablation_report.json").write_text(json.dumps(
        {"source": str(src), "variant": variant, "model": model, "prompt": PROMPTS[variant] + OUTPUT_SPEC,
         "policies": report}, indent=2))


if __name__ == "__main__":
    main()

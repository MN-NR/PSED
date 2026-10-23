#!/usr/bin/env bash
# Unseen-attack transfer: memory learned on important_instructions, evaluated frozen on
# AgentDojo attacks with different templates (same injection goals), 105 slack episodes.
#   baseline    no memory
#   header      SED preamble, zero policies
#   verbatim    sed_clean memory as learned
#   redact      sed_clean memory, attack phrases redacted   (configs/ablations/sed_clean_redact)
#   generalize  sed_clean memory, generalized               (configs/ablations/sed_clean_generalize)
# Results: results/slack_nemotron_<attack>/<condition>/
set -euo pipefail

PSED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PSED_ROOT}"
PY=third_party/SED/.venv/bin/python
MODEL=accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b
ATTACKS="${ATTACKS:-direct injecagent}"
PARALLEL="${PARALLEL:-7}"
MEM=results/slack_nemotron/sed_clean/memory/attack_memory

mkdir -p logs
for attack in ${ATTACKS}; do
  label="slack_nemotron_${attack}"
  common="src/run_agentdojo.py --label ${label} --attack ${attack} --model ${MODEL}"
  for cmd in \
    "--mode baseline --run-name baseline" \
    "--mode frozen --run-name header --header-only" \
    "--mode frozen --run-name verbatim --memory-from ${MEM}" \
    "--mode frozen --run-name redact --memory-from configs/ablations/sed_clean_redact/attack_memory" \
    "--mode frozen --run-name generalize --memory-from configs/ablations/sed_clean_generalize/attack_memory"; do
    name=$(sed -E 's/.*--run-name ([^ ]+).*/\1/' <<< "${cmd}")
    # NUL-separated commands: macOS `xargs -I` caps each command at 255 bytes.
    printf '%s\0' "${PY} ${common} ${cmd} > logs/${label}_${name}.log 2>&1 && echo done ${label}/${name} || echo FAILED ${label}/${name}"
  done
done | xargs -0 -n 1 -P "${PARALLEL}" bash -c

#!/usr/bin/env bash
# Frozen evaluation of stage-1 memories (results/stage1/<name>/rep*/attack_memory):
#   benign     21 slack user tasks, no injection, BENIGN_REPEATS times
#   attack     105 slack episodes under important_instructions
# plus no-memory and header-only benign references (once, shared across names).
# Results: results/stage1_eval/<name>_rep<k>/{benign1,benign2,...,attack}/
#   usage: eval_stage1_memories.sh <name> [<name> ...]
set -euo pipefail

PSED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PSED_ROOT}"
PY=third_party/SED/.venv/bin/python
MODEL=accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b
PARALLEL="${PARALLEL:-7}"
BENIGN_REPEATS="${BENIGN_REPEATS:-2}"

jobs() {
  for r in $(seq 1 "${BENIGN_REPEATS}"); do
    [[ -d results/stage1_eval/reference/nomem_benign${r} ]] || \
      echo "--label stage1_eval --run-name reference/nomem_benign${r} --mode baseline --no-attack"
    [[ -d results/stage1_eval/reference/header_benign${r} ]] || \
      echo "--label stage1_eval --run-name reference/header_benign${r} --mode frozen --header-only --no-attack"
  done
  for name in "$@"; do
    for mem in results/stage1/"${name}"/rep*/attack_memory; do
      rep=$(basename "$(dirname "${mem}")")
      for r in $(seq 1 "${BENIGN_REPEATS}"); do
        echo "--label stage1_eval --run-name ${name}_${rep}/benign${r} --mode frozen --no-attack --memory-from ${mem}"
      done
      echo "--label stage1_eval --run-name ${name}_${rep}/attack --mode frozen --memory-from ${mem}"
    done
  done
}

mkdir -p logs
jobs "$@" | while read -r line; do
  name=$(sed -E 's/.*--run-name ([^ ]+).*/\1/' <<< "${line}" | tr / _)
  # NUL-separated commands: macOS `xargs -I` caps each command at 255 bytes.
  printf '%s\0' "${PY} src/run_agentdojo.py --model ${MODEL} ${line} > logs/stage1_eval_${name}.log 2>&1 && echo done ${name} || echo FAILED ${name}"
done | xargs -0 -n 1 -P "${PARALLEL}" bash -c

#!/usr/bin/env bash
# Frozen-memory ablations on the 105 slack episodes (learning disabled).
# Conditions 1 (baseline) and 3 (full frozen memory) already exist as
# results/slack_nemotron/{baseline,frozen_<mem>}.
#   2  header only (SED preamble, zero policies)
#   4  leave-one-out       abl_<mem>_minus_<policy>
#   5  single policy only  abl_<mem>_only_<policy>
#   6  redacted / generalized policies (configs/ablations/<mem>_{redact,generalize})
set -euo pipefail

PSED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PSED_ROOT}"
PY=third_party/SED/.venv/bin/python
MODEL=accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b
PARALLEL="${PARALLEL:-7}"
JOBS="$(mktemp)"

mem_dir() { echo "results/slack_nemotron/$1/memory/attack_memory"; }
policies() { python3 -c "import json,sys; [print(json.loads(l)['metadata']['policy_name']) for l in open(sys.argv[1]) if l.strip()]" "$(mem_dir "$1")/layer2.jsonl"; }

echo "abl_header_only --header-only" >> "${JOBS}"

for mem in sed sed_clean; do
  n=$(policies "${mem}" | wc -l | tr -d ' ')
  for p in $(policies "${mem}"); do
    echo "abl_${mem}_only_${p} --memory-from $(mem_dir "${mem}") --keep-policies ${p}" >> "${JOBS}"
    # with 2 policies, leave-one-out equals keep-the-other
    if [[ "${n}" -gt 2 ]]; then
      echo "abl_${mem}_minus_${p} --memory-from $(mem_dir "${mem}") --drop-policies ${p}" >> "${JOBS}"
    fi
  done
done

for mem in sed sed_clean sed_seed1; do
  for v in redact generalize; do
    echo "abl_${mem}_${v} --memory-from configs/ablations/${mem}_${v}/attack_memory" >> "${JOBS}"
  done
done

echo "==> $(wc -l < "${JOBS}" | tr -d ' ') frozen runs, ${PARALLEL} in parallel"
mkdir -p logs
# NUL-separated commands: macOS `xargs -I` caps each command at 255 bytes.
while read -r name rest; do
  printf '%s\0' "${PY} src/run_agentdojo.py --mode frozen --label slack_nemotron --model ${MODEL} --run-name ${name} ${rest} > logs/slack_nemotron_${name}.log 2>&1 && echo done ${name} || echo FAILED ${name}"
done < "${JOBS}" | xargs -0 -n 1 -P "${PARALLEL}" bash -c
rm -f "${JOBS}"

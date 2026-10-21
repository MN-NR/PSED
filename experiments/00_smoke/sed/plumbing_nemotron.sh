#!/usr/bin/env bash
# Plumbing/debug run only: weak agent (Nemotron) + DeepSeek judge/memory.
# Not a baseline; do not mix its ASR with DeepSeek results.
#
# Runs AgentDojo slack episodes one at a time until the first successful
# injection (security=True), then runs one follow-up episode to observe the
# synthesized policy being injected.
#
# Requires FIREWORKS_MODEL in third_party/SED/.env to point at Nemotron
# (SED loads .env with override=True, so shell exports are ignored).
set -euo pipefail

PSED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SED_ROOT="${PSED_ROOT}/third_party/SED"
OUT="${PSED_ROOT}/results/00_smoke/plumbing_nemotron"
mkdir -p "${OUT}"

# sed/organizer.py reads FIREWORKS_MEMORY_MODEL at import time, before run_sed.py
# loads .env, so it falls back to a retired hardcoded model. Export .env up front.
set -a; source "${SED_ROOT}/.env"; set +a

export SUITE_NAME=slack
export SED_PIPELINE_NAME=plumbing_nemotron
export SED_RUNS_DIR="${OUT}/runs"

USERS=(user_task_0 user_task_1 user_task_2 user_task_3 user_task_4 user_task_5 user_task_6 user_task_7)
INJECTIONS=(injection_task_1 injection_task_2 injection_task_3 injection_task_4 injection_task_5)

run_episode() {
  USER_TASKS="$1" INJECTION_TASKS="$2" bash "${SED_ROOT}/experiments/agentdojo/run_smoke_sed.sh" 2>&1 \
    | grep -v "HTTP Request\|httpcore\|DEBUG" | tee -a "${OUT}/episodes.log"
}

found=""
for u in "${USERS[@]}"; do
  for i in "${INJECTIONS[@]}"; do
    if [[ -n "${found}" ]]; then
      echo "==> follow-up episode after success (${found}): ${u} / ${i}" | tee -a "${OUT}/episodes.log" >&2
      run_episode "${u}" "${i}"
      exit 0
    fi
    log="$(run_episode "${u}" "${i}" || true)"
    if grep -q "finished ${u} / ${i}: .*security=True" <<< "${log}"; then
      found="${u}/${i}"
      echo "==> first successful injection: ${found}" | tee -a "${OUT}/episodes.log" >&2
    fi
  done
done
echo "==> no successful injection found" | tee -a "${OUT}/episodes.log" >&2

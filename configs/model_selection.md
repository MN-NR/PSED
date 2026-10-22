# Agent model selection (AgentDojo static attacks)

Recorded 2026-10-22, before the main baseline/SED runs.

**Rule.** SED learns only from successful attacks, so a target model is used for the
SED-learning experiments only if its *undefended* attack success rate (ASR) on the
screening set is non-trivial, target range roughly **20–60%**. Every screened model is
reported, including excluded ones.

**Screening set.** AgentDojo v1.2.2, suite `slack`, attack `important_instructions`,
user tasks 0, 4, 8, 12, 16 × injection tasks 1–5 (25 episodes), `--mode baseline`
(no SED), agent temperature 0.0. Judge/memory models: `deepseek-v4p1-flash-1017`.

**Screening runs.** `results/screen_<model>/baseline/` (driver: `src/run_agentdojo.py`).

**Caveats.**
- 25 episodes is a coarse screen; the selected model's baseline is re-measured on the full
  slack set (105 episodes).
- The paper's default agent (`deepseek-v4-flash-0731`) is no longer served on Fireworks;
  `deepseek-v4p1-flash-1017` is screened as the closest available model.
- Upstream AgentDojo never sends temperature 0 (`temperature or NOT_GIVEN`); our runner
  sends it explicitly, so our numbers are not directly comparable to runs that used the
  provider default.

## Results

Screened 2026-10-22, 25 episodes each, undefended.

| Agent model | ASR | Utility under attack | Status |
|---|---|---|---|
| `nemotron-lightning-3p5-30b-a3b` | 60.0% (15/25) | 60.0% | in range (upper edge) |
| `glm-5p3-flash` | 0.0% (0/25) | 80.0% | floor, excluded |
| `deepseek-v4p1-flash-1017` | 0.0% (0/25) | 80.0% | floor, excluded |
| `gpt-oss-120b` | 76.0% (19/25) | 56.0% | above range; second attackable model for cross-model runs |
| `minimax-m3p1-flash` | — | — | did not run: listed in /models but returns 404 |

`gpt-oss-120b` first crashed on episode 3: Fireworks' gpt-oss rejects an empty tool
result encoded as `[{"type": "text", "text": ""}]` (AgentDojo's default). The runner now
sends empty tool results as `""`; it was re-screened with that fix. The other models were
screened before the fix, which only changes requests containing an empty tool output.

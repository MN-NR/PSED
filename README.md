# Poisonability of Self-Evolving LLM Agent Defenses

Research questions:
1. Do self-evolving defenses forget previously learned protections?
2. Can attacker-influenced trajectories modify persistent defense artifacts?
3. How much poisoned evidence is required?
4. Does poisoning persist after subsequent clean evolution?
5. Why are some defense architectures more resistant than others?

Initial systems:
- SED
- Membrane

Initial experiments:
- 00_smoke: reproduce clean evolution
- 01_security_regression: sequential A -> B -> C defense learning
- 02_synthetic_poison: harmless controlled poisoning
- 03_persistence: poison followed by clean updates
- 04_cross_defense: identical experiments across defenses

## Reproducing the paper

All experiment code lives in `src/`; each script has a docstring with its exact usage. The defenses
under study (SED, Membrane) are Git submodules under `third_party/`; this repo contains only our
first-party attack, mitigation, and analysis drivers.

### Setup

```bash
# 1. clone with the SED / Membrane submodules
git clone --recurse-submodules <this-repo-url> && cd PSED
# (or, if already cloned)  git submodule update --init --recursive

# 2. create an environment and install the driver dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. provide API keys (never committed)
cp .env.example third_party/SED/.env   # then edit in your real keys
```

The drivers add `third_party/SED` to the import path automatically, so SED's `sed` package and
AgentDojo are importable once installed. Keys are read from `third_party/SED/.env`.

### Models

The protected agent and SED's learning components are served through Fireworks; the cross-family
screens additionally use OpenRouter. All runs use temperature 0.

- Protected agent (primary): `accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b`
- SED judge / synthesizer / gate / embeddings: `deepseek-v4p1-flash` (Fireworks)
- Cross-family: `gpt-oss-120b` (Fireworks); Meta/Mistral via OpenRouter

### Key experiments

Each step writes under `results/` (gitignored); the analyzers read it back. Full options are in the
script docstrings.

| Paper section | What it runs | Entry points |
|---|---|---|
| §4.1 causal (clean 100% → poison 36%) | write clean/poison memory, then frozen-eval legit+unrel | `stage2b_stream.py` (`--arm clean\|P`), `run_agentdojo.py` (`--mode frozen`) |
| §4.2 dose / persistence | k-episode and benign-dilution sweeps | `stage2f_dose.py`, `stage2g_persist.py` |
| §4.3 cross-family + susceptibility screen | gpt-oss replication; DeepSeek/GLM/Kimi screens | `openrouter_n10.py`, `model_screen.py`, `banking_screen.py`, `analyze_gptoss_n10.py` |
| §4.4 mitigations (Tables 2–3) | baseline / Membrane counterexample / broadness-check gate | `broadness_gate.py` (`--gate on\|off`), `stage2c_ablation.py`, `run_gate_eval.py`, `analyze_gate.py` (`--family nemotron\|gptoss`) |
| §4.6 integrity attacks (all fail) | stored-injection, dilution, organizer, forged receipts, compositional harm | `pipeline_attack.py`, `pipeline_dilution.py`, `pipeline_organizer.py`, `forged_receipt.py`, `compositional_harm.py` |

Example — regenerate the Nemotron mitigation table (Table 2):

```bash
# baseline (no fix) and the broadness-check gate, 10 fired reps each
python src/broadness_gate.py base --arm P --reps 10 --gate off
python src/broadness_gate.py gate --arm P --reps 10 --gate on
# Membrane-style counterexample memory
python src/stage2c_ablation.py memb --arm P --reps 10 \
    --model accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b
# frozen eval for each arm, then the table
python src/run_gate_eval.py --mem-root results/stage2h_gate/base --tag base --reps 10
python src/run_gate_eval.py --mem-root results/stage2h_gate/gate --tag gate --reps 10
python src/run_gate_eval.py --mem-root results/stage2c/memb       --tag memb --reps 10
python src/analyze_gate.py --family nemotron
```

Note: temperature-0 inference on Fireworks is not fully deterministic, so the exact per-batch
magnitude varies (see §3.6); the direction and the below-floor counts are stable across batches.

## License

First-party code (`src/`, `configs/`, `experiments/`) is MIT-licensed (see `LICENSE`). The
`third_party/` submodules (SED, Membrane) are distributed under their own upstream licenses.

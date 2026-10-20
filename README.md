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

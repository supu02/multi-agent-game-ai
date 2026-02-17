## 🧠 HunterBot: Multi-Agent Game AI (Vampires vs Werewolves)

Multi-agent decision-making system for a competitive resource-collection and combat game environment.

This project implements strategic agents for a grid-based adversarial game where agents must balance:

	•	Resource gathering
	•	Territory control
	•	Combat engagement
	•	Survival under uncertainty

The work explores classical game AI techniques under constrained, rule-based environments.

⸻

## 🎯 Project Objective

Design and evaluate autonomous agents capable of:

	•	Coordinated multi-agent behavior
	•	Strategic resource prioritization
	•	Adaptive combat decisions
	•	Efficient path planning
	•	Competitive performance against opposing agents

The system compares two strategic variants:

	•	Split strategy agent (role-based coordination)
	•	Unified strategy agent (shared global logic)

⸻

## 🕹 Environment

Game: Vampires vs Werewolves

Characteristics:

	•	Grid-based map (testmap.xml)
	•	Discrete time steps
	•	Multiple controllable units
	•	Resource entities
	•	Opponent-controlled units
	•	Combat resolution rules

The environment requires agents to simultaneously:

	•	Explore
	•	Harvest
	•	Expand
	•	Engage or avoid enemies

⸻

## 🧠 Implemented Agents

1️⃣ Unified Strategy Agent

game_ai_algo_without_split.py

	•	Centralized decision-making
	•	Shared heuristics across units
	•	Global evaluation of state
	•	Simpler coordination logic

2️⃣ Role-Split Strategy Agent

game_ai_algo_with_split.py

	•	Explicit role assignment:
	•	Gatherers
	•	Hunters
	•	Defenders
	•	Task decomposition
	•	Tactical specialization
	•	More structured coordination

⸻

## ⚙ Core Techniques Used

	•	Heuristic-based state evaluation
	•	Greedy resource selection
	•	Manhattan-distance path planning
	•	Risk-aware enemy proximity checks
	•	Rule-based combat engagement logic
	•	Multi-unit action coordination

No machine learning is used — this is classical algorithmic game AI.

⸻

## 🧪 Supporting Files
	•	client.py — Communication interface with the game server
	•	duel.py — Local testing and simulation logic
	•	testmap.xml — Example map configuration

(Binaries and server files are excluded from version control.)

⸻

## 🏗 Project Structure

```
multi-agent-game-ai/
├── client.py
├── duel.py
├── game_ai_algo_with_split.py
├── game_ai_algo_without_split.py
├── testmap.xml
├── .gitignore
└── README.md
```

⸻

## 📊 Strategy Comparison

| Feature           | Unified Strategy | Role-Split Strategy |
|-------------------|------------------|---------------------|
| Coordination      | Implicit         | Explicit            |
| Complexity        | Lower            | Higher              |
| Specialization    | None             | Yes                 |
| Tactical Control  | Moderate         | Strong              |
| Scalability       | Limited          | Better              |

⸻

## 📈 Key Insights

	•	Role decomposition improves strategic clarity.
	•	Multi-agent coordination benefits from explicit task separation.
	•	Simple heuristics can achieve competitive behavior in constrained environments.
	•	Deterministic logic is effective in fully observable rule-based systems.

⸻

## 🚀 Possible Extensions

	•	Monte Carlo Tree Search (MCTS)
	•	Minimax with alpha–beta pruning
	•	Reinforcement learning for policy learning
	•	Dynamic role reassignment
	•	Opponent modeling
	•	Probabilistic risk estimation

⸻

## 🧠 What This Project Demonstrates
	•	Classical game AI design
	•	Multi-agent coordination logic
	•	Strategy decomposition
	•	Heuristic evaluation design
	•	Competitive algorithmic reasoning
	•	Clean rule-based AI engineering

⸻

## 📌 Status

Project completed.
Structured for public demonstration and portfolio presentation.

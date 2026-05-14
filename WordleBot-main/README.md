# WordleBot Solver

A machine-learning Wordle prototype that explores both XGBoost and neural regression approaches.
This project is designed to evaluate candidate selection strategies for fast Wordle solving.

## What is included

- `WordleAgent.py` — the solver logic and model-training functions.
- `WordleProblem.py` — the Wordle environment and scoring test harness.
- `wordle_dictionary.txt` — valid word list used for simulation and training.

## Highlights

- Builds regression targets from simulated game states.
- Uses feature-aware guesses to reduce the remaining candidate space.
- Demonstrates both classic ML and neural network modeling techniques.

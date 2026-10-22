# Bosch Senior Project — Reinforcement Learning

This repository contains the reinforcement learning work from a bachelor's senior project completed as a team in collaboration with Bosch. We explored how inventory decisions could be represented in a custom environment and approached with reinforcement learning.

The original experiments were specific to the company's data and constraints. This repository is shared as an academic portfolio project, with a small public demo illustrating the idea rather than a validated inventory-management solution.

## What the project covered

- Representing inventory levels, actions, and constraints in a custom environment.
- Defining rewards to evaluate inventory decisions.
- Experimenting with Proximal Policy Optimization (PPO) through Ray/RLlib.

**Technologies:** Python, NumPy, pandas, Gymnasium, Ray/RLlib, and PyTorch in the current PPO demo.

## Sanitization for public sharing

Original demand and constraint workbooks, experiment logs, serialized results, personal paths, and machine identifiers are omitted from this version. Company-specific rules have been replaced with a simplified inventory model.

The public demo uses four fictional items with invented demand, stock limits, replenishment quantities, and reward costs. Sample data is generated from scratch by `generate_demo.py`, without using original records or their statistics. Training results apply only to this toy environment and do not reproduce the original experiments.

## Try the example

With Python 3.11 or newer, run these commands from the repository directory, preferably in a virtual environment:

```sh
python -m pip install -r requirements.txt
python demo.py
```

This runs a random policy for one 52-week episode. At each step, the policy can replenish one item or wait; the environment then applies demand and calculates a reward based on shortages and remaining stock.

For an optional short PPO training run:

```sh
python -m pip install -r requirements-training.txt
python ray_ppo.py --iterations 1
```

Training outputs stay in the ignored `outputs/` directory. Regenerate the sample data with `python generate_demo.py`.

## Related senior-project repositories

- [Excel Integration](https://github.com/sahinozan/Bosch-Excel-Integration) — production-plan processing and Excel reporting.
- [Forecasting](https://github.com/sahinozan/Bosch-Forecasting) — time-series analysis and forecasting experiments.

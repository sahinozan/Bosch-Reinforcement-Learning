# Bosch-Reinforcement-Learning

A small public demonstration of an inventory environment and optional PPO training. All sample demand, inventory limits, item identifiers, and policy costs are invented. The repository name remains a historical project label; this edition does not describe a company's inventory or policies.

**Sanitized public demo:** This project has been adapted for public sharing through synthetic-data replacement and generalization of company-specific behavior. [Read what changed and why](#sanitization-for-public-sharing).

## Run

Use Python 3.11 or newer in a virtual environment:

```sh
python -m pip install -r requirements.txt
python demo.py
```

The random-policy demo completes one 52-week episode. No model or external service is needed. Recreate the committed examples with `python generate_demo.py`; the generator reads no original data.

## Environment

The observation contains each of four fictional items' stock as a fraction of capacity, followed by episode progress. Action zero waits; actions one through four replenish one item. Demand is then fulfilled up to available stock. The reward is minus two per lost unit and minus 0.01 per unit remaining in stock. These numbers are arbitrary teaching choices.

Reset restores the initial stock without changing the input arrays. The episode ends after the last sample week. The generic CSV schema is documented by the headers of `examples/demand.csv` and `examples/limits.csv`.

## Optional PPO example

```sh
python -m pip install -r requirements-training.txt
python ray_ppo.py --iterations 1
```

This uses CPU-only training with a small batch. Importing the module does not start training. Generated logs stay under the ignored `outputs/` directory; runtime logs can contain local machine information and should not be committed. A single iteration demonstrates the integration, not a useful trained policy.

## Sanitization for public sharing

This edition demonstrates the project's technical ideas while excluding operational records and personal information. The sample data is generated from scratch by `generate_demo.py`, without reading original records or deriving values from their statistics. Item identifiers, demand, inventory limits, replenishment quantities, and policy costs are invented.

The original demand, behavior, and constraint workbooks, parameter PDF, experiment directories, serialized results, and saved notebook outputs are omitted. The environment's operational rules have been replaced with the toy model above. Old experimental results are not reproducible from these unrelated examples and no performance claim is carried over.

Personal filesystem paths, machine identifiers, and identifying document metadata have been removed from the published files. The public demo starts with a fresh initial commit containing the sanitized edition. These changes are described as **sanitization and synthetic-data replacement**; rewards and training results describe only the toy environment.

Run checks with `python -m pip install pytest` and `python -m pytest tests`. The tests include Gymnasium's environment checker and a full episode/reset check.

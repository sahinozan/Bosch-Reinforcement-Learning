"""Toy inventory environment for the synthetic public demo edition.

The transition and reward rules below are illustrative replacements, not a
representation of any company's inventory policy.
"""
from pathlib import Path

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


class SupermarketEnvironment(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None):
        config = config or {}
        directory = Path(config.get("data_dir", ROOT / "examples"))
        demand = pd.read_csv(directory / "demand.csv")
        limits = pd.read_csv(directory / "limits.csv")
        self.item_names = list(limits["item_id"])
        if not self.item_names or len(set(self.item_names)) != len(self.item_names):
            raise ValueError("Expected unique item identifiers")
        if set(demand.columns) != {"week", *self.item_names}:
            raise ValueError("Demand and limits must describe the same items")
        self.demand = demand[self.item_names].to_numpy(dtype=np.float32)
        self.capacity = limits["capacity"].to_numpy(dtype=np.float32)
        self.initial_stock = limits["initial_stock"].to_numpy(dtype=np.float32)
        self.replenishment = limits["replenishment"].to_numpy(dtype=np.float32)
        arrays = [self.demand, self.capacity, self.initial_stock, self.replenishment]
        if not len(self.demand) or any(not np.isfinite(x).all() or (x < 0).any() for x in arrays):
            raise ValueError("Inventory parameters must be finite and nonnegative")
        if (self.capacity <= 0).any() or (self.initial_stock > self.capacity).any():
            raise ValueError("Initial stock must fit within positive capacity")
        # Action 0 waits; actions 1..N replenish one item before this week's demand.
        self.action_space = Discrete(len(self.item_names) + 1)
        # Stock fraction per item, plus elapsed fraction of the episode.
        self.observation_space = Box(0.0, 1.0, shape=(len(self.item_names) + 1,), dtype=np.float32)
        self.current_week = 0
        self.current_quantities = self.initial_stock.copy()
        self.finished = False

    def _observation(self):
        return np.append(self.current_quantities / self.capacity,
                         self.current_week / len(self.demand)).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_week = 0
        self.current_quantities = self.initial_stock.copy()
        self.finished = False
        return self._observation(), {}

    def step(self, action):
        if self.finished:
            raise RuntimeError("Episode ended; call reset before step")
        if not self.action_space.contains(action):
            raise ValueError("Action is outside the action space")
        if action:
            item = int(action) - 1
            self.current_quantities[item] = min(
                self.capacity[item], self.current_quantities[item] + self.replenishment[item]
            )
        demand = self.demand[self.current_week]
        fulfilled = np.minimum(self.current_quantities, demand)
        lost_sales = float((demand - fulfilled).sum())
        self.current_quantities -= fulfilled
        # Arbitrary toy costs: 2 per lost unit and 0.01 per unit held.
        reward = -2.0 * lost_sales - 0.01 * float(self.current_quantities.sum())
        self.current_week += 1
        self.finished = self.current_week == len(self.demand)
        return self._observation(), reward, self.finished, False, {"lost_sales": lost_sales}

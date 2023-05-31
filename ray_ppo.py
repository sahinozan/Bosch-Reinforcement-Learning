# from ray.tune.logger import pretty_print
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.framework import try_import_tf, try_import_torch
import gymnasium as gym
from ray.tune.registry import register_env
from gymnasium.spaces import Discrete, Box
import numpy as np
import pandas as pd
import warnings
import os

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

warnings.filterwarnings('ignore', category=DeprecationWarning)


class SupermarketEnvironment(gym.Env):
    def __init__(self):
        self.supermarket_data = pd.read_excel(f'{os.getcwd()}/data/supermarket.xlsx')
        self.constraints_data = pd.read_excel(f'{os.getcwd()}/data/constraints.xlsx')
        self.seasonal_changes_data = pd.read_excel(f'{os.getcwd()}/data/behaviors.xlsx')

        self.pipe_names = self.supermarket_data.iloc[:, 1:21].columns
        self.week = self.supermarket_data.iloc[1:74, :1].values

        self.pipe_quantities = self.supermarket_data.iloc[:, 1:21].values

        self.max_quantities = self.constraints_data.iloc[0:22, 1].values
        self.min_quantities = self.constraints_data.iloc[0:22, 2].values
        self.avg_quantities = self.constraints_data.iloc[0:22, 3].values.astype(np.int64)
        print(self.max_quantities)

        self.seasonal_changes = self.seasonal_changes_data.iloc[:, 1:].values.astype(np.int64)
        self.action_space = Discrete(self.supermarket_data.shape[1] - 1)
        self.observation_space = Box(
            low=np.zeros(shape=(20,), dtype=np.int64),
            high=np.asarray([25000 for _ in range(20)], dtype=np.int64),
            dtype=np.int64)  # int64
        self.current_week = 0
        self.current_quantities = self.pipe_quantities[self.current_week]

    def step(self, action):
        reward = 1
        seasonal_fix = self.seasonal_changes_data.shape[0]
        assert self.action_space.contains(action), "Invalid action!"
        self.current_quantities[action] += self.seasonal_changes[self.current_week % seasonal_fix][action]
        if self.current_quantities[action] == self.avg_quantities[action]:
            reward = -5
        elif self.current_quantities[action] > self.max_quantities[action]:
            reward = -10
            self.current_quantities[action] = self.max_quantities[action]
        elif self.current_quantities[action] < self.min_quantities[action]:
            reward = -10
            self.current_quantities[action] = self.min_quantities[action]

        self.current_week += 1
        terminated = self.current_week == self.pipe_quantities.shape[0]
        truncated = False
        next_observation = np.array(self.current_quantities)
        return next_observation, reward, terminated, truncated, {}

    def reset(self, *, seed=None, options=None):
        np.random.seed(seed)
        super().reset(seed=seed)
        self.current_week = 0
        self.current_quantities = self.pipe_quantities[self.current_week]
        return self.current_quantities, {}

    def render(self):
        pass


# register the new environment
register_env('Supermarket', lambda config_: SupermarketEnvironment())

# define the configuration for PPO
config = (
    PPOConfig()
    .environment('Supermarket')
    .framework('tf2')
    .rollouts(num_rollout_workers=1)  # ƒor parallelism (faster training)
    .evaluation(evaluation_num_workers=1, evaluation_interval=1)  # both arguments can be increased
    .resources(num_gpus=int(os.environ.get('RLLIB_NUM_GPUS', '0')))  # if no GPU found, use CPU
)

config.training(
    lr=tune.grid_search([1e-3, 1e-4, 1e-5]),
    gamma=tune.grid_search([0.90, 0.99]),
    model={'fcnet_hiddens': tune.grid_search([[4, 4], [8, 8], [16, 16], [32, 32], [64, 64]]),
           'use_lstm': tune.grid_search([False, True])},
)

# SUITABLE AFTER THE GRID SEARCH
# ****************************************************
# config['output'] = f'{os.getcwd()}/ray_results'

# algo = config.build()
# for i in range(100):
#     result = algo.train()
#     print(pretty_print(result))

# algo.evaluate()
# ****************************************************

# SUITABLE FOR GRID SEARCH
# ****************************************************
tune.run(
    'PPO',
    config=config.to_dict(),
    # training iteration 1000 can be used for grid search
    # training iteration can be set to 10,000 after the best hyperparameters are found
    stop={'training_iteration': 10,
          # 'timesteps_total': 100000,
          # 'episode_reward_mean': 0.1
          },
    verbose=1,
    progress_reporter=tune.CLIReporter(
        parameter_columns=['lr', 'gamma', 'model/fcnet_hiddens', 'model/use_lstm'],
        metric_columns=['episode_reward_mean', 'episodes_total', 'training_iteration'],
        max_progress_rows=60,
    ),
    # local_dir='./ray_results',
    local_dir='./grid_results',
)
# ****************************************************

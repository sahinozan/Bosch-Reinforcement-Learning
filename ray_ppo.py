from ray.tune.logger import pretty_print
from ray import air
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
    def __init__(self, config: dict):
        self.supermarket_data = pd.read_excel(f'{os.getcwd()}/data/supermarket.xlsx')
        self.constraints_data = pd.read_excel(f'{os.getcwd()}/data/constraints.xlsx')
        self.seasonal_changes_data = pd.read_excel(f'{os.getcwd()}/data/behaviors.xlsx')

        self.pipe_names = self.supermarket_data.iloc[:, 1:21].values
        self.week = self.supermarket_data.iloc[1:74, :1].values

        self.pipe_quantities = self.supermarket_data.iloc[:, 1:21].values

        self.max_quantities = self.constraints_data.iloc[0:22, 1].values
        self.min_quantities = self.constraints_data.iloc[0:22, 2].values
        self.avg_quantities = self.constraints_data.iloc[0:22, 3].values
        print(self.max_quantities)

        self.seasonal_changes = self.seasonal_changes_data.iloc[:, 1:].values
        self.action_space = Discrete(self.supermarket_data.shape[1] - 1)
        self.observation_space = Box(
            low=np.zeros(shape=(20,),
                         dtype=np.int32),
            high=np.asarray([25000 for _ in range(20)]), dtype=np.int32)
        self.current_week = 0
        self.current_quantities = self.pipe_quantities[self.current_week]

    def step(self, action):
        reward = 1
        seasonal_fix = self.seasonal_changes_data.shape[0]
        assert self.action_space.contains(action), "Invalid action!"
        self.current_quantities[action] += self.seasonal_changes[self.current_week % seasonal_fix][action]
        if self.current_quantities[action] == self.avg_quantities[action]:
            reward = -5
            # print('punish for producing average, punish range, production', reward,self.current_quantities -
            # self.avg_quantities)
        elif self.current_quantities[action] > self.max_quantities[action]:
            reward = -10
            # print('punish for exceeding the maximum limit, punish range, production', reward,self.max_quantities -
            # self.current_quantities)
            self.current_quantities[action] = self.max_quantities[action]
        elif self.current_quantities[action] < self.min_quantities[action]:
            reward = -10
            # print('punish for exceeding the minimum limit, punish range, production', reward,self.min_quantities -
            # self.current_quantities)
            self.current_quantities[action] = self.min_quantities[action]

        self.current_week += 1
        terminated = self.current_week == self.pipe_quantities.shape[0]
        truncated = False
        next_observation = np.array(self.current_quantities)
        return next_observation, reward, terminated, truncated, {}

    def print_info(self):
        print("Week Number:", self.current_week)
        # print("Total Reward:", self.total_reward)
        print("Number of Total Pipes:", self.current_quantities)
        print("Number of Produced Pipes:", self.pipe_quantities)

    def reset(self, *, seed=None, options=None):
        np.random.seed(seed)
        super().reset(seed=seed)
        self.current_week = 0
        self.current_quantities = self.pipe_quantities[self.current_week]
        return self.observation_space.sample(), {}

    def render(self):
        pass


# env = SupermarketEnvironment(config={})
# observation = env.reset()
# done = False
# total_reward = 0
# while not done:
#     action = env.action_space.sample()
#     next_observation, reward, done, truncated, info = env.step(action)
#     total_reward += reward
#     env.render()
#     observation = next_observation

#     print(action, next_observation, reward, done, truncated, info)

#     env.close()


# register the new environment
register_env('Supermarket', lambda config: SupermarketEnvironment(config={}))

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
    gamma=tune.grid_search([0.90, 0.95, 0.99]),
    model={'fcnet_hiddens': tune.grid_search([[4, 4], [8, 8], [16, 16], [32, 32], [64, 64]]),
           'use_lstm': tune.grid_search([False, True])},
)

# ****************************************************
# config['output'] = f'{os.getcwd()}/logs'

# algo = config.build()
# for i in range(100):
#     result = algo.train()
#     print(pretty_print(result))

# algo.evaluate()
# ****************************************************

# ****************************************************
# tune.run(
#     'PPO',
#     config=config.to_dict(),
#     # training iteration 1000 can be used for grid search
#     # training iteration can be set to 10,000 after best hyperparameters are found
#     stop={'training_iteration': 1000,
#           'timesteps_total': 100000,
#           'episode_reward_mean': 0.1},
#     verbose=1,
#     name='Supermarket',
#     local_dir='./ray_results',
# )
# ****************************************************

# ****************************************************
# tune.Tuner(
#     trainable="PPO",
#     run_config=air.RunConfig(
#         stop={"training_iteration": 100,
#               "timesteps_total": 100000,
#               "episode_reward_mean": 0.1},
#         local_dir="./ray_results",
#     ),
#     param_space=config.to_dict(),
# ).fit()
# ****************************************************

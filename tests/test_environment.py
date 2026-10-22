from pathlib import Path
import sys

from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from environment import SupermarketEnvironment


def test_gym_contract():
    check_env(SupermarketEnvironment(), skip_render_check=True)


def test_full_episode_and_reset():
    env = SupermarketEnvironment()
    demand_before = env.demand.copy()
    first, _ = env.reset(seed=42)
    for week in range(52):
        observation, reward, terminated, truncated, _ = env.step(week % env.action_space.n)
        assert env.observation_space.contains(observation)
        assert np.isfinite(reward) and not truncated
        assert terminated == (week == 51)
    with pytest.raises(RuntimeError):
        env.step(0)
    restored, _ = env.reset(seed=42)
    np.testing.assert_array_equal(first, restored)
    np.testing.assert_array_equal(demand_before, env.demand)
    assert not np.shares_memory(env.current_quantities, env.initial_stock)


def test_transition_and_invalid_action():
    env = SupermarketEnvironment()
    env.reset()
    initial = env.initial_stock.copy()
    _, reward, _, _, info = env.step(0)
    expected = initial - env.demand[0]
    np.testing.assert_array_equal(env.current_quantities, expected)
    assert reward == pytest.approx(-0.01 * expected.sum())
    assert info["lost_sales"] == 0
    with pytest.raises(ValueError):
        env.step(env.action_space.n)

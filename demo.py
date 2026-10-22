"""Run one bounded synthetic inventory episode with a random policy."""
import json
from environment import SupermarketEnvironment


def main():
    env = SupermarketEnvironment()
    env.action_space.seed(42)
    env.reset(seed=42)
    total_reward = 0.0
    total_lost_sales = 0.0
    for step in range(len(env.demand)):
        _, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total_reward += reward
        total_lost_sales += info["lost_sales"]
        if terminated or truncated:
            break
    env.close()
    print(json.dumps({"synthetic_demo": True, "weeks": step + 1,
                      "reward": round(total_reward, 2), "lost_sales": total_lost_sales}))


if __name__ == "__main__":
    main()

"""Optional bounded PPO training on the fictional inventory environment."""
import argparse
import json
import os

from environment import ROOT, SupermarketEnvironment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    import ray
    from ray import tune
    from ray.rllib.algorithms.ppo import PPO, PPOConfig

    output = ROOT / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    ray.init(num_cpus=1, include_dashboard=False, log_to_driver=False)
    try:
        config = (
            PPOConfig()
            .environment(SupermarketEnvironment, env_config={"data_dir": str(ROOT / "examples")})
            .framework("torch")
            .env_runners(num_env_runners=0, rollout_fragment_length=64)
            .learners(num_learners=0, num_gpus_per_learner=0)
            .training(train_batch_size_per_learner=256, minibatch_size=64, num_epochs=1)
        )
        results = tune.Tuner(
            PPO, param_space=config.to_dict(),
            run_config=tune.RunConfig(
                storage_path=str(output), name="synthetic_ppo",
                stop={"training_iteration": args.iterations}, verbose=0,
            ),
        ).fit()
        if any(result.error for result in results):
            raise RuntimeError("PPO trial failed; see the local output logs")
        print(json.dumps({"synthetic_demo": True, "completed_iterations": args.iterations}))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

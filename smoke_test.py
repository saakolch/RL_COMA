"""Quick smoke test: environment shapes and one training iteration per mode."""
import torch

from env_grid_city import GridCityEnv
from trainers import Config, train


def main():
    torch.set_num_threads(4)
    env = GridCityEnv(seed=0)
    print("obs_dim", env.obs_dim, "n_actions", env.n_actions)
    observations = env.reset()
    print("obs len", len(observations), "obs[0] len", len(observations[0]))

    observations, reward, done, info = env.step(
        [0, 1, 2], compute_counterfactuals=True
    )
    print(
        "with counterfactuals: reward",
        reward,
        "done",
        done,
        "per_alt[0]",
        info["per_alternative"][0],
    )

    env.reset()
    observations, reward, done, info = env.step(
        [0, 1, 2], compute_counterfactuals=False
    )
    print(
        "without counterfactuals: reward",
        reward,
        "done",
        done,
        "per_alt",
        info["per_alternative"],
    )

    cfg = Config(
        iterations=1,
        T=512,
        eval_interval=1,
        eval_episodes=2,
        final_eval_episodes=2,
        results_dir="_smoke",
    )
    for mode in ["shared", "difference", "coma", "coma_a2c"]:
        result = train(mode, 0, cfg, verbose=False)
        row = result["history"][-1]
        print(
            mode,
            "ok",
            "elapsed",
            round(result["elapsed"], 2),
            "success",
            row["success_rate"],
            "delivered",
            row["mean_delivered"],
            "return",
            round(row["mean_return"], 2),
        )

    for coma_critic in ["output_heads", "action_input"]:
        critic_cfg = Config(
            iterations=1,
            T=512,
            eval_interval=1,
            eval_episodes=2,
            final_eval_episodes=2,
            coma_critic=coma_critic,
            results_dir="_smoke",
        )
        result = train("coma", 0, critic_cfg, verbose=False)
        row = result["history"][-1]
        print(
            f"coma[{coma_critic}]",
            "ok",
            "success",
            row["success_rate"],
            "return",
            round(row["mean_return"], 2),
        )


if __name__ == "__main__":
    main()

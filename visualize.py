"""Visual diagnostics: grid schematic, drone trajectories, and visit heatmaps.

The trajectory figure draws one panel per method and episode, with the drone
paths on the same 6x6 grid and the same task layout.  The heatmap aggregates
how often each cell was visited over several episodes.  Both are meant to make
qualitative differences visible: duplication, wandering, and coverage.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from env_grid_city import GridCityEnv
from networks import Actor

DRONE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
TASK_COLORS = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]


def load_config(results_dir):
    path = os.path.join(results_dir, "config.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    return {
        "grid": 6,
        "n_agents": 3,
        "n_tasks": 3,
        "max_steps": 100,
        "hidden": 64,
    }


def load_actors(results_dir, mode, seed, cfg):
    path = os.path.join(results_dir, "models", f"{mode}_seed{seed}.pt")
    if not os.path.exists(path):
        return None
    env = GridCityEnv(
        grid=cfg["grid"],
        n_agents=cfg["n_agents"],
        n_tasks=cfg["n_tasks"],
        max_steps=cfg["max_steps"],
    )
    actors = [
        Actor(env.obs_dim, env.n_actions, hidden=cfg["hidden"])
        for _ in range(cfg["n_agents"])
    ]
    state_dicts = torch.load(path, map_location="cpu")
    for actor, state_dict in zip(actors, state_dicts):
        actor.load_state_dict(state_dict)
        actor.eval()
    return actors


def run_episode(actors, cfg, seed):
    """Run one greedy episode and record the trajectory."""
    env = GridCityEnv(
        grid=cfg["grid"],
        n_agents=cfg["n_agents"],
        n_tasks=cfg["n_tasks"],
        max_steps=cfg["max_steps"],
    )
    observations = env.reset(seed=seed)
    positions = [list(env.positions)]
    carrying = [list(env.carrying)]
    done = False
    while not done:
        actions = []
        with torch.no_grad():
            for i, actor in enumerate(actors):
                logits = actor(
                    torch.as_tensor(observations[i], dtype=torch.float32)
                )
                actions.append(int(logits.argmax().item()))
        observations, _, done, _ = env.step(
            actions, compute_counterfactuals=False
        )
        positions.append(list(env.positions))
        carrying.append(list(env.carrying))

    return {
        "seed": seed,
        "positions": positions,
        "carrying": carrying,
        "tasks": [dict(task) for task in env.tasks],
        "delivered": list(env.task_delivered),
        "n_delivered": sum(env.task_delivered),
        "n_steps": len(positions) - 1,
    }


def _draw_grid(ax, grid):
    for i in range(grid + 1):
        ax.axhline(i - 0.5, color="#d9d9d9", linewidth=0.8, zorder=0)
        ax.axvline(i - 0.5, color="#d9d9d9", linewidth=0.8, zorder=0)
    ax.set_xlim(-0.5, grid - 0.5)
    ax.set_ylim(-0.5, grid - 0.5)
    ax.set_xticks(range(grid))
    ax.set_yticks(range(grid))
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)


def _draw_tasks(ax, episode, grid):
    for task_id, task in enumerate(episode["tasks"]):
        color = TASK_COLORS[task_id % len(TASK_COLORS)]
        pickup = task["pickup"]
        dropoff = task["dropoff"]
        delivered = episode["delivered"][task_id]

        ax.add_patch(
            Circle(
                pickup,
                radius=0.22,
                facecolor="white",
                edgecolor=color,
                linewidth=1.6,
                zorder=4,
            )
        )
        ax.text(
            pickup[0],
            pickup[1],
            str(task_id),
            ha="center",
            va="center",
            fontsize=7,
            color=color,
            zorder=5,
        )
        ax.add_patch(
            Rectangle(
                (dropoff[0] - 0.18, dropoff[1] - 0.18),
                0.36,
                0.36,
                facecolor=color if delivered else "white",
                edgecolor=color,
                linewidth=1.6,
                zorder=4,
            )
        )
        arrow = FancyArrowPatch(
            pickup,
            dropoff,
            arrowstyle="-|>",
            mutation_scale=8,
            linestyle="--",
            linewidth=1.0,
            color=color,
            alpha=0.55,
            zorder=2,
        )
        ax.add_patch(arrow)


def _draw_depot(ax):
    ax.add_patch(
        Rectangle(
            (-0.5, -0.5),
            0.5,
            0.5,
            facecolor="#444444",
            edgecolor="#444444",
            zorder=3,
        )
    )
    ax.text(
        0.0,
        0.0,
        "D",
        ha="center",
        va="center",
        fontsize=7,
        color="white",
        zorder=6,
    )


def plot_env_schematic(path, seed=0):
    """One grid with the depot, the tasks, and the initial drone positions."""
    cfg = {"grid": 6, "n_agents": 3, "n_tasks": 3, "max_steps": 100}
    env = GridCityEnv(**cfg)
    observations = env.reset(seed=seed)
    episode = {
        "seed": seed,
        "positions": [list(env.positions)],
        "tasks": [dict(task) for task in env.tasks],
        "delivered": list(env.task_delivered),
    }

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    _draw_grid(ax, cfg["grid"])
    _draw_tasks(ax, episode, cfg["grid"])
    _draw_depot(ax)
    for i, position in enumerate(env.positions):
        ax.plot(
            position[0],
            position[1],
            marker="o",
            markersize=9,
            color=DRONE_COLORS[i % len(DRONE_COLORS)],
            markeredgecolor="white",
            markeredgewidth=1.0,
            zorder=7,
        )
        ax.text(
            position[0] + 0.12,
            position[1] + 0.12,
            f"d{i}",
            fontsize=7,
            color=DRONE_COLORS[i % len(DRONE_COLORS)],
            zorder=8,
        )
    ax.set_title("Grid-city delivery environment", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories(episodes_by_mode, path):
    """Rows = methods, columns = episodes.  Lines are drone paths."""
    modes = list(episodes_by_mode.keys())
    n_columns = max(len(episodes) for episodes in episodes_by_mode.values())
    n_rows = len(modes)
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4.0 * n_columns, 3.9 * n_rows),
        squeeze=False,
    )

    for row, mode in enumerate(modes):
        for column in range(n_columns):
            ax = axes[row][column]
            episodes = episodes_by_mode[mode]
            if column >= len(episodes):
                ax.axis("off")
                continue
            episode = episodes[column]
            grid = 6
            _draw_grid(ax, grid)
            _draw_tasks(ax, episode, grid)
            _draw_depot(ax)

            for drone_id in range(len(episode["positions"][0])):
                xs = [position[drone_id][0] for position in episode["positions"]]
                ys = [position[drone_id][1] for position in episode["positions"]]
                color = DRONE_COLORS[drone_id % len(DRONE_COLORS)]
                ax.plot(
                    xs,
                    ys,
                    color=color,
                    linewidth=1.7,
                    alpha=0.85,
                    zorder=5,
                )
                ax.plot(
                    xs[0],
                    ys[0],
                    marker="o",
                    markersize=7,
                    color=color,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    zorder=6,
                )
                ax.plot(
                    xs[-1],
                    ys[-1],
                    marker="^",
                    markersize=8,
                    color=color,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    zorder=6,
                )
            ax.set_title(
                f"{mode} | seed {episode['seed']} | "
                f"{episode['n_delivered']} delivered | {episode['n_steps']} steps",
                fontsize=8,
            )

    fig.suptitle(
        "Drone trajectories on the same task layouts "
        "(circle = start, triangle = end)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(episodes_by_mode, path, grid=6):
    """Aggregate cell visits per method over all episodes."""
    modes = list(episodes_by_mode.keys())
    fig, axes = plt.subplots(
        1, len(modes), figsize=(4.2 * len(modes), 4.0), squeeze=False
    )
    for column, mode in enumerate(modes):
        ax = axes[0][column]
        visits = np.zeros((grid, grid), dtype=float)
        episodes = episodes_by_mode[mode]
        for episode in episodes:
            for positions in episode["positions"]:
                for x, y in positions:
                    visits[y, x] += 1.0

        image = ax.imshow(
            visits,
            origin="lower",
            extent=(-0.5, grid - 0.5, -0.5, grid - 0.5),
            cmap="viridis",
            aspect="equal",
        )
        _draw_grid(ax, grid)
        _draw_depot(ax)
        for episode in episodes[:1]:
            _draw_tasks(ax, episode, grid)
        ax.set_title(f"{mode} | visits over {len(episodes)} episodes", fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Where the drones spend their time", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(
    results_dir="results",
    seed=0,
    n_episodes=2,
    n_heatmap_episodes=20,
    modes=None,
):
    cfg = load_config(results_dir)
    modes = modes or ["shared", "difference", "coma"]
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    actors_by_mode = {}
    for mode in modes:
        actors = load_actors(results_dir, mode, seed, cfg)
        if actors is None:
            print(f"  no model for {mode} seed={seed}; skipping")
            continue
        actors_by_mode[mode] = actors

    if not actors_by_mode:
        print("No models found; nothing to plot.")
        return

    episodes_by_mode = {
        mode: [
            run_episode(actors, cfg, seed=seed + offset)
            for offset in range(n_episodes)
        ]
        for mode, actors in actors_by_mode.items()
    }
    heatmap_episodes = {
        mode: [
            run_episode(actors, cfg, seed=seed + 1000 + offset)
            for offset in range(n_heatmap_episodes)
        ]
        for mode, actors in actors_by_mode.items()
    }

    plot_env_schematic(
        os.path.join(figures_dir, "fig_env_schematic.png"), seed=seed
    )
    plot_trajectories(
        episodes_by_mode, os.path.join(figures_dir, "fig_trajectories.png")
    )
    plot_heatmap(
        heatmap_episodes, os.path.join(figures_dir, "fig_heatmap.png")
    )
    print(f"Figures saved in {os.path.abspath(figures_dir)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=2)
    args = parser.parse_args()
    main(args.results_dir, args.seed, args.episodes)

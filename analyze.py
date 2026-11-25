"""Aggregate the runs and draw the comparison figures.

Figures
-------
fig_learning_curves.png   success, deliveries, and return vs environment steps
fig_learning_speed.png    AULC and steps to reach thresholds
fig_final_performance.png final success, deliveries, return, and Gini
fig_per_drone_shares.png  which drone delivered what, per method
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics import load_runs, summarize

MODE_COLORS = {
    "shared": "#1f77b4",
    "difference": "#ff7f0e",
    "coma": "#2ca02c",
    "coma_a2c": "#7f7f7f",
}
MODE_ORDER = ["shared", "difference", "coma", "coma_a2c"]


def _ordered_modes(values):
    present = [mode for mode in MODE_ORDER if mode in set(values)]
    extras = sorted(set(values) - set(present))
    return present + extras


def _bar_with_error(ax, labels, means, errors, colors, ylabel, title):
    positions = np.arange(len(labels))
    means = np.asarray(means, dtype=float)
    errors = np.asarray(errors, dtype=float)
    for position, mean, error, color in zip(positions, means, errors, colors):
        if np.isnan(mean):
            ax.text(
                position,
                0.02,
                "not reached",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=7,
                color="#666666",
                transform=ax.get_xaxis_transform(),
                clip_on=False,
            )
            continue
        ax.bar(
            position,
            mean,
            yerr=0.0 if np.isnan(error) else error,
            color=color,
            capsize=4,
            edgecolor="white",
            linewidth=0.8,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.25)


def plot_learning_curves(runs, path):
    """Mean +/- std across seeds, one line per method."""
    modes = _ordered_modes([run["mode"] for run in runs])
    metrics = [
        ("success_rate", "Success rate"),
        ("mean_delivered", "Deliveries per episode"),
        ("mean_return", "Episode return"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15.0, 4.2))

    for ax, (key, label) in zip(axes, metrics):
        for mode in modes:
            histories = [
                run["history"] for run in runs if run["mode"] == mode
            ]
            if not histories:
                continue
            frame = pd.concat(histories, ignore_index=True)
            grouped = frame.groupby("iteration")
            steps = grouped["steps"].mean()
            mean = grouped[key].mean()
            std = grouped[key].std().fillna(0.0)
            ax.plot(
                steps,
                mean,
                color=MODE_COLORS.get(mode, None),
                linewidth=2.0,
                label=mode,
            )
            ax.fill_between(
                steps,
                mean - std,
                mean + std,
                color=MODE_COLORS.get(mode, None),
                alpha=0.18,
                linewidth=0,
            )
        ax.set_xlabel("Environment steps")
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Learning curves (mean and standard deviation across seeds)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_learning_speed(per_run, path):
    modes = _ordered_modes(per_run["mode"].tolist())
    colors = [MODE_COLORS.get(mode, "#7f7f7f") for mode in modes]
    grouped = per_run.groupby("mode")

    panels = [
        ("aulc_success_rate", "AULC (success rate)", "Area under the learning curve"),
        ("steps_to_50", "Steps to 50% success", "Sample complexity (50%)"),
        (
            "steps_to_90pct_best_success",
            "Steps to 90% of best success",
            "Sample complexity (90% of best)",
        ),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(15.0, 4.2))
    for ax, (key, ylabel, title) in zip(axes, panels):
        means = [grouped[key].mean().get(mode, np.nan) for mode in modes]
        errors = [grouped[key].std().get(mode, np.nan) for mode in modes]
        _bar_with_error(ax, modes, means, errors, colors, ylabel, title)
    fig.suptitle("Learning speed", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_final_performance(per_run, path):
    modes = _ordered_modes(per_run["mode"].tolist())
    colors = [MODE_COLORS.get(mode, "#7f7f7f") for mode in modes]
    grouped = per_run.groupby("mode")

    panels = [
        ("success_rate", "Success rate", "Final success rate"),
        ("mean_delivered", "Deliveries per episode", "Final deliveries"),
        ("mean_return", "Episode return", "Final return"),
        ("gini", "Gini of delivery share", "Credit concentration"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(15.0, 4.2))
    for ax, (key, ylabel, title) in zip(axes, panels):
        means = [grouped[key].mean().get(mode, np.nan) for mode in modes]
        errors = [grouped[key].std().get(mode, np.nan) for mode in modes]
        _bar_with_error(ax, modes, means, errors, colors, ylabel, title)
    fig.suptitle("Final performance (mean and standard deviation across seeds)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_drone_shares(per_run, path):
    modes = _ordered_modes(per_run["mode"].tolist())
    grouped = per_run.groupby("mode")
    share_keys = ["share_0", "share_1", "share_2"]
    drone_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    positions = np.arange(len(modes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for drone_id, key in enumerate(share_keys):
        means = [grouped[key].mean().get(mode, np.nan) for mode in modes]
        errors = [grouped[key].std().get(mode, np.nan) for mode in modes]
        means = np.nan_to_num(np.asarray(means, dtype=float), nan=0.0)
        errors = np.nan_to_num(np.asarray(errors, dtype=float), nan=0.0)
        ax.bar(
            positions + (drone_id - 1) * width,
            means,
            width=width,
            yerr=errors,
            capsize=3,
            color=drone_colors[drone_id],
            edgecolor="white",
            linewidth=0.8,
            label=f"drone {drone_id}",
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(modes)
    ax.set_ylabel("Share of deliveries")
    ax.set_title("Which drone delivered what", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(results_dir="results"):
    runs = load_runs(results_dir)
    if not runs:
        print(f"No runs found in {results_dir}.")
        return None

    per_run, summary = summarize(runs)
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    per_run.to_csv(os.path.join(results_dir, "per_run_metrics.csv"), index=False)
    summary.to_csv(os.path.join(results_dir, "summary.csv"), index=False)

    plot_learning_curves(
        runs, os.path.join(figures_dir, "fig_learning_curves.png")
    )
    plot_learning_speed(
        per_run, os.path.join(figures_dir, "fig_learning_speed.png")
    )
    plot_final_performance(
        per_run, os.path.join(figures_dir, "fig_final_performance.png")
    )
    plot_per_drone_shares(
        per_run, os.path.join(figures_dir, "fig_per_drone_shares.png")
    )

    print(f"Summary saved in {os.path.abspath(results_dir)}")
    print(summary.to_string(index=False))
    return per_run, summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(args.results_dir)

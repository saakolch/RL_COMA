"""Learning-speed and final-performance metrics for the comparison.

Definitions
-----------
sample_complexity(run, threshold)
    Number of environment steps at which the evaluated success rate first
    reaches ``threshold``.  Linear interpolation between adjacent evaluation
    points.  ``None`` if the threshold is never reached.

aulc(run, key)
    Area under the learning curve of ``key`` versus environment steps,
    normalized by the number of steps (so it is on the same scale as the
    metric itself).  A trapezoidal rule is used, with the first evaluation
    value extended back to step 0.

summarize(runs)
    Per-run metrics plus mean/std across seeds for each method.
"""
import json
import os

import numpy as np
import pandas as pd


def sample_complexity(steps, values, threshold):
    """First step at which ``values`` reaches ``threshold`` (interpolated)."""
    steps = np.asarray(steps, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(steps) == 0:
        return None
    if threshold <= 0:
        return 0.0
    if values[0] >= threshold:
        return float(steps[0])
    for i in range(1, len(values)):
        if values[i] >= threshold:
            previous = values[i - 1]
            current = values[i]
            if current <= previous:
                return float(steps[i])
            fraction = (threshold - previous) / (current - previous)
            return float(steps[i - 1] + fraction * (steps[i] - steps[i - 1]))
    return None


def aulc(steps, values):
    """Normalized area under the learning curve."""
    steps = np.asarray(steps, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(steps) == 0:
        return 0.0
    x = np.concatenate([[0.0], steps])
    y = np.concatenate([[values[0]], values])
    area = 0.0
    for i in range(len(x) - 1):
        area += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return float(area / x[-1]) if x[-1] > 0 else 0.0


def plateau_iteration(history, key="success_rate", window=5, tolerance=0.02):
    """First iteration from which the smoothed metric stays within tolerance.

    The metric is smoothed with a moving average of width ``window`` (checkpoint
    spacing).  The plateau is the first smoothed checkpoint whose value is at
    least ``final_smoothed - tolerance`` and that never drops below that level
    afterwards.  Returns ``None`` when there are too few checkpoints.
    """
    values = history[key].values.astype(float)
    iterations = history["iteration"].values
    if len(values) < window + 1:
        return None
    smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
    final_value = smoothed[-1]
    threshold = final_value - tolerance
    for i in range(len(smoothed)):
        if np.all(smoothed[i:] >= threshold):
            return int(iterations[i + window - 1])
    return None


def load_runs(results_dir):
    """Load every ``*_history.csv`` and its matching ``*_final.json``."""
    runs_dir = os.path.join(results_dir, "runs")
    if not os.path.isdir(runs_dir):
        return []

    runs = []
    for name in sorted(os.listdir(runs_dir)):
        if not name.endswith("_history.csv"):
            continue
        history_path = os.path.join(runs_dir, name)
        final_path = history_path.replace("_history.csv", "_final.json")
        if not os.path.exists(final_path):
            continue
        history = pd.read_csv(history_path)
        with open(final_path, "r", encoding="utf-8") as file:
            final = json.load(file)
        runs.append(
            {
                "mode": str(history["mode"].iloc[0]),
                "seed": int(history["seed"].iloc[0]),
                "history": history,
                "final": final,
            }
        )
    return runs


def summarize(runs, thresholds=(0.5, 0.8, 0.9)):
    """Per-run metrics and a mean/std summary per method."""
    if not runs:
        return pd.DataFrame(), pd.DataFrame()

    best_success = max(run["final"]["success_rate"] for run in runs)
    best_delivered = max(run["final"]["mean_delivered"] for run in runs)

    rows = []
    for run in runs:
        history = run["history"]
        final = run["final"]
        row = {"mode": run["mode"], "seed": run["seed"]}

        for key in ("success_rate", "mean_delivered", "mean_return"):
            row[f"aulc_{key}"] = aulc(
                history["steps"].values, history[key].values
            )

        for threshold in thresholds:
            row[f"steps_to_{int(round(threshold * 100))}"] = sample_complexity(
                history["steps"].values,
                history["success_rate"].values,
                threshold,
            )
        row["steps_to_90pct_best_success"] = (
            sample_complexity(
                history["steps"].values,
                history["success_rate"].values,
                0.9 * best_success,
            )
            if best_success > 0
            else None
        )
        row["steps_to_90pct_best_delivered"] = (
            sample_complexity(
                history["steps"].values,
                history["mean_delivered"].values,
                0.9 * best_delivered,
            )
            if best_delivered > 0
            else None
        )

        for key in (
            "success_rate",
            "mean_delivered",
            "mean_return",
            "mean_steps",
            "mean_contests",
            "share_0",
            "share_1",
            "share_2",
            "gini",
        ):
            row[key] = final.get(key)
        row["elapsed"] = final.get("elapsed")
        rows.append(row)

    per_run = pd.DataFrame(rows)

    numeric = per_run.select_dtypes(include="number")
    summary = numeric.groupby(per_run["mode"]).agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    return per_run, summary


def best_final_success(runs):
    return max((run["final"]["success_rate"] for run in runs), default=0.0)

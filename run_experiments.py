"""Run the three credit-assignment methods over several seeds.

Examples
--------
Pilot (fast, one seed):
    python run_experiments.py --pilot

Full experiment (three seeds, 100 iterations):
    python run_experiments.py --seeds 0 1 2 --workers 4

The script writes one history CSV and one final JSON per run, a combined
``all_history.csv`` / ``final_metrics.csv`` / ``summary.csv``, and the trained
actor weights under ``results/models``.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

import pandas as pd
import torch

from metrics import load_runs, summarize
from trainers import (
    Config,
    save_final,
    save_history,
    save_model,
    train,
)


def run_one(task):
    """Train one (mode, seed) pair and save its artefacts."""
    mode, seed, cfg = task
    torch.set_num_threads(4)
    result = train(mode, seed, cfg, verbose=False)
    save_history(result["history"], cfg.results_dir, mode, seed)
    final = {
        **result["final"],
        "elapsed": result["elapsed"],
        "coma_critic": cfg.coma_critic,
    }
    save_final(final, cfg.results_dir, mode, seed)
    save_model(result["actors"], cfg.results_dir, mode, seed)
    return {
        "mode": mode,
        "seed": seed,
        "elapsed": result["elapsed"],
        "total_steps": result["total_steps"],
        "final": result["final"],
    }


def build_config(args):
    cfg = Config(
        iterations=args.iterations,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        final_eval_episodes=args.final_eval_episodes,
        coma_critic=args.coma_critic,
        results_dir=args.results_dir,
    )
    return cfg


def aggregate(results_dir):
    runs = load_runs(results_dir)
    if not runs:
        print("No runs found to aggregate.")
        return None, None

    histories = pd.concat([run["history"] for run in runs], ignore_index=True)
    histories.to_csv(os.path.join(results_dir, "all_history.csv"), index=False)

    per_run, summary = summarize(runs)
    per_run.to_csv(os.path.join(results_dir, "final_metrics.csv"), index=False)
    summary.to_csv(os.path.join(results_dir, "summary.csv"), index=False)
    return per_run, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["shared", "difference", "coma"],
        choices=["shared", "difference", "coma", "coma_a2c"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--final-eval-episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--coma-critic",
        choices=["output_heads", "action_input"],
        default="output_heads",
        help="COMA critic version: output_heads (reference) or action_input (shared Q(s,a))",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="fast smoke run: one seed, 20 iterations, 20 eval episodes",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="run analyze.py after training",
    )
    args = parser.parse_args()

    if args.pilot:
        args.seeds = [0]
        args.iterations = 20
        args.eval_interval = 2
        args.eval_episodes = 20
        args.final_eval_episodes = 50
        args.workers = 3
        args.results_dir = "results_pilot"

    os.makedirs(args.results_dir, exist_ok=True)
    cfg = build_config(args)
    with open(os.path.join(args.results_dir, "config.json"), "w", encoding="utf-8") as file:
        json.dump(asdict(cfg), file, indent=2)

    tasks = [
        (mode, seed, cfg) for mode in args.modes for seed in args.seeds
    ]
    print("=" * 78)
    print(
        f"TRAINING {len(tasks)} RUNS | modes={args.modes} seeds={args.seeds} "
        f"iterations={args.iterations} workers={args.workers} "
        f"coma_critic={args.coma_critic}"
    )
    print(f"Results directory: {os.path.abspath(args.results_dir)}")
    print("=" * 78)

    start = time.time()
    results = []
    if args.workers <= 1 or len(tasks) == 1:
        for task in tasks:
            results.append(run_one(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_one, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    results.append(future.result())
                    print(
                        f"  finished {task[0]:10s} seed={task[1]} "
                        f"({len(results)}/{len(tasks)})"
                    )
                except Exception as error:  # noqa: BLE001
                    print(f"  FAILED {task[0]} seed={task[1]}: {error}")

    elapsed = time.time() - start
    print("=" * 78)
    print(f"Training finished in {elapsed / 60:.1f} min")

    per_run, summary = aggregate(args.results_dir)
    if summary is not None:
        columns = [
            "mode",
            "success_rate_mean",
            "mean_delivered_mean",
            "mean_return_mean",
            "gini_mean",
            "aulc_success_rate_mean",
            "steps_to_80_mean",
            "steps_to_90pct_best_success_mean",
        ]
        columns = [column for column in columns if column in summary.columns]
        print("\nSUMMARY")
        print(summary[columns].to_string(index=False))

    if args.analyze:
        try:
            import analyze

            analyze.main(args.results_dir)
        except Exception as error:  # noqa: BLE001
            print(f"Analysis failed: {error}")


if __name__ == "__main__":
    main()

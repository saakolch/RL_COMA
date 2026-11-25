"""Report where each method's learning curve plateaus.

Usage:
    python plateau_report.py --results-dir results_500

The plateau is the first checkpoint from which the smoothed metric stays
within ``tolerance`` of its final value.  Use it to choose the training
budget for the final multi-seed run.
"""
import argparse

from metrics import load_runs, plateau_iteration


def main(results_dir):
    runs = load_runs(results_dir)
    if not runs:
        print(f"No runs found in {results_dir}")
        return

    for run in runs:
        history = run["history"]
        plateau_success = plateau_iteration(history, "success_rate")
        plateau_delivered = plateau_iteration(
            history, "mean_delivered", tolerance=0.05
        )
        max_success = history["success_rate"].max()
        final_success = history["success_rate"].iloc[-1]
        max_delivered = history["mean_delivered"].max()
        final_delivered = history["mean_delivered"].iloc[-1]
        last_iteration = int(history["iteration"].iloc[-1])
        print(
            f"{run['mode']:10s} seed={run['seed']} | "
            f"success max {max_success:.2f} final {final_success:.2f} "
            f"plateau@{plateau_success} | "
            f"delivered max {max_delivered:.2f} final {final_delivered:.2f} "
            f"plateau@{plateau_delivered} | "
            f"last it {last_iteration}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results_500")
    args = parser.parse_args()
    main(args.results_dir)

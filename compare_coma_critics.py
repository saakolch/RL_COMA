"""Compare the credit-assignment variants across result directories.

Usage:
    python compare_coma_critics.py

Directories:
    results_final        shared, difference, coma (PPO + expected target)
    results_coma_action  coma with the action-input critic
    results_coma_a2c     faithful COMA: A2C actor + TD(lambda) critic
"""
import json
import os

import pandas as pd


def load(results_dir, mode, label):
    runs_dir = os.path.join(results_dir, "runs")
    if not os.path.isdir(runs_dir):
        return pd.DataFrame()
    rows = []
    for name in sorted(os.listdir(runs_dir)):
        if not name.endswith("_final.json"):
            continue
        with open(os.path.join(runs_dir, name), encoding="utf-8") as file:
            data = json.load(file)
        if data.get("mode") != mode:
            continue
        rows.append(
            {
                "variant": label,
                "seed": data.get("seed"),
                "success": data.get("success_rate"),
                "delivered": data.get("mean_delivered"),
                "return": data.get("mean_return"),
                "gini": data.get("gini"),
            }
        )
    return pd.DataFrame(rows)


def main():
    frames = [
        load("results_final", "shared", "shared (PPO)"),
        load("results_final", "difference", "difference (PPO)"),
        load(
            "results_final",
            "coma",
            "COMA output_heads (PPO + expected target)",
        ),
        load(
            "results_coma_action",
            "coma",
            "COMA action_input (PPO + expected target)",
        ),
        load(
            "results_coma_a2c",
            "coma_a2c",
            "COMA A2C + TD(lambda) (faithful)",
        ),
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        print("No results found.")
        return

    df = pd.concat(frames, ignore_index=True)
    print("Per-seed results")
    print(df.to_string(index=False))
    print()
    print("Mean +/- std")
    summary = df.groupby("variant")[["success", "delivered", "return", "gini"]].agg(
        ["mean", "std"]
    )
    print(summary.to_string())
    print()
    for metric in ("success", "delivered", "return"):
        print(
            f"{metric}: "
            + ", ".join(
                f"{variant} {summary.loc[variant, (metric, 'mean')]:.3f}"
                for variant in summary.index
            )
        )


if __name__ == "__main__":
    main()

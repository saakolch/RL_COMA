"""Build the report tables and a results summary from results_final.

Usage:
    python make_tables.py
"""
import os

import pandas as pd

RESULTS = "results_final"
OUT = os.path.join("deliverables", "tables")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(RESULTS, "final_metrics.csv"))
modes = ["shared", "difference", "coma"]
df["mode"] = pd.Categorical(df["mode"], categories=modes, ordered=True)
df = df.sort_values("mode")

# ---------------------------------------------------------------- table 1
t1 = df.groupby("mode", observed=True).agg(
    success_mean=("success_rate", "mean"),
    success_std=("success_rate", "std"),
    delivered_mean=("mean_delivered", "mean"),
    delivered_std=("mean_delivered", "std"),
    return_mean=("mean_return", "mean"),
    return_std=("mean_return", "std"),
    gini_mean=("gini", "mean"),
    gini_std=("gini", "std"),
).reset_index()
t1.to_csv(os.path.join(OUT, "table1_final_performance.csv"), index=False)

# ---------------------------------------------------------------- table 2
t2 = df.groupby("mode", observed=True).agg(
    aulc_mean=("aulc_success_rate", "mean"),
    aulc_std=("aulc_success_rate", "std"),
    steps50_mean=("steps_to_50", "mean"),
    steps50_std=("steps_to_50", "std"),
    steps90_mean=("steps_to_90pct_best_success", "mean"),
    steps90_std=("steps_to_90pct_best_success", "std"),
    elapsed_mean=("elapsed", "mean"),
    elapsed_std=("elapsed", "std"),
).reset_index()
t2.to_csv(os.path.join(OUT, "table2_learning_speed.csv"), index=False)

# ---------------------------------------------------------------- table 3
t3 = df[
    [
        "mode",
        "seed",
        "success_rate",
        "mean_delivered",
        "mean_return",
        "gini",
        "aulc_success_rate",
        "steps_to_50",
        "steps_to_90pct_best_success",
        "elapsed",
    ]
].copy()
t3.to_csv(os.path.join(OUT, "table3_per_seed.csv"), index=False)

# ---------------------------------------------------------------- markdown
lines = ["# Final results summary", ""]
lines.append("## Table 1 - Final performance (mean +/- std over 3 seeds)")
lines.append("")
lines.append("| Method | Success rate | Deliveries /3 | Episode return | Gini |")
lines.append("|---|---|---|---|---|")
for _, row in t1.iterrows():
    lines.append(
        f"| {row['mode']} | {row['success_mean']:.3f} +/- {row['success_std']:.3f} | "
        f"{row['delivered_mean']:.2f} +/- {row['delivered_std']:.2f} | "
        f"{row['return_mean']:.2f} +/- {row['return_std']:.2f} | "
        f"{row['gini_mean']:.3f} +/- {row['gini_std']:.3f} |"
    )
lines.append("")
lines.append("## Table 2 - Learning speed (mean +/- std over 3 seeds)")
lines.append("")
lines.append("| Method | AULC (success) | Steps to 50% | Steps to 90% of best | Wall-clock per run (s) |")
lines.append("|---|---|---|---|---|")
for _, row in t2.iterrows():
    s50 = (
        "not reached"
        if pd.isna(row["steps50_mean"])
        else f"{row['steps50_mean']:.0f} +/- {row['steps50_std']:.0f}"
    )
    s90 = (
        "not reached"
        if pd.isna(row["steps90_mean"])
        else f"{row['steps90_mean']:.0f} +/- {row['steps90_std']:.0f}"
    )
    lines.append(
        f"| {row['mode']} | {row['aulc_mean']:.3f} +/- {row['aulc_std']:.3f} | "
        f"{s50} | {s90} | {row['elapsed_mean']:.0f} |"
    )
lines.append("")
lines.append("## Table 3 - Per-seed final metrics")
lines.append("")
lines.append(
    "| Mode | Seed | Success | Deliveries | Return | Gini | AULC | Steps to 50% |"
)
lines.append("|---|---|---|---|---|---|---|---|")
for _, row in t3.iterrows():
    steps50 = "not reached" if pd.isna(row["steps_to_50"]) else f"{row['steps_to_50']:.0f}"
    lines.append(
        f"| {row['mode']} | {int(row['seed'])} | {row['success_rate']:.2f} | "
        f"{row['mean_delivered']:.2f} | {row['mean_return']:.2f} | "
        f"{row['gini']:.3f} | {row['aulc_success_rate']:.3f} | {steps50} |"
    )
lines.append("")
lines.append(
    "Notes: shared reaches 50% in 2 of 3 seeds; COMA and difference never reach "
    "50% or 90% of the best final success within 340 iterations (~680k steps)."
)

with open(os.path.join("deliverables", "results_summary.md"), "w", encoding="utf-8") as file:
    file.write("\n".join(lines))

print("\n".join(lines))

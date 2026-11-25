# Task Allocation for Delivery Drones: Shared vs Difference vs COMA

A self-contained PyTorch implementation and controlled comparison of three
credit-assignment schemes on a 6x6 grid-city delivery task with 3 drones and
3 tasks.

This folder is a self-contained class-project implementation; no external
reference code is required.

## Headline result

Over three seeds and ~680,000 environment steps per method, **shared reward is
the only scheme that learns reliably**. It reaches 41.3% success, 2.15 deliveries
per episode, and a positive return of +4.57; COMA reaches 10.3% success but is
highly seed-dependent (28%, 1%, 2%); the one-step difference reward reaches 3.0%.
Shared reward is also the fastest (AULC 0.176 vs 0.017 and 0.009) and the most
balanced (Gini 0.071 vs 0.171 and 0.242). The finding contradicts the common
assumption that credit assignment always helps at this scale: the shared signal
is dense enough, and the team is small enough, that simplicity wins.

| Method | Success rate | Deliveries /3 | Episode return | Gini | AULC |
| --- | --- | --- | --- | --- | --- |
| **shared** | **0.413 +/- 0.038** | **2.15 +/- 0.09** | **+4.57 +/- 0.54** | **0.071 +/- 0.019** | **0.176 +/- 0.042** |
| coma | 0.103 +/- 0.153 | 1.12 +/- 0.64 | -3.47 +/- 4.51 | 0.242 +/- 0.095 | 0.017 +/- 0.024 |
| difference | 0.030 +/- 0.030 | 0.78 +/- 0.14 | -5.58 +/- 0.98 | 0.171 +/- 0.050 | 0.009 +/- 0.007 |

## Environment

* 6x6 grid, 3 drones, 3 tasks; each task has a pickup and a dropoff cell.
* Drones start at the depot (0, 0).
* Actions: up, down, left, right.
* A drone carries at most one task; a task is delivered when its carrier reaches
  the dropoff.
* Reward: `-0.1` per step, `+5.0` per delivery. Episode ends when all tasks are
  delivered or after 100 steps.
* Observation per drone: agent id one-hot + all drone states + all task states +
  time (28 dimensions).

## Methods

| mode | reward / advantage | critic |
| --- | --- | --- |
| `shared` | team reward for every drone | one centralized `V(s)` |
| `difference` | `D_i = r - E_{a_i' ~ pi_i}[r(s, (a_i', a_-i))]` | one centralized `V_i(s)` per drone |
| `coma` | `A_i = Q(s,a) - E_{a_i' ~ pi_i}[Q(s, (a_i', a_-i))]` | see the two COMA critic versions below |

All three use the same PPO backbone (clipped surrogate, GAE for the two
value-based modes, 4 epochs, minibatches, entropy bonus) so that the comparison
isolates the credit-assignment mechanism.

### Two explicitly selectable COMA critics

COMA needs a centralized critic `Q(s, .)`. Two parameterisations are provided in

etworks.py`, and the runner records which one is used:

| `--coma-critic` | class | how the critic is trained |
| --- | --- | --- |
| `output_heads` (default, reference) | `JointActionCritic` | one output per joint action; only the taken action's output receives gradient, so the 63 counterfactual values are extrapolations |
| `action_input` | `ActionInputCritic` | `Q(s, a)` with the joint action one-hot as input; parameters are shared across all joint actions, so the counterfactual evaluations use trained parameters |

The final comparison in the report uses `output_heads` (the reference
implementation). The `action_input` version is an additional, clearly labelled
run (`results_coma_action/`). It did not fix the instability: it reached 0.0%
success, 0.04 deliveries and -9.8 return on all three seeds, so the instability
is not an artefact of the critic parameterisation. Compare the two runs with
`python compare_coma_critics.py`.

### Fidelity to the original COMA

The counterfactual advantage is exactly the paper's formula (unit-tested against
brute force). The surrounding training differs in three deliberate ways, all
shared with the baselines so the comparison isolates credit assignment:

- PPO (clipped surrogate, four epochs, minibatches) instead of the paper's
  A2C-style update;
- a one-step expected-Q critic target instead of TD(lambda) on the sampled joint
  action;
- MLPs instead of GRUs, because the grid-city observation is fully observable.

The experiments therefore test COMA-style credit assignment on a common PPO
backbone, not a line-by-line reproduction of the paper's StarCraft results.

### Faithful-paper COMA variant (`coma_a2c`)

A separate mode implements the paper's update more literally:

- an **A2C-style** vanilla policy gradient with the counterfactual advantage
  (no PPO clipping, no epochs);
- a **TD(lambda) critic target on the sampled joint action**
  (`JointActionCritic.td_lambda_targets`, unit-tested against the forward-view
  return).

It uses the same environment, seeds, iterations and evaluation as the main run
so it can be compared directly, but it is **not** part of the main table: it
changes the backbone (A2C vs PPO) and would confound the credit-assignment
comparison.

```bash
python run_experiments.py --modes coma_a2c --seeds 0 1 2 \
    --iterations 340 --eval-interval 10 --eval-episodes 20 \
    --final-eval-episodes 100 --workers 3 --results-dir results_coma_a2c
```

Compare all variants with `python compare_coma_critics.py`.

The variant reached **0.3% success, 0.30 deliveries and -8.5 return** (mean over
three seeds), so the faithful update did not change the ranking: COMA remains far
behind shared reward at this scale.

## Reproduce the final experiment

```bash
pip install -r requirements.txt

# main comparison: 3 methods x 3 seeds x 340 iterations (~680k steps each)
python run_experiments.py --modes shared difference coma --seeds 0 1 2 \
    --iterations 340 --eval-interval 10 --eval-episodes 20 \
    --final-eval-episodes 100 --workers 9 --results-dir results_final

# additional COMA run with the action-input critic
python run_experiments.py --modes coma --seeds 0 1 2 \
    --iterations 340 --eval-interval 10 --eval-episodes 20 \
    --final-eval-episodes 100 --workers 3 \
    --coma-critic action_input --results-dir results_coma_action

# faithful-paper COMA: A2C actor + TD(lambda) critic
python run_experiments.py --modes coma_a2c --seeds 0 1 2 \
    --iterations 340 --eval-interval 10 --eval-episodes 20 \
    --final-eval-episodes 100 --workers 3 --results-dir results_coma_a2c

# report figures and tables
python analyze.py --results-dir results_final
python make_tables.py

# trajectory, heatmap, and environment schematic
python visualize.py --results-dir results_final --seed 0 --episodes 2
```

Fast smoke test: `python run_experiments.py --pilot`.

## Files

| file | purpose |
| --- | --- |
| `env_grid_city.py` | environment (reference dynamics + delivery/contest diagnostics; optional counterfactual evaluation) |
| 
etworks.py` | actors, centralized V critic, joint-action Q critic, COMA helpers, rollout buffer |
| `trainers.py` | rollout collection, GAE / difference / counterfactual advantages, PPO update, evaluation |
| `metrics.py` | sample complexity, AULC, plateau detection, Gini, summary tables |
| `run_experiments.py` | runs all methods/seeds and saves CSVs, JSONs, and models |
| `analyze.py` | learning curves, learning-speed bars, final performance, per-drone shares |
| `visualize.py` | grid schematic, trajectories, visit heatmaps |
| `make_tables.py` | final report tables and `results_summary.md` |
| `smoke_test.py` | quick environment/network sanity check |
| `test_coma_math.py` | unit tests of the COMA advantage against brute force |
| `test_coma_gradient.py` | compares how the two COMA critics receive gradient |
| `plateau_report.py` | reports where each learning curve plateaus |

## Outputs

* `results_final/runs/<mode>_seed<k>_history.csv` and `_final.json`
* `results_final/models/<mode>_seed<k>.pt`
* `results_final/all_history.csv`, `final_metrics.csv`, `summary.csv`, `config.json`
* `results_final/figures/fig_learning_curves.png`, `fig_learning_speed.png`,
  `fig_final_performance.png`, `fig_per_drone_shares.png`,
  `fig_trajectories.png`, `fig_heatmap.png`, `fig_env_schematic.png`

## Deliverables

The `deliverables/` folder contains the submission-ready files:

* `Task_Allocation_COMA_vs_Difference_Report.docx` / `.pdf` — 10-page report
* `Task_Allocation_COMA_vs_Difference_Slides.pptx` / `.pdf` — 14-slide deck
* `figures/` — report figures
* `tables/` — Table 1-3 CSV files
* `results_summary.md` — markdown summary of the final results



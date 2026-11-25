# Deliverables

Submission-ready files for **Task Allocation for Delivery Drones: COMA vs
Difference Rewards**.

| File | Description |
| --- | --- |
| `Task_Allocation_COMA_vs_Difference_Report.docx` | 10-page project report (Word, editable) |
| `Task_Allocation_COMA_vs_Difference_Report.pdf` | 10-page project report (PDF, for submission) |
| `Task_Allocation_COMA_vs_Difference_Slides.pptx` | 14-slide presentation deck (PowerPoint, editable) |
| `Task_Allocation_COMA_vs_Difference_Slides.pdf` | Presentation PDF (backup) |
| `figures/` | Report and slide figures (PNG) |
| `tables/` | Final tables (CSV) |
| `results_summary.md` | Markdown summary of the final results |

## Report structure (10 pages)

1. Abstract
2. Introduction
3. Background and literature review (cooperative MARL, credit assignment,
   shared reward, difference rewards, COMA, related methods, drone delivery)
4. Methodology (environment, problem formulation, the three schemes, training
   protocol, metrics, implementation)
5. Results (learning curves, learning speed, final performance, contribution
   balance, COMA instability, per-seed table)
6. Analysis and discussion (why shared wins, why difference fails, why COMA is
   unstable, cost, relationship to the literature, limitations)
7. Conclusion and future work
8. References (14 entries)
9. Appendix (code structure, reproduction, heatmap)

## Presentation structure (14 slides)

1. Title
2. The problem: who gets the credit?
3. Three answers to the credit-assignment problem
4. Experimental design
5. Learning curves
6. Learning speed
7. Final performance
8. Contribution balance
9. COMA's instability
10. Why shared wins at this scale
11. Cost and limitations
12. Conclusion
13. Backup: hyperparameters
14. Backup: per-seed metrics

## Key numbers

| Method | Success | Deliveries /3 | Return | Gini | AULC |
| --- | --- | --- | --- | --- | --- |
| shared | 0.413 +/- 0.038 | 2.15 +/- 0.09 | +4.57 +/- 0.54 | 0.071 +/- 0.019 | 0.176 +/- 0.042 |
| coma | 0.103 +/- 0.153 | 1.12 +/- 0.64 | -3.47 +/- 4.51 | 0.242 +/- 0.095 | 0.017 +/- 0.024 |
| difference | 0.030 +/- 0.030 | 0.78 +/- 0.14 | -5.58 +/- 0.98 | 0.171 +/- 0.050 | 0.009 +/- 0.007 |

Source of truth: `../results_final/summary.csv`, `../results_final/final_metrics.csv`,
and the per-seed `../results_final/runs/*_final.json`.

## COMA ablations

The report's COMA results use the reference `output_heads` critic. Two
robustness variants were also run with the same environment, seeds, iterations
and evaluation:

| Variant | Success | Deliveries | Return |
| --- | --- | --- | --- |
| COMA output_heads (PPO + expected target, main) | 0.103 | 1.12 | -3.47 |
| COMA action_input (shared parameters) | 0.000 | 0.04 | -9.80 |
| COMA A2C + TD(lambda) (faithful paper) | 0.003 | 0.30 | -8.48 |

Neither variant fixed COMA's instability; the failure is not specific to the
critic parameterisation or the update rule. They are reported as robustness
checks, not in the main table, because they change the backbone.

## Regenerate figures and tables

The report and slides are prebuilt and included here, so they do not need to be
regenerated. The figures and tables can be refreshed from the raw results:

```bash
# from the project root
python analyze.py --results-dir results_final      # figures
python make_tables.py                              # tables + results_summary.md
```

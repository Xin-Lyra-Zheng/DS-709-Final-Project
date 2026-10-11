# Auditing the Costs of Multi-Group Fairness Repair

Code and results for the final project on intersectional equal opportunity, accuracy, and withdrawals of favorable predictions. The paper is [paper.pdf](paper.pdf).

## Reproduce

Python 3.9+:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python code/reproduce.py
```

The archive in `data/` contains frozen scores and baseline predictions for Credit Default and COMPAS: two models and five splits per dataset. The scripts recompute the baseline audit, Retain-EG, and the sensitivity checks. They reuse the saved classifiers and do not retrain the original models or published baselines.

Results are written to `results/`. `baseline_metrics.csv` and `retention_metrics.csv` contain per-fit outcomes; the matching `*_groups.csv` files contain subgroup outcomes. `results/sensitivity/` contains the budget and sample-size checks.

The main Retain-EG setting caps each repair and validation TPR gap at the larger of 0.02 and EG's gap, permits a 0.005 accuracy decrease, and limits the mixture-weight change to 0.25 in L1 distance. Withdrawal is measured among all original favorable predictions.

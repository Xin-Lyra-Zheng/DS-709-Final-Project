"""Recompute the baseline audit from saved scores and predictions."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from run_retention_eg import PARTS, group_definitions

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/baselines"
OUT = ROOT / "results"


def metrics(y, b, q, mask):
    y, b, q = y[mask], b[mask], q[mask]
    accuracy = np.mean(y * q + (1 - y) * (1 - q))
    return dict(
        n=len(y),
        n_positive=int(y.sum()),
        n_base_positive=int(b.sum()),
        tpr=float(q[y == 1].mean()) if y.any() else np.nan,
        accuracy=float(accuracy),
        accuracy_loss=float(np.mean(y == b) - accuracy),
        withdrawal=float(np.sum(b * (1 - q)) / b.sum()) if b.sum() else np.nan,
    )


def run():
    rows, details = [], []
    for dataset in ("Credit", "COMPAS"):
        for model in ("LR", "HGB"):
            for seed in range(709, 714):
                tag = f"{dataset}_{model}_{seed}"
                raw = np.load(ROOT / f"data/scores/expanded_{tag}.npz")
                base = np.load(INPUT / f"{tag}.npz")
                info = json.loads((INPUT / f"{tag}.json").read_text())
                groups = group_definitions(raw, info["meta"])
                for part in PARTS:
                    y = raw[f"{part}_y"]
                    b = base[f"Original__{part}"]
                    for method in ("Original", "Hardt", "Morina", "EG"):
                        q = base[f"{method}__{part}"]
                        key = dict(dataset=dataset, model=model, seed=seed,
                                   partition=part, method=method)
                        row = dict(**key, **metrics(y, b, q, np.ones(len(y), bool)))
                        rates, intersections = {}, []
                        for family, name, masks in groups:
                            result = metrics(y, b, q, masks[part])
                            details.append(dict(**key, family=family, group=name, **result))
                            rates.setdefault(family, []).append(result["tpr"])
                            if family == "intersection":
                                intersections.append(result)
                        row.update({f"gap_{family}": float(np.ptp(values))
                                    for family, values in rates.items()})
                        row.update(
                            min_group_accuracy=min(x["accuracy"] for x in intersections),
                            max_group_withdrawal=max(x["withdrawal"] for x in intersections),
                            max_group_accuracy_loss=max(x["accuracy_loss"] for x in intersections),
                            reported_intersections=len(intersections),
                        )
                        rows.append(row)
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "baseline_metrics.csv", index=False)
    pd.DataFrame(details).to_csv(OUT / "baseline_groups.csv", index=False)


if __name__ == "__main__":
    run()

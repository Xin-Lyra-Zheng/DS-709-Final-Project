"""Fit the three published fairness methods on saved model scores."""

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
from threadpoolctl import threadpool_limits

from baselines import RTDP, eg, hardt, optimal_thresholds, rtdp_predict

ROOT = Path(__file__).resolve().parents[1]
PARTS = ("repair", "validation", "test")


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def fit(tag, input_dir, output_dir, check=False):
    score_path = input_dir / "scores" / f"expanded_{tag}.npz"
    raw = np.load(score_path)
    meta_path = score_path.with_suffix(".json")
    if not meta_path.exists():
        meta_path = ROOT / "data/baselines" / f"{tag}.json"
        meta = json.loads(meta_path.read_text())["meta"]
    else:
        meta = json.loads(meta_path.read_text())
    shape = tuple(len(level) for level in meta["levels"][:2])
    scores = {p: raw[f"{p}_score"] for p in PARTS}
    labels = {p: raw[f"{p}_y"] for p in PARTS}
    groups = {p: raw[f"{p}_groups"] for p in PARTS}
    threshold = float(optimal_thresholds(scores["repair"], labels["repair"],
                                         np.zeros(len(labels["repair"]), int), 1)[0])
    predictions = {f"Original__{p}": (scores[p] >= threshold).astype(float)
                   for p in PARTS}
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        methods = [
            ("Hardt", *hardt(scores, labels, groups, shape, threshold)),
            ("EG", *eg(scores, labels, groups, shape, threshold)),
        ]
        rtdp = RTDP(scores["repair"], labels["repair"], groups["repair"],
                    shape, threshold)
        policy = rtdp.solve()
    methods.append(("Morina", {p: rtdp_predict(policy, scores[p], groups[p])
                                for p in PARTS}, policy))
    info = {"meta": meta, "threshold": threshold,
            "fit": {name: fit_info for name, _, fit_info in methods}}
    for name, values, _ in methods:
        for part in PARTS:
            predictions[f"{name}__{part}"] = values[part]
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / f"{tag}.npz", **predictions)
    (output_dir / f"{tag}.json").write_text(json.dumps(info, default=serializable))
    if check:
        reference = np.load(ROOT / "data/baselines" / f"{tag}.npz")
        difference = max(float(np.max(np.abs(values - reference[name])))
                         for name, values in predictions.items())
        if difference > 1e-8:
            raise AssertionError(f"{tag}: max prediction difference {difference}")
        print(tag, "matched frozen baselines", flush=True)
    else:
        print(tag, "fit", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/refit/baselines")
    parser.add_argument("--only", help="Example: COMPAS_LR_709")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    tags = [f"{dataset}_{model}_{seed}"
            for dataset in ("Credit", "COMPAS")
            for model in ("LR", "HGB")
            for seed in range(709, 714)]
    for tag in tags:
        if args.only is None or args.only == tag:
            fit(tag, args.input_dir, args.output_dir, args.check)

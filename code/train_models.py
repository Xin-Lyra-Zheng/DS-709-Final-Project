"""Train and calibrate LR/HGB scores for Credit Default and COMPAS."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "default_credit.xls": (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls",
        "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933"),
    "compas-scores-two-years.csv": (
        "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv",
        "c451db85908b2f7fef1d83203bedf6b71ecda0d5af468d82ae62178f91d0cc7d"),
}


def ensure_data():
    directory = ROOT / "data/raw"
    directory.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in SOURCES.items():
        path = directory / name
        if not path.exists():
            with urlopen(url, timeout=180) as response:
                path.write_bytes(response.read())
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Unexpected SHA-256 for {name}")


def load_data(name):
    if name == "Credit":
        path = ROOT / "data/raw/default_credit.xls"
        data = pd.read_excel(path, header=1)
        y = (data["default payment next month"] == 0).to_numpy(dtype=int)
        groups = pd.DataFrame({"sex": data.SEX.map({1: "Male", 2: "Female"}),
                               "age": np.where(data.AGE >= 40, "40+", "<40")})
        X = data.drop(columns=["ID", "SEX", "AGE", "default payment next month"]).copy()
        for col in ("EDUCATION", "MARRIAGE"):
            X[col] = X[col].astype(str)
        cluster = np.array([f"Credit:{i}" for i in data.ID])
    else:
        path = ROOT / "data/raw/compas-scores-two-years.csv"
        data = pd.read_csv(path)
        data = data.loc[data.days_b_screening_arrest.between(-30, 30)
                        & (data.is_recid != -1) & (data.c_charge_degree != "O")
                        & (data.score_text != "N/A")].copy().reset_index(drop=True)
        y = (data.two_year_recid == 0).to_numpy(dtype=int)
        groups = data[["race", "sex"]].astype(str).copy()
        groups["age"] = np.where(data.age >= 40, "40+", "<40")
        X = data[["priors_count", "juv_fel_count", "juv_misd_count",
                  "juv_other_count", "c_charge_degree"]].copy()
        cluster = np.array([f"COMPAS:{i}" for i in data.id])
    for col in X:
        if X[col].dtype == object:
            X[col] = X[col].fillna("__missing__").astype(str)
    levels = [sorted(groups[col].unique().tolist()) for col in groups]
    codes = np.column_stack([pd.Categorical(groups[col], categories=levels[i]).codes
                             for i, col in enumerate(groups)])
    meta = {"dataset": name, "n": len(y), "positive": int(y.sum()),
            "attributes": groups.columns.tolist(), "levels": levels,
            "features": X.columns.tolist(),
            "sources": [{"file": path.name,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}],
            "weighted": False}
    return X, y, codes, cluster, meta


def split_data(cluster, seed):
    rng = np.random.default_rng(seed)
    ids = np.unique(cluster)
    ids = ids[rng.permutation(len(ids))]
    fractions = np.array([.5, .1, .15, .1, .15])
    names = ("train", "calibration", "repair", "validation", "test")
    cuts = np.rint(np.r_[0, np.cumsum(fractions)] * len(ids)).astype(int)
    return {name: np.flatnonzero(np.isin(cluster, ids[a:b]))
            for name, a, b in zip(names, cuts[:-1], cuts[1:])}


def fit(tag, output_dir, check=False):
    dataset, model_name, seed_text = tag.rsplit("_", 2)
    seed = int(seed_text)
    X, y, groups, cluster, meta = load_data(dataset)
    partitions = split_data(cluster, seed)
    categorical = [col for col in X if X[col].dtype == object]
    numeric = [col for col in X if col not in categorical]
    if model_name == "LR":
        prep = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)])
        estimator = LogisticRegression(C=1, max_iter=2000, random_state=seed)
    else:
        prep = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
             categorical)])
        estimator = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=.1, max_leaf_nodes=31,
            l2_regularization=1, early_stopping=False, random_state=seed)
    model = make_pipeline(prep, estimator)
    with threadpool_limits(limits=2):
        model.fit(X.iloc[partitions["train"]], y[partitions["train"]])

        def logit(indices):
            p = np.clip(model.predict_proba(X.iloc[indices])[:, 1], 1e-8, 1 - 1e-8)
            return np.log(p / (1 - p)).reshape(-1, 1)

        calibration = LogisticRegression(C=1e6, max_iter=1000)
        calibration.fit(logit(partitions["calibration"]), y[partitions["calibration"]])
        saved = {}
        for part in ("repair", "validation", "test"):
            ix = partitions[part]
            saved[f"{part}_score"] = calibration.predict_proba(logit(ix))[:, 1]
            saved[f"{part}_y"] = y[ix]
            saved[f"{part}_groups"] = groups[ix]
    meta.update({"seed": seed, "model": model_name,
                 "partition_n": {name: len(ix) for name, ix in partitions.items()},
                 "auc_test": roc_auc_score(saved["test_y"], saved["test_score"]),
                 "thresholds_use_validation": True})
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / f"expanded_{tag}.npz", **saved)
    (output_dir / f"expanded_{tag}.json").write_text(json.dumps(meta, indent=2))
    if check:
        reference = np.load(ROOT / "data/scores" / f"expanded_{tag}.npz")
        difference = max(float(np.max(np.abs(values - reference[name])))
                         for name, values in saved.items())
        if difference > 1e-8:
            raise AssertionError(f"{tag}: max score difference {difference}")
        print(tag, "matched frozen scores", flush=True)
    else:
        print(tag, "trained", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/refit/scores")
    parser.add_argument("--only", help="Example: Credit_LR_709")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    ensure_data()
    tags = [f"{dataset}_{model}_{seed}"
            for dataset in ("Credit", "COMPAS")
            for model in ("LR", "HGB")
            for seed in range(709, 714)]
    for tag in tags:
        if args.only is None or args.only == tag:
            fit(tag, args.output_dir, args.check)

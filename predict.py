"""
predict.py
----------
Lightweight inference for the credit card anomaly detector.

The trained "model" is nothing more than two 9-element vectors (mu, sigma)
and one scalar threshold (alpha_opt), stored in model_params.json. This
script loads those parameters and scores new transactions without needing
scikit-learn, a pickled model file, or a server.

Usage:
    python predict.py --values 1.2 -0.5 0.3 -6.2 -1.1 -2.0 -0.8 0.4 3
    python predict.py --csv transactions.csv --out predictions.csv
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

PARAMS_PATH = Path(__file__).parent / "model_params.json"


def load_params(path: Path = PARAMS_PATH) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def log_density(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Log of the joint density under independent per-feature normal
    distributions, computed in log-space for numerical stability
    (raw densities underflow to 0 once epsilon gets as small as
    alpha_opt ** 9 ~ 3.9e-19).

    x can be a single row (shape (n,)) or a batch (shape (m, n)).
    """
    diff = (x - mu) / sigma
    per_feature_log_pdf = -0.5 * diff**2 - np.log(sigma * math.sqrt(2 * math.pi))
    return per_feature_log_pdf.sum(axis=-1)


def predict(x: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (log_density, is_anomaly) for one row or a batch of rows,
    in the feature order given by params["feature_order"].
    """
    mu = np.array(params["mu"])
    sigma = np.array(params["sigma"])
    n_features = len(params["feature_order"])
    log_epsilon = n_features * math.log(params["alpha_opt"])

    ld = log_density(np.asarray(x, dtype=float), mu, sigma)
    is_anomaly = (ld < log_epsilon).astype(int)
    return ld, is_anomaly


def predict_from_dataframe(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    df must contain the columns in params["feature_order"] (extra columns
    are ignored). Returns df with two new columns: log_density, is_fraud.
    """
    x = df[params["feature_order"]].values
    ld, is_anomaly = predict(x, params)
    out = df.copy()
    out["log_density"] = ld
    out["is_fraud"] = is_anomaly
    return out


def main():
    parser = argparse.ArgumentParser(description="Score credit card transactions for anomalies.")
    parser.add_argument(
        "--values", nargs="+", type=float,
        help="Single transaction as space-separated values, in order: " + ", ".join(load_params()["feature_order"]),
    )
    parser.add_argument("--csv", type=str, help="Path to a CSV containing the required feature columns.")
    parser.add_argument("--out", type=str, help="Where to write predictions when using --csv (defaults to stdout).")
    args = parser.parse_args()

    params = load_params()

    if args.values:
        if len(args.values) != len(params["feature_order"]):
            raise SystemExit(f"Expected {len(params['feature_order'])} values in order "
                              f"{params['feature_order']}, got {len(args.values)}.")
        ld, is_anomaly = predict(np.array(args.values), params)
        verdict = "FRAUD" if is_anomaly else "AUTHENTIC"
        print(f"log_density = {ld:.4f}   verdict = {verdict}")

    elif args.csv:
        df = pd.read_csv(args.csv)
        result = predict_from_dataframe(df, params)
        if args.out:
            result.to_csv(args.out, index=False)
            print(f"Wrote {len(result)} predictions to {args.out}")
        else:
            print(result.to_string(index=False))

    else:
        parser.error("Provide either --values or --csv.")


if __name__ == "__main__":
    main()

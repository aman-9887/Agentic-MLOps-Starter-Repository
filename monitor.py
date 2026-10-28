"""Calculate PSI and KS drift metrics and emit a machine-readable alert."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PSI_THRESHOLD = float(os.getenv("PSI_THRESHOLD", "0.2"))
KS_THRESHOLD = float(os.getenv("KS_THRESHOLD", "0.1"))


def calculate_psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """Calculate population stability index using baseline quantile bins."""
    expected_values = pd.to_numeric(expected, errors="coerce").dropna().to_numpy()
    actual_values = pd.to_numeric(actual, errors="coerce").dropna().to_numpy()
    if not len(expected_values) or not len(actual_values):
        raise ValueError("PSI requires non-empty numeric baseline and inference data.")

    # Quantile edges can repeat when the baseline has many tied values.
    edges = np.unique(np.quantile(expected_values, np.linspace(0, 1, buckets + 1)))
    if len(edges) < 2:
        return 0.0 if np.allclose(expected_values.mean(), actual_values.mean()) else float("inf")
    edges[0] = -np.inf
    edges[-1] = np.inf
    expected_counts, _ = np.histogram(expected_values, bins=edges)
    actual_counts, _ = np.histogram(actual_values, bins=edges)
    expected_pct = np.maximum(expected_counts / expected_counts.sum(), 1e-6)
    actual_pct = np.maximum(actual_counts / actual_counts.sum(), 1e-6)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def calculate_ks(expected: pd.Series, actual: pd.Series) -> float:
    """Return the two-sample Kolmogorov–Smirnov statistic."""
    expected_values = pd.to_numeric(expected, errors="coerce").dropna().to_numpy()
    actual_values = pd.to_numeric(actual, errors="coerce").dropna().to_numpy()
    if not len(expected_values) or not len(actual_values):
        raise ValueError("KS requires non-empty numeric baseline and inference data.")
    return float(ks_2samp(expected_values, actual_values).statistic)


def detect_drift(
    baseline_path: str | Path,
    current_path: str | Path,
    psi_threshold: float = PSI_THRESHOLD,
    ks_threshold: float = KS_THRESHOLD,
    feature: str = "trade_value",
) -> dict[str, Any]:
    """Compare a baseline and current batch and return metrics plus alert status."""
    baseline_df = pd.read_csv(baseline_path)
    current_df = pd.read_csv(current_path)
    if feature not in baseline_df or feature not in current_df:
        raise ValueError(f"Both files must contain the '{feature}' column.")
    psi_value = calculate_psi(baseline_df[feature], current_df[feature])
    ks_value = calculate_ks(baseline_df[feature], current_df[feature])
    detected = psi_value > psi_threshold or ks_value > ks_threshold
    return {
        "event": "Drift Alert" if detected else "No Drift",
        "drift_detected": detected,
        "feature": feature,
        "psi": psi_value,
        "ks": ks_value,
        "psi_threshold": psi_threshold,
        "ks_threshold": ks_threshold,
    }


def main() -> int:
    """Parse command-line options, print JSON, and expose GitHub Actions outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="data/baseline_data.csv")
    parser.add_argument("--current", default="data/inference_data.csv")
    parser.add_argument("--feature", default="trade_value")
    parser.add_argument("--psi-threshold", type=float, default=PSI_THRESHOLD)
    parser.add_argument("--ks-threshold", type=float, default=KS_THRESHOLD)
    args = parser.parse_args()
    result = detect_drift(
        args.baseline, args.current, args.psi_threshold, args.ks_threshold, args.feature
    )
    print(json.dumps(result, sort_keys=True))
    output_path = os.getenv("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as output:
            output.write(f"drift_detected={str(result['drift_detected']).lower()}\n")
            output.write(f"psi={result['psi']:.8f}\n")
            output.write(f"ks={result['ks']:.8f}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

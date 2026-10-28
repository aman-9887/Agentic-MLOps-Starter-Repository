"""Retrain the investment-agent model from feedback and track it with MLflow."""
from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import time
from pathlib import Path
from typing import Iterable

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = ["agent_id", "tool_name", "status"]
TARGET = "trade_value"
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "investment-agent")
MSE_THRESHOLD = float(os.getenv("MSE_THRESHOLD", "2000"))


def load_feedback_data(feedback_dir: str | Path) -> pd.DataFrame:
    """Read all feedback CSVs in one deterministic pass.

    Args:
        feedback_dir: Directory containing feedback CSV files.

    Returns:
        Combined feedback rows, sorted by filename order.

    Raises:
        FileNotFoundError: If the directory or CSV files are missing.
        ValueError: If no CSV files are found or required columns are absent.
    """
    directory = Path(feedback_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Feedback directory does not exist: {directory}")
    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No feedback CSV files found in {directory}")
    required = set(FEATURES + [TARGET])
    frames = [pd.read_csv(path, usecols=lambda col: col in required) for path in csv_files]
    data = pd.concat(frames, ignore_index=True, copy=False)
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Feedback data is missing required columns: {sorted(missing)}")
    data = data.dropna(subset=list(required))
    if data.empty:
        raise ValueError("Feedback data contains no usable rows after removing missing values.")
    return data


def build_model() -> Pipeline:
    """Build a categorical preprocessing and linear-regression pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocessor), ("regressor", LinearRegression())])


def latest_version(client: MlflowClient, model_name: str, stage: str | None = None):
    """Return the newest registered version, optionally restricted to a stage."""
    try:
        versions = client.get_latest_versions(model_name, stages=[stage] if stage else None)
    except Exception:
        return None
    return max(versions, key=lambda version: int(version.version)) if versions else None


def promote_challenger(model_name: str, candidate_version: str, candidate_mse: float) -> dict:
    """Stage the challenger and promote only when its MSE beats the champion.

    Lower MSE is better. If a Production champion exists and the challenger
    performs worse or equal, the previous Production version remains in place.
    """
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name, version=candidate_version, stage="Staging", archive_existing_versions=False
    )
    challenger = client.get_model_version(model_name, candidate_version)
    client.set_model_version_tag(model_name, candidate_version, "role", "Challenger")
    client.set_model_version_tag(model_name, candidate_version, "mse", str(candidate_mse))

    if candidate_mse >= MSE_THRESHOLD:
        return {"decision": "rejected_quality_gate", "challenger_version": candidate_version,
                "champion_mse": None, "challenger_mse": candidate_mse,
                "required_mse": f"< {MSE_THRESHOLD}"}

    champion = latest_version(client, model_name, "Production")
    champion_mse = None
    if champion is not None:
        try:
            champion_mse = float(champion.tags.get("mse", "nan"))
        except (TypeError, ValueError):
            champion_mse = float("nan")

    if champion is None or (champion_mse == champion_mse and candidate_mse < champion_mse):
        client.transition_model_version_stage(
            name=model_name, version=candidate_version, stage="Production",
            archive_existing_versions=True,
        )
        client.set_model_version_tag(model_name, candidate_version, "role", "Champion")
        return {"decision": "promoted", "challenger_version": candidate_version,
                "previous_champion": None if champion is None else champion.version,
                "champion_mse": champion_mse, "challenger_mse": candidate_mse}
    # The existing Production champion remains active; the candidate stays in Staging.
    return {"decision": "rollback_to_previous_champion", "challenger_version": candidate_version,
            "previous_champion": champion.version, "champion_mse": champion_mse,
            "challenger_mse": candidate_mse}


def run_training(feedback_dir: str | Path, profile_path: str | Path | None = None) -> dict:
    """Train, evaluate, log metrics/artifacts, and apply registry promotion rules."""
    data = load_feedback_data(feedback_dir)
    X = data[FEATURES]
    y = pd.to_numeric(data[TARGET], errors="raise")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = build_model()
    mlflow.set_experiment("agentic-mlops-retraining")
    start = time.perf_counter()
    profiler = cProfile.Profile()
    profiler.enable()
    with mlflow.start_run() as run:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = float(mean_squared_error(y_test, predictions))
        elapsed = time.perf_counter() - start
        profiler.disable()
        mlflow.log_params({
            "model": "LinearRegression", "test_size": 0.2, "random_state": 42,
            "feedback_rows": len(data), "feedback_files": len(list(Path(feedback_dir).glob("*.csv"))),
        })
        logged_metrics = {
            "mse": mse, "train_rows": len(X_train), "test_rows": len(X_test),
            "retraining_seconds": elapsed, "mse_gate_pass": int(mse < MSE_THRESHOLD),
        }
        for env_name, metric_name in (("DRIFT_PSI", "drift_psi"), ("DRIFT_KS", "drift_ks")):
            metric_value = os.getenv(env_name)
            if metric_value is not None:
                logged_metrics[metric_name] = float(metric_value)
        mlflow.log_metrics(logged_metrics)
        mlflow.set_tags({"role": "Challenger", "pipeline": "agentic-mlops"})
        model_info = mlflow.sklearn.log_model(
            sk_model=model, artifact_path="model", registered_model_name=MODEL_NAME
        )
        version = getattr(model_info, "registered_model_version", None)
        if version is None:
            client = MlflowClient()
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            version = max(versions, key=lambda item: int(item.version)).version
        if profile_path:
            profiler.disable() if profiler.enabled else None
            stream = io.StringIO()
            pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(20)
            Path(profile_path).write_text(stream.getvalue(), encoding="utf-8")
            mlflow.log_artifact(str(profile_path), artifact_path="profiling")
        promotion = promote_challenger(MODEL_NAME, str(version), mse)
        mlflow.log_param("promotion_decision", promotion["decision"])
        result = {"run_id": run.info.run_id, "mse": mse, "seconds": elapsed,
                  "model_version": str(version), "promotion": promotion}
        if mse >= MSE_THRESHOLD:
            raise RuntimeError(
                f"MSE gate failed: {mse:.3f} must be < {MSE_THRESHOLD:.3f}. "
                f"Candidate remains in Staging; Production is unchanged."
            )
        return result


def main() -> int:
    """Run retraining from CLI and print a machine-readable result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback-dir", default="data/feedback")
    parser.add_argument("--profile-out", default="retrain.prof.txt")
    args = parser.parse_args()
    result = run_training(args.feedback_dir, args.profile_out)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

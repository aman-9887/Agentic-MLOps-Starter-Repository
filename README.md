# Agentic-MLOps-Starter — Production Monitoring and Retraining

This implementation computes population stability index (PSI) and two-sample
Kolmogorov–Smirnov (KS) statistics on `trade_value`, emits JSON, and conditionally
runs retraining when either configured drift threshold is exceeded.

## Requirements and setup

Python 3.11 is recommended. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For local monitoring, files are expected at `data/baseline_data.csv` and
`data/inference_data.csv`; feedback CSVs are read from `data/feedback/`.

## Thresholds

- PSI threshold: `0.2`; drift if `PSI > 0.2`.
- KS statistic threshold: `0.1`; drift if `KS > 0.1`.
- A drift alert is emitted if **either** condition is true.
- Retraining quality gate: MSE must be `< 2000`.
- Operational targets: monitor under 10 seconds and retraining under 30 seconds.

Thresholds can be overridden with `PSI_THRESHOLD` and `KS_THRESHOLD`. CLI flags
`--psi-threshold` and `--ks-threshold` override environment defaults.

## Manual commands

```bash
python monitor.py
python monitor.py --baseline data/baseline_data.csv --current data/inference_data.csv
python retrain.py --feedback-dir data/feedback --profile-out retrain.prof.txt
python -m cProfile -o retrain.prof python retrain.py --feedback-dir data/feedback
python -m pstats retrain.prof
```

The retraining command needs an MLflow tracking server with Model Registry support.
Set `MLFLOW_TRACKING_URI` to the server URI before running. GitHub Actions reads it
from the repository secret of the same name. Never commit credentials or tokens.

## GitHub Actions

The workflow is `.github/workflows/retrain.yml`. It runs on push/pull request to
`main`, scheduled hourly, or manual dispatch. The `monitor` job exports
`drift_detected`, `psi`, and `ks` as job outputs. The `retrain` job runs only when
`drift_detected == 'true'`, loads all CSVs from `data/feedback/`, logs model
parameters/metrics/artifacts to MLflow, and enforces the MSE gate.

To configure the remote registry, add repository secret `MLFLOW_TRACKING_URI`.
The GitHub runner must be able to reach that server; use a secure network endpoint
and configure authentication as appropriate for your MLflow deployment.

## Interpreting MLflow runs

Experiment: `agentic-mlops-retraining`.
- Parameters: model type, feedback row count, test split, random seed.
- Metrics: `mse`, train/test row counts, `retraining_seconds`, `mse_gate_pass`.
- Tags: `role=Challenger` before promotion and `role=Champion` after promotion.
- Artifacts: fitted scikit-learn pipeline and profiling report (when enabled).
- Model Registry: candidate is first assigned to `Staging`. If candidate MSE is
  lower than the Production champion's stored `mse` tag, it is promoted to
  `Production`; otherwise the current Production version remains active and the
  challenger stays in Staging.

Lower MSE is better. The first registered model can become Production because
there is no prior champion to compare against. Ensure the registry has a stable
Production version with an `mse` tag for subsequent comparisons.

## Security and dependency audit

The pinned dependency list includes `requests==2.19.0`, an old release. Run
`python -m pip install safety && safety scan` in the project environment and
save the dated output. This repo package list is not a substitute for an actual
vulnerability scan: review all findings, update vulnerable dependencies in a
controlled change, then rerun tests and the scan. Avoid claiming a clean audit
without retaining the scanner output.

## Reproducibility and limitations

The data files are supplied examples. Results depend on their contents and the
configured MLflow server. The GitHub Actions job deliberately fails if the
MLflow URI secret is missing. MLflow model-stage APIs are supported by the
pinned MLflow 2.x dependency; review migration guidance before upgrading to
newer MLflow releases.

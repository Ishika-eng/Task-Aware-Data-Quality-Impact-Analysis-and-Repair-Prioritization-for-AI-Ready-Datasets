"""
Training provenance log.

Random Forest doesn't train in "epochs" (that's a neural-network concept --
RF builds a fixed number of independent trees in one fit() call, with no
iterative passes over the data to log per-epoch loss). What IS meaningful
and loggable here: exactly when each model was trained, how long it took,
on what data, with which hyperparameters and seed. This module appends one
row per model fit to results/training_log.csv, so the project has a
concrete, timestamped record of every training run performed -- not just
the final aggregated metrics.
"""

import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "training_log.csv")

_COLUMNS = ["timestamp_utc", "phase", "context", "model", "n_estimators", "random_state",
            "n_rows", "n_features", "duration_seconds", "notes"]


@contextmanager
def log_training(phase: str, context: str, model, X, seed: int, notes: str = ""):
    """Wrap a pipeline.fit(X, y) call to log it automatically:

        with log_training("phase9", "label_errors rate=10% seed=42", pipeline, X_dirty, seed=42):
            pipeline.fit(X_dirty, y_dirty)
    """
    start = time.monotonic()
    yield
    duration = time.monotonic() - start

    clf = model.named_steps.get("clf") or model.named_steps.get("reg") if hasattr(model, "named_steps") else model
    n_estimators = getattr(clf, "n_estimators", None)
    model_name = type(clf).__name__ if clf is not None else type(model).__name__

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "phase": phase, "context": context, "model": model_name,
        "n_estimators": n_estimators, "random_state": seed,
        "n_rows": len(X), "n_features": X.shape[1] if hasattr(X, "shape") else None,
        "duration_seconds": round(duration, 3), "notes": notes,
    }
    _append_row(row)


def _append_row(row: dict):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    df = pd.DataFrame([row], columns=_COLUMNS)
    write_header = not os.path.exists(LOG_PATH)
    df.to_csv(LOG_PATH, mode="a", header=write_header, index=False)


if __name__ == "__main__":
    if os.path.exists(LOG_PATH):
        log = pd.read_csv(LOG_PATH)
        print(f"{len(log)} logged training runs across {log['phase'].nunique()} phase(s)")
        print(f"Total logged training time: {log['duration_seconds'].sum():.1f}s")
        print(log.groupby("phase")["duration_seconds"].agg(["count", "sum"]).round(1))
    else:
        print("No training log yet -- run a phase script that uses log_training() first.")

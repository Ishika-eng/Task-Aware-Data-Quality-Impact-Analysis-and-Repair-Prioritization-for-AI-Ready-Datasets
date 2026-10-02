"""
Clean-data baseline models (Phase 2).

We standardize on one preprocessing pipeline shared by both models so
comparisons are fair: impute (median/most-frequent) then OneHotEncoder for
categoricals (handle_unknown="ignore" so a category seen only in val/test
doesn't crash the pipeline), StandardScaler for numerics. Trees don't
strictly need scaling, but sharing one pipeline keeps the two models'
inputs identical, which matters once we're comparing how each model reacts
to injected errors later.

The imputers are a no-op on Phase 2's clean data (nothing to impute, so
Phase 2's recorded baseline numbers are unaffected) but are required from
Phase 7 onward, where this same preprocessor gets reused on data that may
contain injected missing values -- RandomForestClassifier (unlike
HistGradientBoostingClassifier) can't accept NaN natively.

Two models, matching the brief:
- LogisticRegression: a simple linear baseline.
- RandomForestClassifier: a stronger nonlinear baseline, and the model
  we'll carry forward for the rest of the project's experiments.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import column_types

POS_LABEL = ">50K"  # the minority / "interesting" class for precision/recall/AUC


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols, numeric_cols = column_types(X)
    return ColumnTransformer([
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols),
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_cols),
    ])


def build_models(preprocessor: ColumnTransformer, seed: int = 42) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]),
        "random_forest": Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)),
        ]),
    }


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    preds = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)
    pos_idx = list(pipeline.classes_).index(POS_LABEL)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, pos_label=POS_LABEL),
        "recall": recall_score(y_test, preds, pos_label=POS_LABEL),
        "f1": f1_score(y_test, preds, pos_label=POS_LABEL),
        "roc_auc": roc_auc_score((y_test == POS_LABEL).astype(int), proba[:, pos_idx]),
    }


def run_baseline(X_train, y_train, X_test, y_test, seed: int = 42) -> pd.DataFrame:
    from training_log import log_training  # local import: avoids a cycle at module load time

    preprocessor = build_preprocessor(X_train)
    models = build_models(preprocessor, seed=seed)
    rows = []
    for name, pipeline in models.items():
        with log_training("phase2", f"clean_baseline model={name}", pipeline, X_train, seed=seed):
            pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        rows.append({"model": name, **metrics})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import os

    from data import load_adult_clean, three_way_split

    print("Loading UCI Adult (native missing rows dropped for a clean baseline)...")
    X, y = load_adult_clean()
    print(f"Clean dataset: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Class balance: {y.value_counts(normalize=True).round(3).to_dict()}")

    X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(X, y)
    print(f"Split -> train: {len(X_train)} | val: {len(X_val)} | test: {len(X_test)}")

    print("\nTraining clean baseline models (evaluated on held-out TEST split)...")
    results = run_baseline(X_train, y_train, X_test, y_test)
    print("\n" + "=" * 70)
    print("CLEAN BASELINE RESULTS")
    print("=" * 70)
    print(results.round(4).to_string(index=False))

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "phase2_clean_baseline.csv")
    results.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")

"""Shared setup for the syllabus companion notebooks (unit1 ... unit6).

These notebooks are additive: they read data/adult_clean.csv and, where it helps, import
functions from src/ (data.py, baseline.py, detect.py) read-only. Nothing in src/, results/
or app.py is modified, so the frozen research phases are untouched.
"""
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))          # lets notebooks reuse project code read-only
warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42                                      # same split seed as the research phases
NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
TARGET = "class"

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 90, "savefig.dpi": 90, "figure.autolayout": True})


def load_adult(categorical: bool = True) -> pd.DataFrame:
    """UCI Adult with natively-missing rows already dropped (45,222 rows), as used by the project.

    With categorical=True the text columns become the `category` dtype, which is the dtype the
    project's detectors were built on (pandas 3 would otherwise load them as `str`).
    """
    df = pd.read_csv(ROOT / "data" / "adult_clean.csv")
    if categorical:
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype("category")
    return df


def fmt_p(p: float) -> str:
    """A p-value below float precision prints as 0.0; say what that really means."""
    return "p < 1e-300" if p < 1e-300 else f"p = {p:.2e}"

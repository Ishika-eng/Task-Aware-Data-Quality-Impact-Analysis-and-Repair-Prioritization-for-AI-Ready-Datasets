"""
Model training and evaluation.

We deliberately use HistGradientBoostingClassifier because it natively
supports NaN values in features. That matters for this project: if we used
a model that can't handle NaN, we'd be forced to impute missing values
before training, which would erase the very damage we're trying to measure.
Using a NaN-tolerant model lets "missing values" actually behave like a
data-quality problem that the model feels, not a preprocessing footnote.
"""

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def train_and_evaluate(X, y, test_X, test_y, seed: int = 0) -> float:
    """Train on (X, y), evaluate on a held-out (test_X, test_y), return macro F1.

    The test set is always the CLEAN held-out split, for every experiment.
    That's important: we're asking "if I train on dirty data, how well does
    the model do on real, clean, unseen data?" -- not "how well does it do
    on more dirty data?". Keeping the test set fixed and clean is what makes
    F1 scores comparable across error types.
    """
    model = HistGradientBoostingClassifier(random_state=seed)
    model.fit(X, y)
    preds = model.predict(test_X)
    return f1_score(test_y, preds, average="macro")


def make_clean_split(X, y, test_size: float = 0.25, seed: int = 0):
    """One fixed train/test split used as the basis for every experiment."""
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

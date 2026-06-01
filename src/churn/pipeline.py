"""Preprocessing + model pipeline for churn prediction (scikit-learn)."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC


def build_pipeline(random_state: int = 42, **model_kwargs) -> Pipeline:
    """A full preprocessing + classifier pipeline.

    Numeric features are imputed (median) and scaled; categoricals are imputed
    (most frequent) and one-hot encoded. The classifier is a histogram-based
    gradient-boosting model.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
    ])
    params = dict(learning_rate=0.05, max_iter=400, max_depth=None,
                  l2_regularization=1.0, random_state=random_state)
    params.update(model_kwargs)
    clf = HistGradientBoostingClassifier(**params)
    return Pipeline([("pre", pre), ("clf", clf)])

from churn.data import make_synthetic, FEATURES, TARGET
from churn.pipeline import build_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline


def test_pipeline_fits_and_beats_random():
    df = make_synthetic(n=3000, seed=1)
    X, y = df[FEATURES], (df[TARGET] == "Yes").astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)
    pipe = build_pipeline(max_iter=120)
    pipe.fit(Xtr, ytr)
    auc = roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])
    assert auc > 0.65  # clearly better than random


def test_cv_is_leakage_free_preprocessing_inside_pipeline():
    """The estimator handed to cross_val_score must be a Pipeline whose first
    step is the ColumnTransformer (imputation/scaling/OHE). That guarantees the
    preprocessing is refit *inside each CV fold* rather than once on all data,
    so cross_val_score gives an honest, leakage-free estimate.
    """
    pipe = build_pipeline()
    assert isinstance(pipe, Pipeline)
    first_name, first_step = pipe.steps[0]
    # the transformers must live INSIDE the pipeline, before the classifier
    assert isinstance(first_step, ColumnTransformer), first_step
    assert pipe.steps[-1][0] == "clf"
    # the ColumnTransformer carries imputation + scaling + one-hot encoding
    inner = {name: trans for name, trans, _ in first_step.transformers}
    assert "num" in inner and "cat" in inner
    num_steps = [s for s, _ in inner["num"].steps]
    cat_steps = [s for s, _ in inner["cat"].steps]
    assert "impute" in num_steps and "scale" in num_steps
    assert "impute" in cat_steps and "onehot" in cat_steps

    # smoke: cross_val_score actually runs on the Pipeline (refits per fold)
    df = make_synthetic(n=800, seed=2)
    X, y = df[FEATURES], (df[TARGET] == "Yes").astype(int)
    scores = cross_val_score(build_pipeline(max_iter=60), X, y, cv=3, scoring="roc_auc")
    assert len(scores) == 3 and scores.mean() > 0.6

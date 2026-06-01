from churn.data import make_synthetic, FEATURES, TARGET
from churn.pipeline import build_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def test_pipeline_fits_and_beats_random():
    df = make_synthetic(n=3000, seed=1)
    X, y = df[FEATURES], (df[TARGET] == "Yes").astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)
    pipe = build_pipeline(max_iter=120)
    pipe.fit(Xtr, ytr)
    auc = roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])
    assert auc > 0.65  # clearly better than random

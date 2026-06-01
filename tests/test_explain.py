from churn.data import make_synthetic, FEATURES, TARGET
from churn.pipeline import build_pipeline
from churn.explain import permutation_importances
from sklearn.model_selection import train_test_split

def test_permutation_importance_runs_and_ranks():
    df = make_synthetic(n=1500, seed=3)
    X, y = df[FEATURES], (df[TARGET] == "Yes").astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=3, stratify=y)
    pipe = build_pipeline(max_iter=80).fit(Xtr, ytr)
    rows = permutation_importances(pipe, Xte, yte, n_repeats=3)
    assert len(rows) == len(FEATURES)
    # Contract is the dominant churn driver in the synthetic generator
    names = [r[0] for r in rows]
    assert "Contract" in names[:5]

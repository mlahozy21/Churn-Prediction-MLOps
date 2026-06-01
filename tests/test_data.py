from churn.data import make_synthetic, FEATURES, TARGET

def test_synthetic_schema_and_signal():
    df = make_synthetic(n=2000, seed=0)
    assert len(df) == 2000
    for col in FEATURES + [TARGET]:
        assert col in df.columns
    # target is binary Yes/No and not degenerate
    assert set(df[TARGET].unique()) <= {"Yes", "No"}
    rate = (df[TARGET] == "Yes").mean()
    assert 0.1 < rate < 0.6
    # month-to-month should churn more than two-year (built-in signal)
    m2m = (df[df.Contract == "Month-to-month"][TARGET] == "Yes").mean()
    two = (df[df.Contract == "Two year"][TARGET] == "Yes").mean()
    assert m2m > two

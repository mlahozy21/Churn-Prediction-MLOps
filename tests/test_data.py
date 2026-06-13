import numpy as np
import pandas as pd

from churn.data import make_synthetic, load_data, FEATURES, TARGET


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


def test_total_charges_is_not_near_deterministic():
    """TotalCharges must be correlated with tenure*MonthlyCharges but NOT an
    almost-exact identity of it (otherwise permutation importance is an
    artifact). We check the residual around the ideal product is non-trivial and
    that a few values are missing (NaN), as in the real Telco CSV.
    """
    df = make_synthetic(n=4000, seed=0)
    tc = df["TotalCharges"].to_numpy()
    # some genuinely-missing values exist (new customers)
    assert np.isnan(tc).any()
    mask = ~np.isnan(tc)
    ideal = (df["tenure"] * df["MonthlyCharges"]).to_numpy()[mask]
    obs = tc[mask]
    # ratio obs/ideal should have real spread, not ~1.0 +/- tiny
    nz = ideal > 0
    ratio = obs[nz] / ideal[nz]
    assert ratio.std() > 0.03, ratio.std()
    # correlation stays strong (still a meaningful feature)
    assert np.corrcoef(obs, ideal)[0, 1] > 0.9


def test_load_data_csv_coerces_totalcharges(tmp_path):
    """The CSV path must coerce TotalCharges to numeric and drop blank rows
    (the real Telco CSV stores blanks as ' ' strings)."""
    csv = tmp_path / "telco.csv"
    df = make_synthetic(n=20, seed=3)
    df = df.dropna(subset=["TotalCharges"]).copy()
    # inject a real-Telco-style blank string in one row
    df.iloc[0, df.columns.get_loc("TotalCharges")] = " "
    df.to_csv(csv, index=False)
    loaded = load_data(csv_path=str(csv))
    assert pd.api.types.is_numeric_dtype(loaded["TotalCharges"])
    assert loaded["TotalCharges"].notna().all()
    assert len(loaded) == len(df) - 1  # the blank row was dropped

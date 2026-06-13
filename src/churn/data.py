"""Data loading and a synthetic generator with the Telco Customer Churn schema.

If a real CSV (e.g. the IBM Telco Customer Churn dataset) is available it is
used; otherwise a synthetic dataset with the same columns is generated so the
pipeline is fully runnable and testable without external downloads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "Churn"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC + CATEGORICAL


def make_synthetic(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a Telco-like churn dataset with a realistic signal."""
    rng = np.random.default_rng(seed)

    def pick(options, p=None):
        return rng.choice(options, size=n, p=p)

    contract = pick(["Month-to-month", "One year", "Two year"], [0.55, 0.25, 0.20])
    tenure = np.clip(rng.gamma(2.0, 12.0, n).astype(int), 0, 72)
    monthly = np.round(rng.normal(65, 30, n).clip(18, 120), 2)
    internet = pick(["DSL", "Fiber optic", "No"], [0.34, 0.44, 0.22])
    df = pd.DataFrame({
        "gender": pick(["Male", "Female"]),
        "SeniorCitizen": pick([0, 1], [0.84, 0.16]),
        "Partner": pick(["Yes", "No"]),
        "Dependents": pick(["Yes", "No"], [0.3, 0.7]),
        "tenure": tenure,
        "PhoneService": pick(["Yes", "No"], [0.9, 0.1]),
        "MultipleLines": pick(["Yes", "No", "No phone service"], [0.42, 0.48, 0.10]),
        "InternetService": internet,
        "OnlineSecurity": pick(["Yes", "No", "No internet service"], [0.29, 0.49, 0.22]),
        "OnlineBackup": pick(["Yes", "No", "No internet service"], [0.34, 0.44, 0.22]),
        "DeviceProtection": pick(["Yes", "No", "No internet service"], [0.34, 0.44, 0.22]),
        "TechSupport": pick(["Yes", "No", "No internet service"], [0.29, 0.49, 0.22]),
        "StreamingTV": pick(["Yes", "No", "No internet service"], [0.38, 0.40, 0.22]),
        "StreamingMovies": pick(["Yes", "No", "No internet service"], [0.39, 0.39, 0.22]),
        "Contract": contract,
        "PaperlessBilling": pick(["Yes", "No"], [0.59, 0.41]),
        "PaymentMethod": pick(["Electronic check", "Mailed check",
                               "Bank transfer (automatic)", "Credit card (automatic)"]),
        "MonthlyCharges": monthly,
    })

    # TotalCharges is *roughly* tenure * monthly, but in the real Telco data it
    # is not a deterministic function of the two: plans change, there are
    # promotions/credits and one-off fees, and a handful of brand-new customers
    # have a blank/NaN value. We mirror that with (a) a multiplicative
    # billing-variation factor, (b) additive month-to-month noise that grows with
    # tenure, and (c) an occasional one-off fee. This keeps TotalCharges
    # correlated with -- but not a near-identity of -- tenure and MonthlyCharges,
    # so permutation importance is not measuring an artifact.
    tenure_arr = df["tenure"].to_numpy()
    monthly_arr = df["MonthlyCharges"].to_numpy()
    billing_var = rng.uniform(0.9, 1.05, n)
    additive_noise = (rng.normal(0.0, 1.0, n) * monthly_arr
                      * np.sqrt(np.maximum(tenure_arr, 1)) * 0.15)
    one_off_fee = rng.gamma(1.5, 12.0, n) * (rng.uniform(size=n) < 0.5)
    total = tenure_arr * monthly_arr * billing_var + additive_noise + one_off_fee
    total = np.round(np.clip(total, 0.0, None), 2).astype(float)
    # Brand-new customers (tenure == 0) have not been billed yet, so in the real
    # IBM Telco CSV their TotalCharges cell is blank -> NaN. We reproduce that
    # exactly (plus a tiny fraction of other blanks). The pipeline's median
    # imputer handles these missing values.
    missing = (tenure_arr == 0) | (rng.uniform(size=n) < 0.002)
    total[missing] = np.nan
    df["TotalCharges"] = total

    # churn log-odds: month-to-month, short tenure, high charges, fiber, e-check increase risk
    z = (-1.0
         + 1.4 * (df["Contract"] == "Month-to-month")
         - 0.9 * (df["Contract"] == "Two year")
         - 0.035 * df["tenure"]
         + 0.012 * (df["MonthlyCharges"] - 65)
         + 0.5 * (df["InternetService"] == "Fiber optic")
         + 0.4 * (df["PaymentMethod"] == "Electronic check")
         - 0.3 * (df["TechSupport"] == "Yes")
         + 0.2 * (df["SeniorCitizen"] == 1))
    p = 1 / (1 + np.exp(-z))
    df[TARGET] = np.where(rng.uniform(size=n) < p, "Yes", "No")
    return df


def load_data(csv_path: str | None = None, n_synthetic: int = 5000) -> pd.DataFrame:
    """Load the real CSV if given, else generate a synthetic dataset."""
    if csv_path:
        df = pd.read_csv(csv_path)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df = df.dropna(subset=["TotalCharges"])
        return df
    return make_synthetic(n_synthetic)

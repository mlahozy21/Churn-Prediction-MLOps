import joblib, os
from fastapi.testclient import TestClient
from churn.data import make_synthetic, FEATURES, TARGET
from churn.pipeline import build_pipeline

def _train_tmp_model(path):
    df = make_synthetic(n=1500, seed=2)
    pipe = build_pipeline(max_iter=80)
    pipe.fit(df[FEATURES], (df[TARGET] == "Yes").astype(int))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipe, path)

def test_health_and_predict(tmp_path):
    model_path = str(tmp_path / "m.joblib")
    _train_tmp_model(model_path)
    os.environ["MODEL_PATH"] = model_path
    import importlib, churn.api as api
    importlib.reload(api)
    c = TestClient(api.app)
    assert c.get("/health").status_code == 200
    r = c.post("/predict", json={"Contract": "Month-to-month", "tenure": 1,
                                 "MonthlyCharges": 95.0, "InternetService": "Fiber optic",
                                 "PaymentMethod": "Electronic check"})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["churn"], bool)

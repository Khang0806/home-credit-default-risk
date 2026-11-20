from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)
BASE = {"annual_income": 180000, "credit_amount": 500000, "annuity": 25000, "age_years": 35,
        "years_employed": 5, "ext_source_2": 0.6, "ext_source_3": 0.5}


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_predict_shape():
    r = client.post("/predict", json=BASE)
    assert r.status_code == 200
    body = r.json()
    assert 0 < body["default_probability"] < 1
    assert body["risk_band"] in {"low", "medium", "high"}
    assert len(body["top_factors"]) == 5


def test_low_external_scores_mean_higher_risk():
    good = client.post("/predict", json={**BASE, "ext_source_2": 0.8, "ext_source_3": 0.8}).json()
    bad = client.post("/predict", json={**BASE, "ext_source_2": 0.1, "ext_source_3": 0.1}).json()
    assert bad["default_probability"] > good["default_probability"]


def test_pensioner_and_missing_fields_ok():
    r = client.post("/predict", json={**BASE, "years_employed": None, "ext_source_2": None})
    assert r.status_code == 200


def test_rejects_bad_input():
    assert client.post("/predict", json={**BASE, "ext_source_2": 1.5}).status_code == 422

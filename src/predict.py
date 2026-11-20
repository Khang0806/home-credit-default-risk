"""Load the saved lite model once and score applicants (used by the API and the Streamlit demo)."""
import json
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb

from src.features import applicant_to_raw, build_features

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
# Illustrative cut-offs around the ~8% training default rate - a real lender sets these from cost/approval targets.
BANDS = [(0.05, "low"), (0.15, "medium"), (1.01, "high")]


@lru_cache
def load():
    booster = lgb.Booster(model_file=str(MODEL_DIR / "lite_model.txt"))
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    return booster, meta


def predict(applicant: dict, top_k: int = 5) -> dict:
    booster, meta = load()
    X = build_features(applicant_to_raw(applicant))[meta["features"]]
    p = float(booster.predict(X)[0])
    contrib = booster.predict(X, pred_contrib=True)[0][:-1]   # per-feature SHAP, log-odds
    top = sorted(zip(meta["features"], contrib, X.iloc[0]), key=lambda t: -abs(t[1]))[:top_k]
    return {
        "default_probability": round(p, 4),
        "risk_band": next(name for cut, name in BANDS if p < cut),
        "top_factors": [
            {"feature": f, "value": None if v != v else (v if isinstance(v, str) else float(v)),
             "shap": round(float(s), 4), "effect": "increases risk" if s > 0 else "decreases risk"}
            for f, s, v in top
        ],
        "model": {"oof_auc": meta["cv"]["oof_auc"], "trained_at": meta["trained_at"]},
    }

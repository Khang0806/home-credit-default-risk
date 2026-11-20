# Home Credit Default Risk — LightGBM + deployable API

**Live demo:** https://khang-home-credit.streamlit.app

Predicting loan default probability using the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) dataset (Kaggle),
then serving a form-friendly version of the model as a FastAPI service with per-prediction SHAP explanations.

## Results
| Model | Data used | Features | 5-fold OOF AUC |
|---|---|---|---|
| Full model (`07_HC_Final`) | application + bureau + previous loans + card / POS / instalments | 584 | **0.797** (private LB 0.796) |
| Application table only | every column of `application_train` | 85 | 0.769 |
| **Lite model (deployed)** | 21 fields an applicant can fill in on a form | 67 | **0.765** |

> Full-model submission was made after the competition closed (2018), so it is not officially ranked.

**Takeaways**
- Credit-history tables (bureau, previous applications, payment behaviour) are worth ~+0.03 AUC over the application form alone.
- Cutting from all 85 application columns to 21 form fields costs only 0.004 AUC, so the form-based model keeps almost all of the application-level signal.
- External credit scores (`EXT_SOURCE_*`) dominate both models; in the full model, pruning the 54 zero-gain features left AUC unchanged (0.79722 → 0.79738).

## Deployed model: design decisions
- **One feature module for training and serving.** `src/features.py` is the notebook-01 logic as functions; `train_lite.py` and the API both call `build_features()`, so there is no training–serving skew. A check confirms single-row (API) and batch predictions are identical.
- **Fairness.** `CODE_GENDER` and `NAME_FAMILY_STATUS` are excluded: sex and marital status are protected grounds under Canadian human-rights law.
- **Explainability.** Every prediction returns its top SHAP factors (LightGBM's exact TreeSHAP via `pred_contrib`).
- **Training.** Same protocol as the full model (stratified 5-fold, per-fold early stopping); the final model is refit on all 307,511 labeled rows with 1.1× the mean best iteration.
- Risk bands (low < 5% ≤ medium < 15% ≤ high) are illustrative; a real lender would set cut-offs from approval-rate and loss targets.

## Run it
```bash
pip install -r requirements.txt

# (optional) retrain - needs the Kaggle CSVs
python train_lite.py --data-dir data --benchmark

# API  -> http://127.0.0.1:8000/docs
uvicorn api.main:app --reload

# Demo UI -> http://localhost:8501
streamlit run app/streamlit_app.py

pytest -q
```

Example request:
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "annual_income": 180000, "credit_amount": 500000, "annuity": 25000,
  "age_years": 35, "years_employed": 5, "years_since_registration": 10, "years_since_id_issued": 8,
  "ext_source_2": 0.6, "ext_source_3": 0.5}'
```
```json
{"default_probability": 0.078, "risk_band": "medium",
 "top_factors": [
   {"feature": "EXT_SOURCE_MEAN", "value": 0.55, "shap": -0.1793, "effect": "decreases risk"},
   {"feature": "AMT_GOODS_PRICE", "value": null, "shap": 0.1538, "effect": "increases risk"}, "..."]}
```
Missing fields are allowed (LightGBM handles NaN natively), but they carry signal: leaving `goods_price` empty raises the risk.

### Deploy (Google Cloud Run)
```bash
gcloud run deploy home-credit-api --source . --region northamerica-northeast2 --allow-unauthenticated
```
The `Dockerfile` ships only `src/`, `api/` and `models/`; Cloud Run sets `$PORT`.

## Project structure
```
src/features.py       feature engineering shared by training and serving
src/predict.py        loads the saved model, returns probability + SHAP factors
train_lite.py         trains and saves models/lite_model.txt + metadata.json
api/main.py           FastAPI service (/predict, /health)
app/streamlit_app.py  demo UI
tests/                API tests
01-07_*.ipynb         research notebooks for the full 584-feature model
```

## Research pipeline (full model)
Run notebooks `01`–`06` first (feature aggregation), then `07` (merge + training).

| Notebook | Purpose |
|---|---|
| `01_application_train_test_final_agg.ipynb` | Main application table features |
| `02_bureau_agg_final.ipynb` | Credit bureau history |
| `03_card_balanced_agg_final.ipynb` | Credit card balance |
| `04_instalments_payment_final_agg.ipynb` | Installment payment behavior |
| `05_POS_balanced_agg_final.ipynb` | POS / cash loan balance |
| `06_previous_application_final_agg.ipynb` | Previous Home Credit applications |
| `07_HC_Final.ipynb` | Merge features, train LightGBM, SHAP, feature selection |

## Data
Data is not included (Kaggle rules). Download with:
```
kaggle competitions download -c home-credit-default-risk
```
Place the CSVs in `data/`.

## Tech
Python, pandas, LightGBM, scikit-learn, SHAP, FastAPI, Streamlit, Docker, pytest

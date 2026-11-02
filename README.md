# Home Credit Default Risk — LightGBM

Predicting loan default probability using the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) dataset (Kaggle).

## Results
| Metric | AUC |
|---|---|
| 5-fold OOF | ~0.797 |
| Private LB | ~0.796 |

> Late submission — made after the competition closed (2018), so not officially ranked.

## Approach
- Aggregated 6 source tables to the applicant level (`SK_ID_CURR`)
- Merged all engineered features into a single training table
- Trained a single LightGBM model with stratified K-fold cross-validation

## Pipeline
Run notebooks `01`–`06` first (feature aggregation), then `07` (merge + training).

| Notebook | Purpose |
|---|---|
| `01_application_train_test_final_agg.ipynb` | Main application table features |
| `02_bureau_agg_final.ipynb` | Credit bureau history |
| `03_card_balanced_agg_final.ipynb` | Credit card balance |
| `04_instalments_payment_final_agg.ipynb` | Installment payment behavior |
| `05_POS_balanced_agg_final.ipynb` | POS / cash loan balance |
| `06_previous_application_final_agg.ipynb` | Previous Home Credit applications |
| `07_HC_Final.ipynb` | Merge features, train LightGBM, generate submission |

## Data
Data is not included (Kaggle rules). Download with:
```
kaggle competitions download -c home-credit-default-risk
```
Place the CSVs in `data/`.

## Tech
Python, pandas, NumPy, LightGBM, scikit-learn

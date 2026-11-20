"""Application-level feature engineering, shared by training and serving.

Ported from 01_application_train_test_final_agg.ipynb (+ the DAYS abs() step in 07),
restricted to the fields a loan applicant can actually fill in on a form.
"""
import numpy as np
import pandas as pd

ANOMALY_DAYS = 365243   # Kaggle's placeholder for "not employed / pensioner"

CATEGORICAL_INPUTS = [
    "NAME_CONTRACT_TYPE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE",
]
NUMERIC_INPUTS = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "OWN_CAR_AGE",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
    "REGION_POPULATION_RELATIVE", "REGION_RATING_CLIENT_W_CITY",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
]
# CODE_GENDER and NAME_FAMILY_STATUS are deliberately left out: sex and marital status
# are protected grounds under Canadian human-rights law and shouldn't drive a credit decision.
RAW_INPUTS = CATEGORICAL_INPUTS + NUMERIC_INPUTS


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Kaggle-format application rows (DAYS_* negative) -> model features."""
    df = raw.reindex(columns=RAW_INPUTS).copy()
    for col in NUMERIC_INPUTS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    # Days: anomaly flag, then NaN out the placeholder
    df["APPLICATION_DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == ANOMALY_DAYS).astype(int)
    for col in ["DAYS_EMPLOYED", "DAYS_BIRTH", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH"]:
        df["APPLICATION_" + col] = df[col].replace(ANOMALY_DAYS, np.nan)
    df = df.drop(columns=["DAYS_EMPLOYED", "DAYS_BIRTH", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH"])

    birth, employed = df["APPLICATION_DAYS_BIRTH"], df["APPLICATION_DAYS_EMPLOYED"]
    income, credit = df["AMT_INCOME_TOTAL"], df["AMT_CREDIT"]
    annuity, goods = df["AMT_ANNUITY"], df["AMT_GOODS_PRICE"]

    df["APPLICATION_EMPLOYED_TO_BIRTH_RATIO"] = employed / (birth + 1)
    df["APPLICATION_MONTHS_TO_PAY"] = credit / (annuity + 1)
    df["APPLICATION_INCOME_TO_CREDIT"] = income / (credit + 1e-9)
    df["APPLICATION_INCOME_TO_ANNUITY"] = income / (annuity + 1e-9)
    df["APPLICATION_INCOME_TO_EMPLOYED"] = income / (employed + 1)
    df["APPLICATION_INCOME_TO_BIRTH"] = income / (birth + 1)
    df["APPLICATION_ANNUITY_TO_CREDIT"] = annuity / (credit + 1e-9)
    df["APPLICATION_CREDIT_TO_EMPLOYED"] = credit / (employed + 1)
    df["APPLICATION_CREDIT_TO_BIRTH"] = credit / (birth + 1)
    df["AGE_LABEL"] = pd.cut(-birth / 365, bins=[0, 27, 40, 50, 65, np.inf], labels=False) + 1

    # External credit scores: aggregates + interactions
    e1, e2, e3 = df["EXT_SOURCE_1"], df["EXT_SOURCE_2"], df["EXT_SOURCE_3"]
    ext = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    df["EXT_SOURCE_MEAN"] = ext.mean(axis=1)
    df["EXT_SOURCE_MIN"] = ext.min(axis=1)
    df["EXT_SOURCE_MAX"] = ext.max(axis=1)
    df["EXT_SOURCE_MEDIAN"] = ext.median(axis=1)
    df["EXT_SOURCE_NAN_COUNT"] = ext.isna().sum(axis=1)
    df["EXT_SOURCE_STD"] = ext.std(axis=1)
    df["EXT_1_2_RATE"] = e1 / (e2 + 1e-9)
    df["EXT_2_3_RATE"] = e2 / (e3 + 1e-9)
    df["EXT_1_3_RATE"] = e1 / (e3 + 1e-9)
    df["EXT_1_2_MUL"] = e1 * e2
    df["EXT_2_3_MUL"] = e2 * e3
    df["EXT_1_3_MUL"] = e1 * e3
    df["EXT_SOURCE_WEIGHT"] = 2 * e1 + 1 * e2 + 3 * e3
    df["EXT_SOURCE_PROD"] = e1 * e2 * e3
    for k, e in (("1", e1), ("2", e2), ("3", e3)):
        df[f"EXT_{k}_CREDIT"] = e * credit
        df[f"EXT_{k}_ANNUITY"] = e * annuity
        df[f"EXT_{k}_INCOME"] = e * income
        df[f"EXT_{k}_GOODS"] = e * goods
        df[f"EXT_{k}_BIRTH"] = e * birth
        df[f"EXT_{k}_EMPLOYED"] = e * employed
        df[f"EXT_{k}_ID"] = e * df["APPLICATION_DAYS_ID_PUBLISH"]

    # Same as 07_HC_Final: days become positive after the interactions are built
    for col in ["APPLICATION_DAYS_BIRTH", "APPLICATION_DAYS_EMPLOYED",
                "APPLICATION_DAYS_REGISTRATION", "APPLICATION_DAYS_ID_PUBLISH"]:
        df[col] = df[col].abs()

    for col in CATEGORICAL_INPUTS:
        df[col] = df[col].astype("category")
    return df.replace([np.inf, -np.inf], np.nan)


def applicant_to_raw(a: dict) -> pd.DataFrame:
    """Human-friendly form fields (years, plain amounts) -> one Kaggle-format row."""
    years_employed = a.get("years_employed")
    row = {
        "NAME_CONTRACT_TYPE": a.get("contract_type"),
        "FLAG_OWN_CAR": "Y" if a.get("owns_car") else "N",
        "FLAG_OWN_REALTY": "Y" if a.get("owns_realty") else "N",
        "NAME_INCOME_TYPE": a.get("income_type"),
        "NAME_EDUCATION_TYPE": a.get("education"),
        "OCCUPATION_TYPE": a.get("occupation"),
        "ORGANIZATION_TYPE": a.get("organization"),
        "AMT_INCOME_TOTAL": a.get("annual_income"),
        "AMT_CREDIT": a.get("credit_amount"),
        "AMT_ANNUITY": a.get("annuity"),
        "AMT_GOODS_PRICE": a.get("goods_price"),
        "OWN_CAR_AGE": a.get("car_age_years"),
        "DAYS_BIRTH": -a["age_years"] * 365.25,
        "DAYS_EMPLOYED": ANOMALY_DAYS if years_employed is None else -years_employed * 365.25,
        "DAYS_REGISTRATION": _neg_days(a.get("years_since_registration")),
        "DAYS_ID_PUBLISH": _neg_days(a.get("years_since_id_issued")),
        "REGION_POPULATION_RELATIVE": a.get("region_population_relative"),
        "REGION_RATING_CLIENT_W_CITY": a.get("region_rating"),
        "EXT_SOURCE_1": a.get("ext_source_1"),
        "EXT_SOURCE_2": a.get("ext_source_2"),
        "EXT_SOURCE_3": a.get("ext_source_3"),
    }
    return pd.DataFrame([row])


def _neg_days(years):
    return None if years is None else -years * 365.25

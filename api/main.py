"""FastAPI service for the lite credit-risk model.

    uvicorn api.main:app --reload      # then open http://127.0.0.1:8000/docs
"""
from typing import Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import load, predict

app = FastAPI(title="Home Credit Default Risk API",
              description="LightGBM trained on Kaggle Home Credit application data. "
                          "Returns default probability plus the top SHAP factors behind it.")


class Applicant(BaseModel):
    contract_type: Literal["Cash loans", "Revolving loans"] = "Cash loans"
    annual_income: float = Field(gt=0, examples=[180000])
    credit_amount: float = Field(gt=0, examples=[500000])
    annuity: float = Field(gt=0, description="Yearly loan payment", examples=[25000])
    goods_price: Optional[float] = Field(None, gt=0, examples=[450000])
    age_years: float = Field(ge=18, le=100, examples=[35])
    years_employed: Optional[float] = Field(None, ge=0, le=60, description="null = not employed / pensioner",
                                            examples=[5])
    years_since_registration: Optional[float] = Field(None, ge=0, examples=[10])
    years_since_id_issued: Optional[float] = Field(None, ge=0, examples=[8])
    owns_car: bool = False
    car_age_years: Optional[float] = Field(None, ge=0)
    owns_realty: bool = True
    income_type: Optional[str] = Field(None, examples=["Working"])
    education: Optional[str] = Field(None, examples=["Higher education"])
    occupation: Optional[str] = Field(None, examples=["Laborers"])
    organization: Optional[str] = Field(None, examples=["Business Entity Type 3"])
    region_rating: Optional[Literal[1, 2, 3]] = Field(None, description="1 = best, 3 = worst")
    region_population_relative: Optional[float] = Field(None, ge=0, le=1)
    ext_source_1: Optional[float] = Field(None, ge=0, le=1, description="External credit score (0-1)")
    ext_source_2: Optional[float] = Field(None, ge=0, le=1, examples=[0.6])
    ext_source_3: Optional[float] = Field(None, ge=0, le=1, examples=[0.5])


@app.get("/health")
def health():
    _, meta = load()
    return {"status": "ok", "oof_auc": meta["cv"]["oof_auc"], "trained_at": meta["trained_at"]}


@app.post("/predict")
def score(applicant: Applicant):
    return predict(applicant.model_dump())

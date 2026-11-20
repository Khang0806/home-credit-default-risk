"""Demo UI.  streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt
import pandas as pd
import streamlit as st

from src.predict import load, predict

st.set_page_config(page_title="Credit Risk Demo", page_icon="💳", layout="wide")
_, meta = load()
opts = meta["category_options"]

st.title("Home Credit default risk")
st.caption(f"LightGBM on application-form fields only · 5-fold OOF ROC-AUC {meta['cv']['oof_auc']} · "
           f"trained on {meta['n_train']:,} applicants (default rate {meta['default_rate']:.1%})")


def pick(label, col, default):
    choices = opts[col]
    return st.selectbox(label, choices, index=choices.index(default) if default in choices else 0)


with st.form("applicant"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Loan")
        contract_type = pick("Contract type", "NAME_CONTRACT_TYPE", "Cash loans")
        credit_amount = st.number_input("Credit amount", 1000.0, 5e6, 500000.0, step=10000.0)
        annuity = st.number_input("Yearly payment (annuity)", 100.0, 5e5, 25000.0, step=1000.0)
        goods_price = st.number_input("Goods price", 0.0, 5e6, 450000.0, step=10000.0)
    with c2:
        st.subheader("Applicant")
        age_years = st.slider("Age", 18, 70, 35)
        employed = st.checkbox("Currently employed", True)
        years_employed = st.slider("Years employed (ignored if not employed)", 0.0, 40.0, 5.0, 0.5)
        annual_income = st.number_input("Annual income", 1000.0, 1e7, 180000.0, step=5000.0)
        income_type = pick("Income type", "NAME_INCOME_TYPE", "Working")
        education = pick("Education", "NAME_EDUCATION_TYPE", "Secondary / secondary special")
        occupation = pick("Occupation", "OCCUPATION_TYPE", "Laborers")
        organization = pick("Organization", "ORGANIZATION_TYPE", "Business Entity Type 3")
    with c3:
        st.subheader("Credit history & assets")
        ext_source_2 = st.slider("External score 2", 0.0, 1.0, 0.55, 0.01)
        ext_source_3 = st.slider("External score 3", 0.0, 1.0, 0.50, 0.01)
        has_ext1 = st.checkbox("Has external score 1", False)
        ext_source_1 = st.slider("External score 1 (ignored if unchecked)", 0.0, 1.0, 0.50, 0.01)
        years_since_id_issued = st.slider("Years since ID issued", 0.0, 20.0, 8.0, 0.5)
        years_since_registration = st.slider("Years since registration change", 0.0, 40.0, 10.0, 0.5)
        region_rating = st.radio("Region rating (1 = best)", [1, 2, 3], index=1, horizontal=True)
        owns_realty = st.checkbox("Owns real estate", True)
        owns_car = st.checkbox("Owns a car", False)
    submitted = st.form_submit_button("Score applicant", type="primary")

if submitted:
    result = predict(dict(
        contract_type=contract_type, credit_amount=credit_amount, annuity=annuity,
        goods_price=goods_price or None, age_years=age_years,
        years_employed=years_employed if employed else None,
        annual_income=annual_income, income_type=income_type, education=education,
        occupation=occupation, organization=organization,
        ext_source_1=ext_source_1 if has_ext1 else None,
        ext_source_2=ext_source_2, ext_source_3=ext_source_3,
        years_since_id_issued=years_since_id_issued, years_since_registration=years_since_registration,
        region_rating=region_rating, owns_realty=owns_realty, owns_car=owns_car,
    ), top_k=8)

    p = result["default_probability"]
    m1, m2 = st.columns(2)
    m1.metric("Default probability", f"{p:.1%}",
              f"{p - meta['default_rate']:+.1%} vs average", delta_color="inverse")
    m2.metric("Risk band", result["risk_band"].upper())

    st.subheader("Why: top SHAP factors")
    factors = pd.DataFrame(result["top_factors"])
    factors["value"] = factors["value"].map(
        lambda v: (f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3g}") if isinstance(v, float) else v)
    st.altair_chart(
        alt.Chart(factors).mark_bar().encode(
            x=alt.X("shap:Q", title="SHAP (log-odds)"),
            y=alt.Y("feature:N", sort=None, title=None, axis=alt.Axis(labelLimit=300)),
            color=alt.condition("datum.shap > 0", alt.value("#d62728"), alt.value("#2ca02c")),
            tooltip=["feature", "value", "shap", "effect"],
        ),
        use_container_width=True,
    )
    st.dataframe(factors, hide_index=True, use_container_width=True)
    st.caption("SHAP values are in log-odds: positive pushes the applicant toward default.")

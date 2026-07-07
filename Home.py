"""
Home.py - Streamlit prediction demo for the saved credit-card default classifier.

Loads artifacts/best_model.ckpt, which is the complete preprocessing + model pipeline
created by Task 1.
"""
import sys
from pathlib import Path

import pandas as pd
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))
from preprocessing import NUMERIC_COLS, CATEGORICAL_COLS, ORDINAL_COLS  # noqa: E402

MODEL_PATH = ROOT / "artifacts" / "best_model.ckpt"

EDU_MAP = {"Graduate school": 1, "University": 2, "High school": 3, "Others": 4}
MAR_MAP = {"Married": 1, "Single": 2, "Others": 3}
SEX_MAP = {"Male": 1, "Female": 2}
PAY_HELP = "Repayment status: -2, -1, 0 = paid on time, 1..8 = payment delayed by 1..8 months"

MONTH_LABELS = {
    1: "Last month",
    2: "2 months ago",
    3: "3 months ago",
    4: "4 months ago",
    5: "5 months ago",
    6: "6 months ago",
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def Home():
    st.markdown("## Credit Card Default Predictor")
    st.caption(
        "Enter the customer information and click Predict to run the saved classification pipeline. "
        "The result is a model decision, not a calibrated financial risk score."
    )
    st.divider()

    st.subheader("Customer Information")
    c1, c2, c3 = st.columns(3)
    with c1:
        limit_bal = st.number_input("Credit limit", min_value=0, value=50000, step=1000)
        sex = st.selectbox("Sex", list(SEX_MAP.keys()))
    with c2:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        education = st.selectbox("Education", list(EDU_MAP.keys()))
    with c3:
        marriage = st.selectbox("Marital status", list(MAR_MAP.keys()))

    st.subheader("Repayment History (last 6 months)")
    st.caption(PAY_HELP)
    pay_cols = st.columns(6)
    pay_vals = {}
    for i, col in enumerate(pay_cols, start=1):
        with col:
            pay_vals[f"PAY_{i}"] = st.number_input(
                MONTH_LABELS[i], min_value=-2, max_value=8, value=0, key=f"pay_{i}"
            )

    st.subheader("Bill Statement Amount")
    bill_cols = st.columns(6)
    bill_vals = {}
    for i, col in enumerate(bill_cols, start=1):
        with col:
            bill_vals[f"BILL_AMT{i}"] = st.number_input(
                MONTH_LABELS[i], value=0, step=100, key=f"bill_{i}"
            )

    st.subheader("Previous Payment Amount")
    payamt_cols = st.columns(6)
    payamt_vals = {}
    for i, col in enumerate(payamt_cols, start=1):
        with col:
            payamt_vals[f"PAY_AMT{i}"] = st.number_input(
                MONTH_LABELS[i], min_value=0, value=0, step=100, key=f"payamt_{i}"
            )

    st.divider()

    if st.button("Predict", type="primary", use_container_width=True):
        row = {
            "LIMIT_BAL": limit_bal,
            "SEX": SEX_MAP[sex],
            "EDUCATION": EDU_MAP[education],
            "MARRIAGE": MAR_MAP[marriage],
            "AGE": age,
        }
        row.update(pay_vals)
        row.update(bill_vals)
        row.update(payamt_vals)
        X_input = pd.DataFrame([row])

        try:
            model = load_model()
            pred = int(model.predict(X_input)[0])
            proba = None
            if hasattr(model, "predict_proba"):
                proba = float(model.predict_proba(X_input)[0][1])

            st.subheader("Prediction Result")
            if pred == 1:
                st.error("Prediction: model classifies this customer as DEFAULT next month (DEFAULT = 1)")
            else:
                st.success("Prediction: model classifies this customer as NO DEFAULT next month (DEFAULT = 0)")

            if proba is not None:
                st.metric("Default-class model score", f"{proba * 100:.1f}%")
                st.progress(min(max(proba, 0.0), 1.0))
                st.caption(
                    "This score comes from the model's predict_proba output and default decision threshold. "
                    "It is not calibrated as a real-world probability of default."
                )

        except FileNotFoundError:
            st.error('Could not find "artifacts/best_model.ckpt". Run "python src/train_models.py" first.')
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.caption("Click **Predict** to see the result.")

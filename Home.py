"""
Home.py — BAI 3: Giao dien demo du doan kha nang vo no the tin dung.

Load best_model.ckpt (Pipeline day du: preprocess + model tot nhat tu Bai 1)
va cho phep nguoi dung nhap thong tin khach hang de du doan.
"""
import sys
from pathlib import Path

import pandas as pd
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))
# import de tai su dung dung danh sach cot -> tranh lech giua train va predict
from preprocessing import NUMERIC_COLS, CATEGORICAL_COLS, ORDINAL_COLS  # noqa: E402

MODEL_PATH = ROOT / "artifacts" / "best_model.ckpt"

EDU_MAP = {"Graduate school": 1, "University": 2, "High school": 3, "Others": 4}
MAR_MAP = {"Married": 1, "Single": 2, "Others": 3}
SEX_MAP = {"Male": 1, "Female": 2}
PAY_HELP = "Repayment status: -2, -1, 0 = paid on time, 1..8 = payment delayed by 1..8 months"

# Label thoi gian dung chung cho ca 3 nhom PAY_i / BILL_AMTi / PAY_AMTi,
# vi ca 3 cung dai dien cho 6 thang gan nhat -> dung 1 nguon duy nhat
# de tranh viet lech nhau giua cac nhom (i=1 la thang gan nhat).
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
    st.markdown("## 💳 Credit Card Default Predictor")
    st.caption("Enter the customer information and click Predict to estimate the default risk for next month.")
    st.divider()

    # --- Thong tin ca nhan ---
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

    # --- Lich su tra no PAY_1..PAY_6 ---
    st.subheader("Repayment History (last 6 months)")
    st.caption(PAY_HELP)
    pay_cols = st.columns(6)
    pay_vals = {}
    for i, col in enumerate(pay_cols, start=1):
        with col:
            pay_vals[f"PAY_{i}"] = st.number_input(
                MONTH_LABELS[i], min_value=-2, max_value=8, value=0, key=f"pay_{i}"
            )

    # --- So tien hoa don BILL_AMT ---
    st.subheader("Bill Statement Amount")
    bill_cols = st.columns(6)
    bill_vals = {}
    for i, col in enumerate(bill_cols, start=1):
        with col:
            bill_vals[f"BILL_AMT{i}"] = st.number_input(
                MONTH_LABELS[i], value=0, step=100, key=f"bill_{i}"
            )

    # --- So tien da thanh toan PAY_AMT ---
    st.subheader("Previous Payment Amount")
    payamt_cols = st.columns(6)
    payamt_vals = {}
    for i, col in enumerate(payamt_cols, start=1):
        with col:
            payamt_vals[f"PAY_AMT{i}"] = st.number_input(
                MONTH_LABELS[i], min_value=0, value=0, step=100, key=f"payamt_{i}"
            )

    st.divider()

    if st.button("🔮 Predict", type="primary", use_container_width=True):
        # Gom tat ca input thanh 1 dong DataFrame voi dung ten cot model can
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
            # neu model ho tro predict_proba thi lay xac suat vo no (class 1)
            proba = None
            if hasattr(model, "predict_proba"):
                proba = float(model.predict_proba(X_input)[0][1])

            if pred == 1:
                st.error("⚠️ Prediction: This customer is LIKELY TO DEFAULT next month (DEFAULT = 1)")
            else:
                st.success("✅ Prediction: This customer is NOT likely to default next month (DEFAULT = 0)")

            if proba is not None:
                st.metric("Default probability", f"{proba*100:.1f}%")
                st.progress(min(max(proba, 0.0), 1.0))

        except FileNotFoundError:
            st.error('Could not find "artifacts/best_model.ckpt". Run "python src/train_models.py" first.')
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.caption("Click **Predict** to see the result.")

"""
AboutMe.py — Trang thong tin nhom.
"""
import streamlit as st

DATASET_NAME = "Default of Credit Card Clients (UCI #350)"
DATASET_URL = "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients"

TEAM_MEMBERS = [
    {"name": "Huynh Kien Đong Duy", "mssv": "52200244"},
    {"name": "Tran Trung Hieu", "mssv": "52400047"},
]


def AboutMe():
    st.header("👥 About Us")

    st.subheader("Dataset")
    st.markdown(f"**{DATASET_NAME}**")
    st.markdown(f"Source: [{DATASET_URL}]({DATASET_URL})")
    st.caption("30,000 credit card clients in Taiwan (2005). "
               "Goal: predict whether a client will default next month (binary classification).")

    st.divider()

    st.subheader("Team Members")
    for m in TEAM_MEMBERS:
        st.markdown(f"- **{m['name']}** — Student ID: `{m['mssv']}`")

    st.divider()

    st.subheader("Project Contents")
    st.markdown("""
    - **Task 1**: Compare three classification models (Logistic Regression, Random Forest, Gradient Boosting)
    - **Task 2**: Correlation-based feature selection, compared via MAE
    - **Task 3**: Interactive prediction demo built with Streamlit
    """)

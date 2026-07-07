"""
Statistics.py — Thong ke & kham pha du lieu (EDA) cho dataset Credit Card Default.
Tai cau truc tu project mau, doi sang data/credit_card_default.csv.
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "credit_card_default.csv"


def Statistics():
    st.header("📈 Data Statistics")

    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        st.error(f"File not found: {DATA_PATH}")
        st.stop()

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # --- KPIs ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]}")
    c3.metric("Missing values", f"{int(df.isna().sum().sum()):,}")
    default_rate = df["DEFAULT"].mean() * 100
    c4.metric("Default rate", f"{default_rate:.1f}%")

    st.divider()

    # --- Phan bo target ---
    st.subheader("🎯 Target Distribution")
    vc = df["DEFAULT"].value_counts().reset_index()
    vc.columns = ["DEFAULT", "count"]
    vc["label"] = vc["DEFAULT"].map({0: "No default (0)", 1: "Default (1)"})
    fig = px.pie(vc, names="label", values="count", title="Default vs No-default ratio", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Imbalanced dataset: the default group is the minority (~22%).")

    st.divider()

    # --- Schema ---
    st.subheader("🧱 Dataset Schema")
    schema = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "unique": df.nunique().values,
    })
    st.dataframe(schema, use_container_width=True, hide_index=True)

    st.divider()

    # --- Kham pha cot numeric ---
    st.subheader("🔍 Numeric Explorer")
    num_cols = [c for c in df.columns if c != "DEFAULT"]
    col = st.selectbox("Select a column", num_cols)
    e1, e2 = st.columns(2)
    with e1:
        with st.container(border=True):
            fig = px.histogram(df, x=col, color="DEFAULT", nbins=40,
                            title=f"Distribution of {col} by class")
            st.plotly_chart(fig, use_container_width=True)
    with e2:
        with st.container(border=True):
            fig = px.box(df, x="DEFAULT", y=col, title=f"Box plot of {col} by class")
            st.plotly_chart(fig, use_container_width=True)

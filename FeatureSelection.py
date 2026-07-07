"""
FeatureSelection.py — Hien thi ket qua BAI 2: feature selection dua tren correlation.
Doc tu artifacts/feature_selection_results.json + artifacts/correlation.csv.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULT_PATH = ROOT / "artifacts" / "feature_selection_results.json"
CORR_PATH = ROOT / "artifacts" / "correlation.csv"
DATA_PATH = ROOT / "data" / "credit_card_default.csv"


def FeatureSelection():
    st.header("🎯 Task 2 — Correlation-based Feature Selection")

    try:
        data = json.load(open(RESULT_PATH, encoding="utf-8"))
    except FileNotFoundError:
        st.error('No results found. Run "python src/feature_selection.py" first.')
        st.stop()

    st.info("Target is binary (0/1); LinearRegression + MAE is used to illustrate the effect "
            "of feature selection, not as the main classification model (see Task 1).")

    # --- (a) Correlation cua tung feature voi target ---
    st.subheader("(a) Correlation of Each Feature with the Target")
    corr_df = pd.read_csv(CORR_PATH)
    fig = px.bar(corr_df.sort_values("corr_with_target"),
                 x="corr_with_target", y="feature", orientation="h",
                 title="Pearson correlation with target (DEFAULT)")
    st.plotly_chart(fig, use_container_width=True)

    # --- Heatmap correlation toan bo (minh hoa truc quan) ---
    st.subheader("Correlation Matrix (heatmap)")
    try:
        df = pd.read_csv(DATA_PATH).drop(columns=["ID"])
        corr = df.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                        title="Correlation matrix", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1)
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render heatmap: {e}")

    # --- (b) So sanh MAE theo cac tap feature ---
    st.subheader("(b) MAE Comparison Across Feature Sets (LinearRegression)")
    exp_df = pd.DataFrame(data["experiments"])
    show = exp_df[["feature_set", "num_features", "mae"]].copy()
    show.columns = ["Feature set", "Num features", "MAE"]
    st.dataframe(show, use_container_width=True, hide_index=True)

    fig = px.line(exp_df, x="num_features", y="mae", markers=True,
                  title="MAE vs Number of Selected Features",
                  labels={"num_features": "Number of features", "mae": "MAE"})
    st.plotly_chart(fig, use_container_width=True)

    st.success(f"Feature set with the lowest MAE: **{data['best_feature_set']}**")
    st.caption("Observation: using only the few most correlated features (PAY_1..PAY_6), "
               "the MAE is already close to using all features — showing that feature selection "
               "yields a leaner model with comparable accuracy.")

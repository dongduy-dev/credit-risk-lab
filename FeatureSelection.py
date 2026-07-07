"""
FeatureSelection.py - Display Task 2 correlation-based feature-selection results.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULT_PATH = ROOT / "artifacts" / "feature_selection_results.json"
CORR_PATH = ROOT / "artifacts" / "correlation.csv"
CORR_MATRIX_PATH = ROOT / "artifacts" / "correlation_matrix.csv"


def FeatureSelection():
    st.header("Task 2 - Correlation-based Feature Selection")

    try:
        with open(RESULT_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        st.error('No results found. Run "python src/feature_selection.py" first.')
        st.stop()

    st.info(
        "Target is binary (0/1); LinearRegression + MAE is used to illustrate the effect "
        "of feature selection, not as the main classification model (see Task 1)."
    )

    st.subheader("(a) Training-set Correlation Ranking")
    corr_df = pd.read_csv(CORR_PATH)
    if "abs_corr" not in corr_df.columns:
        corr_df["abs_corr"] = corr_df["corr_with_target"].abs()
    if data.get("correlation_source") == "training_set":
        st.caption("Correlation ranking is calculated on the training split only; the test set is reserved for final MAE evaluation.")
    st.caption(
        "Pearson correlation is a univariate screening method. SEX, EDUCATION, and MARRIAGE are categorical codes, "
        "so their numeric correlations are descriptive only and should not be interpreted as distances between categories."
    )

    fig = px.bar(
        corr_df.sort_values("abs_corr"),
        x="abs_corr",
        y="feature",
        orientation="h",
        color="corr_with_target",
        color_continuous_scale="RdBu_r",
        title="Absolute Pearson correlation with target (training set)",
        labels={"abs_corr": "absolute correlation", "corr_with_target": "signed correlation"},
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Top correlated features used to build subsets**")
    top_corr = corr_df.sort_values("abs_corr", ascending=False).head(10)
    st.dataframe(top_corr[["feature", "corr_with_target", "abs_corr"]], width="stretch", hide_index=True)

    st.subheader("Correlation Matrix (training set heatmap)")
    try:
        corr = pd.read_csv(CORR_MATRIX_PATH, index_col=0)
        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            title="Correlation matrix from training split",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
        )
        fig.update_layout(height=700)
        st.plotly_chart(fig, width="stretch")
    except FileNotFoundError:
        st.warning('No training-set correlation matrix found. Run "python src/feature_selection.py" first.')
    except Exception as e:
        st.warning(f"Could not render heatmap: {e}")

    st.subheader("(b) Held-out MAE Across Feature Sets (LinearRegression)")
    exp_df = pd.DataFrame(data["experiments"])
    show = exp_df[["feature_set", "num_features", "mae"]].rename(columns={
        "feature_set": "Feature set",
        "num_features": "Num features",
        "mae": "MAE",
    })
    if "selected_features" in exp_df.columns:
        show["Selected features"] = exp_df["selected_features"].apply(lambda features: ", ".join(features))
    st.dataframe(show, width="stretch", hide_index=True)

    fig = px.line(
        exp_df,
        x="num_features",
        y="mae",
        markers=True,
        title="MAE vs Number of Selected Features",
        labels={"num_features": "Number of features", "mae": "MAE"},
    )
    st.plotly_chart(fig, width="stretch")

    experiments = data["experiments"]
    best_exp = min(experiments, key=lambda item: item["mae"])
    all_exp = next((item for item in experiments if item["feature_set"] == "all"), None)
    st.success(f"Lowest held-out MAE: **{best_exp['feature_set']}** ({best_exp['mae']:.5f})")
    if all_exp and best_exp["feature_set"] == "all":
        selected_only = [item for item in experiments if item["feature_set"] != "all"]
        best_subset = min(selected_only, key=lambda item: item["mae"]) if selected_only else None
        if best_subset:
            delta = best_subset["mae"] - all_exp["mae"]
            st.info(
                f"The full feature set has the best MAE. The best correlation-selected subset is "
                f"{best_subset['feature_set']} with MAE {best_subset['mae']:.5f}, "
                f"which is {delta:.5f} higher than all features. The subsets are simpler, but they did not improve held-out MAE."
            )
    elif all_exp:
        delta = all_exp["mae"] - best_exp["mae"]
        st.info(
            f"The selected subset improves MAE over all features by {delta:.5f}. "
            "This conclusion is based only on the held-out MAE evaluation."
        )
    st.caption(
        "LinearRegression treats the 0/1 target as a numeric response only for this feature-selection experiment. "
        "It is not the final classifier used in the prediction demo."
    )

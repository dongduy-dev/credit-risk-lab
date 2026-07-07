"""
ModelComparison.py — Hien thi ket qua BAI 1: so sanh 3 model.
Doc tu artifacts/model_comparison.json (sinh boi src/train_models.py).
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULT_PATH = ROOT / "artifacts" / "model_comparison.json"


def ModelComparison():
    st.header("📊 Task 1 — Classification Model Comparison")

    try:
        data = json.load(open(RESULT_PATH, encoding="utf-8"))
    except FileNotFoundError:
        st.error('No results found. Run "python src/train_models.py" first.')
        st.stop()

    results = data["results"]
    selection = data.get("selection", {})
    if selection.get("selection_set") == "validation":
        st.success(f"Best model (selected by validation weighted F1): **{data['best_model']}**")
        st.caption("Final test metrics below are reported after model selection and are not used to choose the saved model.")
    else:
        st.success(f"Best model (by weighted F1): **{data['best_model']}**")

    # --- Bang tong hop ---
    rows = []
    for name, r in results.items():
        row = {
            "Model": name,
            "Accuracy": r["accuracy"],
            "Weighted F1": r["weighted_avg"]["f1"],
            "Precision (avg)": r["weighted_avg"]["precision"],
            "Recall (avg)": r["weighted_avg"]["recall"],
            "Train (s)": r["train_time_sec"],
            "Test (s)": r["test_time_sec"],
        }
        if "selection" in r:
            row["Validation Weighted F1"] = r["selection"].get("validation_weighted_f1")
        rows.append(row)
    df = pd.DataFrame(rows)
    st.subheader("📋 Summary Table")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # --- Bieu do so sanh accuracy & f1 ---
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            fig = px.bar(df, x="Model", y=["Accuracy", "Weighted F1"], barmode="group",
                        title="Accuracy vs Weighted F1")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        with st.container(border=True):
            fig = px.bar(df, x="Model", y=["Train (s)", "Test (s)"], barmode="group",
                        title="Training vs Testing Time (seconds)")
            st.plotly_chart(fig, use_container_width=True)

    # --- Chi tiet per-class ---
    st.subheader("🏷️ Per-class Metrics")
    detail_rows = []
    for name, r in results.items():
        for cls_key, cls_label in [("class_0_no_default", "Class 0 (No default)"),
                                    ("class_1_default", "Class 1 (Default)")]:
            c = r["per_class"][cls_key]
            detail_rows.append({
                "Model": name, "Class": cls_label,
                "Precision": c["precision"], "Recall": c["recall"], "F1": c["f1"],
            })
    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

    st.info("Class 1 (default) is the minority and most important group. "
            "Focus on its recall/F1 when evaluating — high accuracy alone can be misleading "
            "on an imbalanced dataset (~3.5:1).")

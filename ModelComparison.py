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

    rows = []
    for name, r in results.items():
        row = {
            "Model": name,
            "Accuracy": r["accuracy"],
            "Weighted F1": r["weighted_avg"]["f1"],
            "Class 1 Recall": r["per_class"]["class_1_default"]["recall"],
            "Class 1 F1": r["per_class"]["class_1_default"]["f1"],
            "False Negatives": r.get("confusion_matrix", {}).get("false_negatives"),
            "False Positives": r.get("confusion_matrix", {}).get("false_positives"),
            "Train (s)": r["train_time_sec"],
            "Test (s)": r["test_time_sec"],
        }
        if "selection" in r:
            row["Validation Weighted F1"] = r["selection"].get("validation_weighted_f1")
        rows.append(row)
    df = pd.DataFrame(rows)

    st.subheader("📋 Summary Table")
    st.dataframe(df, use_container_width=True, hide_index=True)

    baseline = data.get("majority_baseline")
    interpretation = data.get("risk_interpretation", {})
    if baseline:
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Majority baseline accuracy", baseline["accuracy"])
        b2.metric("Baseline class 1 recall", baseline["per_class"]["class_1_default"]["recall"])
        b3.metric("Actual defaulters in test", baseline["credit_risk"]["actual_defaulters"])
        b4.metric("Baseline missed defaulters", baseline["credit_risk"]["missed_defaulters_false_negatives"])
        st.warning(interpretation.get(
            "majority_baseline_warning",
            "Accuracy alone is misleading for this imbalanced credit-risk dataset.",
        ))

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            fig = px.bar(
                df,
                x="Model",
                y=["Accuracy", "Weighted F1", "Class 1 Recall", "Class 1 F1"],
                barmode="group",
                title="Overall metrics vs default-class metrics",
            )
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        with st.container(border=True):
            fig = px.bar(
                df,
                x="Model",
                y=["Train (s)", "Test (s)"],
                barmode="group",
                title="Training vs Testing Time (seconds)",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("🏷️ Per-class Metrics")
    detail_rows = []
    for name, r in results.items():
        for cls_key, cls_label in [
            ("class_0_no_default", "Class 0 (No default)"),
            ("class_1_default", "Class 1 (Default)"),
        ]:
            c = r["per_class"][cls_key]
            detail_rows.append({
                "Model": name,
                "Class": cls_label,
                "Precision": c["precision"],
                "Recall": c["recall"],
                "F1": c["f1"],
            })
    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

    if all("confusion_matrix" in r for r in results.values()):
        st.subheader("Credit-risk Trade-off")
        risk_rows = []
        for name, r in results.items():
            cm = r["confusion_matrix"]
            risk = r["credit_risk"]
            risk_rows.append({
                "Model": name,
                "True negatives": cm["true_negatives"],
                "False positives": cm["false_positives"],
                "False negatives (missed defaults)": cm["false_negatives"],
                "True positives (caught defaults)": cm["true_positives"],
                "Default recall": risk["default_recall"],
                "Default precision": risk["default_precision"],
                "False negative rate": risk["false_negative_rate"],
                "False positive rate": risk["false_positive_rate"],
                "ROC-AUC": r.get("roc_auc"),
                "PR-AUC": r.get("pr_auc"),
            })
        risk_df = pd.DataFrame(risk_rows)
        st.dataframe(risk_df, use_container_width=True, hide_index=True)

        c3, c4 = st.columns(2)
        with c3:
            fig = px.bar(
                risk_df,
                x="Model",
                y=["False negatives (missed defaults)", "False positives"],
                barmode="group",
                title="False negatives vs false positives",
            )
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            fig = px.bar(
                risk_df,
                x="Model",
                y=["Default recall", "Default precision"],
                barmode="group",
                title="Default class recall vs precision",
            )
            st.plotly_chart(fig, use_container_width=True)

    notes = interpretation.get("risk_objective_notes", [])
    if notes:
        st.subheader("Interpretation for Credit-risk Objectives")
        st.markdown("\n".join(f"- {note}" for note in notes))
        st.caption(
            f"Best by accuracy: {interpretation.get('best_by_accuracy')}; "
            f"best by default recall: {interpretation.get('best_by_default_recall')}; "
            f"best by default F1: {interpretation.get('best_by_default_f1')}."
        )

    st.info("Class 1 (default) is the minority and most important group. "
            "Focus on recall/F1 and false negatives for class 1; high accuracy alone can be misleading "
            "on an imbalanced dataset (~3.5:1).")

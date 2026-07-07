"""
ModelComparison.py - Display Task 1 classification results from artifacts/model_comparison.json.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULT_PATH = ROOT / "artifacts" / "model_comparison.json"


def ModelComparison():
    st.header("Task 1 - Classification Model Comparison")

    try:
        with open(RESULT_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        st.error('No results found. Run "python src/train_models.py" first.')
        st.stop()

    results = data["results"]
    selection = data.get("selection", {})
    baseline = data.get("majority_baseline")
    interpretation = data.get("risk_interpretation", {})

    if selection.get("selection_set") == "validation":
        st.success(f"Saved model (selected by validation weighted F1): **{data['best_model']}**")
        st.caption("Final test metrics below are reported after model selection and are not used to choose the saved model.")
    else:
        st.success(f"Saved model: **{data['best_model']}**")

    best_recall_model = max(results, key=lambda name: results[name]["credit_risk"]["default_recall"])
    fewest_missed_model = min(results, key=lambda name: results[name]["credit_risk"]["missed_defaulters_false_negatives"])
    saved_result = results[data["best_model"]]
    saved_risk = saved_result.get("credit_risk", {})
    actual_defaulters = saved_risk.get("actual_defaulters")

    st.subheader("Credit-risk Snapshot")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Saved model", data["best_model"])
    if actual_defaulters:
        s2.metric("Missed defaulters", f"{saved_risk['missed_defaulters_false_negatives']:,}/{actual_defaulters:,}")
    else:
        s2.metric("Missed defaulters", saved_risk.get("missed_defaulters_false_negatives", "n/a"))
    s3.metric(
        "Best default recall",
        f"{best_recall_model}: {results[best_recall_model]['credit_risk']['default_recall']:.4f}",
    )
    s4.metric("Majority baseline accuracy", f"{baseline['accuracy']:.4f}" if baseline else "n/a")

    if actual_defaulters:
        st.warning(
            f"The saved model still misses {saved_risk['missed_defaulters_false_negatives']:,} "
            f"of {actual_defaulters:,} actual defaulters "
            f"({saved_risk['false_negative_rate']:.1%}). Use this page to compare risk trade-offs, "
            "not just to pick the highest overall score."
        )
    if best_recall_model != data["best_model"]:
        st.info(
            f"{best_recall_model} catches more defaulters than the saved model, but it also creates more false positives. "
            "That may be preferable when missed defaults are more costly than unnecessary reviews."
        )

    rows = []
    baseline_accuracy = baseline["accuracy"] if baseline else None
    for name, r in results.items():
        default_metrics = r["per_class"]["class_1_default"]
        risk = r.get("credit_risk", {})
        row = {
            "Model": name,
            "Accuracy": r["accuracy"],
            "Accuracy vs baseline": round(r["accuracy"] - baseline_accuracy, 4) if baseline_accuracy is not None else None,
            "Weighted F1": r["weighted_avg"]["f1"],
            "Default precision": default_metrics["precision"],
            "Default recall": default_metrics["recall"],
            "Default F1": default_metrics["f1"],
            "Missed defaulters": risk.get("missed_defaulters_false_negatives"),
            "Caught defaulters": risk.get("caught_defaulters_true_positives"),
            "False positives": risk.get("false_alarm_non_defaulters_false_positives"),
            "Train (s)": r["train_time_sec"],
            "Test (s)": r["test_time_sec"],
        }
        if "selection" in r:
            row["Validation weighted F1"] = r["selection"].get("validation_weighted_f1")
        rows.append(row)
    df = pd.DataFrame(rows)

    st.subheader("Assignment Metrics and Credit-risk Summary")
    st.caption("Accuracy and weighted F1 are shown with the required per-class metrics, but default-class recall and missed defaulters are central for this imbalanced credit-risk problem.")
    st.dataframe(df, use_container_width=True, hide_index=True)

    if baseline:
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Majority baseline accuracy", baseline["accuracy"])
        b2.metric("Baseline default recall", baseline["per_class"]["class_1_default"]["recall"])
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
                y=["Accuracy", "Weighted F1", "Default recall", "Default F1"],
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
                title="Training vs testing time (seconds)",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Per-class Metrics")
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
        st.subheader("False-positive and False-negative Trade-off")
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
        if baseline:
            baseline_risk = baseline["credit_risk"]
            baseline_row = {
                "Model": "Majority baseline",
                "True negatives": baseline["confusion_matrix"]["true_negatives"],
                "False positives": baseline["confusion_matrix"]["false_positives"],
                "False negatives (missed defaults)": baseline["confusion_matrix"]["false_negatives"],
                "True positives (caught defaults)": baseline["confusion_matrix"]["true_positives"],
                "Default recall": baseline_risk["default_recall"],
                "Default precision": baseline_risk["default_precision"],
                "False negative rate": baseline_risk["false_negative_rate"],
                "False positive rate": baseline_risk["false_positive_rate"],
                "ROC-AUC": None,
                "PR-AUC": None,
            }
            risk_rows.insert(0, baseline_row)
        risk_display_df = pd.DataFrame(risk_rows)
        st.dataframe(risk_display_df, use_container_width=True, hide_index=True)

        c3, c4 = st.columns(2)
        with c3:
            fig = px.bar(
                risk_display_df,
                x="Model",
                y=["False negatives (missed defaults)", "False positives"],
                barmode="group",
                title="Missed defaulters vs false alarms",
            )
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            fig = px.bar(
                risk_display_df,
                x="Model",
                y=["Default recall", "Default precision"],
                barmode="group",
                title="Default recall vs precision",
            )
            st.plotly_chart(fig, use_container_width=True)

    notes = interpretation.get("risk_objective_notes", [])
    if notes:
        st.subheader("Interpretation for Credit-risk Objectives")
        st.markdown("\n".join(f"- {note}" for note in notes))
        st.caption(
            f"Best by accuracy: {interpretation.get('best_by_accuracy')}; "
            f"best by default recall: {interpretation.get('best_by_default_recall')}; "
            f"best by default F1: {interpretation.get('best_by_default_f1')}; "
            f"fewest missed defaulters: {fewest_missed_model}."
        )

    st.info(
        "Class 1 (default) is the minority and most important group. "
        "Weighted F1 is useful as a summary, but it can hide weak minority-class detection. "
        "A model with lower accuracy may still be better if the business goal is to catch more defaulters."
    )

"""
Statistics.py - Data statistics and EDA for the Credit Card Default dataset.
Uses data/credit_card_default.csv and the raw-data audit helper.
"""
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "credit_card_default.csv"
XLS_PATH = ROOT / "data" / "default_of_credit_card_clients.xls"
sys.path.append(str(ROOT / "src"))
from data_audit import build_audit  # noqa: E402


@st.cache_data
def load_quality_audit():
    return build_audit(str(DATA_PATH), str(XLS_PATH))


def Statistics():
    st.header("Data Statistics")

    try:
        raw_df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        st.error(f"File not found: {DATA_PATH}")
        st.stop()

    df = raw_df.copy()
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
    st.subheader("Target Distribution")
    vc = df["DEFAULT"].value_counts().reset_index()
    vc.columns = ["DEFAULT", "count"]
    vc["label"] = vc["DEFAULT"].map({0: "No default (0)", 1: "Default (1)"})
    fig = px.pie(vc, names="label", values="count", title="Default vs No-default ratio", hole=0.4)
    st.plotly_chart(fig, width="stretch")
    st.caption("Imbalanced dataset: the default group is the minority (~22%).")

    st.divider()

    # --- Schema ---
    st.subheader("Dataset Schema")
    schema = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "unique": df.nunique().values,
    })
    st.dataframe(schema, width="stretch", hide_index=True)

    st.divider()

    # --- Raw data quality audit ---
    st.subheader("Data Quality Audit")
    audit = load_quality_audit()

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Raw rows", f"{audit['dataset']['shape'][0]:,}")
    a2.metric("Raw columns", f"{audit['dataset']['shape'][1]}")
    a3.metric("Unique IDs", f"{audit['identifier_integrity'].get('unique_id_count', 0):,}")
    a4.metric("Exact duplicates", f"{audit['duplicates']['exact_duplicate_rows']:,}")

    st.caption("This audit reports raw CSV values before model preprocessing. It does not remove rows or silently clean rare codes.")

    q1, q2 = st.columns(2)
    with q1:
        st.markdown("**Target distribution**")
        st.dataframe(pd.DataFrame(audit["target_distribution"]), width="stretch", hide_index=True)
    with q2:
        st.markdown("**Missing values and identifiers**")
        id_info = audit["identifier_integrity"]
        id_rows = [
            {"check": "missing values", "value": str(audit["missing_values"]["total_missing_values"])},
            {"check": "duplicate IDs", "value": str(id_info.get("duplicate_id_count"))},
            {"check": "ID is sequential 1..n", "value": str(id_info.get("sequential_1_to_n"))},
            {"check": "rows duplicated after dropping ID", "value": str(audit["duplicates"]["duplicate_rows_excluding_id"])},
            {"check": "feature rows duplicated after dropping ID/target", "value": str(audit["duplicates"]["duplicate_feature_rows_excluding_id_and_target"])},
        ]
        st.dataframe(pd.DataFrame(id_rows), width="stretch", hide_index=True)

    st.markdown("**Unusual categorical codes in raw data**")
    cat_rows = []
    for column, info in audit["categorical_code_audit"].items():
        for item in info.get("raw_counts", []):
            expected = str(item["value"]) in info.get("expected_codes", {})
            cat_rows.append({
                "column": column,
                "value": item["value"],
                "count": item["count"],
                "percent": item["percent"],
                "documented_code": expected,
            })
    st.dataframe(pd.DataFrame(cat_rows), width="stretch", hide_index=True)

    st.markdown("**Raw values versus recoded values**")
    recode_rows = []
    for item in audit["recoding_summary"]:
        recode_rows.append({
            "column": item["column"],
            "mapping": item["mapping"],
            "rows_changed": item["rows_changed"],
            "decision": item["modeling_decision"],
        })
    st.dataframe(pd.DataFrame(recode_rows), width="stretch", hide_index=True)
    selected_recode = st.selectbox(
        "Select recoded categorical column",
        [item["column"] for item in audit["recoding_summary"]],
        key="data_quality_recode_col",
    )
    recode_detail = next(item for item in audit["recoding_summary"] if item["column"] == selected_recode)
    recode_count_rows = []
    for item in recode_detail["raw_counts"]:
        recode_count_rows.append({"stage": "raw", **item})
    for item in recode_detail["recoded_counts"]:
        recode_count_rows.append({"stage": "model-ready", **item})
    st.dataframe(pd.DataFrame(recode_count_rows), width="stretch", hide_index=True)

    st.markdown("**Repayment-status code counts**")
    pay_columns = list(audit["repayment_status_codes"].keys())
    selected_pay = st.selectbox("Select repayment-status column", pay_columns, key="data_quality_pay_col")
    st.dataframe(pd.DataFrame(audit["repayment_status_codes"][selected_pay]["raw_counts"]),
                 width="stretch", hide_index=True)

    naming = audit["schema_naming"]
    st.markdown("**Schema naming traceability**")
    st.write(naming["decision"])
    source_compare = naming.get("source_excel_comparison", {})
    if source_compare.get("available"):
        st.dataframe(pd.DataFrame([source_compare]), width="stretch", hide_index=True)
    else:
        st.warning(source_compare.get("reason", "Source Excel comparison is unavailable."))

    st.divider()

    # --- Kham pha cot numeric ---
    st.subheader("Numeric Explorer")
    num_cols = [c for c in df.columns if c != "DEFAULT"]
    col = st.selectbox("Select a column", num_cols)
    e1, e2 = st.columns(2)
    with e1:
        with st.container(border=True):
            fig = px.histogram(df, x=col, color="DEFAULT", nbins=40,
                            title=f"Distribution of {col} by class")
            st.plotly_chart(fig, width="stretch")
    with e2:
        with st.container(border=True):
            fig = px.box(df, x="DEFAULT", y=col, title=f"Box plot of {col} by class")
            st.plotly_chart(fig, width="stretch")

"""
data_audit.py - raw dataset quality audit for UCI Credit Card Default.

This script reports what exists in the raw CSV before modeling transformations.
It does not remove rows or silently clean values.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from preprocessing import CATEGORY_RECODES, TARGET, DROP_COLS, CATEGORICAL_COLS, ORDINAL_COLS, NUMERIC_COLS

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "credit_card_default.csv"
XLS_PATH = ROOT / "data" / "default_of_credit_card_clients.xls"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

EXPECTED_CATEGORICAL_CODES = {
    "SEX": {1: "male", 2: "female"},
    "EDUCATION": {1: "graduate school", 2: "university", 3: "high school", 4: "others"},
    "MARRIAGE": {1: "married", 2: "single", 3: "others"},
}
EXPECTED_REPAYMENT_CODES = {-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8}


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value


def _counts(series: pd.Series) -> list[dict[str, Any]]:
    total = len(series)
    rows = []
    for value, count in series.value_counts(dropna=False).sort_index().items():
        rows.append({
            "value": _json_value(value),
            "count": int(count),
            "percent": round(float(count / total * 100), 4) if total else 0.0,
        })
    return rows


def _role_for_column(column: str) -> str:
    if column in DROP_COLS:
        return "identifier_dropped_before_modeling"
    if column == TARGET:
        return "target"
    if column in CATEGORICAL_COLS:
        return "categorical_code_one_hot_encoded_for_classification"
    if column in ORDINAL_COLS:
        return "repayment_status_treated_as_ordinal_numeric"
    if column in NUMERIC_COLS:
        return "numeric_scaled_for_modeling"
    return "not_used_or_unknown"


def _schema(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "column": column,
            "dtype": str(df[column].dtype),
            "unique_values": int(df[column].nunique(dropna=False)),
            "role": _role_for_column(column),
        }
        for column in df.columns
    ]


def _missing_values(df: pd.DataFrame) -> dict[str, Any]:
    missing = df.isna().sum()
    return {
        "total_missing_values": int(missing.sum()),
        "columns_with_missing_values": [
            {
                "column": column,
                "missing": int(count),
                "percent": round(float(count / len(df) * 100), 4),
            }
            for column, count in missing.items()
            if int(count) > 0
        ],
    }


def _identifier_integrity(df: pd.DataFrame) -> dict[str, Any]:
    if "ID" not in df.columns:
        return {"id_column_present": False}

    ids = df["ID"]
    expected_ids = pd.Series(range(1, len(df) + 1), index=df.index)
    return {
        "id_column_present": True,
        "unique_id_count": int(ids.nunique(dropna=False)),
        "duplicate_id_count": int(ids.duplicated().sum()),
        "min_id": _json_value(ids.min()),
        "max_id": _json_value(ids.max()),
        "sequential_1_to_n": bool(ids.reset_index(drop=True).equals(expected_ids.reset_index(drop=True))),
        "modeling_decision": "ID is dropped before modeling because it is an identifier, not a predictive feature.",
    }


def _duplicates(df: pd.DataFrame) -> dict[str, Any]:
    feature_cols = [c for c in df.columns if c not in DROP_COLS + [TARGET]]
    non_id_cols = [c for c in df.columns if c not in DROP_COLS]
    return {
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_rows_excluding_id": int(df[non_id_cols].duplicated().sum()),
        "duplicate_feature_rows_excluding_id_and_target": int(df[feature_cols].duplicated().sum()),
        "modeling_decision": "Duplicates are reported for transparency; no duplicate rows are removed by the current project.",
    }


def _categorical_audit(df: pd.DataFrame) -> dict[str, Any]:
    audit = {}
    for column, expected in EXPECTED_CATEGORICAL_CODES.items():
        if column not in df.columns:
            audit[column] = {"present": False}
            continue
        raw_values = set(df[column].dropna().unique().tolist())
        unusual_values = sorted(raw_values - set(expected.keys()))
        audit[column] = {
            "present": True,
            "expected_codes": {str(k): v for k, v in expected.items()},
            "raw_counts": _counts(df[column]),
            "unusual_or_undocumented_codes": [
                {
                    "value": _json_value(value),
                    "count": int((df[column] == value).sum()),
                }
                for value in unusual_values
            ],
        }
    return audit


def _repayment_status_audit(df: pd.DataFrame) -> dict[str, Any]:
    audit = {}
    for column in ORDINAL_COLS:
        if column not in df.columns:
            audit[column] = {"present": False}
            continue
        raw_values = set(df[column].dropna().unique().tolist())
        audit[column] = {
            "present": True,
            "raw_counts": _counts(df[column]),
            "codes_outside_expected_observed_range": sorted(_json_value(v) for v in raw_values - EXPECTED_REPAYMENT_CODES),
            "modeling_decision": "Repayment status is treated as ordinal numeric and scaled in the classification pipeline.",
        }
    return audit


def _recoding_summary(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for column, mapping in CATEGORY_RECODES.items():
        recoded = df[column].replace(mapping)
        rows.append({
            "column": column,
            "mapping": {str(k): v for k, v in mapping.items()},
            "rows_changed": int((df[column] != recoded).sum()),
            "raw_counts": _counts(df[column]),
            "recoded_counts": _counts(recoded),
            "modeling_decision": "Values are grouped into the documented 'others' category before modeling.",
        })
    return rows


def _schema_naming_audit(csv_df: pd.DataFrame, xls_path: Path | None) -> dict[str, Any]:
    audit = {
        "csv_repayment_first_column": "PAY_1" if "PAY_1" in csv_df.columns else None,
        "csv_contains_pay_0": "PAY_0" in csv_df.columns,
        "decision": "The converted CSV uses PAY_1 for the most recent repayment-status column; the UCI Excel header uses PAY_0.",
    }
    if xls_path is None or not xls_path.exists():
        audit["source_excel_comparison"] = {"available": False, "reason": "Source XLS file not found."}
        return audit

    try:
        xls_df = pd.read_excel(xls_path, header=1)
    except Exception as exc:  # Keep the audit usable even if the optional XLS engine is missing.
        audit["source_excel_comparison"] = {
            "available": False,
            "reason": f"Could not read XLS file: {type(exc).__name__}: {exc}",
        }
        return audit

    rename_map = {}
    if "PAY_0" in xls_df.columns and "PAY_1" in csv_df.columns:
        rename_map["PAY_0"] = "PAY_1"
    if "default payment next month" in xls_df.columns and TARGET in csv_df.columns:
        rename_map["default payment next month"] = TARGET

    normalized_xls = xls_df.rename(columns=rename_map)
    comparable = list(csv_df.columns) == list(normalized_xls.columns)
    cell_differences = None
    if comparable and csv_df.shape == normalized_xls.shape:
        cell_differences = int((csv_df.reset_index(drop=True) != normalized_xls.reset_index(drop=True)).to_numpy().sum())

    audit["source_excel_comparison"] = {
        "available": True,
        "xls_shape": [int(xls_df.shape[0]), int(xls_df.shape[1])],
        "csv_shape": [int(csv_df.shape[0]), int(csv_df.shape[1])],
        "rename_map_checked": rename_map,
        "columns_match_after_rename": bool(comparable),
        "cell_differences_after_rename": cell_differences,
    }
    return audit


def _numeric_ranges(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for column in NUMERIC_COLS + ORDINAL_COLS:
        if column in df.columns:
            rows.append({
                "column": column,
                "min": _json_value(df[column].min()),
                "max": _json_value(df[column].max()),
            })
    return rows


def build_audit(csv_path: str | Path = DATA_PATH, xls_path: str | Path | None = XLS_PATH) -> dict[str, Any]:
    csv_path = Path(csv_path)
    xls_path = Path(xls_path) if xls_path is not None else None
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in DROP_COLS + [TARGET]]
    try:
        csv_path_for_artifact = csv_path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        csv_path_for_artifact = csv_path.name

    return {
        "dataset": {
            "csv_path": csv_path_for_artifact,
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "feature_count_excluding_id_and_target": len(feature_cols),
            "target_column": TARGET,
        },
        "schema": _schema(df),
        "target_distribution": _counts(df[TARGET]),
        "missing_values": _missing_values(df),
        "identifier_integrity": _identifier_integrity(df),
        "duplicates": _duplicates(df),
        "categorical_code_audit": _categorical_audit(df),
        "repayment_status_codes": _repayment_status_audit(df),
        "recoding_summary": _recoding_summary(df),
        "schema_naming": _schema_naming_audit(df, xls_path),
        "numeric_ranges": _numeric_ranges(df),
        "model_ready_traceability": [
            "Raw CSV is loaded first.",
            "EDUCATION values 0, 5, and 6 are recoded to 4 (others).",
            "MARRIAGE value 0 is recoded to 3 (others).",
            "ID is dropped before modeling.",
            "SEX, EDUCATION, and MARRIAGE are one-hot encoded for classification.",
            "PAY_1 through PAY_6 are treated as ordinal numeric repayment-status features.",
        ],
    }


def main() -> None:
    audit = build_audit(DATA_PATH, XLS_PATH)
    output_path = ARTIFACTS / "data_quality_audit.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    print(f"Rows: {audit['dataset']['shape'][0]:,}")
    print(f"Columns: {audit['dataset']['shape'][1]}")
    print(f"Target distribution: {audit['target_distribution']}")
    print(f"Missing values: {audit['missing_values']['total_missing_values']}")
    print(f"Exact duplicate rows: {audit['duplicates']['exact_duplicate_rows']}")
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
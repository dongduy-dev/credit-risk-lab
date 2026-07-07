"""
feature_selection.py — BAI 2 (3 diem)

Phuong phap Feature Selection dua tren Correlation, dung dung module
sklearn.feature_selection (link giang vien cung cap).

Y tuong:
  - Tinh correlation cua tung feature voi target -> chon ra cac feature tuong quan manh.
  - Thu nghiem nhieu tap feature (top-5, top-10, top-15, all) voi LinearRegression.
  - So sanh bang do do MAE (Mean Absolute Error).

Luu y ve BAI TOAN: target DEFAULT la nhi phan (0/1). De bai yeu cau LinearRegression + MAE,
nen o day ta dung LinearRegression du doan gia tri lien tuc xap xi nhan 0/1 va do MAE.
Muc dich la MINH HOA anh huong cua feature selection (tap feature tot -> MAE thap hon),
KHONG phai dung LinearRegression lam model phan loai chinh thuc (do la viec cua Bai 1).

Chay:  python src/feature_selection.py
Output: artifacts/feature_selection_results.json  +  artifacts/correlation.csv + artifacts/correlation_matrix.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, r_regression

sys.path.append(str(Path(__file__).resolve().parent))
from preprocessing import load_data, TARGET, DROP_COLS

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "credit_card_default.csv"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def main():
    df = load_data(str(DATA_PATH))

    # Bo cot ID va target. De phuong phap correlation don gian & nhat quan,
    # ta lam viec tren toan bo cot dang so (SEX/EDUCATION/MARRIAGE cung la ma so
    # -> coi nhu numeric cho phan tinh tuong quan tuyen tinh).
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns] + [TARGET])
    y = df[TARGET]
    feature_names = X.columns.tolist()

    # Chia train/test truoc khi tinh correlation. Test set chi dung mot lan de
    # danh gia MAE cuoi cung, khong dung de xep hang hay chon feature.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------------------------------------------------------
    # (a) Phuong phap dua tren Correlation
    # r_regression tra ve he so tuong quan Pearson giua tung feature va target.
    # Gia tri nam trong [-1, 1]; |r| cang lon -> tuong quan tuyen tinh cang manh.
    # Chi fit/tinh tren train set de tranh leakage tu test set.
    # ---------------------------------------------------------------
    corr_scores = r_regression(X_train, y_train)
    corr_df = (
        pd.DataFrame({"feature": feature_names, "corr_with_target": corr_scores})
        .assign(abs_corr=lambda d: d["corr_with_target"].abs())
        .sort_values("abs_corr", ascending=False)
        .reset_index(drop=True)
    )
    corr_df.to_csv(ARTIFACTS / "correlation.csv", index=False)

    corr_matrix_df = X_train.copy()
    corr_matrix_df[TARGET] = y_train
    corr_matrix_df.corr(numeric_only=True).to_csv(ARTIFACTS / "correlation_matrix.csv")

    print("Top 5 feature tuong quan manh nhat voi target (training set only):")
    print(corr_df.head(5).to_string(index=False))
    print()

    # ---------------------------------------------------------------
    # (b) Thu nghiem cac tap feature khac nhau -> so sanh MAE
    # Dung SelectKBest voi score_func=f_regression (dua tren tuong quan tuyen tinh)
    # de chon ra k feature tot nhat mot cach tu dong theo API cua sklearn.
    # Selector fit tren train set; test set chi dung de tinh MAE.
    # ---------------------------------------------------------------
    k_values = [5, 10, 15, X.shape[1]]  # X.shape[1] = dung toan bo feature
    results = []

    for k in k_values:
        # 1) Chon k feature tot nhat theo f_regression (correlation-based)
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X_train, y_train)
        selected = [feature_names[i] for i in selector.get_support(indices=True)]

        X_tr_sel = selector.transform(X_train)
        X_te_sel = selector.transform(X_test)

        # 2) Scale (LinearRegression on dinh hon khi feature cung thang do)
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_sel)
        X_te_sc = scaler.transform(X_te_sel)

        # 3) Train LinearRegression + do MAE tren held-out test set
        lr = LinearRegression()
        lr.fit(X_tr_sc, y_train)
        y_pred = lr.predict(X_te_sc)
        mae = mean_absolute_error(y_test, y_pred)

        label = "all" if k == X.shape[1] else f"top_{k}"
        results.append({
            "feature_set": label,
            "num_features": int(k),
            "mae": round(float(mae), 5),
            "selected_features": selected,
        })
        print(f"{label:8s} (k={k:2d}) -> MAE = {mae:.5f}")

    best = min(results, key=lambda r: r["mae"])
    payload = {
        "correlation_source": "training_set",
        "test_set_role": "Final held-out test set is used only for MAE evaluation.",
        "correlation_ranking": corr_df.to_dict(orient="records"),
        "experiments": results,
        "best_feature_set": best["feature_set"],
        "note": ("Target nhi phan 0/1; LinearRegression + MAE dung de minh hoa "
                 "anh huong cua feature selection, khong phai model phan loai chinh."),
    }
    with open(ARTIFACTS / "feature_selection_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nTap feature cho MAE thap nhat: {best['feature_set']} (MAE={best['mae']})")
    print(f"Saved -> {ARTIFACTS/'feature_selection_results.json'}")
    print(f"Saved -> {ARTIFACTS/'correlation.csv'}")
    print(f"Saved -> {ARTIFACTS/'correlation_matrix.csv'}")

if __name__ == "__main__":
    main()

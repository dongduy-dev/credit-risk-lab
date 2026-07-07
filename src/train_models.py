"""
train_models.py — BAI 1 (5 diem)

Huan luyen va so sanh 3 model phan loai tren dataset Credit Card Default:
  1. Logistic Regression  (ho tuyen tinh)
  2. Random Forest        (ho bagging - nhieu cay quyet dinh)
  3. Gradient Boosting    (ho boosting - cay xay dung tuan tu)

Model selection dung validation set rieng; final test set chi dung mot lan de bao cao
ket qua khach quan. So sanh cac do do: accuracy, precision, recall, f1-score
(tung class + weighted avg), thoi gian training va thoi gian testing.

Chay:  python src/train_models.py
Output: artifacts/model_comparison.json  +  artifacts/best_model.ckpt
"""
from __future__ import annotations

import json
import time
import sys
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

# cho phep import src/preprocessing.py du chay tu bat ky dau
sys.path.append(str(Path(__file__).resolve().parent))
from preprocessing import load_data, split_X_y, build_pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "credit_card_default.csv"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def get_models() -> dict:
    """Khai bao 3 model.

    class_weight='balanced' cho LogReg & RandomForest: dataset mat can bang
    (~3.5 mau class 0 tren 1 mau class 1). Neu khong xu ly, model se thien ve
    du doan het la 'khong vo no' -> recall class 1 (nhom quan trong nhat) rat thap.
    GradientBoosting cua sklearn khong ho tro class_weight nen de mac dinh.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


def main():
    # 1) Doc + lam sach du lieu
    df = load_data(str(DATA_PATH))
    X, y = split_X_y(df)

    # 2) Chia development/test 80/20. Final test set chi dung de bao cao ket qua
    #    cuoi cung, khong dung de chon model.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3) Chia development thanh train/validation. Validation set dung de chon
    #    model tot nhat ma khong nhin vao final test set.
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, random_state=42, stratify=y_dev
    )

    validation_scores = {}
    best_f1 = -1.0
    best_name = None

    for name, estimator in get_models().items():
        pipe = build_pipeline(estimator)
        pipe.fit(X_train, y_train)
        y_val_pred = pipe.predict(X_val)
        val_report = classification_report(
            y_val, y_val_pred, output_dict=True, zero_division=0
        )
        val_wf1 = val_report["weighted avg"]["f1-score"]
        validation_scores[name] = round(val_wf1, 4)

        if val_wf1 > best_f1:
            best_f1, best_name = val_wf1, name

    print("Model selection metric: validation weighted F1")
    for name, score in validation_scores.items():
        print(f"{name:22s} | validation_weighted_f1={score:.4f}")
    print()

    results = {}
    best_pipeline = None

    # 4) Sau khi chon model bang validation, train lai tung model tren toan bo
    #    development set va danh gia mot lan tren final test set.
    for name, estimator in get_models().items():
        pipe = build_pipeline(estimator)

        # --- do thoi gian training ---
        t0 = time.perf_counter()
        pipe.fit(X_dev, y_dev)
        train_time = time.perf_counter() - t0

        # --- do thoi gian testing (predict tren final test set) ---
        t0 = time.perf_counter()
        y_pred = pipe.predict(X_test)
        test_time = time.perf_counter() - t0

        # classification_report tra ve san precision/recall/f1 cho tung class
        # + macro avg + weighted avg. Dung ham co san cua sklearn, khong tu tinh.
        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        acc = accuracy_score(y_test, y_pred)

        results[name] = {
            "accuracy": round(acc, 4),
            "train_time_sec": round(train_time, 4),
            "test_time_sec": round(test_time, 4),
            "per_class": {
                "class_0_no_default": {
                    "precision": round(report["0"]["precision"], 4),
                    "recall": round(report["0"]["recall"], 4),
                    "f1": round(report["0"]["f1-score"], 4),
                },
                "class_1_default": {
                    "precision": round(report["1"]["precision"], 4),
                    "recall": round(report["1"]["recall"], 4),
                    "f1": round(report["1"]["f1-score"], 4),
                },
            },
            "weighted_avg": {
                "precision": round(report["weighted avg"]["precision"], 4),
                "recall": round(report["weighted avg"]["recall"], 4),
                "f1": round(report["weighted avg"]["f1-score"], 4),
            },
            "selection": {
                "validation_weighted_f1": validation_scores[name],
            },
        }

        wf1 = report["weighted avg"]["f1-score"]
        print(f"{name:22s} | test_acc={acc:.4f} | test_weighted_f1={wf1:.4f} "
              f"| train={train_time:.2f}s | test={test_time:.3f}s")

        if name == best_name:
            best_pipeline = pipe

    # 5) Luu ket qua so sanh final test + model duoc chon bang validation.
    payload = {
        "best_model": best_name,
        "selection": {
            "metric": "weighted_f1",
            "selection_set": "validation",
            "validation_weighted_f1": validation_scores,
            "final_test_note": "Final test set is reserved for unbiased evaluation and is not used for model selection.",
        },
        "results": results,
    }
    with open(ARTIFACTS / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    joblib.dump(best_pipeline, ARTIFACTS / "best_model.ckpt")
    print(f"\nBest model: {best_name} (validation_weighted_f1={best_f1:.4f})")
    print(f"Saved -> {ARTIFACTS/'model_comparison.json'}")
    print(f"Saved -> {ARTIFACTS/'best_model.ckpt'}")


if __name__ == "__main__":
    main()

"""
preprocessing.py — Single source of truth cho toàn bộ project.

Cả train_models.py, feature_selection.py va Home.py (app predict) DEU import tu day.
Muc dich: dam bao data khi train va khi predict duoc xu ly GIONG HET nhau.
Neu train mot kieu, predict mot kieu -> model predict sai ma khong bao loi.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

# ------------------------------------------------------------------
# Dinh nghia cot theo dung ban chat du lieu (theo tai lieu UCI #350)
# ------------------------------------------------------------------
TARGET = "DEFAULT"          # nhan: 1 = vo no thang toi, 0 = khong
DROP_COLS = ["ID"]          # ID chi la so thu tu, khong mang thong tin

# Cac gia tri categorical khong ro trong raw data duoc gom vao nhom "others".
# Khai bao rieng de data audit va preprocessing cung noi mot su that.
CATEGORY_RECODES = {
    "EDUCATION": {0: 4, 5: 4, 6: 4},
    "MARRIAGE": {0: 3},
}

# Numerical thuc su (lien tuc): so tien, tuoi
NUMERIC_COLS = [
    "LIMIT_BAL", "AGE",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

# Categorical: gioi tinh, hoc van, hon nhan (dang ma so, khong co thu tu that su)
CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE"]

# Ordinal: trang thai tra no PAY_1..PAY_6 (-2..8) — CO thu tu (cang cao = tre cang nhieu)
# De don gian va nhat quan, ta coi PAY_* nhu numeric (dua thang vao scaler),
# vi gia tri cua no da co y nghia thu tu -> scaler giu duoc quan he lon/nho.
ORDINAL_COLS = ["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]


def load_data(csv_path: str) -> pd.DataFrame:
    """Doc CSV va lam sach cac gia tri categorical bat thuong (documented issue cua dataset).

    - EDUCATION: tai lieu goc chi dinh nghia 1,2,3,4. Cac gia tri 0,5,6 khong ro
      -> gom het vao nhom 4 ('others').
    - MARRIAGE: tai lieu goc dinh nghia 1,2,3. Gia tri 0 khong ro -> gom vao 3 ('others').
    """
    df = pd.read_csv(csv_path)

    # Gom cac gia tri la ve nhom 'others' de tranh categorical rac
    for col, mapping in CATEGORY_RECODES.items():
        df[col] = df[col].replace(mapping)

    return df


def split_X_y(df: pd.DataFrame):
    """Tach features (X) va target (y). Bo cot ID."""
    cols_to_drop = [c for c in DROP_COLS if c in df.columns] + [TARGET]
    X = df.drop(columns=cols_to_drop)
    y = df[TARGET]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Tao ColumnTransformer:

    - StandardScaler cho cot numeric + ordinal (dua ve mean=0, std=1).
      Ly do: cac model nhu Logistic Regression rat nhay voi thang do (scale)
      cua feature; neu khong scale, LIMIT_BAL (hang chuc nghin) se lan at AGE.
    - OneHotEncoder cho cot categorical: bien moi gia tri thanh 1 cot 0/1.
      handle_unknown='ignore' -> luc predict gap gia tri la thi khong crash,
      chi tra ve vector toan 0 cho cot do.
    """
    numeric_all = NUMERIC_COLS + ORDINAL_COLS

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_all),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ],
        remainder="drop",  # cot nao khong khai bao thi bo -> tranh lot ID vao model
    )
    return preprocessor


def build_pipeline(estimator) -> Pipeline:
    """Ghep preprocessor + model thanh 1 Pipeline.

    Nho Pipeline: khi goi .fit(X_train) thi scaler/encoder chi hoc tham so tu
    train set; khi .predict(X_new) thi tu dong ap dung dung tham so do
    -> khong bi data leakage, va app chi can goi .predict() la xong.
    """
    return Pipeline(steps=[
        ("preprocess", build_preprocessor()),
        ("model", estimator),
    ])

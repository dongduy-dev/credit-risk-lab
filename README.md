# Credit Card Default Prediction — Dự án giữa kỳ Nhập môn Học máy

Dataset: **Default of Credit Card Clients** (UCI #350) — 30,000 khách hàng thẻ tín dụng Đài Loan.
Bài toán: **binary classification** — dự đoán khách hàng có vỡ nợ tháng tới hay không.

## Cấu trúc

```
project/
├── data/
│   └── credit_card_default.csv     # dữ liệu (đã convert từ .xls)
├── artifacts/                       # output sinh ra sau khi train
│   ├── model_comparison.json        # kết quả Bài 1
│   ├── feature_selection_results.json  # kết quả Bài 2
│   ├── correlation.csv
│   └── best_model.ckpt              # model tốt nhất cho demo
├── src/
│   ├── preprocessing.py             # xử lý dữ liệu dùng chung (train + app)
│   ├── train_models.py              # Bài 1: train & so sánh 3 model
│   └── feature_selection.py         # Bài 2: feature selection
├── app.py                           # entry point Streamlit
├── Home.py                          # Bài 3: demo predict
├── ModelComparison.py               # hiển thị kết quả Bài 1
├── FeatureSelection.py              # hiển thị kết quả Bài 2
├── Statistics.py                    # thống kê dữ liệu (EDA)
├── AboutMe.py                       # thông tin nhóm
├── styles.css / logo.png
└── requirements.txt
```

## Cách chạy

```bash
# 1. Cài thư viện
pip install -r requirements.txt

# 2. Train models (Bài 1) — sinh artifacts/model_comparison.json + best_model.ckpt
python src/train_models.py

# 3. Feature selection (Bài 2) — sinh artifacts/feature_selection_results.json
python src/feature_selection.py

# 4. Chạy app demo (Bài 3)
streamlit run app.py
```

**Lưu ý**: phải chạy bước 2 và 3 trước khi mở app, vì các trang Model Comparison,
Feature Selection và Home (predict) đều đọc kết quả từ thư mục `artifacts/`.

## Nội dung 3 bài

- **Bài 1**: Logistic Regression + Random Forest + Gradient Boosting. So sánh accuracy,
  precision/recall/f1 (từng class + weighted avg), thời gian train/test. Xử lý mất cân bằng
  dữ liệu bằng `class_weight='balanced'` và `stratify` khi chia train/test.
- **Bài 2**: Feature selection dựa trên correlation (`r_regression`, `SelectKBest` +
  `f_regression` từ `sklearn.feature_selection`). So sánh MAE giữa các tập feature.
- **Bài 3**: Ứng dụng Streamlit demo dự đoán, tái sử dụng đúng pipeline preprocessing
  của lúc train để tránh lệch dữ liệu.

# Dự đoán vỡ nợ thẻ tín dụng (Credit Card Default Prediction)

Mục tiêu: dự đoán liệu một khách hàng thẻ tín dụng có vỡ nợ trong kỳ thanh toán tiếp theo hay không.
Sử dụng bộ dữ liệu UCI **Default of Credit Card Clients** (Dataset ID 350). 

## Cấu trúc dự án

```text
credit-risk-lab/
|-- data/
|   |-- credit_card_default.csv
|   `-- default_of_credit_card_clients.xls
|-- artifacts/
|   |-- best_model.ckpt
|   |-- model_comparison.json
|   |-- feature_selection_results.json
|   |-- correlation.csv
|   |-- correlation_matrix.csv
|   `-- data_quality_audit.json
|-- src/
|   |-- preprocessing.py
|   |-- train_models.py
|   |-- feature_selection.py
|   `-- data_audit.py
|-- app.py
|-- Home.py
|-- ModelComparison.py
|-- FeatureSelection.py
|-- Statistics.py
|-- AboutMe.py
|-- styles.css
|-- logo.png
|-- requirements.txt
`-- README.md
```

## Dataset

Nguồn: UCI Machine Learning Repository, **Default of Credit Card Clients**.

Thông tin tổng quan:

| Thông tin | Giá trị |
|---|---:|
| Số dòng | 30,000 |
| Số cột gốc | 25 |
| Số feature dùng để dự đoán | 23 |
| Cột target | `DEFAULT` |
| Số mẫu class 0 (không vỡ nợ) | 23,364 |
| Tỷ lệ class 0 | 77.88% |
| Số mẫu class 1 (vỡ nợ) | 6,636 |
| Tỷ lệ class 1 | 22.12% |
| Missing values | 0 |
| Số `ID` duy nhất | 30,000 |
| Dòng trùng lặp hoàn toàn | 0 |
| Dòng trùng (bỏ qua `ID`) | 35 |
| Dòng trùng feature (bỏ qua `ID` và target) | 56 |

Lưu ý về schema: File Excel gốc đặt tên cột trạng thái thanh toán gần nhất là `PAY_0`; file CSV trong dự án đổi thành `PAY_1`. Quá trình audit so sánh XLS và CSV sau khi rename `PAY_0 → PAY_1` và `default payment next month → DEFAULT`, kết quả: hai file khớp hoàn toàn, không có cell nào khác biệt.

## Quyết định tiền xử lý (Preprocessing)

Logic tiền xử lý dùng chung được định nghĩa trong `src/preprocessing.py`, được tái sử dụng cho cả quá trình training lẫn demo dự đoán trên Streamlit.

- `ID` bị loại bỏ trước khi train vì đây là định danh, không mang thông tin dự đoán.
- `DEFAULT` là target nhị phân: `0 = không vỡ nợ`, `1 = vỡ nợ`.
- `SEX`, `EDUCATION`, `MARRIAGE` là biến categorical, được one-hot encode.
- `LIMIT_BAL`, `AGE`, các cột bill amount, payment amount, và repayment status được scale bằng StandardScaler.
- Các biến repayment status được coi là feature dạng ordinal (có thứ tự).
- Giá trị `EDUCATION` = `0`, `5`, `6` được gom vào nhóm `4` (`others`).
- Giá trị `MARRIAGE` = `0` được gom vào nhóm `3` (`others`).
- Các dòng trùng lặp được ghi nhận để minh bạch, nhưng không bị xóa.

App Streamlit load `artifacts/best_model.ckpt`, đây là một scikit-learn `Pipeline` đầy đủ, bao gồm cả preprocessing và classifier.

## Thiết kế chống Data Leakage

### Classification

`src/train_models.py` sử dụng thiết kế đánh giá 3 bước:

1. Chia dataset thành tập development và tập final test (có stratify theo label).
2. Chia tập development thành train và validation.
3. Chọn model tốt nhất dựa trên weighted F1 trên tập validation.
4. Refit mỗi model trên toàn bộ tập development.
5. Đánh giá duy nhất một lần trên tập final test.

Tập final test không được sử dụng để chọn model.

### Feature Selection dựa trên Correlation

`src/feature_selection.py` chia train/test trước khi tính correlation. Toàn bộ quá trình xếp hạng correlation, chọn feature, và scaling chỉ được fit trên tập train. Tập test chỉ dùng để đánh giá MAE cuối cùng cho các feature subset đã chọn.

## Kết quả Classification

Model được lưu: Gradient Boosting, chọn theo validation weighted F1.

Majority-class baseline: accuracy đạt `0.7788`, nhưng không phát hiện được bất kỳ khách hàng vỡ nợ nào, bỏ sót toàn bộ 1,327 khách hàng vở nợ trong tập test.

| Model | Accuracy | Weighted F1 | Default Precision | Default Recall | Default F1 | Bỏ sót | False Positives | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6788 | 0.7030 | 0.3681 | 0.6307 | 0.4649 | 490 | 1,437 | 0.7104 | 0.4913 |
| Random Forest | 0.8152 | 0.7927 | 0.6544 | 0.3482 | 0.4545 | 865 | 244 | 0.7613 | 0.5380 |
| Gradient Boosting | 0.8193 | 0.7983 | 0.6676 | 0.3647 | 0.4717 | 843 | 241 | 0.7780 | 0.5502 |

Nhận xét: Gradient Boosting được chọn làm model chính vì có validation weighted F1 cao nhất, đồng thời đạt Default F1 tốt nhất trên tập test. Logistic Regression phát hiện nhiều khách hàng vở nợ hơn nhưng đi kèm số false positive rất cao. Do đó, model tốt nhất còn phụ thuộc vào mục tiêu quản lý rủi ro tín dụng cụ thể.

## Phân tích đánh đổi ngưỡng quyết định (Decision Threshold)

Phần so sánh model ở trên sử dụng ngưỡng mặc định (0.5) và là kết quả chính thức của đồ án. Phần này là phân tích bổ sung: thử nghiệm các ngưỡng khác nhau trên Gradient Boosting để quan sát sự đánh đổi giữa recall và precision. Các ngưỡng chỉ được đánh giá trên tập validation, tập final test không tham gia vào quá trình chọn ngưỡng. Demo Streamlit vẫn sử dụng pipeline đã refit với ngưỡng mặc định.

| Ngưỡng | Default Precision | Default Recall | Default F1 | Phát hiện đúng | Bỏ sót | False Positives |
|---|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.4414 | 0.6473 | 0.5249 | 859 | 468 | 1,087 |
| 0.30 | 0.5415 | 0.5207 | 0.5309 | 691 | 636 | 585 |
| 0.40 | 0.6413 | 0.4190 | 0.5068 | 556 | 771 | 311 |
| 0.50 | 0.6870 | 0.3572 | 0.4700 | 474 | 853 | 216 |
| 0.60 | 0.7079 | 0.2977 | 0.4191 | 395 | 932 | 163 |
| 0.70 | 0.7273 | 0.1688 | 0.2740 | 224 | 1,103 | 84 |
| 0.80 | 0.7241 | 0.0158 | 0.0310 | 21 | 1,306 | 8 |

Nhận xét: Hạ ngưỡng giúp phát hiện nhiều khách hàng vỡ nợ hơn, nhưng false positive tăng mạnh. Nâng ngưỡng giúp giảm false positive, nhưng bỏ sót nhiều khách hàng vỡ nợ hơn. Không có ngưỡng nào là tối ưu tuyệt đối, việc chọn ngưỡng phụ thuộc vào business cost, năng lực xử lý hồ sơ, và chính sách rủi ro tín dụng.

## Kết quả Feature Selection (Correlation-based)

Theo yêu cầu đồ án, phần này sử dụng Linear Regression + MAE để đánh giá các feature subset. Phần này tách biệt với bài toán classification. Target 0/1 được xem như biến số liên tục chỉ nhằm mục đích so sánh giữa các subset.

| Feature set | Số feature | MAE |
|---|---:|---:|
| top_5 | 5 | 0.31201 |
| top_10 | 10 | 0.31025 |
| top_15 | 15 | 0.31001 |
| all | 23 | 0.30661 |

Kết luận: Dùng toàn bộ feature cho MAE thấp nhất trên held-out test. Subset tốt nhất qua correlation là `top_15`, nhưng vẫn không vượt qua full feature set. Các subset dựa trên correlation hữu ích cho việc phân tích và so sánh, nhưng không nên phóng đại hiệu quả của chúng.

Hạn chế phương pháp: Pearson correlation coi các biến categorical đã encode (`SEX`, `EDUCATION`, `MARRIAGE`) như biến số liên tục, điều này không hoàn toàn chính xác. Hạn chế này được ghi nhận trên trang Feature Selection của Streamlit.

## Demo Streamlit

Chạy ứng dụng:

```bash
streamlit run app.py
```

Các trang:

- `Home` — Form dự đoán, sử dụng pipeline artifact đã lưu.
- `Model Comparison` — Classification metrics, so sánh với baseline, hiệu năng trên class default, số khách hàng vở nợ bị bỏ sót, trade-off false positive / false negative, và phân tích ngưỡng trên validation.
- `Feature Selection` — Correlation ranking trên tập train, heatmap, các feature subset đã chọn, và so sánh MAE trên held-out test.
- `Statistics` — Tổng quan dataset, phân bố target, audit dữ liệu thô, các mã bất thường, quyết định recode, phân bố repayment code, và khám phá biến số.
- `About Us` — Thông tin nhóm và dataset.

Lưu ý: Giá trị `predict_proba` hiển thị trên app là model score cho class default, không phải xác suất vỡ nợ đã được calibrate cho ứng dụng tài chính thực tế.

## Hướng dẫn chạy lại dự án (Reproducibility)

Yêu cầu: Python 3.10 trở lên.

Từ thư mục gốc dự án:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Tạo lại artifacts theo thứ tự:

```bash
python -B src\data_audit.py
python -B src\train_models.py
python -B src\feature_selection.py
streamlit run app.py
```

Flag `-B` ngăn Python tạo bytecode cache (file `.pyc`), giúp đảm bảo tính reproducible.

## Artifacts

| Artifact | Tạo bởi | Mục đích |
|---|---|---|
| `artifacts/data_quality_audit.json` | `src/data_audit.py` | Thông tin chất lượng dữ liệu thô, truy vết preprocessing |
| `artifacts/model_comparison.json` | `src/train_models.py` | Classification metrics, baseline, model selection, phân tích rủi ro, phân tích ngưỡng |
| `artifacts/best_model.ckpt` | `src/train_models.py` | Pipeline hoàn chỉnh (preprocessing + classifier) cho Streamlit |
| `artifacts/feature_selection_results.json` | `src/feature_selection.py` | Feature subset và kết quả MAE |
| `artifacts/correlation.csv` | `src/feature_selection.py` | Correlation ranking (feature - target) trên tập train |
| `artifacts/correlation_matrix.csv` | `src/feature_selection.py` | Ma trận tương quan cho heatmap, tính trên tập train |

## Hạn chế

- Dataset bị mất căn bằng, accuracy và weighted F1 có thể giấu đi việc model yếu trong phát hiện khách hàng vở nợ.
- Gradient Boosting vẫn bỏ sót nhiều khách hàng vở nợ ở ngưỡng mặc định.
- Probability từ classifier chưa được calibrate, không nên dùng trực tiếp cho quyết định tài chính thực tế.
- Feature selection bằng correlation là phương pháp univariate, chỉ đo quan hệ tuyến tính, không bắt được interaction hay nonlinear effect.
- Pearson correlation trên biến categorical (đã encode thành số) chỉ mang tính mô tả theo yêu cầu đồ án, không nên diễn giải theo hướng nhân quả hay thứ bậc.
- Hyperparameter tuning được giữ ở mức tối thiểu, đồ án ưu tiên so sánh và diễn giải đúng hơn là tối ưu chỉ số accuracy.

# Credit Risk Lab - Credit Card Default Prediction

University machine-learning midterm project for the UCI **Default of Credit Card Clients** dataset (Dataset ID 350). The goal is to predict whether a credit-card client will default on the next payment.

The project is submission-ready as a reproducible classification experiment, a correlation-based feature-selection experiment, a raw-data quality audit, and a Streamlit demo.

## Project Contents

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

Source: UCI Machine Learning Repository, **Default of Credit Card Clients**.

Current verified dataset facts:

| Fact | Value |
|---|---:|
| Rows | 30,000 |
| Raw columns | 25 |
| Predictive features | 23 |
| Target column | `DEFAULT` |
| Class 0 count | 23,364 |
| Class 0 percent | 77.88% |
| Class 1 count | 6,636 |
| Class 1 percent | 22.12% |
| Missing values | 0 |
| Unique IDs | 30,000 |
| Exact duplicate rows | 0 |
| Duplicate rows excluding ID | 35 |
| Duplicate feature rows excluding ID and target | 56 |

Important schema note: the source Excel file names the most recent repayment-status column `PAY_0`; this project's converted CSV uses `PAY_1`. The audit compares the source XLS and CSV after renaming `PAY_0 -> PAY_1` and `default payment next month -> DEFAULT`; the files match with 0 cell differences.

## Preprocessing Decisions

Shared preprocessing is defined in `src/preprocessing.py` and is reused by training and the Streamlit prediction demo.

- `ID` is dropped before modeling because it is an identifier.
- `DEFAULT` is the binary target, where `0 = no default` and `1 = default`.
- `SEX`, `EDUCATION`, and `MARRIAGE` are categorical codes and are one-hot encoded for classification.
- `LIMIT_BAL`, `AGE`, bill amounts, previous payment amounts, and repayment-status columns are scaled.
- Repayment-status variables are treated as ordinal numeric features.
- Raw `EDUCATION` values `0`, `5`, and `6` are grouped into `4` (`others`) before modeling.
- Raw `MARRIAGE` value `0` is grouped into `3` (`others`) before modeling.
- Duplicate observations are reported for transparency but are not removed.

The Streamlit app loads `artifacts/best_model.ckpt`, which is a full scikit-learn `Pipeline` containing both preprocessing and the selected classifier.

## Leakage-safe Experiment Design

### Classification

`src/train_models.py` uses a three-way evaluation design:

1. Split the dataset into development and final test sets with stratification.
2. Split the development set into train and validation sets.
3. Select the saved model using validation weighted F1.
4. Refit each model on the full development set.
5. Evaluate once on the final held-out test set.

The final test set is not used for model selection.

### Correlation Feature Selection

`src/feature_selection.py` splits train/test before calculating correlations. Correlation ranking, feature selection, and scaling are fit on the training split only. The held-out test split is used only for final MAE evaluation of the selected subsets.

## Classification Results

Current saved model: **Gradient Boosting**, selected by validation weighted F1.

Majority-class baseline: accuracy `0.7788`, but it catches 0 actual defaulters and misses all 1,327 defaulters in the test set.

| Model | Accuracy | Weighted F1 | Default precision | Default recall | Default F1 | Missed defaulters | False positives | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6788 | 0.7030 | 0.3681 | 0.6307 | 0.4649 | 490 | 1,437 | 0.7104 | 0.4913 |
| Random Forest | 0.8152 | 0.7927 | 0.6544 | 0.3482 | 0.4545 | 865 | 244 | 0.7613 | 0.5380 |
| Gradient Boosting | 0.8193 | 0.7983 | 0.6676 | 0.3647 | 0.4717 | 843 | 241 | 0.7780 | 0.5502 |

Interpretation: Gradient Boosting is the saved model because it has the best validation weighted F1 and the best held-out default F1 among the three models. Logistic Regression catches more defaulters, but it also produces many more false positives. Therefore, the best model depends on the credit-risk objective.

## Correlation-based Feature Selection Results

The assignment requires Linear Regression with MAE for the feature-selection experiment. This is separate from the classification task. Here, the 0/1 target is treated as a numeric response only to compare feature subsets.

| Feature set | Number of features | MAE |
|---|---:|---:|
| top_5 | 5 | 0.31201 |
| top_10 | 10 | 0.31025 |
| top_15 | 15 | 0.31001 |
| all | 23 | 0.30661 |

Current conclusion: the full feature set has the lowest held-out MAE. The best correlation-selected subset is `top_15`, but it does not outperform all features. Correlation-based subsets are still useful for interpretation and comparison, but the results should not be overstated.

Methodological limitation: Pearson correlation treats coded categorical variables such as `SEX`, `EDUCATION`, and `MARRIAGE` as numeric. The project reports this limitation in the Streamlit feature-selection page.

## Streamlit Demo

Run the app with:

```bash
streamlit run app.py
```

Pages:

- `Home`: prediction form using the saved full pipeline artifact.
- `Model Comparison`: classification metrics, baseline comparison, default-class performance, missed defaulters, and false-positive/false-negative trade-offs.
- `Feature Selection`: training-set correlation ranking, heatmap, selected feature subsets, and held-out MAE comparison.
- `Statistics`: dataset overview, target distribution, raw-data audit, unusual codes, recoding decisions, repayment-code counts, and numeric exploration.
- `About Us`: dataset and team information.

Prediction caveat: the app labels `predict_proba` as a default-class model score, not as a calibrated real-world probability of default.

## Reproducible Execution

Recommended environment: Python 3.10 or newer. The project was validated with Python 3.13.

From the project root:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Then regenerate artifacts in this order:

```bash
python -B src\data_audit.py
python -B src\train_models.py
python -B src\feature_selection.py
streamlit run app.py
```

The `-B` flag prevents Python bytecode cache files from being generated during reproducibility checks.

## Generated Artifacts

| Artifact | Produced by | Purpose |
|---|---|---|
| `artifacts/data_quality_audit.json` | `src/data_audit.py` | Raw dataset facts and preprocessing traceability |
| `artifacts/model_comparison.json` | `src/train_models.py` | Classification metrics, baseline, validation selection, risk interpretation |
| `artifacts/best_model.ckpt` | `src/train_models.py` | Saved full preprocessing + classifier pipeline used by Streamlit |
| `artifacts/feature_selection_results.json` | `src/feature_selection.py` | Feature subsets and MAE results |
| `artifacts/correlation.csv` | `src/feature_selection.py` | Training-set feature-to-target correlation ranking |
| `artifacts/correlation_matrix.csv` | `src/feature_selection.py` | Training-set correlation heatmap input |

## Assignment Coverage

| Requirement | Status | Evidence |
|---|---|---|
| Read UCI dataset | Complete | `data/credit_card_default.csv`; source XLS retained |
| Preprocess and normalize data | Complete | `ColumnTransformer` with `StandardScaler` and `OneHotEncoder` |
| Use at least 3 classifiers | Complete | Logistic Regression, Random Forest, Gradient Boosting |
| Report required per-class metrics | Complete | `artifacts/model_comparison.json`, Streamlit Model Comparison page |
| Report training and testing time | Complete | `train_time_sec`, `test_time_sec` in model artifact |
| Meaningful classification comparison | Complete | Baseline, default recall/F1, false positives, false negatives, risk notes |
| Correlation-based feature selection | Complete | Training-set `r_regression` ranking and `SelectKBest(f_regression)` subsets |
| Correlation visualizations | Complete | Ranking bar chart and correlation heatmap in Streamlit |
| Multiple feature subsets | Complete | top_5, top_10, top_15, all |
| Linear Regression + MAE | Complete | `src/feature_selection.py` and Streamlit Feature Selection page |
| Streamlit demo | Complete | `app.py` with pages for prediction, metrics, feature selection, statistics, and team information |

## Known Limitations

- The dataset is imbalanced, so accuracy and weighted F1 can hide weak default-class detection.
- The saved Gradient Boosting model still misses many actual defaulters at the default decision threshold.
- Probabilities shown by the classifier are not calibrated for real financial decision-making.
- Correlation feature selection is univariate and linear; it does not capture interactions or nonlinear effects.
- Pearson correlation on categorical codes is reported only as a descriptive assignment method, not as a causal or ordinal interpretation.
- Hyperparameter tuning is intentionally limited; the assignment focuses on correct comparison and interpretation rather than leaderboard performance.

## Presentation Checklist

Suggested slide flow:

1. Dataset and problem statement.
2. Raw dataset facts and data-quality findings.
3. Preprocessing decisions and leakage-safe experiment design.
4. Classification model comparison and majority baseline.
5. Credit-risk interpretation: default recall, missed defaulters, false positives.
6. Correlation feature-selection method and visualizations.
7. Linear Regression + MAE feature-subset comparison.
8. Streamlit demo walkthrough.
9. Limitations and honest conclusion.
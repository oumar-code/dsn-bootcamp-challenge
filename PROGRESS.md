# 📋 DSN Bootcamp Challenge - Progress Tracker

**Project**: Sales Forecasting for DSN Mart  
**Challenge Type**: Regression (Time-Series)  
**Evaluation Metric**: RMSE (Root Mean Squared Error)  
**Last Updated**: 2026-09-09

---

## 🎯 Milestones Overview

### ✅ Milestone 1: Setup
**Status**: IN PROGRESS

- [x] Repository structure created
- [x] README.md with project overview
- [x] Submission format documented
- [ ] Data files added to `data/` directory
  - [ ] `data/train.csv` uploaded
  - [ ] `data/test.csv` uploaded
  - [ ] `data/README.md` with data dictionary created
- [ ] `requirements.txt` finalized with dependencies
- [ ] `.gitignore` configured for data & outputs
- [ ] Virtual environment setup documented

**Subtasks**:
```
- [ ] Clone repo & create venv
- [ ] Install dependencies
- [ ] Verify data/ directory structure
- [ ] Test data loading in Python
```

**Notes**:
- Waiting for train.csv and test.csv files
- Consider creating data/README.md to document schema, features, and target variable

---

### 📊 Milestone 2: Data Exploration & Cleaning
**Status**: NOT STARTED

**Deliverables**:
- [ ] Data ingest & schema check
- [ ] EDA report (distributions, missing values, correlations)
- [ ] Data quality assessment
- [ ] Identify outliers and handle missing values
- [ ] Create `notebooks/01_EDA.ipynb`

**Subtasks**:
```
- [ ] Load train.csv and inspect shape, dtypes, missing values
- [ ] Statistical summary (describe, value_counts)
- [ ] Visualize target distribution (sales)
- [ ] Check for data leakage indicators
- [ ] Document data cleaning decisions in notebook
```

**Success Criteria**:
- Dataset fully understood (features, target, temporal range)
- Cleaned dataset ready for feature engineering
- EDA report saved with plots

**Blockers**: None (depends on Milestone 1)

---

### 🛠️ Milestone 3: Feature Engineering
**Status**: NOT STARTED

**Deliverables**:
- [ ] Feature engineering scripts in `src/features.py`
- [ ] Create `notebooks/02_Feature_Engineering.ipynb`
- [ ] Generate lag features (sales history)
- [ ] Rolling statistics (moving averages)
- [ ] Store & product aggregates
- [ ] Date-based features (month, day-of-week, seasonality)
- [ ] Target encoding for categorical variables

**Subtasks**:
```
- [ ] Time-based feature extraction (date decomposition)
- [ ] Lag features: sales_lag_1, sales_lag_7, sales_lag_30, etc.
- [ ] Rolling mean/std: rolling_mean_7, rolling_mean_30, etc.
- [ ] Store-level aggregates: store_avg_sales, store_std_sales
- [ ] Product-level aggregates: product_avg_sales, product_std_sales
- [ ] Handle NaN values in lag/rolling features (forward-fill or drop early rows)
- [ ] Feature correlation analysis
```

**Success Criteria**:
- Feature set captures temporal patterns & seasonality
- No data leakage from test set
- Features documented in notebook

**Blockers**: Milestone 2 (EDA complete)

---

### 🤖 Milestone 4: Baseline Model
**Status**: NOT STARTED

**Deliverables**:
- [ ] Baseline LightGBM model
- [ ] Time-series cross-validation setup
- [ ] Create `notebooks/03_Baseline_Model.ipynb`
- [ ] Out-of-fold (OOF) predictions
- [ ] CV RMSE score recorded

**Subtasks**:
```
- [ ] Implement TimeSeriesSplit (sklearn)
- [ ] Train LightGBM on training set
- [ ] Evaluate with cross-validation (CV RMSE)
- [ ] Generate OOF predictions for future ensembles
- [ ] Record feature importance
- [ ] Baseline predictions on test set
- [ ] Save submission: submissions/submission_v1.csv
```

**Success Criteria**:
- Time-series CV prevents data leakage
- CV RMSE recorded and documented
- Baseline score established (benchmark)
- Feature importance identified

**Blockers**: Milestone 3 (features ready)

---

### 🚀 Milestone 5: Model Improvements
**Status**: NOT STARTED

**Deliverables**:
- [ ] Target encoding for categorical features
- [ ] Hyperparameter tuning (Optuna or grid search)
- [ ] Model ensembles (LightGBM + XGBoost)
- [ ] Create `notebooks/04_Model_Improvements.ipynb`
- [ ] Improved CV RMSE score
- [ ] Intermediate submissions tracked

**Subtasks**:
```
- [ ] Implement target encoding (mean encoding per category)
- [ ] Hyperparameter tuning for LightGBM:
  - [ ] num_leaves, learning_rate, max_depth
  - [ ] min_data_in_leaf, feature_fraction, bagging_fraction
  - [ ] Use Optuna for automated tuning
- [ ] Train XGBoost as secondary model
- [ ] Ensemble strategies:
  - [ ] Weighted average of LightGBM + XGBoost
  - [ ] Stacking with meta-learner
- [ ] Compare CV RMSE with baseline
- [ ] Save submission: submissions/submission_v2.csv (or higher)
```

**Success Criteria**:
- CV RMSE improved over baseline
- Hyperparameters documented & reproducible
- Ensemble outperforms single model
- All experiments logged

**Blockers**: Milestone 4 (baseline complete)

---

### 🏁 Milestone 6: Final Training & Submission
**Status**: NOT STARTED

**Deliverables**:
- [ ] Final model trained on full training set
- [ ] Test predictions generated
- [ ] Final submission file: `submissions/submission_final.csv`
- [ ] Leaderboard entry updated in submissions/README.md
- [ ] Summary report of approach

**Subtasks**:
```
- [ ] Train final model on entire train.csv (best hyperparameters)
- [ ] Generate predictions for test.csv
- [ ] Validate submission format (ID, predicted_sales)
- [ ] Check for NaN or extreme values
- [ ] Upload to competition platform
- [ ] Record public & private RMSE scores
- [ ] Document final approach in README
```

**Success Criteria**:
- Submission accepted by platform
- Score recorded on leaderboard
- All files version-controlled in submissions/
- Final summary documented

**Blockers**: Milestone 5 (improvements complete)

---

## 📈 Leaderboard Tracking

| Version | Milestone | Date | CV RMSE | Notes |
|---------|-----------|------|---------|-------|
| v1 | 4 | — | — | Baseline LightGBM |
| v2 | 5 | — | — | With target encoding |
| v3 | 5 | — | — | Hyperparameter tuned |
| v4 | 5 | — | — | LightGBM + XGBoost ensemble |
| final | 6 | — | — | Final submission |

---

## 🔑 Key Reminders

### ⚠️ Critical Rules
- **Time-Series CV**: ALWAYS use time-based splits (train dates < validation dates)
- **No Data Leakage**: Never use test set information in training
- **RMSE Metric**: Evaluate on raw target (not log-transformed unless explicitly transformed)
- **Reproducibility**: Lock random seeds in every notebook:
  ```python
  import numpy as np
  import random
  np.random.seed(42)
  random.seed(42)
  ```

### 📝 Documentation Checklist
- [ ] Record random seeds & package versions
- [ ] Save notebook outputs (metrics, plots, feature importance)
- [ ] Document feature engineering decisions
- [ ] Log hyperparameters & model configurations
- [ ] Track submission versions & scores

### 🛠️ Tools & Libraries
- **Data**: pandas, NumPy
- **ML**: scikit-learn, LightGBM, XGBoost
- **EDA**: matplotlib, seaborn
- **Tuning**: Optuna
- **Metrics**: sklearn.metrics (mean_squared_error, mean_absolute_error)

---

## 📅 Timeline & Deadlines

**Expected Flow** (adjust based on progress):
1. **Setup**: Complete today
2. **EDA**: 1-2 days
3. **Feature Engineering**: 2-3 days
4. **Baseline & Improvements**: 3-4 days
5. **Final Training**: 1 day

**Total Estimated Time**: 1-2 weeks

---

## 🚀 Next Steps

### Immediate (Today):
1. [ ] Add `data/train.csv` and `data/test.csv` to repo
2. [ ] Create `data/README.md` with feature descriptions
3. [ ] Create `requirements.txt` with dependencies
4. [ ] Set up virtual environment locally

### Short-term (This week):
1. [ ] Complete Milestone 2 (EDA)
2. [ ] Begin Milestone 3 (Feature Engineering)

---

**Status Summary**: Setup phase in progress. Awaiting data files to begin exploration.

**Questions/Blockers**: None at this stage.


# Data Dictionary & Schema

**Dataset**: DSN Mart Historical Sales Data  
**Challenge**: Sales Forecasting (Regression)  
**Evaluation Metric**: RMSE (Root Mean Squared Error)  

---

## 📊 Dataset Overview

This directory contains the training and test datasets for the DSN Bootcamp Challenge.

### Files

- **`train.csv`** — Historical sales data with target variable (training set)
- **`test.csv`** — Test set without target variable (for predictions)

---

## 🔍 Feature Dictionary

### Target Variable
| Feature | Type | Description | Range/Notes |
|---------|------|-------------|-------------|
| `sales` | float | **Total product-store sales** (target) | Continuous, positive values |

### Store Features
| Feature | Type | Description | Notes |
|---------|------|-------------|-------|
| `store_id` | int | Unique identifier for store/outlet | Categorical |
| `store_type` | categorical | Type of store (e.g., Supermarket, Grocery) | TBD - check unique values |
| `store_location` | categorical | Geographic location/city of store | TBD - check unique values |
| `store_size` | categorical | Size category (e.g., Small, Medium, Large) | TBD - check unique values |

### Product Features
| Feature | Type | Description | Notes |
|---------|------|-------------|-------|
| `product_id` | int | Unique identifier for product/SKU | Categorical |
| `product_name` | string | Name/description of product | TBD - check for missing values |
| `product_category` | categorical | Product category (e.g., Food, Electronics) | TBD - check unique values |
| `product_subcategory` | categorical | Subcategory within product category | TBD - check unique values |
| `product_weight` | float | Weight of product (kg or lbs) | TBD - check units and missing values |
| `product_visibility` | float | Product shelf visibility (% or score) | TBD - check scale |
| `product_price` | float | Price of product (currency) | TBD - check currency and range |

### Temporal Features
| Feature | Type | Description | Notes |
|---------|------|-------------|-------|
| `date` | datetime | Date of sale | Format: YYYY-MM-DD (verify in EDA) |
| `year` | int | Year of sale | Extract from date |
| `month` | int | Month of sale | 1-12 |
| `week` | int | Week of year | 1-52/53 |
| `day_of_week` | int | Day of week | 0=Monday, 6=Sunday (or TBD) |
| `quarter` | int | Quarter of year | Q1, Q2, Q3, Q4 |

### Additional Features (if present)
| Feature | Type | Description | Notes |
|---------|------|-------------|-------|
| `promotion_active` | bool | Whether promotion is active | 0 or 1 (verify) |
| `discount_percentage` | float | Discount applied | 0-100 or 0-1 (verify) |
| `customer_count` | int | Number of customers (foot traffic) | TBD - check range |
| `is_holiday` | bool | Holiday indicator | 0 or 1 |

---

## 📈 Data Quality Checklist

**To complete during EDA (Milestone 2):**

- [ ] Shape: Number of rows and columns
- [ ] Date range: First and last dates in dataset
- [ ] Missing values: Count and percentage per column
- [ ] Data types: Verify all column types (int, float, datetime, categorical)
- [ ] Duplicates: Check for duplicate rows or transactions
- [ ] Outliers: Identify extreme values in numerical columns
- [ ] Target distribution: Skewness, range, and quartiles of `sales`
- [ ] Temporal coverage: Are there gaps in date range?
- [ ] Store/Product coverage: How many unique stores and products?

---

## 🔗 Relationships & Cardinality

- **One store** can have **multiple products** over time
- **One product** can be sold at **multiple stores**
- **One store-product pair** can have **multiple sales records** across different dates
- **Temporal order**: Sales records are time-stamped and should maintain chronological order

---

## ⚠️ Key Considerations

### Time-Series Nature
- This is a **time-series regression** problem
- **Data leakage risk**: Test set dates must NOT overlap with training dates
- Use **time-based cross-validation** (e.g., `TimeSeriesSplit`) to avoid leakage

### Feature Engineering Opportunities
1. **Lag features**: Previous sales (lag_1, lag_7, lag_30)
2. **Rolling statistics**: Moving averages, rolling std dev
3. **Store aggregates**: Average sales per store, store sales variance
4. **Product aggregates**: Average sales per product, product popularity
5. **Seasonality**: Month, day-of-week, holiday effects
6. **Trend**: Long-term sales trend per store-product pair

### Target Encoding
- Categorical features (store_type, product_category) may benefit from **mean target encoding**
- Use **cross-validation folds** to prevent leakage when creating encoded features

---

## 📝 Data Loading Example

```python
import pandas as pd

# Load data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Basic inspection
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"\nTrain dtypes:\n{train_df.dtypes}")
print(f"\nMissing values:\n{train_df.isnull().sum()}")
print(f"\nTarget (sales) stats:\n{train_df['sales'].describe()}")
```

---

## 🚀 Next Steps

1. **Load data** using the example above
2. **Inspect schema** — verify all expected columns are present
3. **Run EDA** — complete Milestone 2 (Data Exploration & Cleaning)
4. **Document findings** — update this file with actual data statistics

---

**Last Updated**: 2026-09-09  
**Status**: Awaiting data files

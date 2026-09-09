# Submissions Directory

This directory contains prediction files to submit to the competition platform.

## Submission Format

Each CSV file should have the following structure:

```
ID,predicted_sales
1,1250.50
2,980.25
3,1125.00
...
```

### Format Requirements:
- **Header Row**: `ID,predicted_sales`
- **ID Column**: Unique identifier from test set
- **Predictions Column**: Float values representing forecasted total sales
- **No Index Column**: CSV should not include row indices

## File Naming Convention

Use versioning to track submission iterations:

- `submission_v1.csv` — First baseline submission
- `submission_v2.csv` — After feature engineering
- `submission_v3.csv` — After hyperparameter tuning
- `submission_v4.csv` — Final ensemble model
- `submission_final.csv` — Final competition submission

## Before Submission

Checklist before uploading to the competition platform:

- [ ] Predictions are for all rows in test.csv
- [ ] No missing values (NaN) in predictions
- [ ] Predictions are reasonable (no extreme outliers)
- [ ] CSV format is correct (ID, predicted_sales)
- [ ] File is not corrupted and can be read
- [ ] Model was trained using proper time-series CV (no leakage)

## Leaderboard Tracking

| Version | Date | CV RMSE | Public RMSE | Private RMSE | Notes |
|---------|------|---------|------------|-------------|-------|
| v1 | YYYY-MM-DD | — | — | — | Baseline LightGBM |
| v2 | YYYY-MM-DD | — | — | — | With feature engineering |
| v3 | YYYY-MM-DD | — | — | — | Hyperparameter tuned |
| final | YYYY-MM-DD | — | — | — | Final submission |

---

**Last Updated**: 2026-09-09

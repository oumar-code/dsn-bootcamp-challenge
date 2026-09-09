"""
Utility functions for DSN Bootcamp Challenge.

Includes:
- Reproducibility & seeding
- Time-series cross-validation
- Evaluation metrics
- Data handling helpers
"""

import numpy as np
import pandas as pd
import random
from typing import Tuple, List, Optional, Union
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# REPRODUCIBILITY & SEEDING
# ============================================================================

def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    Parameters:
    -----------
    seed : int, default=42
        Seed value for random number generators
        
    Example:
    --------
    >>> set_seed(42)
    """
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def print_seed_info(seed: int = 42) -> None:
    """
    Print seed information for reproducibility documentation.
    
    Parameters:
    -----------
    seed : int, default=42
        Seed value used
    """
    print("=" * 60)
    print("REPRODUCIBILITY INFO")
    print("=" * 60)
    print(f"Random Seed: {seed}")
    print(f"NumPy Version: {np.__version__}")
    print(f"Pandas Version: {pd.__version__}")
    try:
        import sklearn
        print(f"Scikit-learn Version: {sklearn.__version__}")
    except:
        pass
    try:
        import lightgbm
        print(f"LightGBM Version: {lightgbm.__version__}")
    except:
        pass
    try:
        import xgboost
        print(f"XGBoost Version: {xgboost.__version__}")
    except:
        pass
    print("=" * 60)


# ============================================================================
# TIME-SERIES CROSS-VALIDATION
# ============================================================================

class TimeSeriesCV:
    """
    Time-series aware cross-validation splitter.
    
    Ensures training dates predate validation dates to avoid data leakage.
    
    Parameters:
    -----------
    n_splits : int, default=5
        Number of CV folds
    test_size : float or int, default=0.2
        Size of test set (fraction or number of samples)
    gap : int, default=0
        Number of samples to exclude between train and test (for buffer)
    """
    
    def __init__(self, n_splits: int = 5, test_size: Union[float, int] = 0.2, gap: int = 0):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.splitter = TimeSeriesSplit(n_splits=n_splits)
    
    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None, groups=None):
        """
        Generate train/test indices for time-series CV.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series, optional
            Target variable
        groups : array-like, optional
            Group labels (ignored for time-series)
            
        Yields:
        -------
        train_idx : np.ndarray
            Training indices
        test_idx : np.ndarray
            Test indices
        """
        for train_idx, test_idx in self.splitter.split(X):
            # Apply gap between train and test
            if self.gap > 0:
                train_idx = train_idx[:-self.gap]
            yield train_idx, test_idx


def create_time_series_folds(
    df: pd.DataFrame,
    date_column: str,
    n_splits: int = 5,
    test_fraction: float = 0.2,
    verbose: bool = True
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create time-series cross-validation folds based on date.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    date_column : str
        Name of date column
    n_splits : int, default=5
        Number of CV folds
    test_fraction : float, default=0.2
        Fraction of data to use for validation in each fold
    verbose : bool, default=True
        Print fold information
        
    Returns:
    --------
    folds : List[Tuple[pd.DataFrame, pd.DataFrame]]
        List of (train_df, val_df) tuples
        
    Example:
    --------
    >>> folds = create_time_series_folds(df, date_column='date', n_splits=5)
    >>> for train, val in folds:
    ...     print(f"Train: {train.shape}, Val: {val.shape}")
    """
    df_sorted = df.sort_values(date_column).reset_index(drop=True)
    total_samples = len(df_sorted)
    test_size = int(total_samples * test_fraction)
    
    folds = []
    
    for fold_idx in range(n_splits):
        # Calculate split point
        split_point = total_samples - (n_splits - fold_idx) * test_size
        
        train_fold = df_sorted.iloc[:split_point]
        val_fold = df_sorted.iloc[split_point:split_point + test_size]
        
        folds.append((train_fold, val_fold))
        
        if verbose:
            print(f"Fold {fold_idx + 1}:")
            print(f"  Train: {len(train_fold)} samples | "
                  f"Date range: {train_fold[date_column].min()} to {train_fold[date_column].max()}")
            print(f"  Val:   {len(val_fold)} samples | "
                  f"Date range: {val_fold[date_column].min()} to {val_fold[date_column].max()}")
    
    return folds


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Parameters:
    -----------
    y_true : np.ndarray or pd.Series
        True values
    y_pred : np.ndarray or pd.Series
        Predicted values
        
    Returns:
    --------
    float
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Parameters:
    -----------
    y_true : np.ndarray or pd.Series
        True values
    y_pred : np.ndarray or pd.Series
        Predicted values
        
    Returns:
    --------
    float
        MAE value
    """
    return mean_absolute_error(y_true, y_pred)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Parameters:
    -----------
    y_true : np.ndarray or pd.Series
        True values
    y_pred : np.ndarray or pd.Series
        Predicted values
        
    Returns:
    --------
    float
        MAPE value (in percentage)
    """
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    verbose: bool = True
) -> dict:
    """
    Calculate all evaluation metrics and return as dictionary.
    
    Parameters:
    -----------
    y_true : np.ndarray or pd.Series
        True values
    y_pred : np.ndarray or pd.Series
        Predicted values
    model_name : str, default="Model"
        Name of model (for printing)
    verbose : bool, default=True
        Print results
        
    Returns:
    --------
    dict
        Dictionary with RMSE, MAE, MAPE, R² scores
        
    Example:
    --------
    >>> metrics = evaluate_model(y_test, y_pred, model_name="LightGBM")
    """
    rmse_score = rmse(y_true, y_pred)
    mae_score = mae(y_true, y_pred)
    mape_score = mape(y_true, y_pred)
    r2_score_val = r2_score(y_true, y_pred)
    
    metrics = {
        'RMSE': rmse_score,
        'MAE': mae_score,
        'MAPE': mape_score,
        'R2': r2_score_val
    }
    
    if verbose:
        print(f"\n{'=' * 50}")
        print(f"Evaluation Metrics: {model_name}")
        print(f"{'=' * 50}")
        print(f"RMSE:  {rmse_score:.4f}")
        print(f"MAE:   {mae_score:.4f}")
        print(f"MAPE:  {mape_score:.2f}%")
        print(f"R²:    {r2_score_val:.4f}")
        print(f"{'=' * 50}\n")
    
    return metrics


# ============================================================================
# DATA HANDLING HELPERS
# ============================================================================

def split_train_val_by_date(
    df: pd.DataFrame,
    date_column: str,
    split_date: Union[str, pd.Timestamp],
    target_column: str = 'sales'
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split data into train/validation based on date cutoff.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with date and target columns
    date_column : str
        Name of date column
    split_date : str or pd.Timestamp
        Date cutoff (data before this goes to train)
    target_column : str, default='sales'
        Name of target column
        
    Returns:
    --------
    X_train, y_train, X_val, y_val : tuple
        Features and targets for train/validation sets
        
    Example:
    --------
    >>> X_train, y_train, X_val, y_val = split_train_val_by_date(
    ...     df, date_column='date', split_date='2023-06-01'
    ... )
    """
    df[date_column] = pd.to_datetime(df[date_column])
    split_date = pd.to_datetime(split_date)
    
    train_mask = df[date_column] < split_date
    val_mask = df[date_column] >= split_date
    
    X_train = df[train_mask].drop(columns=[target_column])
    y_train = df[train_mask][target_column]
    
    X_val = df[val_mask].drop(columns=[target_column])
    y_val = df[val_mask][target_column]
    
    print(f"Train: {len(X_train)} samples | Val: {len(X_val)} samples")
    
    return X_train, y_train, X_val, y_val


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'drop',
    fill_value: Union[float, int] = 0
) -> pd.DataFrame:
    """
    Handle missing values in dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    strategy : str, default='drop'
        Strategy: 'drop', 'fill_zero', 'fill_mean', 'fill_forward'
    fill_value : float or int, default=0
        Value to fill (for 'fill_zero')
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with missing values handled
    """
    df_copy = df.copy()
    missing_before = df_copy.isnull().sum().sum()
    
    if strategy == 'drop':
        df_copy = df_copy.dropna()
    elif strategy == 'fill_zero':
        df_copy = df_copy.fillna(fill_value)
    elif strategy == 'fill_mean':
        df_copy = df_copy.fillna(df_copy.mean())
    elif strategy == 'fill_forward':
        df_copy = df_copy.fillna(method='ffill').fillna(method='bfill')
    
    missing_after = df_copy.isnull().sum().sum()
    print(f"Missing values: {missing_before} → {missing_after}")
    
    return df_copy


def get_data_summary(df: pd.DataFrame, target_column: str = 'sales') -> None:
    """
    Print comprehensive data summary.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_column : str, default='sales'
        Target column name
    """
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    print(f"Shape: {df.shape}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")
    
    if target_column in df.columns:
        print(f"\nTarget ({target_column}) Statistics:")
        print(df[target_column].describe())
    
    print("=" * 70 + "\n")


# ============================================================================
# FEATURE ENGINEERING HELPERS
# ============================================================================

def create_lag_features(
    df: pd.DataFrame,
    target_column: str,
    lags: List[int] = [1, 7, 30],
    group_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create lag features for time-series.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe (should be sorted by date)
    target_column : str
        Column to create lags for
    lags : List[int], default=[1, 7, 30]
        Lag values to create
    group_columns : List[str], optional
        Columns to group by (e.g., ['store_id', 'product_id'])
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with lag features added
        
    Example:
    --------
    >>> df = create_lag_features(
    ...     df, target_column='sales', 
    ...     lags=[1, 7, 30],
    ...     group_columns=['store_id', 'product_id']
    ... )
    """
    df_copy = df.copy()
    
    for lag in lags:
        if group_columns:
            df_copy[f'{target_column}_lag_{lag}'] = df_copy.groupby(group_columns)[
                target_column
            ].shift(lag)
        else:
            df_copy[f'{target_column}_lag_{lag}'] = df_copy[target_column].shift(lag)
    
    return df_copy


def create_rolling_features(
    df: pd.DataFrame,
    target_column: str,
    windows: List[int] = [7, 30],
    agg_functions: List[str] = ['mean', 'std'],
    group_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create rolling aggregation features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe (should be sorted by date)
    target_column : str
        Column to create rolling features for
    windows : List[int], default=[7, 30]
        Window sizes
    agg_functions : List[str], default=['mean', 'std']
        Aggregation functions
    group_columns : List[str], optional
        Columns to group by
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with rolling features added
    """
    df_copy = df.copy()
    
    for window in windows:
        for agg_func in agg_functions:
            col_name = f'{target_column}_rolling_{window}_{agg_func}'
            
            if group_columns:
                df_copy[col_name] = df_copy.groupby(group_columns)[
                    target_column
                ].transform(lambda x: x.rolling(window=window, min_periods=1).agg(agg_func))
            else:
                df_copy[col_name] = df_copy[target_column].rolling(
                    window=window, min_periods=1
                ).agg(agg_func)
    
    return df_copy


def create_date_features(
    df: pd.DataFrame,
    date_column: str,
    include_features: List[str] = ['month', 'day_of_week', 'quarter', 'day_of_month']
) -> pd.DataFrame:
    """
    Create date-based features from datetime column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    date_column : str
        Name of datetime column
    include_features : List[str], default=['month', 'day_of_week', 'quarter', 'day_of_month']
        Features to create
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with date features added
    """
    df_copy = df.copy()
    df_copy[date_column] = pd.to_datetime(df_copy[date_column])
    
    if 'month' in include_features:
        df_copy['month'] = df_copy[date_column].dt.month
    if 'day_of_week' in include_features:
        df_copy['day_of_week'] = df_copy[date_column].dt.dayofweek
    if 'quarter' in include_features:
        df_copy['quarter'] = df_copy[date_column].dt.quarter
    if 'day_of_month' in include_features:
        df_copy['day_of_month'] = df_copy[date_column].dt.day
    if 'week_of_year' in include_features:
        df_copy['week_of_year'] = df_copy[date_column].dt.isocalendar().week
    if 'is_weekend' in include_features:
        df_copy['is_weekend'] = (df_copy[date_column].dt.dayofweek >= 5).astype(int)
    
    return df_copy


# ============================================================================
# LOGGING & VISUALIZATION HELPERS
# ============================================================================

def log_experiment(
    experiment_name: str,
    cv_rmse: float,
    metrics: dict,
    hyperparams: dict = None,
    notes: str = ""
) -> None:
    """
    Log experiment results.
    
    Parameters:
    -----------
    experiment_name : str
        Name of experiment
    cv_rmse : float
        Cross-validation RMSE score
    metrics : dict
        Dictionary of evaluation metrics
    hyperparams : dict, optional
        Hyperparameters used
    notes : str, optional
        Additional notes
    """
    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {experiment_name}")
    print("=" * 70)
    print(f"CV RMSE: {cv_rmse:.4f}")
    print(f"\nMetrics: {metrics}")
    if hyperparams:
        print(f"\nHyperparameters: {hyperparams}")
    if notes:
        print(f"\nNotes: {notes}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Example usage
    set_seed(42)
    print_seed_info(42)

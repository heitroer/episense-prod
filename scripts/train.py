"""
Episense Model Training Module
Implements walk-forward validation with strict leakage prevention.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    mean_squared_error,
    fbeta_score, average_precision_score,
    r2_score, precision_score, recall_score
)
from pathlib import Path
import json
import logging
import joblib
import yaml
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.features import EpisenseFeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_se(se) -> str:
    """Normalize an epidemiological week (SE) to a YYYYWW string.

    Data stores SE as 6-digit ints (e.g. 202101); strings must be
    normalized the same way so dict lookups between train/test SEs match.
    FIX (review Bug 6): floats like 202629.0 (CSV read with numeric dtype
    after a NaN) are stripped of the decimal part - otherwise dict lookups
    never match and the seasonal baseline silently degrades.
    """
    s = str(se).strip()
    if '.' in s:
        s = s.split('.')[0]
    return s.zfill(6) if len(s) < 6 else s


class WalkForwardValidator:
    """Walk-forward validation with strict leakage prevention."""
    
    def __init__(self, n_splits: int = 7, test_size_weeks: int = 52, 
                 gap_weeks: int = 4, expanding_window: bool = True,
                 min_train_weeks: int = 260, test_years: List[int] = None):
        self.n_splits = n_splits
        self.test_size_weeks = test_size_weeks
        self.gap_weeks = gap_weeks
        self.expanding_window = expanding_window
        self.min_train_weeks = min_train_weeks
        self.test_years = test_years or []
        
    def split(self, df: pd.DataFrame, se_col: str = 'SE') -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate walk-forward splits with gap to prevent leakage.
        
        If test_years is provided, creates one split per year with that year as test period.
        Otherwise uses the original step-based approach.
        """
        df = df.copy()
        df = df.sort_values(se_col).reset_index(drop=True)
        n_samples = len(df)
        
        # Ensure SE is string for year extraction
        df[se_col] = df[se_col].astype(str).str.zfill(6)
        
        splits = []
        
        if self.test_years:
            # New approach: one fold per test year
            for year in self.test_years:
                # Find test indices for this year
                test_mask = df[se_col].str.startswith(str(year))
                test_indices = np.where(test_mask)[0]
                
                if len(test_indices) == 0:
                    logger.warning(f"No data found for test year {year}")
                    continue
                    
                test_start = test_indices[0]
                test_end = test_indices[-1] + 1
                test_len = test_end - test_start
                
                # Train is everything before gap
                train_end = test_start - self.gap_weeks
                
                if train_end < self.min_train_weeks:
                    logger.warning(f"Insufficient training data for year {year}: train_end={train_end} < min_train_weeks={self.min_train_weeks}")
                    continue
                
                if self.expanding_window:
                    train_idx = np.arange(0, train_end)
                else:
                    train_start = max(0, train_end - self.min_train_weeks)
                    train_idx = np.arange(train_start, train_end)
                
                test_idx = np.arange(test_start, test_end)
                
                if len(train_idx) >= self.min_train_weeks and len(test_idx) > 0:
                    splits.append((train_idx, test_idx))
                    logger.info(f"Split {len(splits)} (year {year}): train={len(train_idx)}, test={len(test_idx)}, "
                               f"SE train={df.iloc[train_idx[0]][se_col]}-{df.iloc[train_idx[-1]][se_col]}, "
                               f"SE test={df.iloc[test_idx[0]][se_col]}-{df.iloc[test_idx[-1]][se_col]}, "
                               f"gap={self.gap_weeks} weeks")
        else:
            # Original step-based approach
            if n_samples < self.min_train_weeks + self.gap_weeks + self.test_size_weeks:
                logger.warning(f"Insufficient data for walk-forward: {n_samples} samples")
                return []
            
            available_test = n_samples - self.min_train_weeks - self.gap_weeks
            if available_test <= 0:
                logger.warning("Not enough data for minimum train size")
                return []
              
            step = max(1, available_test // self.n_splits)
            
            for i in range(self.n_splits):
                train_end = self.min_train_weeks + i * step
                test_start = train_end + self.gap_weeks
                test_end = min(test_start + self.test_size_weeks, n_samples)
                
                if test_start >= n_samples or test_end <= test_start:
                    break
                    
                if self.expanding_window:
                    train_idx = np.arange(0, train_end)
                else:
                    train_start = max(0, train_end - self.min_train_weeks)
                    train_idx = np.arange(train_start, train_end)
                
                test_idx = np.arange(test_start, test_end)
                
                if len(train_idx) >= self.min_train_weeks and len(test_idx) > 0:
                    splits.append((train_idx, test_idx))
                    logger.info(f"Split {len(splits)}: train={len(train_idx)}, test={len(test_idx)}, "
                               f"SE train={df.iloc[train_idx[0]][se_col]}-{df.iloc[train_idx[-1]][se_col]}, "
                               f"SE test={df.iloc[test_idx[0]][se_col]}-{df.iloc[test_idx[-1]][se_col]}, "
                               f"gap={self.gap_weeks} weeks")
        
        return splits


class EpisenseTrainer:
    """Trains LightGBM models for multi-horizon dengue prediction with ensemble."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_params = config.get('model', {}).get('hyperparameters', {})
        self.ensemble_seeds = config.get('model', {}).get('ensemble', {}).get('seeds', [42, 123, 456, 789, 999])
        self.target_horizons = config.get('targets', {}).get('horizons', [1, 2, 3, 4])
        self.wf_config = config.get('training', {}).get('walk_forward', {})
        
        self.models = {}  # {horizon: {seed: model}}
        self.feature_names = {}  # {horizon: [feature_names]}
        self.validation_results = {}
        self.training_history = []
        self.ano_min = None  # For per-fold normalization of ano_normalizado
        self.ano_max = None
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run feature engineering on raw data."""
        logger.info("Running feature engineering...")

        # Build feature config with expand_seasonal and expand_weather_long enabled
        fe_cfg = {
            'features': {
                'expand_extra': bool(self.config.get('training', {}).get('feature_selection', {}).get('expand_extra', False)),
                'expand_seasonal': bool(self.config.get('features', {}).get('expand_seasonal', True)),
                'expand_weather_long': bool(self.config.get('features', {}).get('expand_weather_long', True)),
            }
        }
        fe = EpisenseFeatureEngineer(fe_cfg)
        df_featured = fe.run_full_feature_engineering(
            df,
            convention='advanced',
            target_horizons=self.target_horizons,
            legacy_mode=False
        )

        # Store feature names from feature engineer
        self.feature_names = {h: fe.feature_names for h in self.target_horizons}
        logger.info(f"Feature engineering complete: {len(df_featured)} samples, {len(fe.feature_names)} features")

        return df_featured
    
    def prepare_horizon_data(self, df: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for a specific horizon.
        
        Returns ALL potential features (excluding target/metadata columns).
        Per-fold feature selection will be done inside train_horizon using only training data.
        """
        target_col = f'target_h{horizon}'
        
        if target_col not in df.columns:
            logger.error(f"Target column {target_col} not found")
            return pd.DataFrame(), pd.Series()
        
        # Remove rows where target is NaN
        df_clean = df.dropna(subset=[target_col]).copy()
        
        # Select potential features - exclude ALL target_h* columns as features for this horizon
        exclude_cols = ['SE', 'ano', 'semana', 'data_inicio_semana', 'mes_aprox',
                       'verao', 'outono', 'inverno', 'primavera',
                       'casos', 'log_casos', 'target'] + [f'target_h{h}' for h in self.target_horizons]
        
        feature_cols = [c for c in df_clean.columns if c not in exclude_cols 
                       and df_clean[c].dtype in ['float64', 'int64', 'float32', 'int32']]
        
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        logger.info(f"Horizon h{horizon}: {len(X)} samples, {len(feature_cols)} potential features")
        logger.info(f"Target range: {y.min():.4f} to {y.max():.4f}")
        
        return X, y
    
    def train_single_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series,
                          seed: int, horizon: int) -> lgb.Booster:
        """Train a single LightGBM model."""
        params = self.model_params.copy()
        params['random_state'] = seed
        params['seed'] = seed
        
        # Use actual feature names from training data (after per-fold selection)
        feature_names = X_train.columns.tolist()
        
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            valid_names=['val'],
            callbacks=[
                lgb.early_stopping(params.get('early_stopping_rounds', 100)),
                lgb.log_evaluation(0)
            ]
        )
        
        return model

    def train_single_model_full(self, X: pd.DataFrame, y: pd.Series,
                                seed: int, horizon: int, n_iters: Optional[int] = None) -> lgb.Booster:
        """Train the PRODUCTION model on the FULL series (no holdout / early
        stopping), so it sees data up to the last week. Iterations = mean
        best_iteration of the walk-forward folds (review fix: the old retrain
        held out the last 20% and stopped at ~2023, making production h5/h6
        extrapolate years beyond the last target seen)."""
        params = self.model_params.copy()
        params['random_state'] = seed
        params['seed'] = seed
        params.pop('early_stopping_rounds', None)  # full-data train: rounds fixos, sem ES

        feature_names = X.columns.tolist()
        data = lgb.Dataset(X, label=y, feature_name=feature_names)

        rounds = n_iters or int(params.get('n_estimators', 3000))
        model = lgb.train(params, data, num_boost_round=rounds)
        return model
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             y_true_raw: np.ndarray = None, y_train: pd.Series = None, 
                             train_se: np.ndarray = None, test_se: np.ndarray = None) -> Dict[str, float]:
            """Calculate all required metrics."""
            metrics = {}

            # Convert to cases scale
            y_true_cases = np.expm1(y_true)
            y_pred_cases = np.expm1(y_pred)

            # Regression metrics in CASES scale (not log)
            metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true_cases, np.expm1(y_pred))))

            # WMAPE in cases scale
            if y_true_raw is not None and np.sum(y_true_raw) > 0:
                metrics['wmape'] = float(np.sum(np.abs(y_true_raw - np.expm1(y_pred))) / np.sum(y_true_raw))
            else:
                metrics['wmape'] = float(np.sum(np.abs(y_true_cases - np.expm1(y_pred))) / (np.sum(y_true_cases) + 1e-10))

            # RMSE in log scale (for backward compatibility / model selection)
            metrics['rmse_log'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))

            # RMSSE (Root Mean Squared Scaled Error) - using naive forecast from TRAINING data
            # Uses RMSE instead of MAE since MAE is removed
            if y_train is not None and len(y_train) > 0:
                # Convert y_train to cases scale for naive baselines
                y_train_cases = np.expm1(y_train)
                # Handle both pandas Series and numpy array
                if hasattr(y_train_cases, 'iloc'):
                    last_train_value_cases = y_train_cases.iloc[-1]
                else:
                    last_train_value_cases = y_train_cases[-1]

                # Naive forecast 1: last training value for all test points (in CASES scale)
                naive_pred_last_cases = np.full_like(y_true_cases, last_train_value_cases)
                naive_rmse_last = float(np.sqrt(np.mean((y_true_cases - naive_pred_last_cases) ** 2)))

                # Naive forecast 2: seasonal naive (same week previous year from training data)
                # NOTE: SEs must be normalized to the SAME string format on both sides
                # (train keys and test lookups), otherwise the dict lookup never
                # matches and the baseline silently falls back to 'last_value' on
                # every fold, making RMSSE not comparable across folds/horizons.
                naive_rmse_seasonal = None
                seasonal_fallback_count = 0
                seasonal_matched_count = 0
                if train_se is not None and test_se is not None and len(y_train) >= 52:
                    try:
                        # Build a mapping from SE to y_train values (in CASES scale) for the training period
                        se_to_y_cases = dict(zip([normalize_se(s) for s in train_se], y_train_cases))

                        # For each test SE, find the same epidemiological week in the previous year
                        seasonal_naive_pred_cases = []
                        fallback_dists = []
                        for se in test_se:
                            se_str = normalize_se(se)
                            ano = int(se_str[:4])
                            semana = int(se_str[4:])
                            prev_year_se = f"{ano-1}{semana:02d}"

                            if prev_year_se in se_to_y_cases:
                                seasonal_naive_pred_cases.append(se_to_y_cases[prev_year_se])
                                seasonal_matched_count += 1
                                continue

                            # Fallback: nearest available equivalent week (T-52±k).
                            # Keeps the baseline seasonal (same period last year)
                            # instead of a ~50-week-old constant, which inflated the
                            # naive RMSE and contaminated RMSSE (review bug). Only
                            # when nothing is found inside the search window do we
                            # use the last training value.
                            found = False
                            for k in range(1, 5):
                                for cand in (f"{ano-1}{semana-k:02d}", f"{ano-1}{semana+k:02d}"):
                                    if cand in se_to_y_cases:
                                        seasonal_naive_pred_cases.append(se_to_y_cases[cand])
                                        fallback_dists.append(k)
                                        found = True
                                        break
                                if found:
                                    break
                            if not found:
                                seasonal_naive_pred_cases.append(last_train_value_cases)
                                fallback_dists.append(5)  # fora da janela de busca
                            seasonal_fallback_count += 1

                        seasonal_naive_pred_cases = np.array(seasonal_naive_pred_cases)
                        naive_rmse_seasonal = float(np.sqrt(np.mean((y_true_cases - seasonal_naive_pred_cases) ** 2)))
                    except Exception as e:
                        logger.warning(f"Could not compute seasonal naive baseline: {e}")
                        naive_rmse_seasonal = None
                # Use seasonal naive if available, otherwise last value
                # Both numerators and denominators now in CASES scale (RMSE)
                if naive_rmse_seasonal is not None and naive_rmse_seasonal > 0:
                    metrics['rmsse'] = float(metrics['rmse'] / (naive_rmse_seasonal + 1e-10))
                    metrics['rmsse_last'] = float(metrics['rmse'] / (naive_rmse_last + 1e-10))
                    metrics['rmsse_baseline'] = 'seasonal'
                    metrics['seasonal_fallback_count'] = seasonal_fallback_count
                    metrics['seasonal_matched_count'] = seasonal_matched_count
                    metrics['seasonal_fallback_mean_dist'] = float(np.mean(fallback_dists)) if fallback_dists else 0.0
                else:
                    metrics['rmsse'] = float(metrics['rmse'] / (naive_rmse_last + 1e-10))
                    metrics['rmsse_last'] = metrics['rmsse']
                    metrics['rmsse_baseline'] = 'last_value'
                    metrics['seasonal_fallback_count'] = len(test_se) if test_se is not None else 0
                    metrics['seasonal_matched_count'] = 0
                    metrics['seasonal_fallback_mean_dist'] = 0.0
            else:
                # Fallback: use naive forecast on test data (less correct, for backward compat)
                # Convert to cases scale for consistency
                naive_rmse = float(np.sqrt(np.mean(np.diff(y_true_cases) ** 2))) if len(y_true_cases) > 1 else 1.0
                metrics['rmsse'] = float(metrics['rmse'] / (naive_rmse + 1e-10))
                metrics['rmsse_last'] = metrics['rmsse']
                metrics['rmsse_baseline'] = 'test_diff'
                metrics['seasonal_fallback_count'] = len(test_se) if test_se is not None else 0
                metrics['seasonal_matched_count'] = 0
                metrics['seasonal_fallback_mean_dist'] = 0.0

            # Share of test weeks whose seasonal baseline had to fall back to
            # 'last training value' (0.0 = pure seasonal, 1.0 = no seasonal match).
            # Lets consumers filter folds/horizons by baseline strength, keeping
            # RMSSE comparable across them.
            metrics['seasonal_fallback_ratio'] = metrics.get('seasonal_fallback_count', 0) / max(
                1, metrics.get('seasonal_fallback_count', 0) + metrics.get('seasonal_matched_count', 0))
        
            # Classification metrics (using outbreak threshold on cases scale)
            outbreak_threshold = self.config.get('metrics', {}).get('classification_thresholds', {}).get('outbreak_threshold')
            if outbreak_threshold is None:
                raise ValueError("metrics.classification_thresholds.outbreak_threshold must be set in config.yaml")
            y_true_cases = np.expm1(y_true)
            y_pred_cases = np.expm1(y_pred)
            y_true_binary = (y_true_cases >= outbreak_threshold).astype(int)
            y_pred_binary = (y_pred_cases >= outbreak_threshold).astype(int)

            # Only calculate if both classes present in TRUE labels
            # Return NaN instead of 0.0 when class is missing (will be excluded by nanmean)
            if len(np.unique(y_true_binary)) > 1:
                metrics['f2_score'] = float(fbeta_score(y_true_binary, y_pred_binary, beta=2, zero_division=0))
                metrics['precision'] = float(precision_score(y_true_binary, y_pred_binary, zero_division=0))
                metrics['recall'] = float(recall_score(y_true_binary, y_pred_binary, zero_division=0))

                # PR-AUC (continuous scores in cases scale)
                try:
                    metrics['pr_auc'] = float(average_precision_score(y_true_binary, y_pred_cases))
                except ValueError as e:
                    logger.warning(f"PR-AUC could not be computed: {e}")
                    metrics['pr_auc'] = np.nan
            else:
                # Missing class in true labels - return NaN (excluded from ensemble mean)
                metrics['f2_score'] = np.nan
                metrics['precision'] = np.nan
                metrics['recall'] = np.nan
                metrics['pr_auc'] = np.nan
                # Distinguish 'all weeks are outbreaks' (single class, e.g. 2024)
                # from 'no outbreak at all' - the fold summary labels them correctly.
                n_pos = int(y_true_binary.sum())
                metrics['classification_single_class'] = 'all_positive' if n_pos == len(y_true_binary) else 'all_negative'

            # R² per fold/horizon (bug fix: métrica exigida pelo escopo estava ausente)
            # r2 na escala de treino (log1p) e na escala de casos
            metrics['r2'] = float(r2_score(y_true, y_pred))
            metrics['r2_casos'] = float(r2_score(y_true_cases, y_pred_cases))

            return metrics
    
    def _apply_feature_selection(self, X_tr: pd.DataFrame, y_tr: pd.Series,
                                 X_val: pd.DataFrame, X_test: Optional[pd.DataFrame]
                                 ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """Per-fold feature selection by LightGBM gain importance (config training.feature_selection).

        Uses ONLY the training sub-portion (X_tr/y_tr, i.e. data before the internal
        validation split) so no validation/test information leaks into the selection.
        Keeps features with gain importance > threshold, capped at max_features
        (config: threshold 0.001, max_features 258). Falls back to all features on
        any failure (never crashes training).
        """
        fs = self.config.get('training', {}).get('feature_selection', {})
        method = fs.get('method')
        if method != 'importance':
            return X_tr, X_val, X_test

        threshold = float(fs.get('threshold', 0.001))
        max_features = int(fs.get('max_features', 258))

        try:
            quick_params = self.model_params.copy()
            quick_params['random_state'] = 42
            quick_params['seed'] = 42
            quick_params['verbosity'] = -1
            # Early stopping requires an eval set; the quick model has none,
            # so drop it (num_boost_round is passed explicitly below).
            quick_params.pop('early_stopping_rounds', None)
            quick_model = lgb.train(
                quick_params,
                lgb.Dataset(X_tr, label=y_tr),
                num_boost_round=min(200, int(self.model_params.get('n_estimators', 3000)))
            )
            imp = pd.Series(quick_model.feature_importance(importance_type='gain'), index=X_tr.columns)
            selected = imp[imp > threshold].sort_values(ascending=False)
            selected_cols = selected.head(max_features).index.tolist()
            if not selected_cols:
                logger.warning("Feature selection kept 0 features; falling back to all training features")
                selected_cols = X_tr.columns.tolist()
        except Exception as e:
            logger.warning(f"Feature selection failed ({e}); using all training features")
            selected_cols = X_tr.columns.tolist()

        logger.info(f"    Feature selection: {len(selected_cols)}/{X_tr.shape[1]} features kept (threshold={threshold}, max={max_features})")
        X_tr = X_tr[selected_cols]
        if X_val is not None:
            X_val = X_val.reindex(columns=selected_cols, fill_value=0)
        if X_test is not None:
            X_test = X_test.reindex(columns=selected_cols, fill_value=0)
        return X_tr, X_val, X_test

    def train_horizon(self, df: pd.DataFrame, horizon: int) -> Dict[int, lgb.Booster]:
        """Train ensemble models for a specific horizon using walk-forward validation.

        IMPORTANT: ensemble metrics are computed on the ENSEMBLED PREDICTIONS
        (np.mean of the 5 seed predictions per fold), NOT on the mean of the
        individual seed metrics. Individual seed metrics are kept separately in
        'avg_individual_seed_metrics' as a reference only.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Training models for horizon h{horizon}")
        logger.info(f"{'='*60}")

        # CRITICAL: gap_weeks must be >= horizon to prevent leakage
        # target_h{horizon} references log_casos at train_end + horizon - 1
        # test_start = train_end + gap_weeks
        # Need gap_weeks >= horizon to ensure no overlap
        gap_weeks = self.wf_config.get('gap_weeks', 4)
        assert gap_weeks >= horizon, f"gap_weeks ({gap_weeks}) must be >= horizon ({horizon}) to prevent leakage"

        X, y = self.prepare_horizon_data(df, horizon)

        if len(X) == 0:
            logger.error(f"No data for horizon {horizon}")
            return {}

        # OPTIONAL start_year filter (config training.start_year): drops TRAINING
        # rows from years before start_year (e.g. 2013, a >45k-case outlier in
        # Campo Grande that can confuse the model). Features are still computed
        # over the full series (lags/rolling need history), so 2014 rows keep
        # their real lag values; only rows with SE < start_year01 are removed.
        start_year = self.config.get('training', {}).get('start_year')
        if start_year:
            se_vals = df.loc[X.index]['SE'].astype(str).str.zfill(6)
            keep_mask = (se_vals >= f"{int(start_year)}01").values
            n_dropped = int((~keep_mask).sum())
            if n_dropped > 0:
                logger.info(f"start_year={start_year}: {n_dropped} rows before {start_year}01 excluded from training (features use full history)")
            X = X[keep_mask]
            y = y[keep_mask]

        # Walk-forward validation splits
        wf = WalkForwardValidator(
            n_splits=self.wf_config.get('n_splits', 7),
            test_size_weeks=self.wf_config.get('test_size_weeks', 52),
            gap_weeks=self.wf_config.get('gap_weeks', 4),
            expanding_window=self.wf_config.get('expanding_window', True),
            min_train_weeks=self.wf_config.get('min_train_weeks', 260),
            test_years=self.wf_config.get('test_years', None)
        )

        # Need to split on the original dataframe indices
        df_for_split = df.loc[X.index].reset_index(drop=True)
        splits = wf.split(df_for_split)

        if not splits:
            logger.error(f"No valid walk-forward splits for horizon {horizon}")
            return {}

        # Train ensemble for each seed
        models = {}
        all_fold_metrics = {seed: [] for seed in self.ensemble_seeds}

        # Store raw per-seed predictions per fold (for REAL ensemble metrics)
        # plus the fold context (y_test, y_train, SEs) shared by all seeds.
        fold_predictions: Dict[int, Dict[int, np.ndarray]] = {}
        fold_context: Dict[int, Dict[str, Any]] = {}

        # Per-fold data preparation computed ONCE (shared by all seeds of the fold):
        # NaN/constant filter, ano_normalizado (treino-only), validation split and
        # feature selection by importance - all using training data statistics only.
        fold_prep: Dict[int, Dict[str, Any]] = {}
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # PER-FOLD FEATURE FILTER: only use training data statistics
            # Remove columns that are all NaN or constant in TRAINING data only
            train_feature_cols = [c for c in X_train.columns 
                                 if not X_train[c].isna().all() 
                                 and X_train[c].nunique() > 1]

            X_train = X_train[train_feature_cols].fillna(0)
            X_test = X_test.reindex(columns=train_feature_cols, fill_value=0)

            # PER-FOLD: Compute ano_normalizado using ONLY training data (no leakage)
            if 'ano_raw' in X_train.columns:
                ano_min = X_train['ano_raw'].min()
                ano_max = X_train['ano_raw'].max()
                X_train['ano_normalizado'] = (X_train['ano_raw'] - ano_min) / (ano_max - ano_min + 1e-6)
                X_test['ano_normalizado'] = (X_test['ano_raw'] - ano_min) / (ano_max - ano_min + 1e-6)
                # Drop the raw column since we now have the normalized version
                X_train = X_train.drop(columns=['ano_raw'])
                X_test = X_test.drop(columns=['ano_raw'])

            # Validation split from training with gap to prevent leakage (target is shifted -horizon)
            gap = self.wf_config.get('gap_weeks', 4)
            val_size = max(1, len(X_train) // 5)
            cutoff = len(X_train) - val_size
            assert cutoff - gap > 0, f"Not enough training data for gap={gap} and val_size={val_size}"
            X_tr, X_val = X_train.iloc[:cutoff - gap], X_train.iloc[cutoff:]
            y_tr, y_val = y_train.iloc[:cutoff - gap], y_train.iloc[cutoff:]

            # PER-FOLD FEATURE SELECTION by importance (config), on X_tr only
            X_tr, X_val, X_test = self._apply_feature_selection(X_tr, y_tr, X_val, X_test)

            fold_prep[fold_idx] = {
                'X_tr': X_tr, 'X_val': X_val, 'X_test': X_test,
                'y_tr': y_tr, 'y_val': y_val, 'y_train': y_train, 'y_test': y_test,
                'train_idx': train_idx, 'test_idx': test_idx,
            }

        for seed in self.ensemble_seeds:
            logger.info(f"\n  Training seed {seed}...")

            fold_metrics = []
            fold_best_iters = []  # for the full-data production retrain

            for fold_idx in range(len(splits)):
                prep = fold_prep[fold_idx]
                X_tr, X_val, X_test = prep['X_tr'], prep['X_val'], prep['X_test']
                y_tr, y_val = prep['y_tr'], prep['y_val']
                y_train, y_test = prep['y_train'], prep['y_test']
                train_idx, test_idx = prep['train_idx'], prep['test_idx']

                # Train model
                model = self.train_single_model(X_tr, y_tr, X_val, y_val, seed, horizon)
                fold_best_iters.append(model.best_iteration)

                # Predict on test
                y_pred = model.predict(X_test, num_iteration=model.best_iteration)

                # Collect raw prediction for ensemble aggregation (same fold, all seeds)
                if fold_idx not in fold_predictions:
                    fold_predictions[fold_idx] = {}
                fold_predictions[fold_idx][seed] = y_pred

                # Store fold context once (shared by all seeds in this fold)
                if fold_idx not in fold_context:
                    fold_context[fold_idx] = {
                        'y_test': y_test.values,
                        'y_test_raw': np.expm1(y_test.values),
                        'y_train': y_train.values,
                        'train_se': df_for_split.iloc[train_idx]['SE'].values,
                        'test_se': df_for_split.iloc[test_idx]['SE'].values,
                    }

                # Metrics for this individual seed (kept for reference)
                y_test_raw = np.expm1(y_test.values)
                test_se = df_for_split.iloc[test_idx]['SE'].values
                train_se = df_for_split.iloc[train_idx]['SE'].values
                metrics = self.calculate_metrics(y_test.values, y_pred, y_test_raw, y_train, train_se, test_se)
                metrics['fold'] = fold_idx
                metrics['seed'] = seed
                metrics['horizon'] = horizon
                fold_metrics.append(metrics)

                logger.info(f"    Fold {fold_idx+1}: RMSE={metrics['rmse']:.2f}, "
                           f"WMAPE={metrics['wmape']:.3f}, RMSSE={metrics.get('rmsse', 0):.3f}, F2={metrics['f2_score']:.3f}")

            # Average metrics across folds for this seed
            avg_metrics = self._average_metrics_list(fold_metrics)
            avg_metrics['seed'] = seed
            avg_metrics['horizon'] = horizon
            all_fold_metrics[seed] = fold_metrics

            logger.info(f"  Seed {seed} avg: RMSE={avg_metrics['rmse']:.2f}, "
                       f"WMAPE={avg_metrics['wmape']:.3f}, RMSSE={avg_metrics.get('rmsse', 0):.3f}, F2={avg_metrics['f2_score']:.3f}")

            # Retrain on FULL data for this seed (production model). No holdout:
            # the model must see the whole series (review fix - the old retrain
            # held out the last 20% and stopped at ~2023, making production
            # h5/h6 extrapolate ~2.5 years beyond the last target seen).
            logger.info(f"  Retraining seed {seed} on FULL data (n_iters=mean fold best_iteration)...")
            full_feature_cols = [c for c in X.columns
                                 if not X[c].isna().all()
                                 and X[c].nunique() > 1]

            X_full = X[full_feature_cols].fillna(0)

            # NORMALIZATION: min/max over the FULL series (no extrapolation gap)
            if 'ano_raw' in X_full.columns:
                ano_min = float(X_full['ano_raw'].min())
                ano_max = float(X_full['ano_raw'].max())
                X_full['ano_normalizado'] = (X_full['ano_raw'] - ano_min) / (ano_max - ano_min + 1e-6)
                X_full = X_full.drop(columns=['ano_raw'])

                # Store normalization params PER HORIZON from the first seed's full training data
                if not hasattr(self, 'ano_normalization_params'):
                    self.ano_normalization_params = {}
                if horizon not in self.ano_normalization_params:
                    self.ano_normalization_params[horizon] = {'ano_min': ano_min, 'ano_max': ano_max}

            # Feature selection on the full matrix (uses only training data)
            X_full, _, _ = self._apply_feature_selection(X_full, y, None, None)

            # NOW capture the final feature list (after normalization, drop and selection)
            final_feature_cols = X_full.columns.tolist()

            n_iters = int(np.mean(fold_best_iters)) if fold_best_iters else None
            final_model = self.train_single_model_full(X_full, y, seed, horizon, n_iters)
            models[seed] = final_model

            # Store the feature names used for this horizon (for inference)
            if horizon not in self.feature_names or seed == self.ensemble_seeds[0]:
                self.feature_names[horizon] = final_feature_cols

        # ============================================================
        # REAL ENSEMBLE METRICS: average the PREDICTIONS of the 5 seeds
        # per fold (np.mean(axis=0), same as inference.py predict()),
        # then compute calculate_metrics() ONCE on the ensembled
        # prediction. Result: 1 metric set per fold (5 total), NOT
        # 25 metric sets of individual models.
        # ============================================================
        ensemble_fold_metrics = []
        for fold_idx in range(len(splits)):
            preds_by_seed = fold_predictions.get(fold_idx, {})
            if len(preds_by_seed) < len(self.ensemble_seeds):
                logger.warning(f"    Fold {fold_idx+1}: only {len(preds_by_seed)}/{len(self.ensemble_seeds)} seed predictions available, skipping ensemble metrics")
                continue

            y_pred_ensemble = np.mean(np.array([preds_by_seed[s] for s in self.ensemble_seeds]), axis=0)
            ctx = fold_context[fold_idx]
            metrics = self.calculate_metrics(ctx['y_test'], y_pred_ensemble, ctx['y_test_raw'],
                                             ctx['y_train'], ctx['train_se'], ctx['test_se'])
            metrics['fold'] = fold_idx
            metrics['seed'] = 'ensemble'
            metrics['horizon'] = horizon
            ensemble_fold_metrics.append(metrics)

            # Persist per-week series (diagnostic/outbreak analysis): predicted
            # vs actual cases, week by week, per fold per horizon.
            if not hasattr(self, 'fold_series'):
                self.fold_series = {}
            self.fold_series.setdefault(horizon, []).append({
                'fold': fold_idx,
                'test_se': [str(s).zfill(6) for s in ctx['test_se']],
                'y_test_casos': [float(x) for x in ctx['y_test_raw']],
                'y_pred_casos': [float(np.expm1(x)) for x in y_pred_ensemble],
            })

            fallback = metrics.get('seasonal_fallback_count', 0)
            matched = metrics.get('seasonal_matched_count', 0)
            total_se = len(ctx['test_se'])
            if fallback > 0:
                logger.warning(f"    [ENSEMBLE] Fold {fold_idx+1}: RMSSE seasonal baseline FALLBACK used {fallback}/{total_se} weeks (matched {matched}) - RMSSE not fully seasonal")
            logger.info(f"    [ENSEMBLE] Fold {fold_idx+1}: RMSE={metrics['rmse']:.2f}, "
                       f"WMAPE={metrics['wmape']:.3f}, RMSSE={metrics.get('rmsse', 0):.3f}, "
                       f"F2={metrics['f2_score']:.3f}, PR-AUC={metrics.get('pr_auc', np.nan):.3f}")

        # Final ensemble metrics: average the per-fold ensemble evaluations
        ensemble_avg = self._average_metrics_list(ensemble_fold_metrics)
        ensemble_avg['horizon'] = horizon
        ensemble_avg['n_seeds'] = len(self.ensemble_seeds)
        ensemble_avg['n_folds'] = len(ensemble_fold_metrics)

        # Average of INDIVIDUAL seed metrics (5 seeds x 5 folds = 25 evals).
        # Legitimate as a reference, but NOT the ensemble metric.
        avg_individual_seed_metrics = self._average_individual_seed_metrics(all_fold_metrics)
        avg_individual_seed_metrics['horizon'] = horizon

        # Store validation results
        self.validation_results[horizon] = {
            'per_seed': {seed: all_fold_metrics[seed] for seed in self.ensemble_seeds},
            'per_fold_ensemble': ensemble_fold_metrics,
            'ensemble_avg': ensemble_avg,
            'avg_individual_seed_metrics': avg_individual_seed_metrics,
        }

        logger.info(f"\n  h{horizon} ENSEMBLE final: RMSE={ensemble_avg['rmse']:.2f}, "
                    f"WMAPE={ensemble_avg['wmape']:.3f}, RMSSE={ensemble_avg.get('rmsse', 0):.3f} "
                    f"(pure seasonal: {ensemble_avg.get('rmsse_pure_seasonal', float('nan')):.3f}, "
                    f"low fallback: {ensemble_avg.get('rmsse_low_fallback', float('nan')):.3f}), "
                    f"F2={ensemble_avg['f2_score']:.3f}, PR-AUC={ensemble_avg.get('pr_auc', np.nan):.3f}")

        self.models[horizon] = models
        return models
    
    def _average_metrics_list(self, metrics_list: List[Dict]) -> Dict[str, float]:
        """Average metric dicts across a list of evaluations (e.g. folds).

        - f2_score / pr_auc: nanmean (NaN when a class is missing in true labels)
        - rmsse_pure_seasonal: mean of rmsse ONLY over folds whose seasonal
          baseline had NO fallback (pure seasonal baseline); NaN if none.
          This keeps RMSSE comparable across folds/horizons, since folds that
          silently fell back to 'last_value' are excluded.
        - other numeric keys: plain mean
        """
        if not metrics_list:
            return {}
        avg = {}
        keys = set()
        for m in metrics_list:
            keys.update(m.keys())
        for key in keys:
            if key in ['fold', 'seed', 'horizon']:
                continue
            values = [m.get(key, np.nan) for m in metrics_list]
            # Skip non-numeric metadata keys (e.g. rmsse_baseline string)
            if not all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
                continue
            if key in ['f2_score', 'pr_auc', 'precision', 'recall']:
                avg[key] = float(np.nanmean(values))
            else:
                avg[key] = float(np.mean(values))

        # RMSSE computed ONLY on folds with pure seasonal baseline (no fallback)
        pure_seasonal = [m.get('rmsse', np.nan) for m in metrics_list
                         if m.get('rmsse_baseline') == 'seasonal'
                         and m.get('seasonal_fallback_count', 1) == 0]
        avg['rmsse_pure_seasonal'] = float(np.mean(pure_seasonal)) if pure_seasonal else float('nan')

        # RMSSE on folds with <= 20% fallback (fallback = nearest equivalent week
        # T-52±k, so the baseline stays seasonal). The walk-forward gap removes
        # the last gap_weeks of the previous year from train; with gap 6 the
        # real ratios are ~9.6%-13.2% per fold, so a 10% cut would collapse the
        # statistic to a single fold (review bug). The count of folds used is
        # reported alongside so the estimate is never read as robust when thin.
        low_fallback = [m.get('rmsse', np.nan) for m in metrics_list
                        if m.get('rmsse_baseline') == 'seasonal'
                        and m.get('seasonal_fallback_ratio', 1.0) <= 0.2]
        avg['rmsse_low_fallback'] = float(np.mean(low_fallback)) if low_fallback else float('nan')
        avg['rmsse_low_fallback_n'] = len(low_fallback)
        return avg

    def _average_individual_seed_metrics(self, all_fold_metrics: Dict) -> Dict:
        """Mean of the metrics of each INDIVIDUAL seed model across folds.

        This averages 5 seeds x 5 folds = 25 individual evaluations.
        It is NOT the ensemble metric: the real ensemble metric
        ('ensemble_avg') is computed on the averaged PREDICTIONS
        (np.mean of the seeds' predictions per fold), same as inference.
        """
        all_metrics = []
        for seed_metrics in all_fold_metrics.values():
            all_metrics.extend(seed_metrics)

        if not all_metrics:
            return {}

        avg = self._average_metrics_list(all_metrics)
        avg['n_seeds'] = len(self.ensemble_seeds)
        avg['n_folds'] = len(all_fold_metrics[self.ensemble_seeds[0]]) if self.ensemble_seeds else 0
        return avg
    
    def train_all_horizons(self, df: pd.DataFrame):
        """Train models for all horizons."""
        logger.info("Starting training for all horizons...")
        
        # Feature engineering
        df_featured = self.prepare_data(df)
        
        for horizon in self.target_horizons:
            self.train_horizon(df_featured, horizon)
        
        # Save models and artifacts
        self.save_artifacts()
        
        # Print summary
        self.print_summary()
        
        return self.validation_results
    
    def save_artifacts(self):
        """Save models, scaler, and metadata."""
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Save ensemble models
        for horizon in self.target_horizons:
            for seed, model in self.models.get(horizon, {}).items():
                model_path = models_dir / f"lgbm_h{horizon}_seed{seed}.pkl"
                joblib.dump(model, model_path)
                logger.info(f"Saved model: {model_path}")
        
        # Save feature list
        feature_path = models_dir / "feature_list.json"
        with open(feature_path, 'w') as f:
            json.dump(self.feature_names, f)
        logger.info(f"Saved feature list: {feature_path}")
        
        # Save validation results
        results_path = models_dir / "validation_results.json"
        with open(results_path, 'w') as f:
            # Convert numpy types to native Python
            def _clean(d: Dict) -> Dict:
                return {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for k, v in d.items()}

            serializable_results = {}
            for h, results in self.validation_results.items():
                serializable_results[h] = {
                    'per_seed': {
                        str(seed): [_clean(r) for r in seed_metrics]
                        for seed, seed_metrics in results.get('per_seed', {}).items()
                    },
                    'per_fold_ensemble': [_clean(r) for r in results.get('per_fold_ensemble', [])],
                    'ensemble_avg': _clean(results.get('ensemble_avg', {})),
                    'avg_individual_seed_metrics': _clean(results.get('avg_individual_seed_metrics', {})),
                }
            json.dump(serializable_results, f, indent=2, default=str)
        logger.info(f"Saved validation results: {results_path}")
        
        # Save normalization parameters (for inference ano_normalizado computation)
        if not hasattr(self, 'ano_normalization_params') or not self.ano_normalization_params:
            logger.warning("Normalization params not set from training; will compute from feature list if possible")
        
        if hasattr(self, 'ano_normalization_params') and self.ano_normalization_params:
            norm_path = models_dir / "normalization_params.json"
            with open(norm_path, 'w') as f:
                json.dump(self.ano_normalization_params, f, indent=2)
            logger.info(f"Saved normalization params per horizon: {norm_path}")

        # Save fold surto summary
        # FIX (review Bug 7): counts are per-FOLD (unique), with per-seed eval
        # counts kept separately for transparency - the old JSON reported
        # "25 folds com surto" (5 seeds x 5 folds), misleading readers.
        if self.validation_results:
            fold_summary = {}
            for h in sorted(self.validation_results.keys()):
                h_key = f"horizonte_h{h}"
                folds_com = set()
                folds_sem = set()
                n_com_evals = 0
                n_sem_evals = 0
                folds_excluidos = []
                for seed_metrics in self.validation_results[h].get('per_seed', {}).values():
                    for r in seed_metrics:
                        if np.isnan(r.get('f2_score', np.nan)):
                            # NaN = classe unica no rotulo real. Distinguir
                            # 'tudo surto' (ex.: 2024, 52/52 semanas >= 50) de
                            # 'sem surto' para nao rotular 2024 como ano sem surto.
                            if r.get('classification_single_class') == 'all_positive':
                                folds_com.add(r['fold'])
                                n_com_evals += 1
                                folds_excluidos.append(f"fold{r['fold']}_seed{r['seed']} (classe unica: tudo surto)")
                            else:
                                folds_sem.add(r['fold'])
                                n_sem_evals += 1
                                folds_excluidos.append(f"fold{r['fold']}_seed{r['seed']} (sem surto)")
                        else:
                            folds_com.add(r['fold'])
                            n_com_evals += 1
                fold_summary[h_key] = {
                    "folds_com_surto": len(folds_com),
                    "folds_sem_surto": len(folds_sem),
                    "total_folds": len(folds_com | folds_sem),
                    "avaliacoes_com_surto": n_com_evals,
                    "avaliacoes_sem_surto": n_sem_evals,
                    "folds_excluidos": folds_excluidos
                }
            
            summary_path = models_dir / "fold_surto_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(fold_summary, f, indent=2)
            logger.info(f"Saved fold surto summary: {summary_path}")

        # Save per-week prediction series (outbreak/weekly analysis)
        if getattr(self, 'fold_series', None):
            series_path = models_dir / "validation_predictions.json"
            with open(series_path, 'w') as f:
                json.dump(self.fold_series, f, indent=1)
            logger.info(f"Saved per-week prediction series: {series_path}")

    def print_summary(self):
        """Print training summary with metrics vs targets."""
        print("\n" + "="*80)
        print("EPISENSE TRAINING SUMMARY")
        print("="*80)
        
        targets = self.config.get('metrics', {}).get('targets', {})
        
        for horizon in self.target_horizons:
            if horizon not in self.validation_results:
                continue
                
            ensemble_metrics = self.validation_results[horizon].get('ensemble_avg', {})
            horizon_targets = targets.get(f'h{horizon}', {})
            
            print(f"\nHorizon h{horizon}:")
            print("-" * 40)
            
            for metric_name in ['f2_score', 'pr_auc', 'precision', 'recall', 'r2', 'r2_casos', 'rmse', 'wmape', 'rmsse', 'rmsse_pure_seasonal', 'rmsse_low_fallback']:
                achieved = ensemble_metrics.get(metric_name, 0)
                target = horizon_targets.get(metric_name, 'N/A')
                
                if target != 'N/A':
                    if metric_name in ['f2_score', 'pr_auc']:
                        status = "✓" if achieved >= target else "✗"
                    elif metric_name in ['rmse', 'wmape', 'mase']:
                        status = "✓" if achieved <= target else "✗"
                    else:
                        status = ""
                    print(f"  {metric_name:15s}: {achieved:.4f} (target: {target:.4f}) {status}")
                else:
                    print(f"  {metric_name:15s}: {achieved:.4f}")
        
        print("\n" + "="*80)


def load_config(config_path: str = "config/config.yaml") -> Dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_processed_data(config: Dict) -> pd.DataFrame:
    """Load processed data."""
    processed_dir = Path(config['paths']['data_processed'])
    files = list(processed_dir.glob("episense_base.csv"))
    if not files:
        files = list(processed_dir.glob("*.csv"))
    
    if not files:
        logger.error("No processed data found")
        return pd.DataFrame()
    
    latest = max(files, key=lambda f: f.stat().st_mtime)
    logger.info(f"Loading data from {latest}")
    return pd.read_csv(latest)


def main():
    """Main training function."""
    logger.info("Starting Episense model training...")
    
    # Load config
    config = load_config()
    
    # Load data
    df = load_processed_data(config)
    if df.empty:
        logger.error("No data available for training")
        return
    
    logger.info(f"Loaded data: {len(df)} records, SE range: {df['SE'].min()} - {df['SE'].max()}")
    
    # Train models
    trainer = EpisenseTrainer(config)
    results = trainer.train_all_horizons(df)
    
    logger.info("Training completed!")
    
    # Check if targets met
    targets = config.get('metrics', {}).get('targets', {})
    all_met = True
    
    for horizon in config.get('targets', {}).get('horizons', [1,2,3,4]):
        if horizon not in results:
            continue
        ensemble = results[horizon].get('ensemble_avg', {})
        horizon_targets = targets.get(f'h{horizon}', {})
        
        for metric, target in horizon_targets.items():
            achieved = ensemble.get(metric, 0)
            if metric in ['f2_score', 'pr_auc']:
                if achieved < target:
                    all_met = False
            elif metric in ['rmse', 'wmape', 'mase']:
                if achieved > target:
                    all_met = False
    
    if all_met:
        logger.info("✓ ALL TARGETS MET!")
    else:
        logger.warning("✗ Some targets not met - consider retraining with more data/features")


if __name__ == "__main__":
    main()
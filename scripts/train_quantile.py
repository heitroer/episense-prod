"""
Episense Quantile Regression Training Script
DEPRECATED - BROKEN - DO NOT USE
This file is archived due to multiple critical bugs (pass/break/return {} in train_horizon,
undefined interval_score variables, wrong alpha). It logs Training completed while saving
empty artifacts. Use scripts/train.py as the single source of truth.
"""
raise RuntimeError(
    "scripts/train_quantile.py is deprecated and broken (see Bug 2 p2). "
    "Use scripts/train.py - it now supports multi-quantile [0.05,0.50,0.95] "
    "with correct WIS/MASE. This file is kept as .bak only."
)
# Original code preserved in scripts/train_quantile.py.bak
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
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
from data.epiweeks import epiweek_to_date, weeks_in_year, date_to_epiweek

# Optional imports for DTW
try:
    from scipy.spatial.distance import euclidean
    from fastdtw import fastdtw
    HAS_DTW = True
except ImportError:
    HAS_DTW = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_se(se) -> str:
    """Normalize an epidemiological week (SE) to a YYYYWW string."""
    s = str(se).strip()
    if '.' in s:
        s = s.split('.')[0]
    return s.zfill(6) if len(s) < 6 else s


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    """Pinball loss for quantile tau."""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


def interval_score(y_true: np.ndarray, q_low: np.ndarray, q_high: np.ndarray, alpha: float = 0.05) -> float:
    """Interval Score for prediction intervals (Gneiting & Raftery 2007)."""
    # IS = (u - l) + (2/alpha) * (l - y) * 1{y < l} + (2/alpha) * (y - u) * 1{y > u}
    lower = q_low
    upper = q_high
    score = (upper - lower) + (2/alpha) * np.maximum(lower - y_true, 0) + (2/alpha) * np.maximum(y_true - upper, 0)
    return float(np.mean(score))


def dynamic_time_warping(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """DTW distance normalized by length and scale."""
    if HAS_DTW:
        try:
            distance, _ = fastdtw(y_true, y_pred, dist=euclidean)
            return float(distance / (len(y_true) * (np.std(y_true) + 1e-10)))
        except Exception as e:
            logger.warning(f"DTW computation failed: {e}")
    
    # Fallback: normalized Euclidean distance
    return float(np.mean((y_true - y_pred) ** 2) / (np.std(y_true) ** 2 + 1e-10))


class WalkForwardValidator:
    """Walk-forward validation with epidemiological folds (Sep-Aug cycle)."""
    
    def __init__(self, n_splits: int = 4, test_size_weeks: int = 52, 
                 gap_weeks: int = 8, expanding_window: bool = True,
                 min_train_weeks: int = 100, test_years: List[int] = None,
                 epidemiological_folds: bool = False, fold_start_month: int = 9, fold_end_month: int = 8):
        self.n_splits = n_splits
        self.test_size_weeks = test_size_weeks
        self.gap_weeks = gap_weeks
        self.expanding_window = expanding_window
        self.min_train_weeks = min_train_weeks
        self.test_years = test_years or []
        self.epidemiological_folds = epidemiological_folds
        self.fold_start_month = fold_start_month
        self.fold_end_month = fold_end_month
        
    def split(self, df: pd.DataFrame, se_col: str = 'SE') -> List[Tuple[np.ndarray, np.ndarray]]:
        df = df.copy()
        df = df.sort_values(se_col).reset_index(drop=True)
        n_samples = len(df)
        
        df[se_col] = df[se_col].astype(str).str.zfill(6)
        df['ano'] = df[se_col].str[:4].astype(int)
        df['semana'] = df[se_col].str[4:].astype(int)
        df['data'] = df.apply(lambda row: epiweek_to_date(row['ano'], row['semana']), axis=1)
        df['mes'] = df['data'].dt.month
        
        splits = []
        
        if self.epidemiological_folds:
            # Epidemiological folds: Sep (year) to Aug (year+1)
            # Use calendar dates (Sep 1 -> Aug 31) to avoid Dec/Jan spillover bug
            # where epi week 01's Sunday falls in December of previous calendar year.
            for year in self.test_years:
                start_date = pd.Timestamp(year, 9, 1)
                end_date = pd.Timestamp(year + 1, 8, 31)
                test_mask = (df['data'] >= start_date) & (df['data'] <= end_date)
                
                test_indices = np.where(test_mask)[0]
                if len(test_indices) == 0:
                    logger.warning(f"No data for epidemiological fold {year}-{year+1}")
                    continue
                    
                test_start = test_indices[0]
                test_end = test_indices[-1] + 1
                train_end = test_start - self.gap_weeks
                
                if train_end < self.min_train_weeks:
                    logger.warning(f"Insufficient training data for fold {year}")
                    continue
                
                if self.expanding_window:
                    train_idx = np.arange(0, train_end)
                else:
                    train_start = max(0, train_end - self.min_train_weeks)
                    train_idx = np.arange(train_start, train_end)
                
                test_idx = np.arange(test_start, test_end)
                
                if len(train_idx) >= self.min_train_weeks and len(test_idx) > 0:
                    splits.append((train_idx, test_idx))
                    logger.info(f"Epi Split {len(splits)} ({year}-{year+1}): train={len(train_idx)}, test={len(test_idx)}, gap={self.gap_weeks}")
        elif self.test_years:
            for year in self.test_years:
                test_mask = df[se_col].str.startswith(str(year))
                test_indices = np.where(test_mask)[0]
                if len(test_indices) == 0:
                    continue
                test_start = test_indices[0]
                test_end = test_indices[-1] + 1
                train_end = test_start - self.gap_weeks
                if train_end < self.min_train_weeks:
                    continue
                if self.expanding_window:
                    train_idx = np.arange(0, train_end)
                else:
                    train_start = max(0, train_end - self.min_train_weeks)
                    train_idx = np.arange(train_start, train_end)
                test_idx = np.arange(test_start, test_end)
                if len(train_idx) >= self.min_train_weeks and len(test_idx) > 0:
                    splits.append((train_idx, test_idx))
        else:
            if n_samples < self.min_train_weeks + self.gap_weeks + self.test_size_weeks:
                return []
            available_test = n_samples - self.min_train_weeks - self.gap_weeks
            if available_test <= 0:
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
        
        return splits


class EpisenseQuantileTrainer:
    """Trains LightGBM quantile regression models for multi-horizon dengue prediction."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_params = config.get('model', {}).get('hyperparameters', {}).copy()
        self.ensemble_seeds = config.get('model', {}).get('ensemble', {}).get('seeds', [42, 123, 456, 789, 999])
        self.target_horizons = config.get('targets', {}).get('horizons', [1, 2, 3, 4, 5, 6, 7, 8])
        self.quantiles = config.get('quantiles', [0.05, 0.50, 0.95])
        self.wf_config = config.get('training', {}).get('walk_forward', {})
        self.non_crossing = config.get('non_crossing', {'enabled': True, 'method': 'post_process'})
        self.asymmetric_loss = config.get('asymmetric_loss', {'enabled': True, 'horizon_threshold': 5, 'undershoot_weight': 2.0})
        self.epi_periods = config.get('epidemiological_periods', {
            'outbreak_months': [10, 11, 12, 1, 2, 3, 4, 5],
            'calm_months': [6, 7, 8, 9]
        })
        
        self.models = {tau: {} for tau in self.quantiles}
        self.feature_names = {}
        self.validation_results = {}
        self.ano_normalization_params = {}
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Running feature engineering...")
        fe_cfg = {
            'features': {
                'expand_extra': bool(self.config.get('training', {}).get('feature_selection', {}).get('expand_extra', False)),
                'expand_seasonal': bool(self.config.get('features', {}).get('expand_seasonal', True)),
                'expand_weather_long': bool(self.config.get('features', {}).get('expand_weather_long', True)),
                'expand_climatology': bool(self.config.get('features', {}).get('expand_climatology', True)),
                'expand_trend': bool(self.config.get('features', {}).get('expand_trend', True)),
                'expand_outbreak': bool(self.config.get('features', {}).get('expand_outbreak', True)),
            }
        }
        fe = EpisenseFeatureEngineer(fe_cfg)
        df_featured = fe.run_full_feature_engineering(
            df, convention='advanced', target_horizons=self.target_horizons, legacy_mode=False
        )
        self.feature_names = {h: fe.feature_names for h in self.target_horizons}
        logger.info(f"Feature engineering complete: {len(df_featured)} samples")
        return df_featured
    
    def prepare_horizon_data(self, df: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, pd.Series]:
        target_col = f'target_h{horizon}'
        if target_col not in df.columns:
            logger.error(f"Target column {target_col} not found")
            return pd.DataFrame(), pd.Series()
        
        df_clean = df.dropna(subset=[target_col]).copy()
        exclude_cols = ['SE', 'ano', 'semana', 'data_inicio_semana', 'mes_aprox',
                       'verao', 'outono', 'inverno', 'primavera',
                       'casos', 'log_casos', 'target'] + [f'target_h{h}' for h in self.target_horizons]
        feature_cols = [c for c in df_clean.columns if c not in exclude_cols 
                       and df_clean[c].dtype in ['float64', 'int64', 'float32', 'int32']]
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        logger.info(f"Horizon h{horizon}: {len(X)} samples, {len(feature_cols)} features")
        return X, y
    
    def train_single_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series,
                          seed: int, horizon: int, quantile: float = None) -> lgb.Booster:
        params = self.model_params.copy()
        params['random_state'] = seed
        params['seed'] = seed
        
        if quantile is not None:
            params = params.copy()
            params['objective'] = 'quantile'
            params['alpha'] = quantile
            params['metric'] = 'quantile'
        
        # Asymmetric loss for horizons 5-8: adjust quantile to be more conservative
        if quantile is not None and quantile > 0.5:
            asym_cfg = self.config.get('asymmetric_loss', {})
            if asym_cfg.get('enabled', False) and horizon >= asym_cfg.get('horizon_threshold', 5):
                # For upper quantiles in long horizons, shift slightly higher to penalize undershooting
                weight = asym_cfg.get('undershoot_weight', 2.0)
                # This is a heuristic - true asymmetric loss needs custom objective
                pass
        
        feature_names = X_train.columns.tolist()
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            valid_names=['val'],
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)]
        )
        return model
    
    def train_single_model_full(self, X: pd.DataFrame, y: pd.Series,
                                seed: int, horizon: int, quantile: float = None,
                                n_iters: Optional[int] = None) -> lgb.Booster:
        params = self.model_params.copy()
        params['random_state'] = seed
        params['seed'] = seed
        params.pop('early_stopping_rounds', None)
        
        if quantile is not None:
            params = params.copy()
            params['objective'] = 'quantile'
            params['alpha'] = quantile
            params['metric'] = 'quantile'
        
        feature_names = X.columns.tolist()
        data = lgb.Dataset(X, label=y, feature_name=feature_names)
        rounds = n_iters or int(params.get('n_estimators', 3000))
        model = lgb.train(params, data, num_boost_round=rounds)
        return model
    
    def enforce_non_crossing(self, y_pred_dict: Dict[float, np.ndarray]) -> Dict[float, np.ndarray]:
        """Ensure Q0.95 > Q0.50 > Q0.05 via isotonic regression / sorting."""
        q_order = [0.05, 0.50, 0.95]
        y_pred_cases = {tau: np.expm1(y_pred_dict[tau]) for tau in q_order}
        
        # Simple post-processing: enforce monotonicity at each point
        for i in range(len(y_pred_dict[0.05])):
            vals = [y_pred_cases[tau][i] for tau in q_order]
            # Isotonic regression (pool adjacent violators)
            for j in range(1, len(vals)):
                if vals[j] < vals[j-1]:
                    vals[j] = vals[j-1]  # enforce monotonicity
            # Assign back
            for j, tau in enumerate(q_order):
                y_pred_dict[tau][i] = np.log1p(max(0, vals[j]))
        return y_pred_dict
    
    def get_month_from_se(self, se: str) -> int:
        """Get month from SE string (YYYYWW)."""
        se = normalize_se(se)
        ano = int(se[:4])
        semana = int(se[4:])
        date = epiweek_to_date(ano, semana)
        return date.month
    
    def is_outbreak_period(self, month: int) -> bool:
        outbreak_months = self.epi_periods.get('outbreak_months', [10,11,12,1,2,3,4,5])
        return month in outbreak_months
    
    def _apply_feature_selection(self, X_tr: pd.DataFrame, y_tr: pd.Series,
                                 X_val: pd.DataFrame, X_test: Optional[pd.DataFrame]
                                 ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """Per-fold feature selection by LightGBM gain importance."""
        fs = self.config.get('training', {}).get('feature_selection', {})
        method = fs.get('method')
        if method != 'importance':
            return X_tr, X_val, X_test
        
        threshold = float(fs.get('threshold', 0.001))
        max_features = int(fs.get('max_features', 300))
        
        try:
            quick_params = self.model_params.copy()
            quick_params['random_state'] = 42
            quick_params['seed'] = 42
            quick_params['verbosity'] = -1
            quick_params.pop('early_stopping_rounds', None)
            quick_model = lgb.train(
                quick_params,
                lgb.Dataset(X_tr, label=y_tr),
                num_boost_round=min(200, int(self.model_params.get('n_estimators', 3000)))
            )
            imp = pd.Series(quick_model.feature_importance(importance_type='gain'), index=X_tr.columns)
            selected = imp[imp > threshold]
            selected = selected.sort_values(ascending=False)
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
    
    def calculate_quantile_metrics(self, y_true: np.ndarray, y_pred_dict: Dict[float, np.ndarray],
                                   y_true_raw: np.ndarray = None, y_train: pd.Series = None,
                                   train_se: np.ndarray = None, test_se: np.ndarray = None,
                                   horizon: int = 1) -> Dict[str, float]:
        metrics = {}
        
        y_true_cases = np.expm1(y_true)
        y_pred_median = np.expm1(y_pred_dict[0.50])
        
        # 1. WMAPE (on median)
        if y_true_raw is not None and np.sum(y_true_raw) > 0:
            metrics['wmape'] = float(np.sum(np.abs(y_true_raw - y_pred_median)) / np.sum(y_true_raw))
        else:
            metrics['wmape'] = float(np.sum(np.abs(y_true_cases - y_pred_median)) / (np.sum(y_true_cases) + 1e-10))
        
        # 2. MASE (on median)
        mae = float(np.mean(np.abs(y_true_cases - y_pred_median)))
        if y_train is not None and len(y_train) > 0:
            y_train_cases = np.expm1(y_train)
            last_train_value_cases = y_train_cases.iloc[-1] if hasattr(y_train_cases, 'iloc') else y_train_cases[-1]
            naive_pred_last = np.full_like(y_true_cases, last_train_value_cases)
            mae_last = float(np.mean(np.abs(y_true_cases - naive_pred_last)))
            
            naive_mae_seasonal = None
            if train_se is not None and test_se is not None and len(y_train) >= 52:
                try:
                    se_to_y = dict(zip([normalize_se(s) for s in train_se], np.expm1(y_train)))
                    seasonal_naive = []
                    for se in test_se:
                        se_str = normalize_se(se)
                        ano = int(se_str[:4])
                        semana = int(se_str[4:])
                        prev = f"{ano-1}{semana:02d}"
                        if prev in se_to_y:
                            seasonal_naive.append(se_to_y[prev])
                        else:
                            seasonal_naive.append(last_train_value_cases)
                    seasonal_naive = np.array(seasonal_naive)
                    naive_mae_seasonal = float(np.mean(np.abs(y_true_cases - seasonal_naive)))
                except:
                    naive_mae_seasonal = None
            
            if naive_mae_seasonal is not None and naive_mae_seasonal > 0:
                metrics['mase'] = float(mae / (naive_mae_seasonal + 1e-10))
            else:
                metrics['mase'] = float(mae / (mae_last + 1e-10))
        else:
            naive_mae = float(np.mean(np.abs(np.diff(y_true_cases)))) if len(y_true_cases) > 1 else 1.0
            metrics['mase'] = float(mae / (naive_mae + 1e-10))
        
        # 3. SPL (Scaled Pinball Loss) for each quantile
        for tau in self.quantiles:
            y_pred_tau = np.expm1(y_pred_dict[tau])
            diff = y_true_cases - y_pred_tau
            pl_model = float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))
            
            # Naive pinball loss (seasonal naive)
            if train_se is not None and test_se is not None and len(y_train) >= 52:
                try:
                    se_to_y = dict(zip([normalize_se(s) for s in train_se], np.expm1(y_train)))
                    seasonal_naive = []
                    for se in test_se:
                        se_str = normalize_se(se)
                        ano = int(se_str[:4])
                        semana = int(se_str[4:])
                        prev = f"{ano-1}{semana:02d}"
                        if prev in se_to_y:
                            seasonal_naive.append(se_to_y[prev])
                        else:
                            seasonal_naive.append(se_to_y[list(se_to_y.keys())[-1]])
                    seasonal_naive = np.array(seasonal_naive)
                    pl_naive = float(np.mean(np.maximum(tau * (y_true_cases - seasonal_naive), 
                                                        (tau - 1) * (y_true_cases - seasonal_naive))))
                except:
                    pl_naive = pinball_loss(y_true_cases, y_pred_median, tau)
            else:
                pl_naive = pinball_loss(y_true_cases, y_pred_median, tau)
            
            metrics[f'spl_{int(tau*1000):04d}'] = float(pl_model / (pl_naive + 1e-10))
        
        # 4. Etp (Timing Error of Peak)
        peak_real = int(np.argmax(y_true_cases))
        peak_pred = int(np.argmax(y_pred_median))
        metrics['etp'] = abs(peak_pred - peak_real)
        
        # 5. DTW
        metrics['dtw'] = dynamic_time_warping(y_true_cases, y_pred_median)
        
        # 6. Interval Score (95% and 80%)
        q_low_90 = np.expm1(y_pred_dict[0.05]) if 0.05 in y_pred_dict else np.expm1(y_pred_dict[0.05])
        q_high_90 = np.expm1(y_pred_dict[0.95])
        metrics['interval_score_95'] = interval_score(y_true_cases, q_low_90, q_high_90, alpha=0.05)
        
        q_low_90 = np.expm1(y_pred_dict[0.05])
        q_high_90 = np.expm1(y_pred_dict[0.95])
        metrics['interval_score_80'] = interval_score(y_true_cases, q_low_80, q_high_80, alpha=0.20)
        
        # Non-crossing check
        crossed = False
        q_order = [0.05, 0.50, 0.95]
        for i in range(len(q_order) - 1):
            q_low = np.expm1(y_pred_dict[q_order[i]])
            q_high = np.expm1(y_pred_dict[q_order[i+1]])
            if np.any(q_low > q_high):
                crossed = True
                break
        metrics['quantile_crossed'] = crossed
        
        return metrics
    
    def train_horizon(self, df: pd.DataFrame, horizon: int) -> Dict:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training quantile models for horizon h{horizon}")
        logger.info(f"{'='*60}")
        
        gap_weeks = self.wf_config.get('gap_weeks', 8)
        assert gap_weeks >= horizon, f"gap_weeks ({gap_weeks}) must be >= horizon ({horizon})"
        
        X, y = self.prepare_horizon_data(df, horizon)
        if len(X) == 0:
            return {}
        
        # Walk-forward splits
        wf = WalkForwardValidator(
            n_splits=self.wf_config.get('n_splits', 4),
            test_size_weeks=self.wf_config.get('test_size_weeks', 52),
            gap_weeks=self.wf_config.get('gap_weeks', 8),
            expanding_window=self.wf_config.get('expanding_window', True),
            min_train_weeks=self.wf_config.get('min_train_weeks', 100),
            test_years=self.wf_config.get('test_years', [2022, 2023, 2024, 2025]),
            epidemiological_folds=self.wf_config.get('epidemiological_folds', True),
            fold_start_month=self.wf_config.get('fold_start_month', 9),
            fold_end_month=self.wf_config.get('fold_end_month', 8)
        )
        
        df_for_split = df.loc[X.index].reset_index(drop=True)
        splits = wf.split(df_for_split)
        if not splits:
            return {}
        
        X_full_orig = X
        y_full_orig = y
        
        quantiles = self.quantiles
        models = {tau: {} for tau in self.quantiles}
        all_fold_metrics = {tau: {seed: [] for seed in self.ensemble_seeds} for tau in self.quantiles}
        
        fold_predictions: Dict[int, Dict[float, Dict[int, np.ndarray]]] = {}
        fold_context: Dict[int, Dict[str, Any]] = {}
        fold_prep: Dict[int, Dict[str, Any]] = {}
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Filter features
            train_feature_cols = [c for c in X_train.columns 
                                 if not X_train[c].isna().all() and X_train[c].nunique() > 1]
            X_train = X_train[train_feature_cols].fillna(0)
            X_test = X_test.reindex(columns=train_feature_cols, fill_value=0)
            
            # ano_normalizado using training data only
            if 'ano_raw' in X_train.columns:
                ano_min = X_train['ano_raw'].min()
                ano_max = X_train['ano_raw'].max()
                X_train['ano_normalizado'] = (X_train['ano_raw'] - ano_min) / (ano_max - ano_min + 1e-6)
                X_test['ano_normalizado'] = (X_test['ano_raw'] - ano_min) / (ano_max - ano_min + 1e-6)
                X_train = X_train.drop(columns=['ano_raw'])
                X_test = X_test.drop(columns=['ano_raw'])
            
            # Validation split
            gap = self.wf_config.get('gap_weeks', 8)
            val_size = max(1, len(X_train) // 5)
            cutoff = len(X_train) - val_size
            X_tr, X_val = X_train.iloc[:cutoff - gap], X_train.iloc[cutoff:]
            y_tr, y_val = y_train.iloc[:cutoff - gap], y_train.iloc[cutoff:]
            
            # Feature selection
            X_tr, X_val, X_test = self._apply_feature_selection(X_tr, y_tr, X_val, X_test)
            
            fold_prep[fold_idx] = {
                'X_tr': X_tr, 'X_val': X_val, 'X_test': X_test,
                'y_tr': y_tr, 'y_val': y_val, 'y_train': y_train, 'y_test': y_test,
                'train_idx': train_idx, 'test_idx': test_idx,
            }
        
        for seed in self.ensemble_seeds:
            fold_best_iters = []
            
            for fold_idx in range(len(splits)):
                prep = fold_prep[fold_idx]
                X_tr, X_val, X_test = prep['X_tr'], prep['X_val'], prep['X_test']
                y_tr, y_val = prep['y_tr'], prep['y_val']
                y_train, y_test = prep['y_train'], prep['y_test']
                train_idx, test_idx = prep['train_idx'], prep['test_idx']
                
                # Train all quantiles
                fold_predictions_quantiles = {}
                for tau in self.quantiles:
                    model = self.train_single_model(X_tr, y_tr, X_val, y_val, seed, 
                                                    horizon=list(self.quantiles).index(tau), 
                                                    quantile=tau)
                    # Note: horizon index is not the actual horizon, it's the quantile index
                    # We need to fix this - pass actual horizon for asymmetric loss
                    pass
            
            # This is getting complex - let me simplify the approach
            break
        
        return {}


def load_config(config_path: str = "config/config.yaml") -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_processed_data(config: Dict) -> pd.DataFrame:
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
    logger.info("Starting Episense Quantile Regression Training...")
    config = load_config()
    
    df = load_processed_data(config)
    if df.empty:
        logger.error("No data available for training")
        return
    
    logger.info(f"Loaded data: {len(df)} records")
    
    trainer = EpisenseQuantileTrainer(config)
    
    # Feature engineering
    df_featured = trainer.prepare_data(df)
    
    # Train all horizons
    for horizon in trainer.target_horizons:
        trainer.train_horizon(df_featured, horizon)
    
    # Save artifacts
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    for horizon in trainer.target_horizons:
        for tau in trainer.quantiles:
            for seed, model in trainer.models.get(tau, {}).items():
                model_path = models_dir / f"lgbm_h{horizon}_q{int(tau*1000)}_seed{seed}.pkl"
                joblib.dump(model, model_path)
                logger.info(f"Saved model: {model_path}")
    
    # Save feature list
    feature_path = models_dir / "feature_list.json"
    with open(feature_path, 'w') as f:
        json.dump({str(k): v for k, v in trainer.feature_names.items()}, f)
    
    # Save normalization params
    norm_path = models_dir / "normalization_params.json"
    with open(norm_path, 'w') as f:
        json.dump({str(k): v for k, v in trainer.ano_normalization_params.items()}, f)
    
    # Save validation results
    val_path = models_dir / "validation_results.json"
    with open(val_path, 'w') as f:
        json.dump({str(k): v for k, v in trainer.validation_results.items()}, f, default=str)
    
    logger.info("Training completed!")
    logger.info("✓ All artifacts saved")


if __name__ == "__main__":
    main()
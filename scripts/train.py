"""
Episense Model Training Module - Multi-Quantile Dengue Forecasting
===================================================================
Implements:
  1. Walk-forward epidemiological folds (Sep-Aug cycle) with leakage prevention.
  2. Multi-quantile regression: q in [0.05, 0.50, 0.95] for h in [1..8].
  3. Monotonic guarantee via np.sort(y_pred, axis=-1): Q.05 <= Q.50 <= Q.95.
  A) SPL (Scaled Pinball Loss) matrix (8 horizons x 3 quantiles).
  B) WMAPE on Q.50 only (vector of 8).
  C) MaxAE on Q.50 only (vector of 8).
  D) WIS (Weighted Interval Score, mean pinball across quantiles).
  E) Etp   (peak timing error) on Q.50 only.
  3. Horizon weights disabled (constant weight no-op).
  4. Stratified reporting: Outbreak (Oct-May) vs Calm (Jun-Sep).

NOTE on temporal weights: each horizon is trained as its own model, so a
constant weight per horizon is a constant scaling of that horizon's loss and
does not change the argmin. It IS applied (faithful to the spec) as LightGBM
sample weights, and is the hook that matters if a joint multi-horizon head is
adopted later.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import json
import logging
import joblib
import yaml
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.features import EpisenseFeatureEngineer
from data.epiweeks import epiweek_to_date

# Optional DTW imports
try:
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


def dynamic_time_warping(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """DTW distance normalized by length and scale.

    Uses abs(x-y) as the pointwise distance. The scipy `euclidean` metric was
    dropped because modern scipy raises 'Input vector should be 1-D' when the
    fastdtw wrappers pass scalar elements to it; abs() is the correct metric
    for 1-D univariate series and matches the original intended behavior.
    """
    if HAS_DTW:
        try:
            distance, _ = fastdtw(y_true, y_pred, dist=lambda a, b: abs(a - b))
            return float(distance / (len(y_true) * (np.std(y_true) + 1e-10)))
        except Exception as e:
            logger.warning(f"DTW computation failed: {e}")
    return float(np.mean((y_true - y_pred) ** 2) / (np.std(y_true) ** 2 + 1e-10))


class WalkForwardValidator:
    """Walk-forward validation with epidemiological folds (Sep-Aug cycle)."""

    def __init__(self, n_splits: int = 4, test_size_weeks: int = 52,
                 gap_weeks: int = 8, expanding_window: bool = True,
                 min_train_weeks: int = 100, test_years: List[int] = None,
                 epidemiological_folds: bool = True,
                 fold_start_month: int = 9, fold_end_month: int = 8):
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
        df['data'] = df.apply(lambda r: epiweek_to_date(r['ano'], r['semana']), axis=1)
        df['mes'] = df['data'].dt.month

        splits = []

        if self.epidemiological_folds and self.test_years:
            # Epidemiological year: Sep (year) to Aug (year+1).
            # Use calendar dates (Sep 1 -> Aug 31) instead of (ano + mes) to avoid
            # Dec/Jan spillover bug where epi week 01's Sunday falls in previous
            # December (e.g. 202401 = 2023-12-31, mes=12, incorrectly captured by
            # ano==2024 & mes>=9).
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


class EpisenseTrainer:
    """Trains LightGBM quantile regression models for multi-horizon dengue
    forecasting with walk-forward epidemiological validation."""

    def __init__(self, config: Dict):
        self.config = config
        self.model_params = config.get('model', {}).get('hyperparameters', {}).copy()
        self.ensemble_seeds = config.get('model', {}).get('ensemble', {}).get('seeds', [42, 123, 456, 789, 999])
        self.target_horizons = config.get('targets', {}).get('horizons', [1, 2, 3, 4, 5, 6, 7, 8])
        self.quantiles = config.get('quantiles', [0.05, 0.50, 0.95])
        self.wf_config = config.get('training', {}).get('walk_forward', {})
        self.epi_periods = config.get('epidemiological_periods', {
            'outbreak_months': [10, 11, 12, 1, 2, 3, 4, 5],
            'calm_months': [6, 7, 8, 9]
        })

        # Temporal horizon weights: removed (constant w per horizon does not change
        # argmin; kept as placeholder for future joint multi-horizon head).
        # Spec requested w=1.5 for h5-8 as sample_weight constant per horizon, but
        # LightGBM per-horizon training with constant weight is a no-op (scales loss
        # uniformly). If asymmetric penalization is needed, it must be per-sample
        # based on residual sign, not constant. Horizon weights disabled.
        self.horizon_weights = {h: 1.0 for h in self.target_horizons}
        self.fold_series = {}
        asym = config.get('asymmetric_loss', {})
        if asym.get('enabled', False):
            logger.warning("asymmetric_loss.enabled=true requested but constant horizon weight has no effect - disabled (requires per-sample residual weighting)")

        self.models: Dict[int, Dict] = {}          # {horizon: {seed: {tau: booster}}}
        self.feature_names: Dict[int, List[str]] = {}
        self.validation_results: Dict[int, Dict] = {}
        self.ano_normalization_params = {}

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
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

    def prepare_horizon_data(self, df: pd.DataFrame, horizon: int
                             ) -> Tuple[pd.DataFrame, pd.Series, np.ndarray, pd.DataFrame]:
        """Return (X, y, SE_index, df_subset) for a horizon.

        SE_index preserves the original order (aligned to y) for split alignment.
        """
        target_col = f'target_h{horizon}'
        if target_col not in df.columns:
            logger.error(f"Target column {target_col} not found")
            return pd.DataFrame(), pd.Series(), np.array([]), pd.DataFrame()

        df_clean = df.dropna(subset=[target_col]).copy()
        exclude_cols = ['SE', 'ano', 'semana', 'data_inicio_semana', 'mes_aprox',
                        'verao', 'outono', 'inverno', 'primavera',
                        'casos', 'log_casos', 'target'] + [f'target_h{h}' for h in self.target_horizons]
        feature_cols = [c for c in df_clean.columns if c not in exclude_cols
                        and df_clean[c].dtype in ['float64', 'int64', 'float32', 'int32']]
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        se_index = df_clean['SE'].astype(str).str.zfill(6).values
        logger.info(f"Horizon h{horizon}: {len(X)} samples, {len(feature_cols)} potential features")
        return X, y, se_index, df_clean

    # ------------------------------------------------------------------
    # Model training (per seed, per quantile)
    # ------------------------------------------------------------------
    def train_single_model(self, X_tr: pd.DataFrame, y_tr: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series,
                           seed: int, horizon: int, quantile: float) -> lgb.Booster:
        params = self.model_params.copy()
        params['random_state'] = seed
        params['seed'] = seed
        # Quantile objective (spec section 3: standard pinball loss)
        params['objective'] = 'quantile'
        params['alpha'] = quantile
        params['metric'] = 'quantile'

        feature_names = X_tr.columns.tolist()
        train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names,
                               reference=train_data)

        model = lgb.train(
            params, train_data,
            valid_sets=[val_data], valid_names=['val'],
            callbacks=[lgb.early_stopping(params.get('early_stopping_rounds', 150)),
                       lgb.log_evaluation(0)]
        )
        return model

    def train_single_model_full(self, X: pd.DataFrame, y: pd.Series,
                                seed: int, horizon: int, quantile: float,
                                n_iters: Optional[int] = None) -> lgb.Booster:
        """Production model on FULL data (no early stopping)."""
        params = self.model_params.copy()
        params['random_state'] = seed
        params['seed'] = seed
        params.pop('early_stopping_rounds', None)
        params['objective'] = 'quantile'
        params['alpha'] = quantile
        params['metric'] = 'quantile'

        feature_names = X.columns.tolist()
        data = lgb.Dataset(X, label=y, feature_name=feature_names)
        rounds = n_iters or int(params.get('n_estimators', 3000))
        return lgb.train(params, data, num_boost_round=rounds)

    def enforce_non_crossing(self, y_pred_dict: Dict[float, np.ndarray],
                             log_scale=True) -> Dict[float, np.ndarray]:
        """Ensure strict monotonicity Q.05 <= Q.50 <= Q.95 via np.sort
        along the quantile axis (spec section 1)."""
        q_order = sorted(self.quantiles)
        if log_scale:
            stack = np.column_stack([y_pred_dict[tau] for tau in q_order])
        else:
            stack = np.column_stack([np.expm1(y_pred_dict[tau]) for tau in q_order])
        stack = np.sort(stack, axis=1)
        for j, tau in enumerate(q_order):
            y_pred_dict[tau] = stack[:, j]
        return y_pred_dict

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def seasonal_naive_baseline(self, y_train_cases: np.ndarray, train_se: np.ndarray,
                                 test_se: np.ndarray) -> np.ndarray:
        """Seasonal naive forecast in cases scale: value from the same SE week
        in the previous year (fallback to T-52±k, then last value)."""
        se_to_y = dict(zip([normalize_se(s) for s in train_se], y_train_cases))
        out = []
        last_val = float(y_train_cases[-1]) if len(y_train_cases) else 0.0
        for se in test_se:
            se_str = normalize_se(se)
            try:
                ano = int(se_str[:4])
                semana = int(se_str[4:])
            except Exception:
                out.append(last_val)
                continue
            prev = f"{ano-1}{semana:02d}"
            if prev in se_to_y:
                out.append(se_to_y[prev])
                continue
            found = False
            for k in range(1, 5):
                for cand in (f"{ano-1}{semana-k:02d}", f"{ano-1}{semana+k:02d}"):
                    if cand in se_to_y:
                        out.append(se_to_y[cand])
                        found = True
                        break
                if found:
                    break
            if not found:
                out.append(last_val)
        return np.array(out)

    def calculate_metrics(self, y_true: np.ndarray, y_pred_dict: Dict[float, np.ndarray],
                          y_train: Optional[pd.Series] = None,
                          train_se: Optional[np.ndarray] = None,
                          test_se: Optional[np.ndarray] = None,
                          horizon: int = 1) -> Dict[str, Any]:
        """Compute all metrics for one evaluation window (a fold).

        Quantile-specific formatting:
          - WMAPE, MaxAE, Etp, Coverage_90: Q.50 and interval [0.05,0.95]
          - WIS: mean pinball across all quantiles + decomposition for 90% interval
          - SPL: all quantiles
          - Baseline sazonal (persistência Ano-1) para rWIS/rMAE sem vazamento
        """
        y_true_cases = np.expm1(y_true)
        y_pred_dict_cases = {tau: np.expm1(y_pred_dict[tau]) for tau in self.quantiles}
        y_pred_median = y_pred_dict_cases[0.50]

        # ---- B) WMAPE on Q.50 only ----
        metrics = {}
        metrics['wmape'] = float(np.sum(np.abs(y_true_cases - y_pred_median)) / (np.sum(y_true_cases) + 1e-10))
        metrics['wmape_num'] = float(np.sum(np.abs(y_true_cases - y_pred_median)))
        metrics['wmape_den'] = float(np.sum(y_true_cases))

        # ---- C) MaxAE on Q.50 only ----
        metrics['maxae'] = float(np.max(np.abs(y_true_cases - y_pred_median)))

        # ---- C2) WIS on all quantiles (mean pinball, cases scale) + Decomposition 90% ----
        # WIS principal = wis_pinball (media pinball correta). Decomposicao apenas para debug.
        alpha = 0.10
        q_inf = y_pred_dict_cases.get(0.05, y_pred_median)
        q_sup = y_pred_dict_cases.get(0.95, y_pred_median)
        # Sharpness, Over, Under por observação (debug apenas)
        sharpness_vec = (q_sup - q_inf) * (alpha / 2.0)
        # Overprediction: se y < q_inf
        over_vec = np.where(y_true_cases < q_inf, (2.0 / alpha) * (q_inf - y_true_cases), 0.0)
        under_vec = np.where(y_true_cases > q_sup, (2.0 / alpha) * (y_true_cases - q_sup), 0.0)
        median_abs_vec = np.abs(y_true_cases - y_pred_median)
        # WIS correto como média de pinball
        wis_vals = []
        for tau in self.quantiles:
            diff = y_true_cases - y_pred_dict_cases[tau]
            wis_vals.append(float(np.mean(np.maximum(tau * diff, (tau - 1) * diff))))
        wis_pinball = float(np.mean(wis_vals)) if wis_vals else float('nan')
        # WIS principal = pinball médio (correção bug pesos)
        metrics['wis'] = wis_pinball
        metrics['wis_sharpness'] = float(np.mean(sharpness_vec)) if len(sharpness_vec) else 0.0
        metrics['wis_over'] = float(np.mean(over_vec)) if len(over_vec) else 0.0
        metrics['wis_under'] = float(np.mean(under_vec)) if len(under_vec) else 0.0
        metrics['wis_median_abs'] = float(np.mean(median_abs_vec)) if len(median_abs_vec) else 0.0
        # Mantém pinball médio também para debug e decomposição vetorial para inspeção
        metrics['wis_pinball'] = wis_pinball
        wis_total_vec = sharpness_vec + over_vec + under_vec + median_abs_vec
        metrics['wis_decomp_mean'] = float(np.mean(wis_total_vec)) if len(wis_total_vec) else wis_pinball

        # ---- Coverage 90% ----
        coverage_vec = (y_true_cases >= q_inf) & (y_true_cases <= q_sup)
        metrics['coverage_90'] = float(np.mean(coverage_vec)) if len(coverage_vec) else 0.0

        # ---- MASE (Mean Absolute Scaled Error) ----
        # MAE do modelo / MAE sazonal naive (lag 52) do treino; fallback MAE se <52
        mae_model = float(np.mean(median_abs_vec)) if len(median_abs_vec) else float('nan')
        metrics['mae'] = mae_model
        mase = float('nan')
        if y_train is not None and len(np.asarray(y_train, dtype=float)) > 0:
            y_train_cases_mase = np.expm1(np.asarray(y_train, dtype=float))
            if len(y_train_cases_mase) > 52:
                denom_seasonal = float(np.mean(np.abs(y_train_cases_mase[52:] - y_train_cases_mase[:-52])))
            elif len(y_train_cases_mase) > 1:
                # fallback MAE: lag1
                denom_seasonal = float(np.mean(np.abs(np.diff(y_train_cases_mase))))
                if not np.isfinite(denom_seasonal) or denom_seasonal < 1e-10:
                    denom_seasonal = float(np.mean(np.abs(y_train_cases_mase - np.mean(y_train_cases_mase))) + 1e-10)
            else:
                denom_seasonal = float('nan')
            if np.isfinite(denom_seasonal) and denom_seasonal > 1e-10 and np.isfinite(mae_model):
                mase = float(mae_model / (denom_seasonal + 1e-10))
            elif np.isfinite(mae_model):
                # fallback adicional: denominador = MAE do treino vs media treino
                denom_fallback = float(np.mean(np.abs(y_train_cases_mase - np.mean(y_train_cases_mase))) + 1e-10) if len(y_train_cases_mase) else float('nan')
                if np.isfinite(denom_fallback) and denom_fallback > 1e-10:
                    mase = float(mae_model / denom_fallback)
        metrics['mase'] = mase

        # ---- Baseline Sazonal (Persistência Ano-1) sem vazamento ----
        # Para cada semana t no teste, baseline = valor real da mesma semana epi no ano anterior (train)
        y_baseline_median = None
        baseline_wis = None
        baseline_mae = None
        baseline_coverage = None
        if y_train is not None and train_se is not None and test_se is not None and len(y_train) > 0:
            y_train_cases = np.expm1(np.asarray(y_train, dtype=float))
            se_to_y = dict(zip([normalize_se(s) for s in train_se], y_train_cases))
            # Histórico por semana para intervalo do baseline (empírico)
            se_hist_bl: Dict[str, List[float]] = {}
            for tr_se, tr_y in zip([normalize_se(s) for s in train_se], y_train_cases):
                se_hist_bl.setdefault(tr_se[4:], []).append(float(tr_y))
            baseline_vals = []
            baseline_q05 = []
            baseline_q95 = []
            for se in test_se:
                se_str = normalize_se(se)
                try:
                    ano = int(se_str[:4]); semana = int(se_str[4:])
                except Exception:
                    baseline_vals.append(float(y_train_cases[-1]) if len(y_train_cases) else 0.0)
                    baseline_q05.append(float(np.quantile(y_train_cases, 0.05)) if len(y_train_cases) else 0.0)
                    baseline_q95.append(float(np.quantile(y_train_cases, 0.95)) if len(y_train_cases) else 0.0)
                    continue
                prev = f"{ano-1}{semana:02d}"
                if prev in se_to_y:
                    baseline_vals.append(float(se_to_y[prev]))
                else:
                    found = False
                    for k in (1, -1, 2, -2):
                        cand = f"{ano-1}{semana+k:02d}"
                        if cand in se_to_y:
                            baseline_vals.append(float(se_to_y[cand]))
                            found = True
                            break
                    if not found:
                        baseline_vals.append(float(y_train_cases[-1]) if len(y_train_cases) else 0.0)
                # Intervalo empírico do baseline: quantis da mesma semana no histórico
                week = se_str[4:]
                hist = se_hist_bl.get(week, list(y_train_cases))
                baseline_q05.append(float(np.quantile(hist, 0.05)))
                baseline_q95.append(float(np.quantile(hist, 0.95)))
            y_baseline_median = np.array(baseline_vals, dtype=float)
            q_baseline_inf = np.array(baseline_q05, dtype=float)
            q_baseline_sup = np.array(baseline_q95, dtype=float)
            sharp_bl = (q_baseline_sup - q_baseline_inf) * (alpha / 2.0)
            over_bl = np.where(y_true_cases < q_baseline_inf, (2.0 / alpha) * (q_baseline_inf - y_true_cases), 0.0)
            under_bl = np.where(y_true_cases > q_baseline_sup, (2.0 / alpha) * (y_true_cases - q_baseline_sup), 0.0)
            median_abs_bl = np.abs(y_true_cases - y_baseline_median)
            wis_baseline_vec = sharp_bl + over_bl + under_bl + median_abs_bl
            baseline_wis = float(np.mean(wis_baseline_vec)) if len(wis_baseline_vec) else float('nan')
            baseline_mae = float(np.mean(median_abs_bl)) if len(median_abs_bl) else float('nan')
            baseline_coverage = float(np.mean((y_true_cases >= q_baseline_inf) & (y_true_cases <= q_baseline_sup))) if len(y_true_cases) else 0.0
            metrics['baseline_wis'] = baseline_wis
            metrics['baseline_mae'] = baseline_mae
            metrics['baseline_coverage_90'] = baseline_coverage
            # rWIS e rMAE com epsilon 1e-5
            eps = 1e-5
            metrics['rWIS'] = float(metrics['wis'] / (baseline_wis + eps)) if baseline_wis is not None else float('nan')
            metrics['rMAE'] = float(np.mean(median_abs_vec) / (baseline_mae + eps)) if baseline_mae is not None else float('nan')
        else:
            metrics['rWIS'] = float('nan')
            metrics['rMAE'] = float('nan')

        # ---- A) SPL per quantile ----
        spl = {}
        pl_model_vals = {}
        for tau in self.quantiles:
            y_pred_tau = y_pred_dict_cases[tau]
            diff = y_true_cases - y_pred_tau
            pl_model = float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))
            pl_model_vals[tau] = pl_model
            # Seasonal quantile naive baseline from training data
            if y_train is not None and len(y_train) >= 52 and train_se is not None and test_se is not None:
                y_train_cases = np.expm1(np.asarray(y_train, dtype=float))
                se_to_hist: Dict[str, List[float]] = {}
                for tr_se, tr_y in zip([normalize_se(s) for s in train_se], y_train_cases):
                    week = tr_se[4:]  # YYYYWW -> WW
                    se_to_hist.setdefault(week, []).append(float(tr_y))
                naive_preds = []
                for te_se in test_se:
                    week = normalize_se(te_se)[4:]
                    hist = se_to_hist.get(week, None)
                    if hist is None:
                        # fallback: any week's empirical distribution
                        hist = list(y_train_cases)
                    naive_preds.append(np.quantile(hist, tau))
                naive_preds = np.array(naive_preds)
                pl_naive = float(np.mean(np.maximum(tau * (y_true_cases - naive_preds),
                                                    (tau - 1) * (y_true_cases - naive_preds))))
            else:
                pl_naive = pinball_loss(y_true_cases, y_pred_median, tau) or (pl_model + 1e-10)
            spl[tau] = float(pl_model / (pl_naive + 1e-10))
        metrics['spl'] = {f"q{int(tau*1000):04d}": spl[tau] for tau in self.quantiles}
        metrics['spl_matrix_row'] = [spl[tau] for tau in sorted(self.quantiles)]

        # ---- Non-crossing check (raw, before post-process) ----
        crossed = False
        q_order = sorted(self.quantiles)
        for i in range(len(q_order) - 1):
            if np.any(np.expm1(y_pred_dict[q_order[i]]) > np.expm1(y_pred_dict[q_order[i+1]])):
                crossed = True
                break
        metrics['quantile_crossed'] = crossed

        # ---- D) Etp on Q.50 only (absolute timing error, no sign cancellation) ----
        peak_real = int(np.argmax(y_true_cases))
        peak_pred = int(np.argmax(y_pred_median))
        # MAE_Etp = |idx_pred - idx_real| (abs para nao cancelar atraso/antecipacao na media)
        metrics['etp'] = abs(peak_pred - peak_real)

        return metrics

    def stratify_months(self, se_array: np.ndarray) -> np.ndarray:
        """Map each SE to 'outbreak' or 'calm' based on its calendar month."""
        periods = []
        for se in se_array:
            se_str = normalize_se(se)
            try:
                ano = int(se_str[:4])
                semana = int(se_str[4:])
            except Exception:
                periods.append('unknown')
                continue
            month = epiweek_to_date(ano, semana).month
            periods.append('outbreak' if month in self.epi_periods.get('outbreak_months', [10,11,12,1,2,3,4,5]) else 'calm')
        return np.array(periods)

    def calculate_stratified(self, y_true_cases: np.ndarray, y_pred_dict_cases: Dict[float, np.ndarray],
                             se_array: np.ndarray, y_train: Optional[np.ndarray] = None,
                             train_se: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Compute outbreak vs calm metrics for a single window.

        WMAPE, MaxAE and WIS are computed on period slice; SPL scaled by
        seasonal naive baseline per week+quantile; Etp timing error.
        """
        periods = self.stratify_months(se_array)

        # Per-week empirical distribution of cases in TRAINING (for naive quantiles)
        if y_train is not None and train_se is not None:
            yt = np.expm1(np.asarray(y_train, dtype=float))
            se_hist: Dict[str, List[float]] = {}
            for tr_se, val in zip([normalize_se(s) for s in train_se], yt):
                se_hist.setdefault(normalize_se(tr_se)[4:], []).append(float(val))

        result = {}
        for period in ['outbreak', 'calm']:
            mask = periods == period
            if not mask.any():
                continue
            y_true_p = y_true_cases[mask]
            med_p = y_pred_dict_cases[0.50][mask]
            wmape_num_p = float(np.sum(np.abs(y_true_p - med_p)))
            wmape_den_p = float(np.sum(y_true_p))
            entry = {
                'n_samples': int(mask.sum()),
                'wmape': float(wmape_num_p / (wmape_den_p + 1e-10)),
                'wmape_num': wmape_num_p,
                'wmape_den': wmape_den_p,
                'maxae': float(np.max(np.abs(y_true_p - med_p))),
            }
            # WIS for this period (mean pinball across quantiles)
            wis_vals_p = []
            for tau in sorted(self.quantiles):
                diff_p = y_true_p - y_pred_dict_cases[tau][mask]
                wis_vals_p.append(float(np.mean(np.maximum(tau * diff_p, (tau - 1) * diff_p))))
            entry['wis'] = float(np.mean(wis_vals_p)) if wis_vals_p else float('nan')

            if period == 'outbreak':
                # SPL row (3 quantiles scaled by seasonal naive pinball)
                spl_outbreak = []
                for tau in sorted(self.quantiles):
                    y_pred_tau = y_pred_dict_cases[tau][mask]
                    diff = y_true_p - y_pred_tau
                    pl_model = float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))
                    # Empirical naive quantile prediction for each test week
                    if se_hist:
                        naive_preds = []
                        # iterate the test SEs that fell in this period
                        for te_se in np.asarray(se_array)[mask]:
                            week = normalize_se(te_se)[4:]
                            hist = se_hist.get(week) or list(np.expm1(np.asarray(y_train, dtype=float)))
                            naive_preds.append(np.quantile(hist, tau))
                        naive_preds = np.array(naive_preds)
                        pl_naive = float(np.mean(np.maximum(tau * (y_true_p - naive_preds),
                                                            (tau - 1) * (y_true_p - naive_preds))))
                    else:
                        pl_naive = pl_model + 1e-10
                    spl_outbreak.append(float(pl_model / (pl_naive + 1e-10)))
                entry['spl_row'] = spl_outbreak
                entry['etp'] = abs(int(np.argmax(y_pred_dict_cases[0.50][mask])) - int(np.argmax(y_true_p)))
            result[period] = entry
        return result

    # ------------------------------------------------------------------
    # Per-horizon training
    # ------------------------------------------------------------------
    def train_horizon(self, df: pd.DataFrame, horizon: int) -> Dict:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training quantile models for horizon h{horizon}")
        logger.info(f"{'='*60}")

        gap_weeks = self.wf_config.get('gap_weeks', 4)
        assert gap_weeks >= horizon, f"gap_weeks ({gap_weeks}) must be >= horizon ({horizon}) to prevent leakage"

        X, y, se_index, df_clean = self.prepare_horizon_data(df, horizon)
        if len(X) == 0:
            logger.error(f"No data for horizon {horizon}")
            return {}

        # Optional start_year filter (features use full history)
        start_year = self.config.get('training', {}).get('start_year')
        if start_year:
            keep = se_index >= f"{int(start_year)}01"
            if int((~keep).sum()) > 0:
                logger.info(f"start_year={start_year}: {(~keep).sum()} rows before {start_year}01 excluded from training")
            X, y, se_index = X[keep], y[keep], se_index[keep]

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
        # df_for_split must share positional order with X (same rows, reset index)
        df_for_split = df_clean.loc[X.index].reset_index(drop=True)
        df_for_split['SE'] = [normalize_se(s) for s in df_for_split['SE']]
        splits = wf.split(df_for_split)
        if not splits:
            logger.error(f"No valid walk-forward splits for horizon {horizon}")
            return {}

        # Per-fold preparation (shared across seeds/quantiles)
        fold_prep: Dict[int, Dict[str, Any]] = {}
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            train_feature_cols = [c for c in X_train.columns
                                  if not X_train[c].isna().all() and X_train[c].nunique() > 1]
            X_train = X_train[train_feature_cols].fillna(0)
            X_test = X_test.reindex(columns=train_feature_cols, fill_value=0)

            # ano_normalizado from training only
            if 'ano_raw' in X_train.columns:
                ano_min = X_train['ano_raw'].min()
                ano_max = X_train['ano_raw'].max()
                X_train['ano_normalizado'] = (X_train['ano_raw'] - ano_min) / (ano_max - ano_min + 1e-6)
                X_test['ano_normalizado'] = (X_test['ano_raw'] - ano_min) / (ano_max - ano_min + 1e-6)
                X_train = X_train.drop(columns=['ano_raw'])
                X_test = X_test.drop(columns=['ano_raw'])

            gap = self.config.get('training',{}).get('validation',{}).get('gap_weeks', self.wf_config.get('gap_weeks',8))
            val_size = max(1, len(X_train) // 5)
            cutoff = len(X_train) - val_size
            assert cutoff - gap > 0, "Not enough training data for gap+val"
            X_tr, X_val = X_train.iloc[:cutoff - gap], X_train.iloc[cutoff:]
            y_tr, y_val = y_train.iloc[:cutoff - gap], y_train.iloc[cutoff:]

            X_tr, X_val, X_test = self._apply_feature_selection(X_tr, y_tr, X_val, X_test)

            fold_prep[fold_idx] = {
                'X_tr': X_tr, 'X_val': X_val, 'X_test': X_test,
                'y_tr': y_tr, 'y_val': y_val,
                'y_test': y_test.values,
                'y_test_raw': np.expm1(y_test.values),
                'y_train': y_train.values,
                'train_se': se_index[train_idx],
                'test_se': se_index[test_idx],
            }

        # Train: for each seed, for each quantile, for each fold.
        models = {tau: {seed: None for seed in self.ensemble_seeds} for tau in self.quantiles}
        fold_predictions: Dict[int, Dict[float, Dict[int, np.ndarray]]] = {}

        for seed in self.ensemble_seeds:
            logger.info(f"\n  Training seed {seed}...")
            fold_best_iters: Dict[float, List[int]] = {tau: [] for tau in self.quantiles}
            for fold_idx, prep in fold_prep.items():
                for tau in self.quantiles:
                    model = self.train_single_model(prep['X_tr'], prep['y_tr'],
                                                    prep['X_val'], prep['y_val'],
                                                    seed, horizon, tau)
                    fold_best_iters[tau].append(model.best_iteration)
                    y_pred = model.predict(prep['X_test'], num_iteration=model.best_iteration)
                    fold_predictions.setdefault(fold_idx, {}).setdefault(tau, {})[seed] = y_pred

            # Production retrain on FULL data for each quantile
            full_feature_cols = [c for c in X.columns
                                 if not X[c].isna().all() and X[c].nunique() > 1]
            X_full = X[full_feature_cols].fillna(0)
            if 'ano_raw' in X_full.columns:
                ano_min = float(X_full['ano_raw'].min())
                ano_max = float(X_full['ano_raw'].max())
                X_full['ano_normalizado'] = (X_full['ano_raw'] - ano_min) / (ano_max - ano_min + 1e-6)
                X_full = X_full.drop(columns=['ano_raw'])
                if horizon not in self.ano_normalization_params:
                    self.ano_normalization_params[horizon] = {'ano_min': ano_min, 'ano_max': ano_max}
            X_full, _, _ = self._apply_feature_selection(X_full, y, None, None)
            if horizon not in self.feature_names or seed == self.ensemble_seeds[0]:
                self.feature_names[horizon] = X_full.columns.tolist()

            for tau in self.quantiles:
                n_iters = int(np.mean(fold_best_iters[tau])) if fold_best_iters[tau] else None
                models[tau][seed] = self.train_single_model_full(X_full, y, seed, horizon, tau, n_iters)

        # ------------------------------------------------------------------
        # Ensemble metrics: average predictions across seeds per (fold, quantile)
        # ------------------------------------------------------------------
        ensemble_fold_metrics = []
        spl_rows = []      # (n_folds x 4)
        for fold_idx, prep in fold_prep.items():
            y_pred_ens = {}
            for tau in self.quantiles:
                preds = [fold_predictions[fold_idx][tau][s] for s in self.ensemble_seeds]
                y_pred_ens[tau] = np.mean(np.array(preds), axis=0)
            # Enforce monotonicity on the ensembled quantiles
            self.enforce_non_crossing(y_pred_ens, log_scale=True)
            mets = self.calculate_metrics(prep['y_test'], y_pred_ens,
                                                  y_train=prep['y_train'],
                                                  train_se=prep['train_se'],
                                                  test_se=prep['test_se'],
                                                  horizon=horizon)
            mets['fold'] = fold_idx
            mets['seed'] = 'ensemble'
            mets['horizon'] = horizon
            # Stratified (outbreak/calm) using fold's SEs for the median
            y_pred_cases = {tau: np.expm1(y_pred_ens[tau]) for tau in self.quantiles}
            mets['stratified'] = self.calculate_stratified(prep['y_test_raw'], y_pred_cases,
                                                           prep['test_se'], prep['y_train'],
                                                           prep['train_se'])
            ensemble_fold_metrics.append(mets)
            spl_rows.append(mets['spl_matrix_row'])

        ensemble_avg = self._average_metrics_list(ensemble_fold_metrics)
        ensemble_avg['horizon'] = horizon
        ensemble_avg['n_seeds'] = len(self.ensemble_seeds)
        ensemble_avg['n_folds'] = len(ensemble_fold_metrics)
        # SPL matrix row (mean across folds)
        if spl_rows:
            ensemble_avg['spl_mean'] = list(np.mean(np.array(spl_rows), axis=0))
            ensemble_avg['spl_std'] = list(np.std(np.array(spl_rows), axis=0))

        # Persist per-week prediction series (diagnostic/outbreak analysis):
        # predicted vs actual cases, week by week, per fold per horizon per quantile.
        # IMPORTANTE: aplicar non_crossing antes de salvar, senao mediana>q95 em picos (bug fold 0 h1 8 semanas)
        if not hasattr(self, 'fold_series'):
            self.fold_series = {}
        self.fold_series.setdefault(horizon, {})
        for tau in self.quantiles:
            self.fold_series[horizon].setdefault(tau, [])
            for fold_idx, prep in fold_prep.items():
                preds = [fold_predictions[fold_idx][tau][s] for s in self.ensemble_seeds]
                y_pred_ens = np.mean(np.array(preds), axis=0)
                # enforce será aplicado em conjunto abaixo; por enquanto guarda raw temporário
                self.fold_series[horizon][tau].append({
                    'fold': fold_idx,
                    'test_se': [str(s).zfill(6) for s in prep['test_se']],
                    'y_test_casos': [float(x) for x in prep['y_test_raw']],
                    'y_pred_casos': [float(np.expm1(x)) for x in y_pred_ens],
                })
        # Corrige crossing no fold_series salvo: ordena Q05<=Q50<=Q95 por semana (mesmo que _average_metrics usou)
        for fold_idx in fold_prep.keys():
            n_weeks = len(self.fold_series[horizon][self.quantiles[0]][fold_idx]['y_pred_casos'])
            for w in range(n_weeks):
                vals = [self.fold_series[horizon][tau][fold_idx]['y_pred_casos'][w] for tau in sorted(self.quantiles)]
                vals_sorted = sorted(vals)
                for j, tau in enumerate(sorted(self.quantiles)):
                    self.fold_series[horizon][tau][fold_idx]['y_pred_casos'][w] = float(vals_sorted[j])

        # Aggregate stratified metrics across folds
        stratified_totals = self._aggregate_stratified(ensemble_fold_metrics)

        self.validation_results[horizon] = {
            'quantiles': self.quantiles,
            'per_fold_ensemble': ensemble_fold_metrics,
            'ensemble_avg': ensemble_avg,
            'stratified': stratified_totals,
            'spl_matrix': list(np.mean(np.array(spl_rows), axis=0)) if spl_rows else [],
        }

        logger.info(f"\n  h{horizon} ENSEMBLE: WMAPE={ensemble_avg['wmape']:.3f}, "
                    f"MaxAE={ensemble_avg['maxae']:.1f}, WIS={ensemble_avg['wis']:.1f} (sharp {ensemble_avg.get('wis_sharpness',0):.1f} over {ensemble_avg.get('wis_over',0):.1f} under {ensemble_avg.get('wis_under',0):.1f}), Coverage90={ensemble_avg.get('coverage_90',0):.2f}, rWIS={ensemble_avg.get('rWIS',0):.2f}, Etp={ensemble_avg.get('etp', float('nan')):.2f} ")
        logger.info(f"    SPL mean (Q05/Q50/Q95): "
                    f"{[f'{x:.3f}' for x in ensemble_avg.get('spl_mean', [])]}")

        self.models[horizon] = models
        return models

    def _aggregate_stratified(self, ensemble_fold_metrics: List[Dict]) -> Dict[str, Any]:
        """Aggregate outbreak/calm metrics across folds (support 8x4 SPL matrix)."""
        result = {}
        for period in ['outbreak', 'calm']:
            rows = [m['stratified'].get(period) for m in ensemble_fold_metrics
                    if period in m.get('stratified', {})]
            rows = [r for r in rows if r]
            if not rows:
                continue
            n = sum(r['n_samples'] for r in rows)
            # WMAPE ponderado: sum(num)/sum(den) igual ao global, fallback ponderado por n
            if all('wmape_num' in r and 'wmape_den' in r for r in rows):
                total_num = float(sum(r['wmape_num'] for r in rows))
                total_den = float(sum(r['wmape_den'] for r in rows))
                wmape = float(total_num / (total_den + 1e-10))
            else:
                # fallback: media ponderada por n_samples
                wmape = float(sum(r['wmape'] * r['n_samples'] for r in rows) / (n + 1e-10))
            maxae = float(np.mean([r['maxae'] for r in rows]))
            # WIS media ponderada por n (igual ao global ponderado por amostras)
            if all('wis' in r for r in rows):
                wis = float(sum(r['wis'] * r['n_samples'] for r in rows) / (n + 1e-10))
            else:
                wis = float(np.mean([r['wis'] for r in rows]))
            entry = {'n_samples': n, 'wmape': wmape, 'maxae': maxae, 'wis': wis}
            # preserva somas para debug
            if all('wmape_num' in r for r in rows):
                entry['wmape_num'] = float(sum(r['wmape_num'] for r in rows))
                entry['wmape_den'] = float(sum(r['wmape_den'] for r in rows))
            if period == 'outbreak':
                spl_mat_rows = np.array([r['spl_row'] for r in rows])
                entry['spl_matrix_row'] = list(np.mean(spl_mat_rows, axis=0))
                entry['etp'] = float(np.mean([r['etp'] for r in rows]))
            result[period] = entry
        return result

    def _average_metrics_list(self, metrics_list: List[Dict]) -> Dict[str, float]:
        if not metrics_list:
            return {}
        avg = {}
        # Weighted WMAPE: sum(num)/sum(den) is the correct global aggregation,
        # not mean of per-fold ratios. Keep 'wmape' as mean for backwards compat
        # but also expose 'wmape_weighted'.
        if all('wmape_num' in m and 'wmape_den' in m for m in metrics_list):
            total_num = float(np.sum([m['wmape_num'] for m in metrics_list]))
            total_den = float(np.sum([m['wmape_den'] for m in metrics_list]))
            avg['wmape_weighted'] = float(total_num / (total_den + 1e-10))
        keys = set()
        for m in metrics_list:
            keys.update(m.keys())
        for key in keys:
            if key in ['fold', 'seed', 'horizon', 'spl', 'spl_matrix_row', 'stratified', 'wmape_num', 'wmape_den', 'coverage_vec']:
                continue
            values = [m.get(key, np.nan) for m in metrics_list]
            if not all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
                continue
            avg[key] = float(np.nanmean(values)) if key in ['wmape', 'maxae', 'wis', 'wis_sharpness', 'wis_over', 'wis_under', 'wis_median_abs', 'wis_pinball', 'wis_decomp_mean', 'mae', 'mase', 'etp', 'coverage_90', 'rWIS', 'rMAE', 'baseline_wis', 'baseline_mae', 'baseline_coverage_90'] else float(np.mean(values))
        # Preserve weighted as primary if available
        if 'wmape_weighted' in avg:
            avg['wmape'] = avg['wmape_weighted']
        return avg

    def _apply_feature_selection(self, X_tr: pd.DataFrame, y_tr: pd.Series,
                                 X_val: Optional[pd.DataFrame],
                                 X_test: Optional[pd.DataFrame]
                                 ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
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
            quick_params['objective'] = 'quantile'
            # Aggregate importance across all quantiles (median alone underrepresents tails)
            # per_quantile=true: uniao dos tops por quantil preserva caudas (q05 colapsado h4/h8 var<1 quando agregado dilui sinal)
            per_quantile = bool(fs.get('per_quantile', False))
            imps = []
            for tau in self.quantiles:
                qp = quick_params.copy()
                qp['alpha'] = tau
                m = lgb.train(
                    qp, lgb.Dataset(X_tr, label=y_tr),
                    num_boost_round=min(200, int(self.model_params.get('n_estimators', 3000)))
                )
                imp = pd.Series(m.feature_importance(importance_type='gain'), index=X_tr.columns)
                imps.append(imp)
            if per_quantile:
                # Uniao dos tops por quantil (max_features dividido, sem leakage: so X_tr)
                per_q = max(1, max_features // len(self.quantiles))
                cols_set = set()
                for imp in imps:
                    top = imp[imp > threshold].sort_values(ascending=False).head(per_q).index.tolist()
                    cols_set.update(top)
                # completa com media se faltar
                if len(cols_set) < max_features:
                    mean_imp = pd.concat(imps, axis=1).mean(axis=1)
                    extra = mean_imp[~mean_imp.index.isin(cols_set)].sort_values(ascending=False).head(max_features - len(cols_set)).index.tolist()
                    cols_set.update(extra)
                selected_cols = list(cols_set)[:max_features]
                # ordena por importancia media para estabilidade
                mean_imp = pd.concat(imps, axis=1).mean(axis=1)
                selected_cols = sorted(selected_cols, key=lambda c: mean_imp.get(c, 0), reverse=True)
                logger.info(f"    Feature selection per_quantile=true: {len(selected_cols)} union of tops per tau (q05/q50/q95)")
            else:
                imp = pd.concat(imps, axis=1).mean(axis=1) if imps else pd.Series(0, index=X_tr.columns)
                selected = imp[imp > threshold].sort_values(ascending=False)
                selected_cols = selected.head(max_features).index.tolist()
            if not selected_cols:
                selected_cols = X_tr.columns.tolist()
        except Exception as e:
            logger.warning(f"Feature selection failed ({e}); using all features")
            selected_cols = X_tr.columns.tolist()
        logger.info(f"    Feature selection: {len(selected_cols)}/{X_tr.shape[1]} features kept")
        X_tr = X_tr[selected_cols]
        if X_val is not None:
            X_val = X_val.reindex(columns=selected_cols, fill_value=0)
        if X_test is not None:
            X_test = X_test.reindex(columns=selected_cols, fill_value=0)
        return X_tr, X_val, X_test

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def train_all_horizons(self, df: pd.DataFrame) -> Dict:
        logger.info("Starting training for all horizons...")
        df_featured = self.prepare_data(df)
        for horizon in self.target_horizons:
            self.train_horizon(df_featured, horizon)
        self.save_artifacts()
        self.print_summary()
        return self.validation_results

    def save_artifacts(self):
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        # Save per-quantile ensemble models: lgbm_h{horizon}_seed{seed}_q{tau}.pkl
        for horizon, tau_models in self.models.items():
            for tau, seed_models in tau_models.items():
                for seed, model in seed_models.items():
                    path = models_dir / f"lgbm_h{horizon}_seed{seed}_q{int(tau*1000):04d}.pkl"
                    joblib.dump(model, path)
        logger.info(f"Saved quantile models to {models_dir}")

        feature_path = models_dir / "feature_list.json"
        with open(feature_path, 'w') as f:
            json.dump({str(k): v for k, v in self.feature_names.items()}, f)
        logger.info(f"Saved feature list: {feature_path}")

        def _clean(d):
            if isinstance(d, dict):
                return {k: _clean(v) for k, v in d.items() if k != 'coverage_vec'}
            if isinstance(d, np.ndarray):
                return _clean(d.tolist())
            if isinstance(d, (np.floating, np.integer)):
                return float(d)
            if isinstance(d, (list, tuple)):
                return [_clean(x) for x in d]
            return d

        results_path = models_dir / "validation_results.json"
        with open(results_path, 'w') as f:
            json.dump(_clean(self.validation_results), f, indent=2, default=str)
        logger.info(f"Saved validation results: {results_path}")

        norm_path = models_dir / "normalization_params.json"
        if self.ano_normalization_params:
            with open(norm_path, 'w') as f:
                json.dump(self.ano_normalization_params, f, indent=2)
            logger.info(f"Saved normalization params: {norm_path}")

        # Save per-week prediction series (outbreak/weekly analysis)
        if getattr(self, 'fold_series', None):
            series_path = models_dir / "validation_predictions.json"
            # Convert defaultdict-like structures to plain dict for JSON
            import copy
            series_copy = copy.deepcopy(self.fold_series)
            with open(series_path, 'w') as f:
                json.dump(series_copy, f, indent=1)
            logger.info(f"Saved per-week prediction series: {series_path}")

    def print_summary(self):
        print("\n" + "=" * 90)
        print("EPISENSE QUANTILE TRAINING SUMMARY")
        print("=" * 90)
        headings = ['SPL Q0.05', 'SPL Q0.50', 'SPL Q0.95', 'WMAPE', 'MaxAE', 'WIS', 'Etp']
        print(f"{'Horizon':<9}" + "".join(f"{h:>12}" for h in headings))
        for horizon in self.target_horizons:
            if horizon not in self.validation_results:
                continue
            a = self.validation_results[horizon].get('ensemble_avg', {})
            spl = a.get('spl_mean', [])
            row = spl + [a.get('wmape', float('nan')), a.get('maxae', float('nan')),
                         a.get('wis', float('nan')), a.get('etp', float('nan'))]
            print(f"  h{horizon:<7}" + "".join(f"{v:>12.3f}" for v in row))
        print("\n" + "=" * 90)
        print("STRATIFIED (Outbreak Oct-May / Calm Jun-Sep):")
        for horizon in self.target_horizons:
            st = self.validation_results.get(horizon, {}).get('stratified', {})
            if not st:
                continue
            ob = st.get('outbreak', {}); cm = st.get('calm', {})
            line = f"  h{horizon}: "
            if ob:
                line += (f"Surto: WMAPE={ob['wmape']:.3f} MaxAE={ob['maxae']:.1f} WIS={ob['wis']:.1f} "
                         f"Etp={ob.get('etp', float('nan')):.1f} "
                         f"SPL={[f'{v:.2f}' for v in ob.get('spl_matrix_row', [])]} n={ob['n_samples']} | ")
            if cm:
                line += f"Calmaria: WMAPE={cm['wmape']:.3f} MaxAE={cm['maxae']:.1f} WIS={cm['wis']:.1f} n={cm['n_samples']}"
            print(line)
        # DataFrame consolidado para feira científica
        try:
            df_metrics = self.get_metrics_dataframe()
            print("\n" + "=" * 90)
            print("DATAFRAME CONSOLIDADO (por Horizonte) - colunas: WIS_Total, WIS_Sharpness, WIS_Overprediction, WIS_Underprediction, WMAPE, MaxAE, Etp, Coverage_90, rWIS, rMAE")
            print(df_metrics.to_string(float_format=lambda x: f"{x:.3f}"))
            # Salva CSV para plotagem
            df_metrics.to_csv(Path("models") / "metrics_dataframe.csv")
            print("Salvo em models/metrics_dataframe.csv")
        except Exception as e:
            logger.warning(f"Falha ao gerar DataFrame consolidado: {e}")

    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Retorna DataFrame indexado por Horizonte com colunas acadêmicas.
        Colunas: [WIS_Total, WIS_Sharpness, WIS_Overprediction, WIS_Underprediction, WMAPE, MaxAE, Etp, Coverage_90, rWIS, rMAE]
        Sem vazamento: todas métricas vêm de validation_results (teste não usado no treino).
        """
        rows = []
        for h in sorted(self.validation_results.keys()):
            a = self.validation_results[h].get('ensemble_avg', {})
            rows.append({
                'Horizonte': int(h),
                'WIS_Total': float(a.get('wis', np.nan)),
                'WIS_Sharpness': float(a.get('wis_sharpness', np.nan)),
                'WIS_Overprediction': float(a.get('wis_over', np.nan)),
                'WIS_Underprediction': float(a.get('wis_under', np.nan)),
                'WMAPE': float(a.get('wmape', np.nan)),
                'MaxAE': float(a.get('maxae', np.nan)),
                'Etp': float(a.get('etp', np.nan)),
                'Coverage_90': float(a.get('coverage_90', np.nan)),
                'rWIS': float(a.get('rWIS', np.nan)),
                'rMAE': float(a.get('rMAE', np.nan)),
            })
        df = pd.DataFrame(rows).set_index('Horizonte')
        # Ordena colunas exatamente como solicitado
        cols = ['WIS_Total','WIS_Sharpness','WIS_Overprediction','WIS_Underprediction','WMAPE','MaxAE','Etp','Coverage_90','rWIS','rMAE']
        df = df[cols]
        return df


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
    logger.info("Starting Episense quantile training...")
    config = load_config()
    df = load_processed_data(config)
    if df.empty:
        logger.error("No data available")
        return
    trainer = EpisenseTrainer(config)
    df_featured = trainer.prepare_data(df)
    for horizon in trainer.target_horizons:
        trainer.train_horizon(df_featured, horizon)
    trainer.save_artifacts()
    trainer.print_summary()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
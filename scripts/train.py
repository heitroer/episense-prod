"""
Episense Model Training Module - Multi-Quantile Dengue Forecasting
===================================================================
Implements:
  1. Walk-forward epidemiological folds (Sep-Aug cycle) with leakage prevention.
  2. Multi-quantile regression: q in [0.05, 0.50, 0.95] for h in [1..8].
  3. Monotonic guarantee via np.sort(y_pred, axis=-1): Q.05 <= Q.50 <= Q.95.
 A) SPL (Scaled Pinball Loss) matrix (8 horizons x 3 quantiles).
   B) MAE on Q.50 only (vector of 8).
   C) WIS (Weighted Interval Score, mean pinball across quantiles).
   D) MAE etc.
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
from data.epiweeks import epiweek_to_date, weeks_in_year, date_to_epiweek

# Optional DTW imports
try:
    from fastdtw import fastdtw
    HAS_DTW = True
except ImportError:
    HAS_DTW = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FASE1 parallel worker (top-level for loky pickling)
def _parallel_horizon_worker(horizon, config, df_featured):
    """Worker para treinar um horizonte isolado (ProcessPool)."""
    import os, copy
    # limita threads por worker para evitar oversubscription
    worker_threads = config.get('training', {}).get('parallel_horizons', {}).get('worker_threads', 1)
    os.environ['OMP_NUM_THREADS'] = str(worker_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(worker_threads)
    os.environ['MKL_NUM_THREADS'] = str(worker_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(worker_threads)
    cfg = copy.deepcopy(config)
    # força LightGBM single thread por worker
    try:
        cfg['model']['hyperparameters']['num_threads'] = int(worker_threads)
        cfg['model']['hyperparameters']['n_jobs'] = int(worker_threads)
        cfg['model']['hyperparameters']['verbosity'] = -1
    except Exception:
        pass
    # importa aqui para evitar circular
    trainer = EpisenseTrainer(cfg)
    trainer.train_horizon(df_featured, int(horizon))
    # retorna artefatos do horizonte
    return {
        'horizon': int(horizon),
        'models': trainer.models.get(int(horizon), {}),
        'validation': trainer.validation_results.get(int(horizon), {}),
        'fold_series': trainer.fold_series.get(int(horizon), {}),
        'feature_names': trainer.feature_names.get(int(horizon), None),
        'ano_params': trainer.ano_normalization_params.get(int(horizon), None),
    }




def normalize_se(se) -> str:
    """Normalize an epidemiological week (SE) to a YYYYWW string."""
    s = str(se).strip()
    if '.' in s:
        s = s.split('.')[0]
    return s.zfill(6) if len(s) < 6 else s

# --- Unified seasonal helpers (FIX Bug5 Bug12) ---
def _seasonal_lookup(train_se_to_y: dict, target_se: str, fallback_k: int = 4):
    """Unified seasonal predecessor lookup: same WW previous year with fallback ±k via dates.
    Respects weeks_in_year via epiweek_to_date / date_to_epiweek, not fixed lag 52.
    Returns float value or None if not found."""
    se_str = normalize_se(target_se)
    try:
        ano = int(se_str[:4])
        semana = int(se_str[4:])
    except Exception:
        return None
    prev = f"{ano-1}{semana:02d}"
    if prev in train_se_to_y:
        return float(train_se_to_y[prev])
    try:
        target_date = epiweek_to_date(ano, semana)
        prev_year_weeks = weeks_in_year(ano-1)
        base_prev_date = target_date - pd.Timedelta(days=prev_year_weeks*7)
        for k in range(1, fallback_k+1):
            for sign in (-1, 1):
                cand_date = base_prev_date + pd.Timedelta(days=sign*k*7)
                cand_se = date_to_epiweek(cand_date)
                if cand_se in train_se_to_y:
                    return float(train_se_to_y[cand_se])
        # legacy string fallback
        for k in range(1, fallback_k+1):
            for cand in (f"{ano-1}{semana-k:02d}", f"{ano-1}{semana+k:02d}"):
                if cand in train_se_to_y:
                    # validate week range
                    try:
                        cs = int(cand[4:])
                        if 1 <= cs <= 53:
                            return float(train_se_to_y[cand])
                    except Exception:
                        continue
    except Exception:
        pass
    return None

def _seasonal_mase_denom(y_train_cases: np.ndarray, train_se: np.ndarray):
    """Compute MASE denom via SE-mapped seasonal naive (not lag 52 index).
    Returns (denom, n_pairs)."""
    # filter only finite y_train_cases for dict building
    y_train_cases_arr = np.asarray(y_train_cases, dtype=float)
    train_se_arr = np.asarray(train_se)
    if len(y_train_cases_arr) == len(train_se_arr):
        _finite = np.isfinite(y_train_cases_arr)
        _y_f = y_train_cases_arr[_finite]
        _se_f = train_se_arr[_finite]
    else:
        _y_f = y_train_cases_arr[np.isfinite(y_train_cases_arr)]
        _se_f = train_se_arr
        if len(_se_f) > len(_y_f):
            _se_f = _se_f[:len(_y_f)]
    train_se_to_y = dict(zip([normalize_se(s) for s in _se_f], _y_f))
    diffs = []
    for se, y in zip(train_se, y_train_cases):
        if not np.isfinite(y):
            continue
        pred = _seasonal_lookup(train_se_to_y, se, fallback_k=4)
        if pred is not None and np.isfinite(pred):
            _d = abs(float(y) - float(pred))
            if np.isfinite(_d):
                diffs.append(_d)
    if diffs:
        _arr = np.asarray(diffs, dtype=float)
        _arr = _arr[np.isfinite(_arr)]
        if len(_arr):
            return float(np.nanmean(_arr)), len(_arr)
        else:
            return float('nan'), 0
    else:
        return float('nan'), 0



def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    """Pinball loss for quantile tau."""
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    _m = np.isfinite(diff)
    if not np.any(_m):
        return float('nan')
    _pin = np.maximum(tau * diff[_m], (tau - 1) * diff[_m])
    _pin = _pin[np.isfinite(_pin)]
    return float(np.nanmean(_pin)) if len(_pin) else float('nan')


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
        self.asymmetric_loss = config.get('asymmetric_loss', {})
        # asymmetric now implemented via custom quantile gradient (per-sample residual weighting)
        if self.asymmetric_loss.get('enabled', False):
            logger.info(f"asymmetric_loss enabled for h>={self.asymmetric_loss.get('horizon_threshold',5)} undershoot x{self.asymmetric_loss.get('undershoot_weight',2.0)}")

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
                'expand_regime_conditional': bool(self.config.get('features', {}).get('expand_regime_conditional', True)),
                'expand_peak_timing': bool(self.config.get('features', {}).get('expand_peak_timing', True)),
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

    def _compute_sample_weights(self, y_tr: pd.Series, X_tr: Optional[pd.DataFrame] = None, se_index: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """v3: peso por recencia + pico. Retorna None se desabilitado (peso uniforme)."""
        cfg = self.config.get('training', {})
        tw_cfg = cfg.get('temporal_weighting', {})
        pw_cfg = cfg.get('peak_weighting', {})
        tw_enabled = bool(tw_cfg.get('enabled', False)) if isinstance(tw_cfg, dict) else False
        pw_enabled = bool(pw_cfg.get('enabled', False)) if isinstance(pw_cfg, dict) else False
        if not tw_enabled and not pw_enabled:
            return None
        n = len(y_tr)
        w = np.ones(n, dtype=float)
        # Peak weighting (em casos escala) - finite guard
        if pw_enabled:
            _y_arr_pw = np.asarray(y_tr, dtype=float)
            _y_fin_pw = np.where(np.isfinite(_y_arr_pw), _y_arr_pw, np.nan)
            # only finite before expm1
            y_cases = np.full_like(_y_arr_pw, np.nan, dtype=float)
            _m_pw = np.isfinite(_y_arr_pw)
            if np.any(_m_pw):
                y_cases[_m_pw] = np.expm1(_y_arr_pw[_m_pw])
            # replace nan with 0 to avoid weight nan
            y_cases = np.where(np.isfinite(y_cases), y_cases, 0.0)
            w200 = float(pw_cfg.get('weight_200', 1.3))
            w500 = float(pw_cfg.get('weight_500', 1.8))
            w1000 = float(pw_cfg.get('weight_1000', 2.2))
            # aplica hierarquico: >1000 sobrescreve >500 etc
            w = np.where(y_cases > 200, w200, w)
            w = np.where(y_cases > 500, w500, w)
            w = np.where(y_cases > 1000, w1000, w)
        # Temporal decay por SE (exp(-lambda * anos_atras))
        if tw_enabled and se_index is not None and len(se_index) == n:
            try:
                half = float(tw_cfg.get('half_life_years', 3))
                lam = np.log(2) / max(half, 0.5)
                # extrai ano da SE
                anos = np.array([int(str(s)[:4]) for s in se_index], dtype=float)
                ano_max = float(np.max(anos))
                anos_atras = ano_max - anos
                w_time = np.exp(-lam * anos_atras)
                # normaliza para media 1.0 para não mudar escala global do loss
                w_time = w_time / (np.mean(w_time) + 1e-10)
                w = w * w_time
            except Exception as e:
                logger.warning(f"temporal weighting failed: {e}")
        # normaliza para media 1
        w = w / (np.mean(w) + 1e-10)
        # clip para evitar peso extremo
        w = np.clip(w, 0.3, 4.0)
        return w

    def _get_quantile_hparams(self, quantile: float) -> Dict:
        """Overrides de hyperparams por quantil (v3)."""
        qp_cfg = self.config.get('training', {}).get('quantile_hparams', {})
        # chaves podem ser string \"0.05\" ou float
        for k, v in qp_cfg.items():
            try:
                kf = float(k)
            except Exception:
                continue
            if abs(kf - quantile) < 1e-6 and isinstance(v, dict):
                return v
        return {}

    # ------------------------------------------------------------------
    # Model training (per seed, per quantile)
    # ------------------------------------------------------------------
    def train_single_model(self, X_tr: pd.DataFrame, y_tr: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series,
                           seed: int, horizon: int, quantile: float,
                           se_tr: Optional[np.ndarray] = None,
                           se_val: Optional[np.ndarray] = None) -> lgb.Booster:
        params = self.model_params.copy()
        params['random_state'] = seed
        params['seed'] = seed
        # v3: override por quantil (q05 mais sensivel)
        qh = self._get_quantile_hparams(quantile)
        if qh:
            params.update(qh)
            logger.debug(f"h{horizon} q{quantile} quantile_hparams {qh}")
        # Quantile objective - asymmetric custom grad for h>=5 when enabled (LightGBM 4.5: objective is callable via params)
        fobj = self._asymmetric_quantile_obj(quantile, horizon)
        if fobj is not None:
            # custom objective via params['objective'] callable (fobj removed in lgb 4.x)
            params['objective'] = fobj
            params['alpha'] = quantile  # kept so metric 'quantile' can use it
            params['metric'] = 'quantile'
        else:
            params['objective'] = 'quantile'
            params['alpha'] = quantile
            params['metric'] = 'quantile'

        feature_names = X_tr.columns.tolist()
        w_tr = self._compute_sample_weights(y_tr, X_tr, se_tr)
        w_val = self._compute_sample_weights(y_val, X_val, se_val)
        if w_tr is not None:
            train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names, weight=w_tr)
        else:
            train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names)
        if w_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, weight=w_val, reference=train_data)
        else:
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
                                n_iters: Optional[int] = None,
                                se_full: Optional[np.ndarray] = None) -> lgb.Booster:
        """Production model on FULL data (no early stopping)."""
        params = self.model_params.copy()
        params['random_state'] = seed
        params['seed'] = seed
        params.pop('early_stopping_rounds', None)
        # v3 per-quantile hparams
        qh = self._get_quantile_hparams(quantile)
        if qh:
            params.update(qh)
        # FULL nunca usa early stopping (sem val), então remove se qh trouxe
        params.pop('early_stopping_rounds', None)
        fobj = self._asymmetric_quantile_obj(quantile, horizon)
        if fobj is not None:
            params['objective'] = fobj  # callable objective for LGBM 4.5
            params['alpha'] = quantile
            params['metric'] = 'quantile'
        else:
            params['objective'] = 'quantile'
            params['alpha'] = quantile
            params['metric'] = 'quantile'

        feature_names = X.columns.tolist()
        w_full = self._compute_sample_weights(y, X, se_full)
        if w_full is not None:
            data = lgb.Dataset(X, label=y, feature_name=feature_names, weight=w_full)
        else:
            data = lgb.Dataset(X, label=y, feature_name=feature_names)
        rounds = n_iters or int(params.get('n_estimators', 3000))
        # FASE1 FIX: pop n_estimators para num_boost_round controlar (FULL usa mediana ~322, não 3000)
        params.pop('n_estimators', None)
        params.pop('num_iterations', None)
        params.pop('num_iteration', None)
        return lgb.train(params, data, num_boost_round=rounds)

    def enforce_non_crossing(self, y_pred_dict: Dict[float, np.ndarray],
                             log_scale=True) -> Dict[float, np.ndarray]:
        """Ensure strict monotonicity Q.05 <= Q.50 <= Q.95 via np.sort
        along the quantile axis (spec section 1)."""
        q_order = sorted(self.quantiles)
        if log_scale:
            stack = np.column_stack([y_pred_dict[tau] for tau in q_order])
        else:
            # finite guard before expm1
            cols = []
            for tau in q_order:
                _a = np.asarray(y_pred_dict[tau], dtype=float)
                _out = np.full_like(_a, np.nan, dtype=float)
                _m = np.isfinite(_a)
                if np.any(_m):
                    _out[_m] = np.expm1(_a[_m])
                cols.append(_out)
            stack = np.column_stack(cols)
        stack = np.sort(stack, axis=1)
        for j, tau in enumerate(q_order):
            y_pred_dict[tau] = stack[:, j]
        return y_pred_dict

    def _asymmetric_quantile_obj(self, tau: float, horizon: int):
        """Custom gradient for asymmetric quantile loss (undershoot penalized).
        WIS_under 8-15x indica subestimacao cronica; para h>=threshold penaliza
        residual positivo (y_true > y_pred) com peso 2x. Per-sample, nao constante.
        v3: only_median true => só mediana tem peso extra (q05/q95 já têm assimetria natural)
        """
        asym = getattr(self, 'asymmetric_loss', {})
        if not asym.get('enabled', False):
            return None
        # v3: só mediana se flag
        if asym.get('only_median', False) and abs(tau - 0.5) > 1e-6:
            return None
        h_thr = int(asym.get('horizon_threshold', 5))
        if horizon < h_thr:
            return None
        w_under = float(asym.get('undershoot_weight', 2.0))
        w_over = float(asym.get('overshoot_weight', 1.0))
        if w_under == 1.0 and w_over == 1.0:
            return None
        def _obj(preds, dataset):
            y_true = dataset.get_label()
            residual = y_true - preds  # >0 = underpredict (subestimou)
            # Pinball grad: -tau if residual>=0 else (1-tau); scale per residual sign
            grad = np.where(residual >= 0, -tau * w_under, (1.0 - tau) * w_over)
            hess = np.ones_like(grad)  # quantile hess=0 -> use 1 for LGBM
            return grad, hess
        return _obj

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def seasonal_naive_baseline(self, y_train_cases: np.ndarray, train_se: np.ndarray,
                                 test_se: np.ndarray) -> np.ndarray:
        """Seasonal naive forecast in cases scale: value from the same SE week
        in the previous year (fallback to T-52±k via epiweek_to_date/weeks_in_year, then last value).
        Unified via _seasonal_lookup (FIX Bug5)."""
        train_se_to_y = dict(zip([normalize_se(s) for s in train_se], y_train_cases))
        out = []
        last_val = float(y_train_cases[-1]) if len(y_train_cases) else 0.0
        for se in test_se:
            val = _seasonal_lookup(train_se_to_y, se, fallback_k=4)
            if val is not None:
                out.append(float(val))
            else:
                out.append(last_val)
        return np.array(out)

    def calculate_metrics(self, y_true: np.ndarray, y_pred_dict: Dict[float, np.ndarray],
                          y_train: Optional[pd.Series] = None,
                          train_se: Optional[np.ndarray] = None,
                          test_se: Optional[np.ndarray] = None,
                          horizon: int = 1) -> Dict[str, Any]:
        """Compute all metrics for one evaluation window (a fold).

        Quantile-specific formatting:
          - mae, Coverage_90: Q.50 and interval [0.05,0.95]
          - WIS: mean pinball across all quantiles + decomposition for 90% interval
          - SPL: all quantiles
          - Baseline sazonal (persistência Ano-1) para baseline
        """
        # NaN/Inf guard before expm1: only finite log values are transformed
        y_true_arr = np.asarray(y_true, dtype=float)
        y_true_cases = np.full_like(y_true_arr, np.nan, dtype=float)
        _mask_true = np.isfinite(y_true_arr)
        if np.any(_mask_true):
            y_true_cases[_mask_true] = np.expm1(y_true_arr[_mask_true])
        y_pred_dict_cases = {}
        for tau in self.quantiles:
            _arr = np.asarray(y_pred_dict[tau], dtype=float)
            _out = np.full_like(_arr, np.nan, dtype=float)
            _m = np.isfinite(_arr)
            if np.any(_m):
                _out[_m] = np.expm1(_arr[_m])
            y_pred_dict_cases[tau] = _out
        y_pred_median = y_pred_dict_cases[0.50]

        # ---- B) MAE on Q.50 only ----
        metrics = {}

        # ---- C2) WIS on all quantiles (mean pinball, cases scale) + Decomposition 90% ----
        # WIS principal = wis_pinball (media pinball correta). Decomposicao apenas para debug.
        alpha = 0.10
        q_inf = y_pred_dict_cases.get(0.05, y_pred_median)
        q_sup = y_pred_dict_cases.get(0.95, y_pred_median)
        # Sharpness, Over, Under por observação (debug apenas) - Interval Score alpha=0.10: IS=(q95-q05)+(2/alpha)*(q05-y)+...
        # finite guard: only finite triples contribute
        _finite_wis_mask = np.isfinite(y_true_cases) & np.isfinite(q_inf) & np.isfinite(q_sup) & np.isfinite(y_pred_median)
        # debug vectors filtered to finite
        sharpness_vec = np.where(_finite_wis_mask, (q_sup - q_inf), np.nan)
        over_vec = np.where(_finite_wis_mask & (y_true_cases < q_inf), (2.0 / alpha) * (q_inf - y_true_cases), np.where(_finite_wis_mask, 0.0, np.nan))
        under_vec = np.where(_finite_wis_mask & (y_true_cases > q_sup), (2.0 / alpha) * (y_true_cases - q_sup), np.where(_finite_wis_mask, 0.0, np.nan))
        median_abs_vec = np.where(_finite_wis_mask, np.abs(y_true_cases - y_pred_median), np.nan)
        # WIS correto como média de pinball (finite only, nanmean)
        wis_vals = []
        for tau in self.quantiles:
            diff = y_true_cases - y_pred_dict_cases[tau]
            _m = np.isfinite(diff)
            if not np.any(_m):
                continue
            pin = np.maximum(tau * diff[_m], (tau - 1) * diff[_m])
            # filter pin finite as well
            pin = pin[np.isfinite(pin)]
            if len(pin):
                wis_vals.append(float(np.mean(pin)))
        wis_pinball = float(np.nanmean(wis_vals)) if wis_vals else float('nan')
        # WIS principal = pinball médio (correção bug pesos)
        metrics['wis'] = wis_pinball
        metrics['wis_sharpness'] = float(np.nanmean(sharpness_vec)) if np.any(np.isfinite(sharpness_vec)) else 0.0
        metrics['wis_over'] = float(np.nanmean(over_vec)) if np.any(np.isfinite(over_vec)) else 0.0
        metrics['wis_under'] = float(np.nanmean(under_vec)) if np.any(np.isfinite(under_vec)) else 0.0
        metrics['wis_median_abs'] = float(np.nanmean(median_abs_vec)) if np.any(np.isfinite(median_abs_vec)) else 0.0
        # Mantém pinball médio também para debug e decomposição vetorial para inspeção
        metrics['wis_pinball'] = wis_pinball
        wis_total_vec = sharpness_vec + over_vec + under_vec  # FIX Bug4: sem median_abs (IS puro para 90% interval)
        metrics['wis_decomp_mean'] = float(np.nanmean(wis_total_vec)) if np.any(np.isfinite(wis_total_vec)) else wis_pinball

        # ---- Coverage 90% ----
        _cov_mask = np.isfinite(y_true_cases) & np.isfinite(q_inf) & np.isfinite(q_sup)
        if not np.any(_cov_mask):
            metrics['coverage_90'] = float('nan')
        else:
            coverage_vec = (y_true_cases[_cov_mask] >= q_inf[_cov_mask]) & (y_true_cases[_cov_mask] <= q_sup[_cov_mask])
            metrics['coverage_90'] = float(np.mean(coverage_vec)) if len(coverage_vec) else float('nan')
        # ---- Coverage 50% (q25-q75) - corrige fallback: se 0.25/0.75 não existem retorna nan
        try:
            if 0.25 not in y_pred_dict_cases or 0.75 not in y_pred_dict_cases:
                metrics['coverage_50'] = float('nan')
            else:
                q25 = y_pred_dict_cases[0.25]
                q75 = y_pred_dict_cases[0.75]
                # only finite pairs count
                _finite_cov = np.isfinite(y_true_cases) & np.isfinite(q25) & np.isfinite(q75)
                if not np.any(_finite_cov):
                    metrics['coverage_50'] = float('nan')
                else:
                    coverage50_vec = (y_true_cases[_finite_cov] >= q25[_finite_cov]) & (y_true_cases[_finite_cov] <= q75[_finite_cov])
                    metrics['coverage_50'] = float(np.mean(coverage50_vec)) if len(coverage50_vec) else float('nan')
        except Exception:
            metrics['coverage_50'] = float('nan')

        # ---- MASE (Mean Absolute Scaled Error) - FIX Bug5+Bug12 unified SE-mapped denom with floor ----
        # Unified: denom via SE mapping (weeks_in_year + fallback ±k via datas) not lag52 index; floor to avoid 1e12
        mae_model = float(np.nanmean(median_abs_vec)) if np.any(np.isfinite(median_abs_vec)) else float('nan')
        metrics['mae'] = mae_model
        mase = float('nan')
        denom_floor = float('nan')
        n_pairs = 0
        denom_seasonal = float('nan')
        if y_train is not None and len(np.asarray(y_train, dtype=float)) > 0 and train_se is not None and test_se is not None:
            _y_train_arr_mase = np.asarray(y_train, dtype=float)
            _y_train_finite_mase = _y_train_arr_mase[np.isfinite(_y_train_arr_mase)]
            if len(_y_train_finite_mase) == 0:
                y_train_cases_mase = np.array([], dtype=float)
            else:
                y_train_cases_mase = np.expm1(_y_train_finite_mase)
            try:
                denom_seasonal, n_pairs = _seasonal_mase_denom(y_train_cases_mase, np.asarray(train_se))
            except Exception as e:
                logger.warning(f"seasonal mase denom failed: {e}")
                denom_seasonal, n_pairs = float('nan'), 0
            # Floor: max(denom_seasonal, P10(|y_train|), 5 casos)
            try:
                p10 = float(np.percentile(np.abs(y_train_cases_mase), 10)) if len(y_train_cases_mase) >= 5 else 5.0
            except Exception:
                p10 = 5.0
            floor_val = max(5.0, p10)
            if np.isfinite(denom_seasonal):
                denom_floor = max(denom_seasonal, floor_val)
            else:
                denom_floor = float('nan')
            # Confiabilidade: precisa >=5 pares sazonais e denom finito >=5.0
            if n_pairs < 5 or not np.isfinite(denom_floor) or denom_floor < 5.0:
                mase = float('nan')
                logger.debug(f"MASE NaN (unreliable denom): n_pairs={n_pairs} denom={denom_floor:.2f} floor={floor_val:.2f}")
            elif np.isfinite(mae_model):
                mase = float(mae_model / denom_floor)
                if mase > 1e3:
                    logger.warning(f"MASE extreme {mase:.1f} clipped to NaN (denom {denom_floor:.2f})")
                    mase = float('nan')
        elif y_train is not None and len(np.asarray(y_train, dtype=float)) > 0:
            # sem SE disponível: marca NaN (não confiável)
            mase = float('nan')
        metrics['mase'] = mase
        metrics['mase_denom'] = float(denom_floor) if np.isfinite(denom_floor) else float('nan')
        metrics['mase_n_pairs'] = int(n_pairs)

        # ---- FBias: viés fracionário 2*Sum(P-R)/Sum(P+R) em [-2,2] ----
        try:
            sum_pred = float(np.sum(y_pred_median))
            sum_true = float(np.sum(y_true_cases))
            denom = sum_pred + sum_true
            if abs(denom) < 1e-10:
                fbias = float('nan')
            else:
                fbias = float(2.0 * (sum_pred - sum_true) / denom)
                # clip para [-2,2] por segurança numérica
                fbias = float(max(-2.0, min(2.0, fbias)))
            # mantém compat: fbias_rel/frac espelham fbias principal
            fbias_rel = fbias
            fbias_frac = fbias
        except Exception:
            fbias = float('nan'); fbias_rel=float('nan'); fbias_frac=float('nan')
        metrics['fbias'] = fbias
        metrics['fbias_rel'] = fbias_rel
        metrics['fbias_frac'] = fbias_frac

        # ---- FBias estratificado por limiar dinâmico P75 global por fold (FIX Bug7) ----
        # FIX Bug7: thr único P75 de todo y_true do fold (52 ou 49 semanas), não por fatia calendário.
        # Fold epidemiológico Sep-Ago quebrado em 2 anos calendário gerava thr instável.
        try:
            if test_se is not None and len(test_se) and len(y_true_cases) >= 4:
                thr = float(np.percentile(y_true_cases, 75))
                surto_mask = y_true_cases > thr
                calm_mask = ~surto_mask
                def _fbias_mask(mask):
                    if not np.any(mask):
                        return float('nan')
                    sp=float(np.sum(y_pred_median[mask]))
                    st=float(np.sum(y_true_cases[mask]))
                    denom=sp+st
                    if abs(denom)<1e-10:
                        return float('nan')
                    v=float(2*(sp-st)/denom)
                    return float(max(-2,min(2,v)))
                metrics['fbias_surto'] = _fbias_mask(surto_mask)
                metrics['fbias_calmaria'] = _fbias_mask(calm_mask)
                metrics['fbias_surto_n'] = int(np.sum(surto_mask))
                metrics['fbias_calmaria_n'] = int(np.sum(calm_mask))
                metrics['fbias_p75_threshold'] = thr
                metrics['fbias_p75_thresholds'] = {"global": thr}
            else:
                metrics['fbias_surto']=float('nan'); metrics['fbias_calmaria']=float('nan')
                metrics['fbias_p75_threshold']=float('nan')
                metrics['fbias_p75_thresholds'] = {}
        except Exception as e:
            metrics['fbias_surto']=float('nan'); metrics['fbias_calmaria']=float('nan')

        # ---- Etp: time-to-peak error (semanas até pico) ----
        try:
            # pico na janela de teste (52 sem): compara semana do max real vs max previsto (mediana)
            if len(y_true_cases) and len(y_pred_median):
                true_peak = int(np.argmax(y_true_cases))
                pred_peak = int(np.argmax(y_pred_median))
                etp = float(abs(true_peak - pred_peak))
            else:
                etp = float('nan')
        except Exception:
            etp = float('nan')
        metrics['etp'] = etp

        # ---- Baseline Sazonal (Persistência Ano-1) sem vazamento ----
        # Baseline usa MESMA escala que modelo: wis = mean(pinball) sobre q05/median/q95 históricos
        # com fallback unificado _seasonal_lookup (datas ±4 + string ±4) idêntico a SPL/MASE
        y_baseline_median = None
        baseline_wis = None
        baseline_mae = None
        baseline_coverage = None
        if y_train is not None and train_se is not None and test_se is not None and len(y_train) > 0:
            _y_train_arr_bl = np.asarray(y_train, dtype=float)
            _mask_bl = np.isfinite(_y_train_arr_bl)
            if not np.any(_mask_bl):
                pass
            else:
                _y_train_filt = _y_train_arr_bl[_mask_bl]
                y_train_cases_bl = np.expm1(_y_train_filt)
                y_train_cases_bl = y_train_cases_bl[np.isfinite(y_train_cases_bl)]
                _train_se_arr = np.asarray(train_se)
                # align se to finite mask if lengths match
                if len(_train_se_arr) == len(_y_train_arr_bl):
                    _train_se_bl = _train_se_arr[_mask_bl]
                else:
                    _train_se_bl = _train_se_arr
                # filter y_train_cases_bl finite for dict
                y_train_cases_for_dict = np.expm1(_y_train_filt)
                y_train_cases_for_dict = y_train_cases_for_dict[np.isfinite(y_train_cases_for_dict)]
                # if filtering removed some, need to also filter _train_se_bl accordingly to keep alignment
                # easiest: rebuild dict from _train_se_bl and y_train_cases_for_dict truncated to min len
                _min_len = min(len(_train_se_bl), len(y_train_cases_for_dict))
                _train_se_bl = _train_se_bl[:_min_len]
                y_train_cases_for_dict = y_train_cases_for_dict[:_min_len]
                se_to_y = dict(zip([normalize_se(s) for s in _train_se_bl], y_train_cases_for_dict))
                # Histórico por semana para quantis do baseline (empírico) - mesmo fallback que SPL
                se_hist_bl: Dict[str, List[float]] = {}
                for tr_se, tr_y in zip([normalize_se(s) for s in _train_se_bl], y_train_cases_for_dict):
                    if np.isfinite(tr_y):
                        se_hist_bl.setdefault(tr_se[4:], []).append(float(tr_y))
                # Build baseline predictions per quantile using se_hist_bl with unified fallback
                baseline_pred_dict = {tau: [] for tau in self.quantiles}
                for se in test_se:
                    week = normalize_se(se)[4:]
                    hist = se_hist_bl.get(week)
                    if hist is None or len(hist) == 0:
                        hist = None
                        try:
                            se_str = normalize_se(se)
                            ano_t = int(se_str[:4]); semana_t = int(se_str[4:])
                            tgt_date = epiweek_to_date(ano_t, semana_t)
                            for k in range(1, 5):
                                for sign in (-1, 1):
                                    cand_date = tgt_date + pd.Timedelta(days=sign*k*7)
                                    cand_se = date_to_epiweek(cand_date)
                                    cand_week = cand_se[4:]
                                    if cand_week in se_hist_bl and se_hist_bl[cand_week]:
                                        hist = se_hist_bl[cand_week]
                                        break
                                if hist is not None:
                                    break
                            if hist is None:
                                for k in range(1, 5):
                                    for cand_w in (f"{semana_t-k:02d}", f"{semana_t+k:02d}"):
                                        if cand_w in se_hist_bl and se_hist_bl[cand_w]:
                                            hist = se_hist_bl[cand_w]
                                            break
                                    if hist is not None:
                                        break
                        except Exception:
                            hist = None
                    if hist is None or len(hist) == 0:
                        hist = list(y_train_cases_bl) if len(y_train_cases_bl) else [0.0]
                    for tau in self.quantiles:
                        hist_arr = np.asarray(hist, dtype=float)
                        hist_arr = hist_arr[np.isfinite(hist_arr)]
                        if len(hist_arr) == 0:
                            hist_arr = y_train_cases_bl[np.isfinite(y_train_cases_bl)] if len(y_train_cases_bl) else np.array([0.0])
                        baseline_pred_dict[tau].append(float(np.quantile(hist_arr, tau)) if len(hist_arr) else float('nan'))
                for tau in self.quantiles:
                    baseline_pred_dict[tau] = np.array(baseline_pred_dict[tau], dtype=float)
                y_baseline_median = baseline_pred_dict[0.50]
                q_baseline_inf = baseline_pred_dict.get(0.05, y_baseline_median)
                q_baseline_sup = baseline_pred_dict.get(0.95, y_baseline_median)
                # Baseline WIS correto = mean(pinball) idêntico ao modelo
                wis_bl_vals = []
                for tau in self.quantiles:
                    _pred = baseline_pred_dict[tau]
                    _diff = y_true_cases - _pred
                    _m = np.isfinite(_diff)
                    if not np.any(_m):
                        continue
                    _pin = np.maximum(tau * _diff[_m], (tau - 1) * _diff[_m])
                    _pin = _pin[np.isfinite(_pin)]
                    if len(_pin):
                        wis_bl_vals.append(float(np.nanmean(_pin)))
                baseline_wis = float(np.nanmean(wis_bl_vals)) if wis_bl_vals else float('nan')
                # Debug IS decomposition kept separately but not as principal
                _mae_mask = np.isfinite(y_true_cases) & np.isfinite(y_baseline_median)
                baseline_mae = float(np.nanmean(np.abs(y_true_cases[_mae_mask] - y_baseline_median[_mae_mask]))) if np.any(_mae_mask) else float('nan')
                _cov_mask_bl = np.isfinite(y_true_cases) & np.isfinite(q_baseline_inf) & np.isfinite(q_baseline_sup)
                baseline_coverage = float(np.mean((y_true_cases[_cov_mask_bl] >= q_baseline_inf[_cov_mask_bl]) & (y_true_cases[_cov_mask_bl] <= q_baseline_sup[_cov_mask_bl]))) if np.any(_cov_mask_bl) else float('nan')
                metrics['baseline_wis'] = baseline_wis
                metrics['baseline_mae'] = baseline_mae
                metrics['baseline_coverage_90'] = baseline_coverage
                # optional debug IS
                try:
                    _sharp_bl = np.where(_cov_mask_bl, q_baseline_sup - q_baseline_inf, np.nan)
                    _over_bl = np.where(_cov_mask_bl & (y_true_cases < q_baseline_inf), (2.0 / alpha) * (q_baseline_inf - y_true_cases), np.where(_cov_mask_bl, 0.0, np.nan))
                    _under_bl = np.where(_cov_mask_bl & (y_true_cases > q_baseline_sup), (2.0 / alpha) * (y_true_cases - q_baseline_sup), np.where(_cov_mask_bl, 0.0, np.nan))
                    metrics['baseline_wis_sharpness'] = float(np.nanmean(_sharp_bl)) if np.any(np.isfinite(_sharp_bl)) else float('nan')
                    metrics['baseline_wis_over'] = float(np.nanmean(_over_bl)) if np.any(np.isfinite(_over_bl)) else float('nan')
                    metrics['baseline_wis_under'] = float(np.nanmean(_under_bl)) if np.any(np.isfinite(_under_bl)) else float('nan')
                except Exception:
                    pass
        else:
            pass

        # ---- A) SPL per quantile ----
        spl = {}
        pl_model_vals = {}
        for tau in self.quantiles:
            y_pred_tau = y_pred_dict_cases[tau]
            diff = y_true_cases - y_pred_tau
            _mask_pl = np.isfinite(diff)
            if np.any(_mask_pl):
                _pin = np.maximum(tau * diff[_mask_pl], (tau - 1) * diff[_mask_pl])
                _pin = _pin[np.isfinite(_pin)]
                pl_model = float(np.nanmean(_pin)) if len(_pin) else float('nan')
            else:
                pl_model = float('nan')
            pl_model_vals[tau] = pl_model
            # Seasonal quantile naive baseline from training data
            if y_train is not None and len(np.asarray(y_train, dtype=float)) >= 52 and train_se is not None and test_se is not None:
                _y_train_arr_spl = np.asarray(y_train, dtype=float)
                _mask_spl = np.isfinite(_y_train_arr_spl)
                _y_train_filt_spl = _y_train_arr_spl[_mask_spl]
                if len(_y_train_filt_spl) == 0:
                    y_train_cases = np.array([], dtype=float)
                    _train_se_filt = np.asarray(train_se)
                else:
                    y_train_cases = np.expm1(_y_train_filt_spl)
                    y_train_cases = y_train_cases[np.isfinite(y_train_cases)]
                    _train_se_arr = np.asarray(train_se)
                    if len(_train_se_arr) == len(_y_train_arr_spl):
                        _train_se_filt = _train_se_arr[_mask_spl]
                        # also filter to keep only finite expm1
                        _finite_exp = np.isfinite(np.expm1(_y_train_filt_spl))
                        y_train_cases = y_train_cases[_finite_exp]
                        _train_se_filt = _train_se_filt[_finite_exp]
                    else:
                        _train_se_filt = _train_se_arr
                se_to_hist: Dict[str, List[float]] = {}
                for tr_se, tr_y in zip([normalize_se(s) for s in _train_se_filt], y_train_cases):
                    if np.isfinite(tr_y):
                        week = tr_se[4:]  # YYYYWW -> WW
                        se_to_hist.setdefault(week, []).append(float(tr_y))
                naive_preds = []
                for te_se in test_se:
                    week = normalize_se(te_se)[4:]
                    hist = se_to_hist.get(week)
                    if hist is None or len(hist) == 0:
                        # FIX Bug5: fallback ±k via datas (weeks_in_year) antes de global
                        hist = None
                        try:
                            se_str = normalize_se(te_se)
                            ano_t = int(se_str[:4]); semana_t = int(se_str[4:])
                            tgt_date = epiweek_to_date(ano_t, semana_t)
                            for k in range(1,5):
                                for sign in (-1, 1):
                                    cand_date = tgt_date + pd.Timedelta(days=sign*k*7)
                                    cand_se = date_to_epiweek(cand_date)
                                    cand_week = cand_se[4:]
                                    if cand_week in se_to_hist and se_to_hist[cand_week]:
                                        hist = se_to_hist[cand_week]
                                        break
                                if hist is not None:
                                    break
                            if hist is None:
                                for k in range(1,5):
                                    for cand_w in (f"{semana_t-k:02d}", f"{semana_t+k:02d}"):
                                        if cand_w in se_to_hist and se_to_hist[cand_w]:
                                            hist = se_to_hist[cand_w]
                                            break
                                    if hist is not None:
                                        break
                        except Exception:
                            hist = None
                    if hist is None or len(hist)==0:
                        hist = list(y_train_cases) if len(y_train_cases) else [0.0]
                    # filter hist finite
                    hist_arr = np.asarray(hist, dtype=float)
                    hist_arr = hist_arr[np.isfinite(hist_arr)]
                    if len(hist_arr)==0:
                        hist_arr = y_train_cases[np.isfinite(y_train_cases)] if len(y_train_cases) else np.array([0.0])
                    naive_preds.append(float(np.quantile(hist_arr, tau)) if len(hist_arr) else float('nan'))
                naive_preds = np.array(naive_preds, dtype=float)
                _diff_naive = y_true_cases - naive_preds
                _m_naive = np.isfinite(_diff_naive)
                if np.any(_m_naive):
                    _pin_naive = np.maximum(tau * _diff_naive[_m_naive], (tau - 1) * _diff_naive[_m_naive])
                    _pin_naive = _pin_naive[np.isfinite(_pin_naive)]
                    pl_naive = float(np.nanmean(_pin_naive)) if len(_pin_naive) else float('nan')
                else:
                    pl_naive = float('nan')
            else:
                # corrige or bug: se len<52 ou train_se None, pl_naive = nan e spl = nan, não fallback para median
                pl_naive = float('nan')
                # explicitly handle finite: if pl_model is finite we could fallback? Task says nan not fallback.
                # keep nan
            # explicit nan handling for spl
            if not np.isfinite(pl_naive) or abs(pl_naive) < 1e-12:
                spl[tau] = float('nan')
            elif not np.isfinite(pl_model):
                spl[tau] = float('nan')
            else:
                spl[tau] = float(pl_model / (pl_naive + 1e-10))
        metrics['spl'] = {f"q{int(tau*1000):04d}": spl[tau] for tau in self.quantiles}
        metrics['spl_matrix_row'] = [spl[tau] for tau in sorted(self.quantiles)]

        # ---- Non-crossing check (raw, before post-process) - finite guard
        crossed = False
        q_order = sorted(self.quantiles)
        for i in range(len(q_order) - 1):
            _a = np.asarray(y_pred_dict[q_order[i]], dtype=float)
            _b = np.asarray(y_pred_dict[q_order[i+1]], dtype=float)
            _m = np.isfinite(_a) & np.isfinite(_b)
            if not np.any(_m):
                continue
            # only finite before expm1
            _a_c = np.full_like(_a, np.nan, dtype=float)
            _b_c = np.full_like(_b, np.nan, dtype=float)
            _a_c[_m] = np.expm1(_a[_m])
            _b_c[_m] = np.expm1(_b[_m])
            if np.any(_a_c[_m] > _b_c[_m]):
                crossed = True
                break
        metrics['quantile_crossed'] = crossed

        # mae removido

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

        MAE and WIS are computed on period slice; SPL scaled by
        seasonal naive baseline per week+quantile; mae timing error.
        """
        periods = self.stratify_months(se_array)

        # Per-week empirical distribution of cases in TRAINING (for naive quantiles) - finite guard
        se_hist: Dict[str, List[float]] = {}
        if y_train is not None and train_se is not None:
            _y_train_arr = np.asarray(y_train, dtype=float)
            _mask_yt = np.isfinite(_y_train_arr)
            _y_train_filt = _y_train_arr[_mask_yt]
            _train_se_arr = np.asarray(train_se)
            if len(_train_se_arr) == len(_y_train_arr):
                _train_se_filt = _train_se_arr[_mask_yt]
            else:
                _train_se_filt = _train_se_arr
            yt = np.full_like(_y_train_filt, np.nan, dtype=float)
            _m = np.isfinite(_y_train_filt)
            if np.any(_m):
                yt[_m] = np.expm1(_y_train_filt[_m])
            yt = yt[np.isfinite(yt)]
            _train_se_filt = _train_se_filt[:len(yt)] if len(_train_se_filt) > len(yt) else _train_se_filt
            for tr_se, val in zip([normalize_se(s) for s in _train_se_filt], yt):
                if np.isfinite(val):
                    se_hist.setdefault(normalize_se(tr_se)[4:], []).append(float(val))

        result = {}
        for period in ['outbreak', 'calm']:
            mask = periods == period
            if not mask.any():
                continue
            y_true_p = y_true_cases[mask]
            med_p = y_pred_dict_cases[0.50][mask]
            # finite guard for mae
            _mae_mask = np.isfinite(y_true_p) & np.isfinite(med_p)
            if np.any(_mae_mask):
                mae_p = float(np.nansum(np.abs(y_true_p[_mae_mask] - med_p[_mae_mask])))
                mae_den_p = float(np.nansum(y_true_p[_mae_mask]))
                mae_val = float(np.nanmean(np.abs(y_true_p[_mae_mask] - med_p[_mae_mask])))
            else:
                mae_p = float('nan')
                mae_den_p = float('nan')
                mae_val = float('nan')
            entry = {
                'n_samples': int(mask.sum()),
                'mae': mae_val,
                'mae_den': mae_den_p,
            }
            # WIS for this period (mean pinball across quantiles) - finite
            wis_vals_p = []
            for tau in sorted(self.quantiles):
                diff_p = y_true_p - y_pred_dict_cases[tau][mask]
                _m_p = np.isfinite(diff_p)
                if not np.any(_m_p):
                    continue
                _pin = np.maximum(tau * diff_p[_m_p], (tau - 1) * diff_p[_m_p])
                _pin = _pin[np.isfinite(_pin)]
                if len(_pin):
                    wis_vals_p.append(float(np.nanmean(_pin)))
            entry['wis'] = float(np.nanmean(wis_vals_p)) if wis_vals_p else float('nan')

            if period in ['outbreak', 'calm']:
                # SPL row (3 quantiles scaled by seasonal naive pinball) - agora para ambos os períodos com finite guard
                spl_row = []
                for tau in sorted(self.quantiles):
                    y_pred_tau = y_pred_dict_cases[tau][mask]
                    diff = y_true_p - y_pred_tau
                    _m_model = np.isfinite(diff)
                    if np.any(_m_model):
                        _pin_m = np.maximum(tau * diff[_m_model], (tau - 1) * diff[_m_model])
                        _pin_m = _pin_m[np.isfinite(_pin_m)]
                        pl_model = float(np.nanmean(_pin_m)) if len(_pin_m) else float('nan')
                    else:
                        pl_model = float('nan')
                    # Empirical naive quantile prediction for each test week - FIX Bug5 fallback ±k via datas
                    if se_hist:
                        naive_preds = []
                        # iterate the test SEs that fell in this period
                        for te_se in np.asarray(se_array)[mask]:
                            week = normalize_se(te_se)[4:]
                            hist = se_hist.get(week)
                            if hist is None or len(hist)==0:
                                # fallback ±k via dates before global
                                try:
                                    se_str = normalize_se(te_se)
                                    ano_t = int(se_str[:4]); semana_t = int(se_str[4:])
                                    tgt_date = epiweek_to_date(ano_t, semana_t)
                                    for k in range(1,5):
                                        for sign in (-1, 1):
                                            cand_date = tgt_date + pd.Timedelta(days=sign*k*7)
                                            cand_se = date_to_epiweek(cand_date)
                                            cand_week = cand_se[4:]
                                            if cand_week in se_hist and se_hist[cand_week]:
                                                hist = se_hist[cand_week]
                                                break
                                        if hist is not None and len(hist)>0:
                                            break
                                    if hist is None or len(hist)==0:
                                        for k in range(1,5):
                                            for cand_w in (f"{semana_t-k:02d}", f"{semana_t+k:02d}"):
                                                if cand_w in se_hist and se_hist[cand_w]:
                                                    hist = se_hist[cand_w]
                                                    break
                                            if hist is not None and len(hist)>0:
                                                break
                                except Exception:
                                    hist = None
                            if hist is None or len(hist)==0:
                                _ytr_arr = np.asarray(y_train, dtype=float)
                                _ytr_fin = _ytr_arr[np.isfinite(_ytr_arr)]
                                hist = list(np.expm1(_ytr_fin)) if len(_ytr_fin) else [0.0]
                            hist_arr = np.asarray(hist, dtype=float)
                            hist_arr = hist_arr[np.isfinite(hist_arr)]
                            if len(hist_arr)==0:
                                _ytr_arr2 = np.asarray(y_train, dtype=float)
                                _ytr_fin2 = _ytr_arr2[np.isfinite(_ytr_arr2)]
                                hist_arr = np.expm1(_ytr_fin2) if len(_ytr_fin2) else np.array([0.0])
                            naive_preds.append(float(np.quantile(hist_arr, tau)) if len(hist_arr) else float('nan'))
                        naive_preds = np.array(naive_preds, dtype=float)
                        _diff_n = y_true_p - naive_preds
                        _m_n = np.isfinite(_diff_n)
                        if np.any(_m_n):
                            _pin_n = np.maximum(tau * _diff_n[_m_n], (tau - 1) * _diff_n[_m_n])
                            _pin_n = _pin_n[np.isfinite(_pin_n)]
                            pl_naive = float(np.nanmean(_pin_n)) if len(_pin_n) else float('nan')
                        else:
                            pl_naive = float('nan')
                    else:
                        pl_naive = float('nan')
                    if not np.isfinite(pl_naive) or abs(pl_naive) < 1e-12 or not np.isfinite(pl_model):
                        spl_row.append(float('nan'))
                    else:
                        spl_row.append(float(pl_model / (pl_naive + 1e-10)))
                entry['spl_row'] = spl_row
            result[period] = entry
        return result

    # ------------------------------------------------------------------
    # Per-horizon training
    # ------------------------------------------------------------------
    def train_horizon(self, df: pd.DataFrame, horizon: int) -> Dict:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training quantile models for horizon h{horizon}")
        logger.info(f"{'='*60}")

        # v3: gap por horizonte (gap = h) se habilitado, senão gap fixo config
        base_gap = self.wf_config.get('gap_weeks', 8)
        if self.config.get('training', {}).get('gap_per_horizon', False):
            gap_weeks = int(horizon)
            logger.info(f"  gap_per_horizon=true: gap={gap_weeks} para h{horizon} (base {base_gap})")
        else:
            gap_weeks = int(base_gap)
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
            gap_weeks=gap_weeks,
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

        # Per-fold preparation: separate_quantile_features => 380 feats independentes por tau
        fs_cfg = self.config.get('training', {}).get('feature_selection', {})
        separate_mode = bool(fs_cfg.get('separate_quantile_features', False))
        fold_prep: Dict[int, Dict[str, Any]] = {}
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            train_feature_cols = [c for c in X_train.columns
                                  if not X_train[c].isna().all() and X_train[c].nunique() > 1]
            X_train = X_train[train_feature_cols].fillna(0)
            X_test = X_test.reindex(columns=train_feature_cols, fill_value=0)

            # ano_normalizado from training only (fix: usa max fixo 2030 para futuro sem clip)
            if 'ano_raw' in X_train.columns:
                ano_min = float(X_train['ano_raw'].min())
                ano_max_train = float(X_train['ano_raw'].max())
                # Fixo 2030 para permitir predicao 2026-2030 sem clip [0,1]; per-fold ainda usa treino mas escala ate 2030
                ano_max_fixed = 2030.0
                # den = fixed range, mas mantem ano_min 2014 => futuro 2027 -> ~0.81, 2030->1.0
                # Se treino max <2030, usa 2030; senao usa train max (caso dados futuros)
                ano_max = max(ano_max_train, ano_max_fixed)
                X_train['ano_normalizado'] = (X_train['ano_raw'] - ano_min) / (ano_max - ano_min + 1e-6)
                X_test['ano_normalizado'] = (X_test['ano_raw'] - ano_min) / (ano_max - ano_min + 1e-6)
                X_train = X_train.drop(columns=['ano_raw'])
                X_test = X_test.drop(columns=['ano_raw'])
                # guarda params para debug (sera sobrescrito pelo full depois)
                if horizon not in self.ano_normalization_params:
                    self.ano_normalization_params[horizon] = {'ano_min': ano_min, 'ano_max': ano_max}

            validation_gap = self.config.get('training',{}).get('validation',{}).get('gap_weeks', self.wf_config.get('gap_weeks',8))
            if self.config.get('training', {}).get('gap_per_horizon', False):
                gap = int(horizon)
            else:
                gap = int(validation_gap)
            assert gap >= horizon, f"gap ({gap}) must be >= horizon ({horizon}) to prevent leakage; set gap_per_horizon=true or increase validation.gap_weeks"
            val_size = max(1, len(X_train) // 5)
            cutoff = len(X_train) - val_size
            assert cutoff - gap > 0, "Not enough training data for gap+val"
            X_tr_base, X_val_base = X_train.iloc[:cutoff - gap], X_train.iloc[cutoff:]
            y_tr_base, y_val_base = y_train.iloc[:cutoff - gap], y_train.iloc[cutoff:]
            # v3: SE para temporal weighting (slice alinhado ao X_tr)
            se_train_all = se_index[train_idx]
            se_tr_base = se_train_all[:cutoff - gap]
            se_val = se_train_all[cutoff:]
            # se_full para production retrain
            se_full_all = se_index  # será usado depois

            if separate_mode:
                # 380 por quantil independentes (sem leakage so X_tr_base)
                per_tau_cols = self._get_selected_columns_per_quantile(X_tr_base, y_tr_base)
                X_tr_dict = {tau: X_tr_base[cols] for tau, cols in per_tau_cols.items()}
                X_val_dict = {tau: X_val_base.reindex(columns=cols, fill_value=0) for tau, cols in per_tau_cols.items()}
                X_test_dict = {tau: X_test.reindex(columns=cols, fill_value=0) for tau, cols in per_tau_cols.items()}
                _y_test_vals = np.asarray(y_test.values, dtype=float)
                _y_test_raw = np.full_like(_y_test_vals, np.nan, dtype=float)
                _m_y = np.isfinite(_y_test_vals)
                if np.any(_m_y):
                    _y_test_raw[_m_y] = np.expm1(_y_test_vals[_m_y])
                fold_prep[fold_idx] = {
                    'X_tr_dict': X_tr_dict, 'X_val_dict': X_val_dict, 'X_test_dict': X_test_dict,
                    'X_tr': X_tr_base, 'X_val': X_val_base, 'X_test': X_test,  # base for compat
                    'per_tau_cols': per_tau_cols,
                    'y_tr': y_tr_base, 'y_val': y_val_base,
                    'y_test': y_test.values,
                    'y_test_raw': _y_test_raw,
                    'y_train': y_train.values,
                    'train_se': se_index[train_idx],
                    'test_se': se_index[test_idx],
                    'se_tr_base': se_tr_base,
                    'se_val': se_val,
                    'se_train_all': se_train_all,
                }
            else:
                X_tr, X_val, X_test_sel = self._apply_feature_selection(X_tr_base, y_tr_base, X_val_base, X_test)
                _y_test_vals2 = np.asarray(y_test.values, dtype=float)
                _y_test_raw2 = np.full_like(_y_test_vals2, np.nan, dtype=float)
                _m_y2 = np.isfinite(_y_test_vals2)
                if np.any(_m_y2):
                    _y_test_raw2[_m_y2] = np.expm1(_y_test_vals2[_m_y2])
                fold_prep[fold_idx] = {
                    'X_tr': X_tr, 'X_val': X_val, 'X_test': X_test_sel,
                    'y_tr': y_tr_base, 'y_val': y_val_base,
                    'y_test': y_test.values,
                    'y_test_raw': _y_test_raw2,
                    'y_train': y_train.values,
                    'train_se': se_index[train_idx],
                    'test_se': se_index[test_idx],
                    'se_tr_base': se_tr_base,
                    'se_val': se_val,
                    'se_train_all': se_train_all,
                }

        # Train: for each seed, for each quantile, for each fold.
        models = {tau: {seed: None for seed in self.ensemble_seeds} for tau in self.quantiles}
        fold_predictions: Dict[int, Dict[float, Dict[int, np.ndarray]]] = {}

        for seed in self.ensemble_seeds:
            logger.info(f"\n  Training seed {seed}...")
            fold_best_iters: Dict[float, List[int]] = {tau: [] for tau in self.quantiles}
            for fold_idx, prep in fold_prep.items():
                for tau in self.quantiles:
                    if separate_mode and 'X_tr_dict' in prep:
                        X_tr_tau = prep['X_tr_dict'][tau]
                        X_val_tau = prep['X_val_dict'][tau]
                        X_test_tau = prep['X_test_dict'][tau]
                    else:
                        X_tr_tau = prep['X_tr']
                        X_val_tau = prep['X_val']
                        X_test_tau = prep['X_test']
                    model = self.train_single_model(X_tr_tau, prep['y_tr'],
                                                    X_val_tau, prep['y_val'],
                                                    seed, horizon, tau,
                                                    se_tr=prep.get('se_tr_base'),
                                                    se_val=prep.get('se_val'))
                    fold_best_iters[tau].append(model.best_iteration)
                    y_pred = model.predict(X_test_tau, num_iteration=model.best_iteration)
                    fold_predictions.setdefault(fold_idx, {}).setdefault(tau, {})[seed] = y_pred

            # Production retrain on FULL data for each quantile - FIX Bug1 circular recalibration
            # FIX Bug1: se quantile_recalibration.enabled true, train FULL apenas até max_se do penúltimo fold (holdout)
            # fold_series nunca intersecta X_full quando enabled (holdout 2025 parcial). Se disabled, FULL usa todos dados e recal não é aplicado em inference.
            recal_cfg_full = self.config.get('quantile_recalibration', {}) if isinstance(self.config.get('quantile_recalibration', {}), dict) else {}
            recal_enabled_full = bool(recal_cfg_full.get('enabled', False))
            # determina cutoff holdout (penúltimo fold max SE) se enabled
            _cutoff_se_holdout = None
            if recal_enabled_full and len(splits) >= 2:
                try:
                    # splits[-2] é penúltimo; seu test_idx define max SE antes do holdout
                    penultimate_test_idx = splits[-2][1]
                    penultimate_se = se_index[penultimate_test_idx]
                    _cutoff_se_holdout = str(max(penultimate_se))
                    logger.info(f"  h{horizon} recalibration HOLDOUT enabled: FULL train até SE <= {_cutoff_se_holdout} (exclui último fold holdout)")
                except Exception as e:
                    logger.warning(f"holdout cutoff failed: {e}")
                    _cutoff_se_holdout = None
            # prepara dados FULL filtrados se holdout ativo
            if _cutoff_se_holdout is not None:
                mask_full = np.array([normalize_se(s) <= _cutoff_se_holdout for s in se_index])
                if int(mask_full.sum()) < 100:
                    logger.warning(f"  h{horizon} holdout mask muito pequena ({mask_full.sum()}), fallback para FULL completo")
                    X_full_src = X
                    y_full_src = y
                    se_full_src = se_index
                else:
                    X_full_src = X[mask_full]
                    y_full_src = y[mask_full]
                    se_full_src = se_index[mask_full]
                    logger.info(f"  h{horizon} FULL holdout: {len(X_full_src)}/{len(X)} samples (cutoff {_cutoff_se_holdout})")
            else:
                X_full_src = X
                y_full_src = y
                se_full_src = se_index
            full_feature_cols = [c for c in X_full_src.columns
                                 if not X_full_src[c].isna().all() and X_full_src[c].nunique() > 1]
            X_full_base = X_full_src[full_feature_cols].fillna(0)
            if 'ano_raw' in X_full_base.columns:
                ano_min = float(X_full_base['ano_raw'].min())
                ano_max_train = float(X_full_base['ano_raw'].max())
                ano_max = max(ano_max_train, 2030.0)
                X_full_base['ano_normalizado'] = (X_full_base['ano_raw'] - ano_min) / (ano_max - ano_min + 1e-6)
                X_full_base = X_full_base.drop(columns=['ano_raw'])
                self.ano_normalization_params[horizon] = {'ano_min': ano_min, 'ano_max': ano_max}
            if separate_mode:
                per_tau_full_cols = self._get_selected_columns_per_quantile(X_full_base, y_full_src)
                # feature_names[horizon] passa a ser dict tau->list
                self.feature_names[horizon] = per_tau_full_cols
                logger.info(f"  h{horizon} separate features FULL: " + ", ".join([f"q{int(t*1000):04d}:{len(c)}" for t,c in per_tau_full_cols.items()]))
            else:
                X_full_sel, _, _ = self._apply_feature_selection(X_full_base, y_full_src, None, None)
                if horizon not in self.feature_names or seed == self.ensemble_seeds[0]:
                    self.feature_names[horizon] = X_full_sel.columns.tolist()
                per_tau_full_cols = {tau: self.feature_names[horizon] for tau in self.quantiles}
                X_full_base = X_full_sel

            for tau in self.quantiles:
                n_iters = int(np.mean(fold_best_iters[tau])) if fold_best_iters[tau] else None
                if separate_mode:
                    X_full_tau = X_full_base[per_tau_full_cols[tau]]
                else:
                    X_full_tau = X_full_base
                models[tau][seed] = self.train_single_model_full(X_full_tau, y_full_src, seed, horizon, tau, n_iters, se_full=se_full_src)

        # ------------------------------------------------------------------
        # Ensemble metrics: average predictions across seeds per (fold, quantile)
        # ------------------------------------------------------------------
        ensemble_fold_metrics = []
        spl_rows = []      # (n_folds x 4)
        for fold_idx, prep in fold_prep.items():
            y_pred_ens = {}
            for tau in self.quantiles:
                preds = [fold_predictions[fold_idx][tau][s] for s in self.ensemble_seeds]
                y_pred_ens[tau] = np.nanmean(np.array(preds, dtype=float), axis=0)
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
            # Stratified (outbreak/calm) using fold's SEs for the median - finite guard before expm1
            y_pred_cases = {}
            for tau in self.quantiles:
                _a = np.asarray(y_pred_ens[tau], dtype=float)
                _out = np.full_like(_a, np.nan, dtype=float)
                _m = np.isfinite(_a)
                if np.any(_m):
                    _out[_m] = np.expm1(_a[_m])
                y_pred_cases[tau] = _out
            mets['stratified'] = self.calculate_stratified(prep['y_test_raw'], y_pred_cases,
                                                           prep['test_se'], prep['y_train'],
                                                           prep['train_se'])
            ensemble_fold_metrics.append(mets)
            spl_rows.append(mets['spl_matrix_row'])

        ensemble_avg = self._average_metrics_list(ensemble_fold_metrics)
        ensemble_avg['horizon'] = horizon
        ensemble_avg['n_seeds'] = len(self.ensemble_seeds)
        ensemble_avg['n_folds'] = len(ensemble_fold_metrics)
        # 2025 parcial (49 sem) excluido da media principal: ensemble_avg passa a ser media 3 folds completos (2022-2024)
        # sem leakage, apenas filtragem pos-validacao por tamanho do teste
        partial_folds = [m['fold'] for m in ensemble_fold_metrics if len(fold_prep[m['fold']]['test_se']) < 52]
        if partial_folds:
            filtered = [m for m in ensemble_fold_metrics if m['fold'] not in partial_folds]
            if filtered:
                avg_3 = self._average_metrics_list(filtered)
                # preserva media 4 folds como referencia, mas ensemble_avg principal vira 3 folds
                ensemble_avg_full = ensemble_avg.copy()
                ensemble_avg = avg_3
                ensemble_avg['horizon'] = horizon
                ensemble_avg['n_seeds'] = len(self.ensemble_seeds)
                ensemble_avg['n_folds'] = len(filtered)
                ensemble_avg['n_folds_full'] = len(ensemble_fold_metrics)
                ensemble_avg['excluded_partial_folds'] = partial_folds
                ensemble_avg['full_4folds_avg'] = ensemble_avg_full
                # SPL robusto (3 folds)
                if spl_rows:
                    # recalcula SPL apenas dos folds filtrados - nan guard
                    filtered_spl = [ensemble_fold_metrics[i]['spl_matrix_row'] for i in range(len(ensemble_fold_metrics)) if ensemble_fold_metrics[i]['fold'] not in partial_folds]
                    if filtered_spl:
                        ensemble_avg['spl_mean'] = list(np.nanmean(np.array(filtered_spl, dtype=float), axis=0))
                        ensemble_avg['spl_std'] = list(np.nanstd(np.array(filtered_spl, dtype=float), axis=0))
                logger.info(f"  h{horizon} 2025 parcial {partial_folds} excluido da media: mae 3folds={ensemble_avg.get('mae', float('nan')):.3f} vs 4folds={ensemble_avg_full.get('mae', float('nan')):.3f}")
                # SPL matrix row (mean across folds) ja recalculado acima; pula bloco abaixo
                spl_recalc_done = True
            else:
                spl_recalc_done = False
        else:
            spl_recalc_done = False
        # SPL matrix row (mean across folds) - so se nao recalculado - nan guard
        if not spl_recalc_done and spl_rows:
            ensemble_avg['spl_mean'] = list(np.nanmean(np.array(spl_rows, dtype=float), axis=0))
            ensemble_avg['spl_std'] = list(np.nanstd(np.array(spl_rows, dtype=float), axis=0))

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
                y_pred_ens = np.nanmean(np.array(preds, dtype=float), axis=0)
                # enforce será aplicado em conjunto abaixo; por enquanto guarda raw temporário
                # finite guard for expm1
                _y_pred_cases = np.full_like(y_pred_ens, np.nan, dtype=float)
                _m_pred = np.isfinite(y_pred_ens)
                if np.any(_m_pred):
                    _y_pred_cases[_m_pred] = np.expm1(y_pred_ens[_m_pred])
                self.fold_series[horizon][tau].append({
                    'fold': fold_idx,
                    'test_se': [str(s).zfill(6) for s in prep['test_se']],
                    'y_test_casos': [float(x) if np.isfinite(x) else float('nan') for x in prep['y_test_raw']],
                    'y_pred_casos': [float(v) if np.isfinite(v) else float('nan') for v in _y_pred_cases],
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
            'spl_matrix': list(np.nanmean(np.array(spl_rows, dtype=float), axis=0)) if spl_rows else [],
        }

        logger.info(f"\n  h{horizon} ENSEMBLE: mae={ensemble_avg.get('mae', float('nan')):.3f}, "
                    f"WIS={ensemble_avg['wis']:.1f} (sharp {ensemble_avg.get('wis_sharpness',0):.1f} over {ensemble_avg.get('wis_over',0):.1f} under {ensemble_avg.get('wis_under',0):.1f}), Coverage90={ensemble_avg.get('coverage_90',0):.2f} Etp={ensemble_avg.get('etp',0):.1f} ")
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
            # mae weighted by n with nan handling
            _mae_vals = np.array([r.get('mae', np.nan) for r in rows], dtype=float)
            _mae_weights = np.array([r['n_samples'] if np.isfinite(r.get('mae', np.nan)) else 0 for r in rows], dtype=float)
            if np.any(np.isfinite(_mae_vals)) and np.sum(_mae_weights) > 0:
                mae = float(np.nansum(_mae_vals * _mae_weights) / (np.sum(_mae_weights) + 1e-10))
            else:
                mae = float(np.nanmean(_mae_vals)) if np.any(np.isfinite(_mae_vals)) else float('nan')
            # WIS media ponderada por n (igual ao global ponderado por amostras) com nan guard
            _wis_vals = np.array([r.get('wis', np.nan) for r in rows], dtype=float)
            _wis_weights = np.array([r['n_samples'] if np.isfinite(r.get('wis', np.nan)) else 0 for r in rows], dtype=float)
            if np.any(np.isfinite(_wis_vals)) and np.sum(_wis_weights) > 0:
                wis = float(np.nansum(_wis_vals * _wis_weights) / (np.sum(_wis_weights) + 1e-10))
            else:
                wis = float(np.nanmean(_wis_vals)) if np.any(np.isfinite(_wis_vals)) else float('nan')
            entry = {'n_samples': n, 'mae': mae, 'wis': wis}
            entry['mae_den'] = float(np.nansum([r.get('mae_den', 0) for r in rows if np.isfinite(r.get('mae_den', 0))]))
            # SPL para ambos os períodos (outbreak e calm) - se houver spl_row
            _spl_rows = [r.get('spl_row') for r in rows if r.get('spl_row') is not None]
            if _spl_rows:
                try:
                    spl_mat = np.array(_spl_rows, dtype=float)
                    entry['spl_matrix_row'] = list(np.nanmean(spl_mat, axis=0))
                except Exception:
                    # fallback: try mean ignoring nan
                    try:
                        entry['spl_matrix_row'] = list(np.nanmean(np.array(_spl_rows, dtype=float), axis=0))
                    except Exception:
                        pass

            result[period] = entry
        return result

    def _average_metrics_list(self, metrics_list: List[Dict]) -> Dict[str, float]:
        if not metrics_list:
            return {}
        avg = {}
        keys = set()
        for m in metrics_list:
            keys.update(m.keys())
        for key in keys:
            if key in ['fold', 'seed', 'horizon', 'spl', 'spl_matrix_row', 'stratified', 'mae_den', 'coverage_vec']:
                continue
            values = [m.get(key, np.nan) for m in metrics_list]
            if not all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
                continue
            # corrige agregações: usa nanmean para todas métricas numéricas
            try:
                avg[key] = float(np.nanmean(np.asarray(values, dtype=float)))
            except Exception:
                avg[key] = float(np.nanmean(values))
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
        # v3: threshold por quantil não afeta modo union, mas mantem compat
        try:
            quick_params = self.model_params.copy()
            quick_params['random_state'] = 42
            quick_params['seed'] = 42
            quick_params['verbosity'] = -1
            quick_params.pop('early_stopping_rounds', None)
            # FASE1 FIX: pop n_estimators/num_iterations para num_boost_round controlar (evita 3000 vs 200)
            quick_params.pop('n_estimators', None)
            quick_params.pop('num_iterations', None)
            quick_params.pop('num_iteration', None)
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
                # v3: usa threshold por quantil se disponível
                tpq = fs.get('threshold_per_quantile', {})
                per_q = max(1, max_features // len(self.quantiles))
                cols_set = set()
                for idx, imp in enumerate(imps):
                    tau = self.quantiles[idx]
                    thr = float(tpq.get(str(tau), tpq.get(float(tau), threshold)) if isinstance(tpq, dict) else threshold)
                    top = imp[imp > thr].sort_values(ascending=False).head(per_q).index.tolist()
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

    def _get_selected_columns_per_quantile(self, X_tr: pd.DataFrame, y_tr: pd.Series) -> Dict[float, List[str]]:
        """380 features INDEPENDENTES por quantil (separate_quantile_features).
        Cada tau tem seu top 380 por ganho, sem leakage (so X_tr). Corrige q05 flat h4/h8.
        v3: usa threshold por quantil (0.0005 para q05). Fase1: suporte max_features_per_quantile {0.05:200}."""
        fs = self.config.get('training', {}).get('feature_selection', {})
        max_features_default = int(fs.get('max_features', 380))
        # Fase1: per-quantil max_features (ex: q05/q95 200, q50 380) sem quebrar compat
        mfq = fs.get('max_features_per_quantile', {})
        threshold = float(fs.get('threshold', 0.001))
        tpq = fs.get('threshold_per_quantile', {})
        quick_params = self.model_params.copy()
        quick_params['random_state'] = 42
        quick_params['seed'] = 42
        quick_params['verbosity'] = -1
        quick_params.pop('early_stopping_rounds', None)
        # FASE1 FIX: pop n_estimators para num_boost_round controlar
        quick_params.pop('n_estimators', None)
        quick_params.pop('num_iterations', None)
        quick_params.pop('num_iteration', None)
        quick_params['objective'] = 'quantile'
        per_tau_cols: Dict[float, List[str]] = {}
        for tau in self.quantiles:
            try:
                mf = int(mfq.get(str(tau), mfq.get(float(tau), max_features_default)) if isinstance(mfq, dict) and mfq else max_features_default)
            except Exception:
                mf = max_features_default
            qp = quick_params.copy()
            qp['alpha'] = tau
            m = lgb.train(qp, lgb.Dataset(X_tr, label=y_tr), num_boost_round=min(200, int(self.model_params.get('n_estimators', 3000))))
            imp = pd.Series(m.feature_importance(importance_type='gain'), index=X_tr.columns)
            thr = float(tpq.get(str(tau), tpq.get(float(tau), threshold)) if isinstance(tpq, dict) else threshold)
            filtered = imp[imp > thr].sort_values(ascending=False)
            cols = filtered.head(mf).index.tolist()
            if len(cols) < mf:
                remaining = imp[~imp.index.isin(cols)].sort_values(ascending=False).head(mf - len(cols)).index.tolist()
                cols.extend(remaining)
            if not cols:
                cols = X_tr.columns.tolist()[:mf]
            per_tau_cols[tau] = cols[:mf]
            logger.info(f"    Separate q{int(tau*1000):04d}: {len(cols)}/{X_tr.shape[1]} feats (top gain {imp.max():.1f} thr {thr} mf {mf})")
        return per_tau_cols

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def train_all_horizons(self, df: pd.DataFrame) -> Dict:
        logger.info("Starting training for all horizons...")
        df_featured = self.prepare_data(df)
        horizons = self.target_horizons
        # FASE1 paralelização: se habilitado, treina horizontes em paralelo
        par_cfg = self.config.get('training', {}).get('parallel_horizons', {})
        parallel_enabled = bool(par_cfg.get('enabled', False)) if isinstance(par_cfg, dict) else False
        if parallel_enabled:
            n_jobs = int(par_cfg.get('n_jobs', len(horizons)))
            n_jobs = max(1, min(n_jobs, len(horizons)))
            backend = par_cfg.get('backend', 'loky')
            logger.info(f"Parallel horizons: n_jobs={n_jobs} backend={backend} horizons={horizons}")
            try:
                from joblib import Parallel, delayed
                # joblib precisa que df_featured seja picklável; força cópia para evitar share
                results = Parallel(n_jobs=n_jobs, backend=backend, verbose=10)(
                    delayed(_parallel_horizon_worker)(int(h), self.config, df_featured) for h in horizons
                )
                # merge resultados no self (main process)
                for res in results:
                    h = int(res['horizon'])
                    self.models[h] = res.get('models', {})
                    if res.get('validation'):
                        self.validation_results[h] = res['validation']
                    if res.get('fold_series') is not None:
                        self.fold_series[h] = res['fold_series']
                    if res.get('feature_names') is not None:
                        self.feature_names[h] = res['feature_names']
                    if res.get('ano_params') is not None:
                        self.ano_normalization_params[h] = res['ano_params']
                logger.info(f"Parallel training merged {len(results)} horizons")
            except Exception as e:
                logger.warning(f"Parallel training failed ({e}), fallback sequencial")
                for horizon in horizons:
                    self.train_horizon(df_featured, horizon)
        else:
            for horizon in horizons:
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
        # Handle separate_quantile_features: feature_names[h] may be dict tau->list
        serializable = {}
        for k, v in self.feature_names.items():
            if isinstance(v, dict):
                # per-tau dict: convert float keys to string
                serializable[str(k)] = {str(tau): cols for tau, cols in v.items()}
            else:
                serializable[str(k)] = v
        with open(feature_path, 'w') as f:
            json.dump(serializable, f)
        logger.info(f"Saved feature list: {feature_path} (separate={any(isinstance(v, dict) for v in self.feature_names.values())})")

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

        # ---- Quantile recalibration (global + por regime outbreak/calm) ---- FIX Bug1 circular
        # Gera deltas log-scale para q05/q95 via conformal: delta = quantile(y_true_log - q_pred_log)
        # por regime corrige Cob90 heterogeneo fold0 0.58 vs fold2 1.00 (report P1)
        # FIX Bug1: quando quantile_recalibration.enabled true, usa APENAS holdout (último fold) para deltas
        # e documenta que fold_series holdout nunca intersecta X_full (FULL treinado até penúltimo fold). Se disabled, deltas são salvos mas NÃO aplicados em inference (opt-in).
        try:
            if hasattr(self, 'fold_series') and self.fold_series:
                recal_cfg = self.config.get('quantile_recalibration', {}) if isinstance(self.config.get('quantile_recalibration', {}), dict) else {}
                recal_enabled = bool(recal_cfg.get('enabled', False))
                holdout_only = recal_enabled  # quando enabled, deltas vêm só do holdout (último fold)
                if holdout_only:
                    logger.info("Quantile recalibration HOLDOUT mode: deltas calculados apenas no último fold (não circular, FULL excluiu holdout)")
                else:
                    logger.info("Quantile recalibration: deltas calculados em todos folds mas NÃO serão aplicados em inference (enabled=false, evita circularidade)")
                # global deltas
                recal_global = {}
                recal_regime = {}
                for h, tau_dict in self.fold_series.items():
                    # tau_dict: {tau: [fold_dict,...]}
                    # concat y_true_log e y_pred_log por fold
                    y_true_all = []
                    q05_all = []
                    q95_all = []
                    se_all = []
                    # FIX Bug1: seleciona folds para calibração
                    if holdout_only:
                        # usa apenas último fold (holdout)
                        n_folds = len(tau_dict[self.quantiles[0]])
                        fold_indices = [n_folds - 1] if n_folds >= 1 else []
                    else:
                        fold_indices = range(len(tau_dict[self.quantiles[0]]))
                    for fold_idx in fold_indices:
                        y_true_all.extend(np.log1p(tau_dict[0.5][fold_idx]['y_test_casos']))
                        q05_all.extend(np.log1p(tau_dict[0.05][fold_idx]['y_pred_casos']))
                        q95_all.extend(np.log1p(tau_dict[0.95][fold_idx]['y_pred_casos']))
                        se_all.extend(tau_dict[0.5][fold_idx]['test_se'])
                    y_true_all = np.array(y_true_all)
                    q05_all = np.array(q05_all)
                    q95_all = np.array(q95_all)
                    se_all = np.array(se_all)
                    # global deltas (log scale)
                    r05 = y_true_all - q05_all
                    r95 = y_true_all - q95_all
                    d05 = float(np.quantile(r05, 0.05))
                    d95 = float(np.quantile(r95, 0.95))
                    recal_global[str(h)] = {"0.05": d05, "0.95": d95}
                    # regime split via epiweeks month
                    try:
                        from data.epiweeks import epiweek_to_date, weeks_in_year, date_to_epiweek
                        # classifica cada SE em outbreak (Out-Mai) vs calm (Jun-Set)
                        outbreak_months = self.epi_periods.get('outbreak_months', [10,11,12,1,2,3,4,5])
                        is_outbreak = []
                        for se in se_all:
                            se_str = str(se).zfill(6)
                            try:
                                y = int(se_str[:4]); w = int(se_str[4:])
                                m = epiweek_to_date(y, w).month
                                is_outbreak.append(m in outbreak_months)
                            except Exception:
                                is_outbreak.append(True)
                        is_outbreak = np.array(is_outbreak)
                        is_calm = ~is_outbreak
                        def _delta(mask, r):
                            if mask.sum() < 10:
                                return float(np.quantile(r, 0.05 if r is r05 else 0.95))
                            return float(np.quantile(r[mask], 0.05 if r is r05 else 0.95))
                        # para nao confundir, calcula separado
                        d05_out = float(np.quantile(r05[is_outbreak], 0.05)) if is_outbreak.sum()>5 else d05
                        d95_out = float(np.quantile(r95[is_outbreak], 0.95)) if is_outbreak.sum()>5 else d95
                        d05_calm = float(np.quantile(r05[is_calm], 0.05)) if is_calm.sum()>5 else d05
                        d95_calm = float(np.quantile(r95[is_calm], 0.95)) if is_calm.sum()>5 else d95
                        recal_regime[str(h)] = {
                            "outbreak": {"0.05": d05_out, "0.95": d95_out},
                            "calm": {"0.05": d05_calm, "0.95": d95_calm},
                            "global": {"0.05": d05, "0.95": d95}
                        }
                    except Exception as e:
                        logger.warning(f"h{h} regime recalibration failed: {e}")
                        recal_regime[str(h)] = {"outbreak": recal_global[str(h)], "calm": recal_global[str(h)], "global": recal_global[str(h)]}
                # salva global (compat com inference atual)
                recal_path = models_dir / "quantile_recalibration.json"
                with open(recal_path, 'w') as f:
                    json.dump(recal_global, f, indent=2)
                logger.info(f"Saved quantile recalibration (global): {recal_path}")
                recal_regime_path = models_dir / "quantile_recalibration_regime.json"
                with open(recal_regime_path, 'w') as f:
                    json.dump(recal_regime, f, indent=2)
                logger.info(f"Saved quantile recalibration (regime): {recal_regime_path}")
        except Exception as e:
            logger.warning(f"Quantile recalibration generation failed: {e}")

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
        headings = ['SPL Q0.05', 'SPL Q0.25', 'SPL Q0.50', 'SPL Q0.75', 'SPL Q0.95', 'MAE', 'WIS']
        print(f"{'Horizon':<9}" + "".join(f"{h:>12}" for h in headings))
        for horizon in self.target_horizons:
            if horizon not in self.validation_results:
                continue
            a = self.validation_results[horizon].get('ensemble_avg', {})
            spl = a.get('spl_mean', [])
            row = spl + [a.get('mae', float('nan')),
                         a.get('wis', float('nan'))]
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
                line += (f"Surto: MAE={ob['mae']:.1f} WIS={ob['wis']:.1f} "
                         f"SPL={[f'{v:.2f}' for v in ob.get('spl_matrix_row', [])]} n={ob['n_samples']} | ")
            if cm:
                line += f"Calmaria: MAE={cm['mae']:.1f} WIS={cm['wis']:.1f} n={cm['n_samples']}"
            print(line)
        # DataFrame consolidado para feira científica
        try:
            df_metrics = self.get_metrics_dataframe()
            print("\n" + "=" * 90)
            print("DATAFRAME CONSOLIDADO (por Horizonte) - colunas: WIS_Total, WIS_Sharpness, WIS_Overprediction, WIS_Underprediction, MAE, Coverage_90")
            print(df_metrics.to_string(float_format=lambda x: f"{x:.3f}"))
            # Salva CSV para plotagem
            df_metrics.to_csv(Path("models") / "metrics_dataframe.csv")
            print("Salvo em models/metrics_dataframe.csv")
        except Exception as e:
            logger.warning(f"Falha ao gerar DataFrame consolidado: {e}")

    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Retorna DataFrame indexado por Horizonte com colunas acadêmicas.
        Colunas: [WIS_Total, WIS_Sharpness, WIS_Overprediction, WIS_Underprediction, MAE, Coverage_90]
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
                'MAE': float(a.get('mae', np.nan)),
                'Coverage_90': float(a.get('coverage_90', np.nan)),
                'Coverage_50': float(a.get('coverage_50', np.nan)),
                'FBias': float(a.get('fbias', np.nan)),
                'FBias_surto': float(a.get('fbias_surto', np.nan)),
                'FBias_calmaria': float(a.get('fbias_calmaria', np.nan)),
                'Etp': float(a.get('etp', np.nan)),
            })
        df = pd.DataFrame(rows).set_index('Horizonte')
        cols = ['WIS_Total','WIS_Sharpness','WIS_Overprediction','WIS_Underprediction','MAE','Coverage_90','Coverage_50','FBias','FBias_surto','FBias_calmaria','Etp']
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
    # FASE1: train_all_horizons com paralelismo (8 workers) se habilitado
    trainer.train_all_horizons(df)
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
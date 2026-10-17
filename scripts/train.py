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
import os
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
                'expand_rt': bool(self.config.get('features', {}).get('expand_rt', False)),
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

        # ---- Conjunto canônico (models/metrics.py, fonte única) ----
        # mae, rmse, bias, wis_total, wis_50, wis_90,
        # width_50, width_90, coverage_50, coverage_90.
        # Removidos na spec 2026-10: MASE (e agregados), ETP, SPL, MAPE, CRPS,
        # quantile_crossed e fbias (todas as variantes/estratificações P75).
        from models.metrics import compute_all_metrics
        metrics = {}
        metrics.update(compute_all_metrics(y_true_cases, y_pred_dict_cases))
        # quantile_crossed removido da spec (não é métrica de qualidade)


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
        else:
            pass

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
        """Métricas canônicas por período (surto Out-Mai / calmaria Jun-Set).

        Usa o conjunto canônico (models/metrics.py) por fatia do calendário;
        o bloco stratified mantém apenas n_samples/mae/wis. y_train/train_se
        aceitos por compatibilidade de assinatura (não utilizados — SPL removido).
        """
        from models.metrics import compute_all_metrics
        periods = self.stratify_months(se_array)
        result = {}
        for period in ['outbreak', 'calm']:
            mask = periods == period
            if not mask.any():
                continue
            slice_pred = {tau: np.asarray(v, dtype=float)[mask]
                          for tau, v in y_pred_dict_cases.items()}
            m = compute_all_metrics(y_true_cases[mask], slice_pred)
            result[period] = {
                'n_samples': int(mask.sum()),
                'mae': m.get('mae', float('nan')),
                'wis': m.get('wis_total', float('nan')),
            }
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
                per_tau_cols = self._get_selected_columns_per_quantile(X_tr_base, y_tr_base, horizon=horizon)
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
                per_tau_full_cols = self._get_selected_columns_per_quantile(X_full_base, y_full_src, horizon=horizon)
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
                logger.info(f"  h{horizon} 2025 parcial {partial_folds} excluido da media: mae 3folds={ensemble_avg.get('mae', float('nan')):.3f} vs 4folds={ensemble_avg_full.get('mae', float('nan')):.3f}")

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
        }

        logger.info(f"\n  h{horizon} ENSEMBLE: mae={ensemble_avg.get('mae', float('nan')):.2f}, "
                    f"rmse={ensemble_avg.get('rmse', float('nan')):.2f}, "
                    f"bias={ensemble_avg.get('bias', float('nan')):+.2f}, "
                    f"WIS_total={ensemble_avg.get('wis_total', float('nan')):.2f} "
                    f"(50: {ensemble_avg.get('wis_50', float('nan')):.2f} | 90: {ensemble_avg.get('wis_90', float('nan')):.2f}), "
                    f"Cov50={ensemble_avg.get('coverage_50', 0):.2f} Cov90={ensemble_avg.get('coverage_90', 0):.2f}, "
                    f"Width50={ensemble_avg.get('width_50', 0):.1f} Width90={ensemble_avg.get('width_90', 0):.1f}")

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
            # mae_den / SPL removidos da spec 2026-10 (dependiam de MASE/SPL)

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
            if key in ['fold', 'seed', 'horizon', 'stratified']:
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
        # always_keep: features de regime garantidas (seleção por ganho pode descartá-las)
        _keep_cols = self._always_keep_cols(X_tr)
        if _keep_cols:
            selected_cols = list(dict.fromkeys(list(_keep_cols) + list(selected_cols)))
        logger.info(f"    Feature selection: {len(selected_cols)}/{X_tr.shape[1]} features kept")
        X_tr = X_tr[selected_cols]
        if X_val is not None:
            X_val = X_val.reindex(columns=selected_cols, fill_value=0)
        if X_test is not None:
            X_test = X_test.reindex(columns=selected_cols, fill_value=0)
        return X_tr, X_val, X_test

    def _always_keep_cols(self, X: 'pd.DataFrame') -> List[str]:
        """Features de regime forçadas na seleção (training.feature_selection.always_keep).

        A seleção por ganho pode descartar sinais de regime (YTD-ratio, YoY, slopes,
        anomalias de climatologia) no corte 380->120 — exatamente os sinais que o
        relatório out/2026 apontou como faltantes para picos/mudança de regime (P1/P3/P4).
        Filtrado por existência em runtime: nomes ausentes no df_featured são ignorados.
        """
        train_cfg = self.config.get('training', {}) if isinstance(self.config.get('training', {}), dict) else {}
        fs = train_cfg.get('feature_selection', {}) if isinstance(train_cfg.get('feature_selection', {}), dict) else {}
        keep = [str(c) for c in (fs.get('always_keep') or []) if str(c) in X.columns]
        if keep:
            logger.info(f"    always_keep: garantidas {len(keep)} features de regime")
        return keep

    def _get_selected_columns_per_quantile(self, X_tr: pd.DataFrame, y_tr: pd.Series,
                                           horizon: Optional[int] = None) -> Dict[float, List[str]]:
        """380 features INDEPENDENTES por quantil (separate_quantile_features).
        Cada tau tem seu top 380 por ganho, sem leakage (so X_tr). Corrige q05 flat h4/h8.
        v3: usa threshold por quantil (0.0005 para q05). Fase1: suporte max_features_per_quantile {0.05:200}.
        v4 (out/2026): overrides por horizonte — threshold_per_quantile_per_horizon /
        max_features_per_quantile_per_horizon (ex.: cauda fina em h6-h8 onde a astringente
        original era estável, cauda condicionada em h1-h5)."""
        fs = self.config.get('training', {}).get('feature_selection', {})
        max_features_default = int(fs.get('max_features', 380))
        # Fase1: per-quantil max_features (ex: q05/q95 200, q50 380) sem quebrar compat
        mfq = fs.get('max_features_per_quantile', {})
        threshold = float(fs.get('threshold', 0.001))
        tpq = fs.get('threshold_per_quantile', {})
        # v4: overrides por horizonte (aceita chave str '6' ou int 6)
        tpq_h, mfq_h = tpq, mfq
        if horizon is not None:
            ovr_t = fs.get('threshold_per_quantile_per_horizon', {})
            ovr_m = fs.get('max_features_per_quantile_per_horizon', {})
            for cand in (str(horizon), int(horizon)):
                if isinstance(ovr_t, dict) and isinstance(ovr_t.get(cand), dict):
                    tpq_h = ovr_t[cand]
                if isinstance(ovr_m, dict) and isinstance(ovr_m.get(cand), dict):
                    mfq_h = ovr_m[cand]
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
        keep_all = self._always_keep_cols(X_tr)
        # v6b: ban por horizonte APLICADO ANTES da estimativa de importância —
        # a presença de rt_* no pool de candidatos re-aloca ganho e desloca o
        # ranking dos sobreviventes; filtrar só no corte deixava h2/h5 deslocados.
        _ban_pre = self._banned_feature_patterns(horizon)
        X_imp = X_tr
        if _ban_pre:
            _allowed_cols = [c for c in X_tr.columns if not any(c.startswith(p) for p in _ban_pre)]
            X_imp = X_tr[_allowed_cols]
            logger.info(f"    importance-pool h{horizon}: {X_tr.shape[1]} -> {X_imp.shape[1]} cols (ban {_ban_pre})")
        for tau in self.quantiles:
            try:
                mf = int(mfq_h.get(str(tau), mfq_h.get(float(tau), max_features_default)) if isinstance(mfq_h, dict) and mfq_h else max_features_default)
            except Exception:
                mf = max_features_default
            qp = quick_params.copy()
            qp['alpha'] = tau
            m = lgb.train(qp, lgb.Dataset(X_imp, label=y_tr), num_boost_round=min(200, int(self.model_params.get('n_estimators', 3000))))
            imp = pd.Series(m.feature_importance(importance_type='gain'), index=X_imp.columns)
            thr = float(tpq_h.get(str(tau), tpq_h.get(float(tau), threshold)) if isinstance(tpq_h, dict) else threshold)
            filtered = imp[imp > thr].sort_values(ascending=False)
            cols = filtered.head(mf).index.tolist()
            if len(cols) < mf:
                remaining = imp[~imp.index.isin(cols)].sort_values(ascending=False).head(mf - len(cols)).index.tolist()
                cols.extend(remaining)
            if not cols:
                cols = X_imp.columns.tolist()[:mf]
            # always_keep primeiro (entro do teto mf: substitui os de menor ganho na cauda)
            if keep_all:
                cols = list(dict.fromkeys(list(keep_all) + list(cols)))
            finals = cols[:mf]
            # v6: ban por horizonte (Rt melhora h1/h3/h4/h8 mas regride h2/h5 —
            # família Codeço et al. 2018 aplicada seletivamente em rt_*/p_rt1_*)
            _ban = self._banned_feature_patterns(horizon)
            if _ban:
                n0 = len(finals)
                finals = [c for c in finals if not any(c.startswith(p) for p in _ban)]
                logger.info(f"    ban h{horizon}: {n0} -> {len(finals)} feats (padrões {_ban})")
            per_tau_cols[tau] = finals
            logger.info(f"    Separate q{int(tau*1000):04d}: {len(per_tau_cols[tau])}/{X_tr.shape[1]} feats (top gain {imp.max():.1f} thr {thr} mf {mf})")
        return per_tau_cols

    def _banned_feature_patterns(self, horizon: Optional[int]) -> List[str]:
        """Prefixos banidos na seleção de features por horizonte
        (training.feature_selection.banned_features_per_horizon).

        Ex.: Rt (InfoDengue) entra em h1/h3/h4/h8 mas é banido em h2/h5-h7
        (regressão determinística observada no cut-off per-tau)."""
        if horizon is None:
            return []
        fs = self.config.get('training', {}).get('feature_selection', {})
        bans = fs.get('banned_features_per_horizon', {})
        if not isinstance(bans, dict):
            return []
        for cand in (str(horizon), int(horizon)):
            try:
                patterns = bans.get(cand)
            except Exception:
                patterns = None
            if patterns:
                return [str(p) for p in patterns]
        return []

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def train_all_horizons(self, df: pd.DataFrame) -> Dict:
        logger.info("Starting training for all horizons...")
        df_featured = self.prepare_data(df)
        # referência para a calibração conformal (casos por SE) em save_artifacts
        self._df_featured = df_featured
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
        # EPISENSE_MODELS_DIR: override de destino para smoke tests (não sobrescreve produção)
        models_dir = Path(os.environ.get('EPISENSE_MODELS_DIR', "models"))
        models_dir.mkdir(parents=True, exist_ok=True)

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

        # ---- Calibração conformal v3 (CQR delta-only) + deployment gate ----
        # Substitui a recalibração holdout-único (fraqueza P8). Deltas conformal por
        # horizonte (h1-h3 por padrão) x REGIME-BASE (intensidade da semana-base),
        # calibrados em TODOS os folds walk-forward com pesos temporais exponenciais.
        # Median shift desabilitado (instável entre temporadas — ver models/calibration.py).
        # Deployment gate (P7): pior fold completo (>=52 semanas) vs baseline sazonal.
        try:
            qc_cfg = self.config.get('quantile_conformal', {}) if isinstance(self.config.get('quantile_conformal', {}), dict) else {}
            if hasattr(self, 'fold_series') and self.fold_series and bool(qc_cfg.get('enabled', False)):
                from models.calibration import compute_conformal, make_regime_hook, save_json as _save_conformal
                casos_by_se = {}
                df_ref = getattr(self, '_df_featured', None)
                if df_ref is not None and 'SE' in df_ref.columns and 'casos' in df_ref.columns:
                    _d = df_ref[['SE', 'casos']].copy()
                    _d['SE'] = _d['SE'].astype(str).str.zfill(6).str.replace('.0', '', regex=False)
                    for _se, _v in zip(_d['SE'], _d['casos']):
                        try:
                            _fv = float(_v)
                        except Exception:
                            continue
                        if np.isfinite(_fv):
                            casos_by_se[_se] = _fv
                regime_hook = make_regime_hook(casos_by_se, qc_cfg.get('regime_thresholds') or None)
                production = compute_conformal(self.fold_series, regime_hook,
                                               half_life_years=float(qc_cfg.get('half_life_years', 2.5)),
                                               max_delta=float(qc_cfg.get('max_abs_delta', 1.0)),
                                               use_median_shift=bool(qc_cfg.get('apply_median_shift', False)),
                                               max_shift=float(qc_cfg.get('max_abs_median_shift', 0.8)))
                _save_conformal(models_dir / "quantile_conformal.json", production, None)
                n_h = len([k for k in production if str(k).isdigit()])
                KC = [k for k in production if str(k).isdigit()]
                _keys = sorted(production.get(KC[0], {}).keys()) if KC else []
                logger.info(f"Conformal calibration (CQR delta-only) salva: {n_h} horizontes, regimes {_keys}")
        except Exception as e:
            logger.warning(f"Conformal calibration generation failed: {e}")

        # Deployment gate: compara MAE de cada fold completo contra o baseline
        # sazonal no MESMO horizonte; reprova se o pior fold exceder o budget.
        try:
            from datetime import datetime as _dt
            gate_cfg = self.config.get('deployment_gate', {}) if isinstance(self.config.get('deployment_gate', {}), dict) else {}
            gate_enabled = bool(gate_cfg.get('enabled', False))
            gate = {'enabled': gate_enabled, 'pass': True,
                    'threshold': float(gate_cfg.get('max_baseline_ratio', 1.35)),
                    'per_horizon': {}, 'generated_at': _dt.utcnow().isoformat()}
            if gate_enabled and getattr(self, 'validation_results', None):
                # n_weeks por fold a partir do fold_series (per_fold_ensemble não carrega n_weeks)
                fold_nweeks = {}
                if hasattr(self, 'fold_series') and self.fold_series:
                    for h_fs, td_fs in self.fold_series.items():
                        try:
                            t_keys = [t for t in td_fs.keys()]
                            for blk in td_fs.get(0.5, td_fs.get(t_keys[0], [])):
                                fold_nweeks[(int(h_fs), str(blk.get('fold')))] = len(blk.get('test_se') or [])
                        except Exception:
                            continue
                for h, res in self.validation_results.items():
                    try:
                        rows = [m for m in (res.get('per_fold_ensemble') or [])
                                if fold_nweeks.get((int(h), str(m.get('fold'))), int(m.get('n_weeks', 0) or 0)) >= 52]
                        ratios = {}
                        worst = 0.0
                        for m in rows:
                            mae = m.get('mae')
                            bmae = m.get('baseline_mae')
                            if mae is None or bmae is None:
                                continue
                            if not (np.isfinite(mae) and np.isfinite(bmae) and bmae > 0):
                                continue
                            fold_id = str(m.get('fold', '?'))
                            ratio = float(mae) / float(bmae)
                            ratios[fold_id] = round(ratio, 3)
                            worst = max(worst, ratio)
                        if ratios:
                            gate['per_horizon'][str(h)] = {'worst_fold_ratio': round(worst, 3),
                                                           'folds': ratios}
                            if worst > gate['threshold']:
                                gate['pass'] = False
                    except Exception as e:
                        logger.warning(f"deployment gate h{h}: cálculo falhou ({e})")
            gate_path = models_dir / "deployment_gate.json"
            with open(gate_path, 'w') as f:
                json.dump(gate, f, indent=2)
            if gate.get('pass', True):
                logger.info("Deployment gate: PASS (todos os folds completos dentro do budget vs baseline)")
            else:
                logger.warning(f"Deployment gate: FAIL (pior fold > {gate['threshold']}x baseline) — update_pipeline fará rollback")
        except Exception as e:
            logger.warning(f"Deployment gate generation failed: {e}")

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
        headings = ['MAE', 'RMSE', 'Bias', 'WIS_total', 'WIS_50', 'WIS_90', 'Width50', 'Width90', 'Cov50%', 'Cov90%']
        print(f"{'Horizon':<9}" + "".join(f"{h:>12}" for h in headings))
        keys = ['mae', 'rmse', 'bias', 'wis_total', 'wis_50', 'wis_90', 'width_50', 'width_90', 'coverage_50', 'coverage_90']
        for horizon in self.target_horizons:
            if horizon not in self.validation_results:
                continue
            a = self.validation_results[horizon].get('ensemble_avg', {})
            row = [a.get(k, float('nan')) for k in keys]
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
                         f"n={ob['n_samples']} | ")
            if cm:
                line += f"Calmaria: MAE={cm['mae']:.1f} WIS={cm['wis']:.1f} n={cm['n_samples']}"
            print(line)
        # DataFrame consolidado para feira científica
        try:
            df_metrics = self.get_metrics_dataframe()
            print("\n" + "=" * 90)
            print("DATAFRAME CONSOLIDADO (por Horizonte) - colunas: MAE, RMSE, Bias, WIS_total, WIS_50, WIS_90, Width_50, Width_90, Coverage_50, Coverage_90")
            print(df_metrics.to_string(float_format=lambda x: f"{x:.3f}"))
            # Salva CSV para plotagem
            df_metrics.to_csv(Path("models") / "metrics_dataframe.csv")
            print("Salvo em models/metrics_dataframe.csv")
        except Exception as e:
            logger.warning(f"Falha ao gerar DataFrame consolidado: {e}")

    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Retorna DataFrame indexado por Horizonte com o conjunto canônico.
        Colunas: [MAE, RMSE, Bias, WIS_total, WIS_50, WIS_90,
                  Width_50, Width_90, Coverage_50, Coverage_90]
        Sem vazamento: todas métricas vêm de validation_results (teste não usado no treino).
        """
        keys = {
            'MAE': 'mae', 'RMSE': 'rmse', 'Bias': 'bias',
            'WIS_total': 'wis_total', 'WIS_50': 'wis_50',
            'WIS_90': 'wis_90', 'Width_50': 'width_50', 'Width_90': 'width_90',
            'Coverage_50': 'coverage_50', 'Coverage_90': 'coverage_90',
        }
        rows = []
        for h in sorted(self.validation_results.keys()):
            a = self.validation_results[h].get('ensemble_avg', {})
            row = {'Horizonte': int(h)}
            row.update({k: float(a.get(src, np.nan)) for k, src in keys.items()})
            rows.append(row)
        df = pd.DataFrame(rows).set_index('Horizonte')
        cols = list(keys.keys())
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
    # V6: trava cross-processo (flock em models/.retrain_lock). O fallback de
    # auto-retreino do dashboard (dashboard.app) usa o MESMO arquivo via
    # models/ volume-mountada — impede dois treinadores sobrescrevendo
    # artefatos simultaneamente (host x container). Se já há retreino, sai.
    import fcntl
    _lock_fp = None
    try:
        _models_dir = Path('models')
        _models_dir.mkdir(exist_ok=True)
        _lock_fp = open(_models_dir / '.retrain_lock', 'w')
        fcntl.flock(_lock_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        logger.info("Outro retreino em curso (lock models/.retrain_lock) — saindo sem treinar.")
        return
    logger.info("Starting Episense quantile training...")
    config = load_config()
    # Modo smoke (EPISENSE_QUICK=1): 1 horizonte + 1 seed, para validar o pipeline
    # rapidamente JUNTO com EPISENSE_MODELS_DIR (não sobrescreve artefatos de produção).
    if os.environ.get('EPISENSE_QUICK'):
        # EPISENSE_QUICK_HORIZONS: horizontes alternativos p/ smoke (ex.: "6" ou "1,5")
        _qh = [int(x) for x in os.environ.get('EPISENSE_QUICK_HORIZONS', '1').split(',') if x.strip()]
        _hs = _qh or [1]
        config.setdefault('targets', {})['horizons'] = _hs
        config['targets']['target_cols'] = [f'target_h{h}' for h in _hs]
        ens_cfg = config.setdefault('model', {}).setdefault('ensemble', {})
        ens_cfg['seeds'] = [42]
        ens_cfg['n_seeds'] = 1
        config.setdefault('training', {})['parallel_horizons'] = {'enabled': False}
        logger.info(f"EPISENSE_QUICK=1: modo smoke (h={_hs}, seeds=[42], sequencial)")
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
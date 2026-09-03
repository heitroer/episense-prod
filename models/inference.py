"""
Inference Module for Episense Dengue Prediction
Loads trained ensemble models and makes predictions with proper feature alignment.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EpisenseInference:
    """Inference engine for trained Episense models."""
    
    def __init__(self, model_dir: str = "models", config: Dict = None):
        self.model_dir = Path(model_dir)
        self.config = config or {}
        
        self.models = {}  # {horizon: {quantile: {seed: model}}}
        self.feature_names = {}  # {horizon: [feature_names]}
        self.scaler = None
        self.target_horizons = self.config.get('targets', {}).get('horizons', [1, 2, 3, 4])
        self.ensemble_seeds = self.config.get('model', {}).get('ensemble', {}).get('seeds', [42, 123, 456, 789, 999])
        self.legacy_mode = self.config.get('inference', {}).get('legacy_feature_aliases', True)
        self.quantiles = self.config.get('quantiles', [0.05, 0.50, 0.95])
        self.ano_min = None  # For per-fold normalization of ano_normalizado
        self.ano_max = None
        self.non_crossing = self.config.get('non_crossing', {}).get('enabled', True)
        self.non_crossing_method = self.config.get('non_crossing', {}).get('method', 'post_process')
        self.quantile_recalibration = self.config.get('quantile_recalibration', {})
        
        self._load_models()
        self._load_metadata()
        self._load_normalization_params()
        self._load_quantile_recalibration()
    
    def _load_models(self):
        """Load all ensemble models."""
        for horizon in self.target_horizons:
            self.models[horizon] = {}
            for tau in self.quantiles:
                self.models[horizon][tau] = {}
                for seed in self.ensemble_seeds:
                    model_path = self.model_dir / f"lgbm_h{horizon}_seed{seed}_q{int(tau*1000):04d}.pkl"
                    if model_path.exists():
                        self.models[horizon][tau][seed] = joblib.load(model_path)
                        logger.info(f"Loaded model: {model_path}")
                    else:
                        logger.warning(f"Model not found: {model_path}")
        
        # Fail-fast: se nenhum modelo carregado, falha explícita (config comenta sem sufixo q e só h1-h4)
        total = sum(len(seed_dict) for h_dict in self.models.values() for seed_dict in h_dict.values())
        if total == 0:
            raise RuntimeError(
                f"Nenhum modelo carregado em {self.model_dir} "
                f"(padrão esperado lgbm_h{{h}}_seed{{seed}}_q{{XXXX}}.pkl, "
                f"horizontes={self.target_horizons}, quantis={self.quantiles}, seeds={self.ensemble_seeds})"
            )
        logger.info(f"Total de modelos carregados: {total}")
        for h in self.target_horizons:
            for tau in self.quantiles:
                n = len(self.models.get(h, {}).get(tau, {}))
                logger.info(f"  h{h} q{int(tau*1000):04d}: {n} modelo(s)")
        loaded_horizons = [h for h, m in self.models.items() if any(self.models[h][tau] for tau in self.models[h])]
        logger.info(f"Loaded models for horizons: {loaded_horizons}")
        for h in loaded_horizons:
            loaded_quantiles = [tau for tau, m in self.models[h].items() if m]
            logger.info(f"  Horizon h{h}: quantiles {loaded_quantiles}")
    
    def _load_metadata(self):
        """Load feature names and scaler."""
        feature_path = self.model_dir / "feature_list.json"
        if feature_path.exists():
            with open(feature_path, 'r') as f:
                feature_data = json.load(f)
                # Support both old format (list) and new format (dict per horizon)
                if isinstance(feature_data, dict):
                    # Convert string keys to int for consistency with target_horizons
                    self.feature_names = {int(k): v for k, v in feature_data.items()}
                else:
                    # Backward compatibility: assume same features for all horizons
                    self.feature_names = {h: feature_data for h in self.target_horizons}
            logger.info(f"Loaded feature names for {len(self.feature_names)} horizons")
        
        # NOTE: DataFrameScaler was removed from pipeline (LightGBM doesn't need scaling)
        # scaler_path = self.model_dir / "scaler.pkl"
        # if scaler_path.exists():
        #     self.scaler = joblib.load(scaler_path)
        #     logger.info(f"Loaded scaler from {scaler_path}")
    
    def _load_normalization_params(self):
        """Load normalization parameters (ano_min, ano_max) for per-fold ano_normalizado."""
        norm_path = self.model_dir / "normalization_params.json"
        if norm_path.exists():
            with open(norm_path, 'r') as f:
                params = json.load(f)
                # Support both old format (global) and new format (per horizon)
                if 'ano_min' in params and 'ano_max' in params:
                    # Old global format
                    self.ano_min = params.get('ano_min')
                    self.ano_max = params.get('ano_max')
                    self.ano_normalization_params = {h: {'ano_min': self.ano_min, 'ano_max': self.ano_max} 
                                                   for h in self.target_horizons}
                else:
                    # New per-horizon format
                    self.ano_normalization_params = params
                    # Set global for backward compatibility
                    first_h = min(params.keys()) if params else None
                    if first_h:
                        self.ano_min = params[first_h].get('ano_min')
                        self.ano_max = params[first_h].get('ano_max')
            logger.info(f"Loaded normalization params: {self.ano_normalization_params}")
        else:
            logger.warning("No normalization params found. ano_normalizado will be computed from inference data (potential leakage).")
            self.ano_normalization_params = {}

    def _load_quantile_recalibration(self):
        """Load quantile recalibration deltas for conformal calibration."""
        recal_path = self.model_dir / "quantile_recalibration.json"
        if recal_path.exists():
            with open(recal_path, 'r') as f:
                self.quantile_recalibration = json.load(f)
            logger.info(f"Loaded quantile recalibration: {recal_path}")
        elif self.quantile_recalibration:
            logger.info("Using quantile recalibration from config")
        else:
            logger.info("No quantile recalibration found - using raw quantile predictions")
            self.quantile_recalibration = {}
    
    def prepare_inference_features(self, df: pd.DataFrame) -> pd.DataFrame:
            """Prepare features for inference with strict alignment to training features.
            
            Uses the SAME feature engineering configuration as training (EpisenseTrainer.prepare_data).
            This ensures feature parity between training and inference.
            """
            df = df.copy()
    
            # Check if df already has the full engineered features by checking for target columns
            # which are only created during full feature engineering
            if 'target_h1' in df.columns and 'target_h2' in df.columns:
                # Already engineered, just compute ano_normalizado per horizon during feature selection
                # ano_raw is preserved in the dataframe
                if self.legacy_mode:
                    df = self._apply_legacy_aliases(df)
                return df
    
            # Run full feature engineering with the SAME config as training
            # Training uses: expand_extra=False, expand_seasonal=True, expand_weather_long=True,
            # expand_climatology=True, expand_trend=True, expand_outbreak=True, legacy_mode=False
            # CRITICAL: Must use ALL target_horizons (1-8) to match training feature engineering,
            # since training creates horizon-specific features for ALL horizons before selection.
            from data.features import EpisenseFeatureEngineer
            fe_cfg = {
                'features': {
                    'expand_extra': self.config.get('training', {}).get('feature_selection', {}).get('expand_extra', False),
                    'expand_seasonal': self.config.get('features', {}).get('expand_seasonal', True),
                    'expand_weather_long': self.config.get('features', {}).get('expand_weather_long', True),
                    'expand_climatology': self.config.get('features', {}).get('expand_climatology', True),
                    'expand_trend': self.config.get('features', {}).get('expand_trend', True),
                    'expand_outbreak': self.config.get('features', {}).get('expand_outbreak', True),
                }
            }
            fe = EpisenseFeatureEngineer(fe_cfg)
            df = fe.run_full_feature_engineering(
                df, 
                convention='advanced',
                target_horizons=list(range(1, 9)),  # Use ALL horizons 1-8 to match training
                legacy_mode=False  # Match training
            )
    
            if self.legacy_mode:
                df = self._apply_legacy_aliases(df)
            # ano_raw is preserved; normalization happens per-horizon in _select_horizon_features
            # Return the fully featured dataframe; we'll select per-horizon features in predict()
            return df
    
    def _apply_legacy_aliases(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply legacy feature aliases for 22-feature model compatibility.
        FIX: legacy shift 0 for log_lag1 caused target leakage (same week). Now
        corrected to shift 1 (no leakage). Old 22-feature models must be retrained."""
        df = df.copy()
        
        # Ensure log_casos exists
        if 'log_casos' not in df.columns and 'casos' in df.columns:
            df['log_casos'] = np.log1p(df['casos'])
        
        # Legacy case lags: FIXED - log_lag1 = shift 1 (was shift 0 = leakage)
        if 'log_casos' in df.columns:
            for n in range(1, 9):
                df[f'log_lag{n}'] = df['log_casos'].shift(n)
        
        # Legacy weather lags - FIXED to use correct shifts
        temp_col = None
        for c in ['temp_mean_mean', 'temp_mean', 'tempmed', 'temperature_2m_mean']:
            if c in df.columns:
                temp_col = c
                break
        
        if temp_col:
            df['temp_lag2'] = df[temp_col].shift(2)
            df['temp_lag4'] = df[temp_col].shift(4)
        
        precip_col = None
        for c in ['precip_total', 'precip', 'precipitation_sum']:
            if c in df.columns:
                precip_col = c
                break
              
        if precip_col:
            df['precip_lag2'] = df[precip_col].shift(2)
            df['precip_lag4'] = df[precip_col].shift(4)
        
        hum_col = None
        for c in ['humidity_mean', 'relative_humidity_2m_mean', 'umidmed']:
            if c in df.columns:
                hum_col = c
                break
              
        if hum_col:
            df['humidity_lag2'] = df[hum_col].shift(2)
            df['humidity_lag4'] = df[hum_col].shift(4)
        
        # Rt/nivel/p_rt aliases: REMOVED (nowcast revisado retroativamente = vazamento de dados).
        
        # Seasonal features
        if 'sin_semana' not in df.columns and 'semana' in df.columns:
            df['sin_semana'] = np.sin(2 * np.pi * df['semana'] / 52)
            df['cos_semana'] = np.cos(2 * np.pi * df['semana'] / 52)
        # ano_normalizado is now computed per-fold using training params
        # if 'ano_normalizado' not in df.columns and 'ano' in df.columns:
        #     df['ano_normalizado'] = (df['ano'] - df['ano'].min()) / (df['ano'].max() - df['ano'].min() + 1e-6)
        
        return df
    
    def _select_horizon_features(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Select features for a specific horizon from the fully engineered dataframe."""
        if horizon not in self.feature_names:
            logger.warning(f"No feature names for horizon {horizon}, using all columns")
            return df.fillna(0)
        
        feature_cols = self.feature_names[horizon]
        X = pd.DataFrame(index=df.index)
        
        # Pre-compute ano_normalizado if the trained model uses it. FAIL FAST if the
        # normalization params are missing: silently filling the feature with 0.0 would
        # feed the model a constant value outside its training distribution (contract breach).
        if 'ano_normalizado' in feature_cols:
            params = self.ano_normalization_params.get(str(horizon)) or self.ano_normalization_params.get(horizon)
            if params is None or params.get('ano_min') is None or params.get('ano_max') is None:
                raise ValueError(
                    f"normalization_params.json has no entry for horizon {horizon}: "
                    f"required to compute feature 'ano_normalizado' (contract of the trained model)"
                )
            if 'ano_raw' not in df.columns:
                raise ValueError(
                    "Column 'ano_raw' required to compute 'ano_normalizado' but missing in inference data"
                )
            ano_min = params.get('ano_min')
            ano_max = params.get('ano_max')
            df = df.copy()
            df['ano_normalizado'] = (df['ano_raw'] - ano_min) / (ano_max - ano_min + 1e-6)
            # Clip to [0,1] to avoid extreme extrapolation for years beyond train range
            # (e.g., 2026 predicting with 2014-2025 train). Log first.
            n_extrap = int(((df['ano_normalizado'] < 0) | (df['ano_normalizado'] > 1)).sum())
            if n_extrap > 0:
                logger.warning(
                    f"horizon h{horizon}: ano_normalizado extrapola além da faixa de treino "
                    f"[{ano_min}, {ano_max}] em {n_extrap} linha(s) - predições para anos fora "
                    f"do período de treino são extrapolações (clipped to [0,1])"
                )
                df['ano_normalizado'] = np.clip(df['ano_normalizado'], 0, 1)
        
        n_missing = sum(1 for f in feature_cols if f not in df.columns)
        if n_missing > 0:
            logger.warning(f"h{horizon}: {n_missing}/{len(feature_cols)} features ausentes - preenchidas com 0 (verifique feature engineering)")
        for feat in feature_cols:
            if feat in df.columns:
                X[feat] = df[feat]
            else:
                logger.warning(f"Feature '{feat}' not in inference data for horizon {horizon}, filling with 0")
                X[feat] = 0.0
        
        # Fill NaN with 0 - loga quantos NaNs antes de preencher
        n_nans = int(X.isna().sum().sum())
        if n_nans > 0:
            logger.warning(f"h{horizon}: {n_nans} NaN(s) em features - preenchidos com 0")
        X = X.fillna(0)
        
        return X
    
    def predict(self, df: pd.DataFrame) -> Dict[int, Dict[float, np.ndarray]]:
            """Make predictions for all horizons and quantiles.
        
            IMPORTANT: df should be the FULL base dataset (up to the latest SE available).
            Feature engineering will be run on the full dataset to match training distribution,
            then only the last row will be used for prediction.
            """
            if len(df) == 0:
                logger.error("No valid features for prediction")
                return {}
        
            # Run feature engineering on the FULL dataset to match training distribution
            # This ensures rolling windows, climatology stats, etc. match training
            df_full_feat = self.prepare_inference_features(df)
        
            # Use only the last row for prediction (most recent week)
            # The feature engineering used all history, but we predict only for the latest week
            df_last = df_full_feat.iloc[[-1]].copy()
        
            predictions = {}
            for horizon in self.target_horizons:
                predictions[horizon] = {}
                if horizon not in self.models:
                    logger.warning(f"No models for horizon {horizon}")
                    continue
            
                for tau in self.quantiles:
                    if tau not in self.models[horizon] or not self.models[horizon][tau]:
                        logger.warning(f"No models for horizon {horizon}, quantile {tau}")
                        predictions[horizon][tau] = np.array([])
                        continue
                
                    # Select features for this horizon
                    X = self._select_horizon_features(df_last, horizon)
                
                    # Ensemble predictions for this quantile
                    horizon_preds = []
                    for seed, model in self.models[horizon][tau].items():
                        try:
                            preds = model.predict(X, num_iteration=model.best_iteration if model.best_iteration > 0 else model.num_trees())
                            horizon_preds.append(preds)
                        except Exception as e:
                            logger.error(f"Prediction error for h{horizon} tau={tau} seed {seed}: {e}")
                
                    if horizon_preds:
                        predictions[horizon][tau] = np.mean(horizon_preds, axis=0)
                    else:
                        predictions[horizon][tau] = np.array([])
        
            # Apply non-crossing constraint if enabled
            if self.non_crossing:
                predictions = self._apply_non_crossing(predictions)
        
            # Apply quantile recalibration (split conformal)
            if self.quantile_recalibration:
                predictions = self._apply_quantile_recalibration(predictions)
        
            return predictions

    def _apply_quantile_recalibration(self, predictions: Dict[int, Dict[float, np.ndarray]]) -> Dict[int, Dict[float, np.ndarray]]:
        """Apply split conformal recalibration to q0.05 and q0.95.

        FIX 2026-09-02: delta no arquivo buggy estava em escala de casos (ex: +50 a +210),
        mas era somado diretamente aos logits (log1p). Isso gerava expm1(log+210) astronomico
        e q0.05 negativo. Agora detecta escala: |delta|>5 => assume casos, converte via
        expm1/log1p com clip em 0. Se arquivo nao existir, este metodo nem e chamado.
        Desabilitado por padrao removendo quantile_recalibration.json (modelos raw ja tem
        coverage ~0.85). Se recalibrar, recalcule deltas em escala log.
        """
        if not self.quantile_recalibration:
            return predictions
        logger.warning("quantile_recalibration ativo: deltas serao aplicados em escala de CASOS (expm1/log1p) para evitar bug log+delta")
        for horizon in list(predictions.keys()):
            hkey = str(horizon)
            if hkey not in self.quantile_recalibration:
                continue
            recal = self.quantile_recalibration[hkey]
            for qkey, tau in [('q0.05', 0.05), ('q0.95', 0.95)]:
                if qkey not in recal or tau not in predictions[horizon]:
                    continue
                delta = recal[qkey].get('delta', 0.0)
                if delta == 0.0:
                    continue
                log_pred = predictions[horizon][tau]
                # Heuristica: delta grande => escala casos
                if abs(delta) > 5:
                    cases = np.expm1(log_pred)
                    cases_corr = np.maximum(0, cases + delta)
                    predictions[horizon][tau] = np.log1p(cases_corr)
                    logger.info(f"h{horizon} {qkey}: delta {delta:.1f} aplicado em casos (log->casos->log)")
                else:
                    predictions[horizon][tau] = log_pred + delta
                    logger.info(f"h{horizon} {qkey}: delta {delta:.3f} aplicado em log")
            # Re-apply non-crossing after recalibration
            if self.non_crossing:
                predictions = self._apply_non_crossing(predictions)
        return predictions

    def _apply_non_crossing(self, predictions: Dict[int, Dict[float, np.ndarray]]) -> Dict[int, Dict[float, np.ndarray]]:
        """Apply non-crossing constraint to quantile predictions.
        
        Ensures Q0.05 <= Q0.50 <= Q0.95 for all horizons and samples.
        Uses np.sort along quantile axis for strict monotonicity.
        """
        if self.non_crossing_method == 'post_process':
            for horizon in list(predictions.keys()):
                sorted_taus = sorted(self.quantiles)
                # Pula horizontes onde algum tau tem array vazio (evita column_stack com shape inconsistente)
                if any(tau not in predictions[horizon] or predictions[horizon][tau] is None or getattr(predictions[horizon][tau], 'size', 0) == 0 for tau in sorted_taus):
                    logger.warning(f"h{horizon}: pulando non-crossing - algum quantil vazio")
                    continue
                # Stack predictions: (n_samples, n_quantiles)
                q_preds = np.column_stack([predictions[horizon][tau] for tau in sorted_taus])
                # Sort along quantile axis - guarantees strict monotonicity
                q_preds = np.sort(q_preds, axis=1)
                # Write back
                for i, tau in enumerate(sorted_taus):
                    predictions[horizon][tau] = q_preds[:, i]
        return predictions

    def predict_single_row(self, row: pd.Series) -> Dict[int, float]:
        """Make predictions for a single row (latest data point).
        
        We need the full dataframe to compute lags/rolling features, so we 
        get the last row from the full dataframe after feature engineering.
        """
        # This method should not be called directly with a single row.
        # Instead, use get_latest_predictions which handles the full dataframe.
        raise NotImplementedError("Use get_latest_predictions with full dataframe")
    
    def get_latest_predictions(self, df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """Get predictions for the most recent week with metadata."""
        # Get the last row with valid features
        df_valid = df.dropna(subset=['casos']).copy()
        
        if len(df_valid) == 0:
            logger.error("No valid data for prediction")
            return {}
        
        # Use the full dataframe for feature engineering, then take the last row
        # This is important because lags/rolling features need historical context
        # We need enough historical data to compute lags/rolling features
        # Use the full dataframe (or at least enough history)
        df_full = self.prepare_inference_features(df)
        
        if len(df_full) == 0:
            logger.error("No valid data after feature engineering")
            return {}
        
        # Make predictions on the full dataframe
        preds = self.predict(df_full)
        
        # Get the last row's predictions
        last_row_se = df_full.iloc[-1].get('SE', 'unknown')
        
        results = {}
        for horizon in self.target_horizons:
            if horizon not in preds:
                continue
                
            # Check if we have quantile predictions (new format)
            if isinstance(preds[horizon], dict):
                # New format: preds[horizon] = {tau: array}
                median_preds = preds[horizon].get(0.50, np.array([]))
                if len(median_preds) == 0:
                    continue
                    
                log_casos = median_preds[-1]
                if np.isnan(log_casos):
                    continue
                
                casos_previstos = int(np.round(np.expm1(log_casos)))
                
                # Calculate alert level
                alerta = self._calculate_alert(casos_previstos, horizon)
                
                # Calculate epidemiological week for prediction
                pred_se = self._calculate_future_se(last_row_se, horizon)
                
                # Build quantile predictions dict
                quantile_predictions = {}
                for tau in self.quantiles:
                    if tau in preds[horizon] and len(preds[horizon][tau]) > 0:
                        tau_log = preds[horizon][tau][-1]
                        if not np.isnan(tau_log):
                            quantile_predictions[f'q{int(tau*1000):04d}'] = int(np.round(np.expm1(tau_log)))
                
                results[horizon] = {
                    'horizonte_semanas': horizon,
                    'semana_epidemiologica': str(pred_se),
                    'casos_previstos': max(0, casos_previstos),
                    'alerta': alerta,
                    'casos_previstos_log': float(log_casos),
                    'quantis': quantile_predictions
                }
            else:
                # Old format fallback (shouldn't happen with new models)
                if len(preds[horizon]) == 0:
                    continue
                log_casos = preds[horizon][-1]
                if np.isnan(log_casos):
                    continue
                
                casos_previstos = int(np.round(np.expm1(log_casos)))
                alerta = self._calculate_alert(casos_previstos, horizon)
                pred_se = self._calculate_future_se(last_row_se, horizon)
                
                results[horizon] = {
                    'horizonte_semanas': horizon,
                    'semana_epidemiologica': str(pred_se),
                    'casos_previstos': max(0, casos_previstos),
                    'alerta': alerta,
                    'casos_previstos_log': float(log_casos)
                }
        
        return results
    
    def _calculate_alert(self, casos: int, horizon: int) -> str:
        """Calculate alert level based on predicted cases."""
        thresholds = self.config.get('api', {}).get('response', {}).get('alert_thresholds', {})
        h_thresholds = thresholds.get(f'h{horizon}', [10, 50, 100])
        labels = self.config.get('api', {}).get('response', {}).get('alert_labels', 
                                                                  ["baixo", "medio", "alto", "critico"])
        
        if casos < h_thresholds[0]:
            return labels[0]
        elif casos < h_thresholds[1]:
            return labels[1]
        elif casos < h_thresholds[2]:
            return labels[2]
        else:
            return labels[3]
    
    def _calculate_future_se(self, current_se, horizon: int) -> str:
        """Calculate future epidemiological week SE (YYYYWW string).

        FIX (review Bug 4): computes by DATES on the Brazilian calendar
        (epiweeks), which handles 52/53-week years exactly - the old fixed
        `week > 52` wrap returned the wrong SE in years with 53 weeks.
        Accepts int/float/str SEs (CSVs are often read with numeric dtypes).
        """
        from data.epiweeks import epiweek_to_date, date_to_epiweek
        se_str = str(current_se).strip()
        if '.' in se_str:
            se_str = se_str.split('.')[0]
        se_str = se_str.zfill(6)
        start = epiweek_to_date(int(se_str[:4]), int(se_str[4:]))
        return date_to_epiweek(start + pd.Timedelta(weeks=horizon))


def load_latest_data(config: Dict = None) -> pd.DataFrame:
    """Load the latest processed data for inference."""
    config = config or {}
    processed_dir = Path(config.get('paths', {}).get('data_processed', 'data/processed'))
    
    files = list(processed_dir.glob("episense_base.csv"))
    if not files:
        files = list(processed_dir.glob("*.csv"))
    
    if not files:
        logger.error("No processed data found")
        return pd.DataFrame()
    
    latest = max(files, key=lambda f: f.stat().st_mtime)
    logger.info(f"Loading latest data from {latest}")
    return pd.read_csv(latest)


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    import yaml
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    df = load_latest_data(config)
    
    if not df.empty:
        # Initialize inference
        inference = EpisenseInference("models", config)
        
        # Get latest predictions
        predictions = inference.get_latest_predictions(df)
        
        print("\n=== Episense Predictions ===")
        print(f"Latest data SE: {df.iloc[-1].get('SE', 'N/A')}")
        print(f"Latest cases: {df.iloc[-1].get('casos', 'N/A')}")
        print()
        
        for horizon, pred in predictions.items():
            print(f"Horizon {horizon} week(s) ahead:")
            print(f"  SE: {pred['semana_epidemiologica']}")
            print(f"  Casos previstos: {pred['casos_previstos']}")
            print(f"  Alerta: {pred['alerta']}")
            print()
    else:
        print("No data available for inference")
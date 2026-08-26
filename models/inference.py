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
        
        self.models = {}  # {horizon: {seed: model}}
        self.feature_names = {}  # {horizon: [feature_names]}
        self.scaler = None
        self.target_horizons = self.config.get('targets', {}).get('horizons', [1, 2, 3, 4])
        self.ensemble_seeds = self.config.get('model', {}).get('ensemble', {}).get('seeds', [42, 123, 456, 789, 999])
        self.legacy_mode = self.config.get('inference', {}).get('legacy_feature_aliases', True)
        self.ano_min = None  # For per-fold normalization of ano_normalizado
        self.ano_max = None
        
        self._load_models()
        self._load_metadata()
        self._load_normalization_params()
    
    def _load_models(self):
        """Load all ensemble models."""
        for horizon in self.target_horizons:
            self.models[horizon] = {}
            for seed in self.ensemble_seeds:
                model_path = self.model_dir / f"lgbm_h{horizon}_seed{seed}.pkl"
                if model_path.exists():
                    self.models[horizon][seed] = joblib.load(model_path)
                    logger.info(f"Loaded model: {model_path}")
                else:
                    logger.warning(f"Model not found: {model_path}")
        
        loaded_horizons = [h for h, m in self.models.items() if m]
        logger.info(f"Loaded models for horizons: {loaded_horizons}")
    
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
    
    def prepare_inference_features(self, df: pd.DataFrame) -> pd.DataFrame:
            """Prepare features for inference with strict alignment to training features."""
            df = df.copy()
    
            # Check if df already has the full engineered features by checking for target columns
            # which are only created during full feature engineering
            if 'target_h1' in df.columns and 'target_h2' in df.columns:
                # Already engineered, just compute ano_normalizado per horizon during feature selection
                # ano_raw is preserved in the dataframe
                return df
    
            # Run full feature engineering to generate all features
            from data.features import EpisenseFeatureEngineer
            fe = EpisenseFeatureEngineer()
            df = fe.run_full_feature_engineering(
                df, 
                convention='advanced',
                target_horizons=self.target_horizons,
                legacy_mode=False
            )
    
            # ano_raw is preserved; normalization happens per-horizon in _select_horizon_features
            # Return the fully featured dataframe; we'll select per-horizon features in predict()
            return df
    
    def _apply_legacy_aliases(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply legacy feature aliases for 22-feature model compatibility."""
        df = df.copy()
        
        # Ensure log_casos exists
        if 'log_casos' not in df.columns and 'casos' in df.columns:
            df['log_casos'] = np.log1p(df['casos'])
        
        # Legacy case lags: log_lag1 = log_casos (shift 0), log_lagN = log_casos.shift(N-1)
        if 'log_casos' in df.columns:
            df['log_lag1'] = df['log_casos'].shift(0)  # Legacy: shift 0
            for n in range(2, 9):
                df[f'log_lag{n}'] = df['log_casos'].shift(n - 1)  # Legacy: shift N-1
        
        # Legacy weather lags
        temp_col = None
        for c in ['temp_mean_mean', 'temp_mean', 'tempmed', 'temperature_2m_mean']:
            if c in df.columns:
                temp_col = c
                break
        
        if temp_col:
            df['temp_lag2'] = df[temp_col].shift(1)
            df['temp_lag4'] = df[temp_col].shift(3)
        
        precip_col = None
        for c in ['precip_total', 'precip', 'precipitation_sum']:
            if c in df.columns:
                precip_col = c
                break
              
        if precip_col:
            df['precip_lag2'] = df[precip_col].shift(1)
            df['precip_lag4'] = df[precip_col].shift(3)
        
        hum_col = None
        for c in ['humidity_mean', 'relative_humidity_2m_mean', 'umidmed']:
            if c in df.columns:
                hum_col = c
                break
              
        if hum_col:
            df['humidity_lag2'] = df[hum_col].shift(1)
            df['humidity_lag4'] = df[hum_col].shift(3)
        
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
            n_extrap = int(((df['ano_normalizado'] < 0) | (df['ano_normalizado'] > 1)).sum())
            if n_extrap > 0:
                logger.warning(
                    f"horizon h{horizon}: ano_normalizado extrapola além da faixa de treino "
                    f"[{ano_min}, {ano_max}] em {n_extrap} linha(s) - predições para anos fora "
                    f"do período de treino são extrapolações"
                )
        
        for feat in feature_cols:
            if feat in df.columns:
                X[feat] = df[feat]
            else:
                logger.warning(f"Feature '{feat}' not in inference data for horizon {horizon}, filling with 0")
                X[feat] = 0.0
        
        # Fill NaN with 0
        X = X.fillna(0)
        
        return X
    
    def predict(self, df: pd.DataFrame) -> Dict[int, np.ndarray]:
        """Make predictions for all horizons."""
        # df should already be prepared by prepare_inference_features
        if len(df) == 0:
            logger.error("No valid features for prediction")
            return {}
        
        predictions = {}
        for horizon in self.target_horizons:
            if horizon not in self.models or not self.models[horizon]:
                logger.warning(f"No models for horizon {horizon}")
                predictions[horizon] = np.array([])
                continue
            
            # Select features for this horizon
            X = self._select_horizon_features(df, horizon)
            
            # Ensemble predictions
            horizon_preds = []
            for seed, model in self.models[horizon].items():
                try:
                    preds = model.predict(X, num_iteration=model.best_iteration)
                    horizon_preds.append(preds)
                except Exception as e:
                    logger.error(f"Prediction error for h{horizon} seed {seed}: {e}")
            
            if horizon_preds:
                predictions[horizon] = np.mean(horizon_preds, axis=0)
            else:
                predictions[horizon] = np.array([])
        
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
            if horizon not in preds or len(preds[horizon]) == 0:
                continue
            
            log_casos = preds[horizon][-1]
            if np.isnan(log_casos):
                continue
            
            casos_previstos = int(np.round(np.expm1(log_casos)))
            
            # Calculate alert level
            alerta = self._calculate_alert(casos_previstos, horizon)
            
            # Calculate epidemiological week for prediction
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
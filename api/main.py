"""
Episense API - Production Version
Fetches fresh data from InfoDengue + OpenMeteo on each request for real-time predictions.
No CSV persistence - all in-memory processing like Aedex All.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import yaml
from datetime import datetime, timedelta
import requests
from io import StringIO
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.inference import EpisenseInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load config
with open("config/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

app = FastAPI(
    title="Episense API",
    description="Dengue case prediction API for Campo Grande, MS - Real-time data from APIs (no CSV storage)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference engine
inference_engine: Optional[EpisenseInference] = None

# Simple in-memory cache for API data (60s TTL - atualizado a cada abertura do site, não estático)
_data_cache = {
    'df': None,
    'timestamp': 0,
    'ttl': 60
}

# Config constants
GEOCODE = config['project']['geocode']
INFODENGUE_URL = config['data_sources']['infodengue']['base_url']
OPENMETEO_URL = config['data_sources']['openmeteo']['base_url']
LAT = config['data_sources']['openmeteo']['latitude']
LON = config['data_sources']['openmeteo']['longitude']
TIMEZONE = config['data_sources']['openmeteo']['timezone']
TARGET_HORIZONS = config['targets']['horizons']


def _fetch_infodengue() -> pd.DataFrame:
    """Fetch dengue data: InfoDengue API retorna apenas ~3 semanas recentes.
    Para histórico completo (S18+ correto), carrega full history local
    (infodengue_cg_full.csv) e mescla fresh em memória. Sempre atualizado
    a cada request (TTL 60s), não estático.
    """
    url = f"{INFODENGUE_URL}?geocode={GEOCODE}&disease=dengue&format=json&ew_format=SE"
    # full history local
    full_path = Path("data/raw/infodengue/infodengue_cg_full.csv")
    df_full = None
    if full_path.exists():
        try:
            tmp = pd.read_csv(full_path, dtype={"SE": str})
            keep_full = [c for c in ['SE', 'casos', 'casos_est'] if c in tmp.columns]
            tmp = tmp[keep_full].copy()
            tmp['SE'] = tmp['SE'].astype(str).str.replace(".0","", regex=False).str.zfill(6)
            if 'casos' in tmp.columns:
                tmp['casos'] = pd.to_numeric(tmp['casos'], errors='coerce').fillna(0).astype(int)
            if 'casos_est' in tmp.columns:
                tmp['casos_est'] = pd.to_numeric(tmp['casos_est'], errors='coerce').fillna(0)
            df_full = tmp.drop_duplicates(subset=['SE']).sort_values('SE').reset_index(drop=True)
            logger.info(f"[api live] full history {len(df_full)} rows {df_full['SE'].min()}-{df_full['SE'].max()}")
        except Exception as e:
            logger.warning(f"[api live] failed load full {e}")
            df_full = None
    logger.info(f"Fetching InfoDengue fresh from {url}")
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = response.json()
        df_fresh = pd.DataFrame(data)
        if df_fresh.empty:
            if df_full is not None and not df_full.empty:
                df = df_full.copy()
                df['SE'] = df['SE'].astype(str).str.zfill(6)
                df['ano'] = df['SE'].str[:4].astype(int)
                df['semana'] = df['SE'].str[4:].astype(int)
                return df.sort_values('SE').reset_index(drop=True)
            logger.warning("Empty fresh and no full")
            return df_fresh
        keep_cols = ['SE', 'casos', 'casos_est']
        available_cols = [c for c in keep_cols if c in df_fresh.columns]
        df_fresh = df_fresh[available_cols].copy()
        df_fresh['SE'] = df_fresh['SE'].astype(str).str.zfill(6)
        df_fresh['ano'] = df_fresh['SE'].str[:4].astype(int)
        df_fresh['semana'] = df_fresh['SE'].str[4:].astype(int)
        if 'casos' in df_fresh.columns:
            df_fresh['casos'] = pd.to_numeric(df_fresh['casos'], errors='coerce').fillna(0).astype(int)
        if df_full is not None and not df_full.empty:
            df_full['SE'] = df_full['SE'].astype(str).str.zfill(6)
            df_fresh['SE'] = df_fresh['SE'].astype(str).str.zfill(6)
            df_full_filtered = df_full[~df_full['SE'].isin(df_fresh['SE'])]
            df_merged = pd.concat([df_full_filtered, df_fresh], ignore_index=True)
            df_merged = df_merged.drop_duplicates(subset=['SE']).sort_values('SE').reset_index(drop=True)
            df_merged['ano'] = df_merged['SE'].str[:4].astype(int)
            df_merged['semana'] = df_merged['SE'].str[4:].astype(int)
            logger.info(f"[api live] merged {len(df_full)} + {len(df_fresh)} -> {len(df_merged)} fresh {sorted(df_fresh['SE'].tolist())}")
            return df_merged
        df_fresh = df_fresh.sort_values('SE').reset_index(drop=True)
        logger.info(f"Downloaded {len(df_fresh)} rows fresh (no full to merge)")
        return df_fresh
    except Exception as e:
        logger.warning(f"[api live] fresh fetch failed {e}, fallback full")
        if df_full is not None and not df_full.empty:
            df = df_full.copy()
            df['SE'] = df['SE'].astype(str).str.zfill(6)
            df['ano'] = df['SE'].str[:4].astype(int)
            df['semana'] = df['SE'].str[4:].astype(int)
            return df.sort_values('SE').reset_index(drop=True)
        raise


def _fetch_openmeteo() -> pd.DataFrame:
    """Fetch latest weather data from OpenMeteo API directly (no CSV storage). Com retry 429 e fallback local."""
    # Get last ~171 weeks (1200 days) of daily data — suficiente para 156 semanas (lags/rolling + cobertura minima)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=1200)
    
    params = {
        'latitude': LAT,
        'longitude': LON,
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'daily': ','.join(config['data_sources']['openmeteo']['daily_variables']),
        'timezone': TIMEZONE
    }
    
    logger.info(f"Fetching OpenMeteo data from {OPENMETEO_URL}")
    # Retry com backoff para 429 (rate limit) - crucial para não dar 500
    last_exc = None
    response = None
    for attempt in range(3):
        try:
            response = requests.get(OPENMETEO_URL, params=params, timeout=60)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            last_exc = e
            status = getattr(getattr(e, 'response', None), 'status_code', None) or getattr(response, 'status_code', None)
            if status == 429 and attempt < 2:
                wait = 2 ** attempt * 5  # 5s, 10s
                logger.warning(f"OpenMeteo 429 rate limit, retry {attempt+1}/3 em {wait}s")
                time.sleep(wait)
                continue
            # Se não é 429 ou última tentativa, tenta fallback antes de propagar
            if attempt == 2:
                break
            raise
    if response is None or (hasattr(response, 'status_code') and response.status_code != 200):
        # Se ainda falhou após retries, tenta fallback local
        logger.warning(f"OpenMeteo falhou após retries ({last_exc}), tentando fallback local")
        try:
            from data.collect_openmeteo import latest_weekly_file
            from pathlib import Path
            wf = latest_weekly_file(Path("data/raw/openmeteo"))
            if wf and wf.exists():
                logger.info(f"Fallback OpenMeteo local: {wf}")
                df_fallback = pd.read_csv(wf, dtype={"SE": str})
                df_fallback["SE"] = df_fallback["SE"].astype(str).str.zfill(6)
                return df_fallback.sort_values("SE").reset_index(drop=True)
        except Exception as fe:
            logger.warning(f"Fallback local também falhou: {fe}")
        if last_exc:
            raise last_exc
        raise RuntimeError("OpenMeteo falhou sem resposta")
    
    data = response.json()
    daily = data.get('daily', {})
    
    if not daily:
        logger.warning("No daily data from OpenMeteo")
        return pd.DataFrame()
    
    df = pd.DataFrame(daily)
    df['date'] = pd.to_datetime(df['time'])
    df = df.drop(columns=['time'])
    
    # Aggregate to epidemiological weeks (Brazilian calendar - Sunday start)
    from data.epiweeks import date_to_epiweek
    
    df['SE'] = df['date'].apply(lambda d: date_to_epiweek(d))
    df['SE'] = df['SE'].astype(str).str.zfill(6)
    df['ano'] = df['SE'].str[:4].astype(int)
    df['semana'] = df['SE'].str[4:].astype(int)
    
    # Weekly aggregation
    agg_dict = {}
    for col in df.columns:
        if col in ['date', 'SE', 'ano', 'semana']:
            continue
        if 'precip' in col.lower() or 'precipitation' in col.lower():
            agg_dict[col] = 'sum'
        elif 'wind' in col.lower() or 'speed' in col.lower():
            agg_dict[col] = 'max'
        else:
            agg_dict[col] = 'mean'
    
    weekly = df.groupby(['SE', 'ano', 'semana']).agg(agg_dict).reset_index()
    
    # Rename columns to match expected names
    rename_map = {
        'temperature_2m_max': 'temp_max',
        'temperature_2m_min': 'temp_min',
        'temperature_2m_mean': 'temp_mean',
        'precipitation_sum': 'precip_total',
        'relative_humidity_2m_mean': 'humidity_mean',
        'wind_speed_10m_max': 'wind_max'
    }
    weekly = weekly.rename(columns=rename_map)
    
    weekly = weekly.sort_values('SE').reset_index(drop=True)
    logger.info(f"Aggregated weather to {len(weekly)} epidemiological weeks")
    return weekly


def _merge_and_prepare(dengue_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Merge dengue and weather, ensure continuous weeks, create base features - all in memory."""
    dengue_df['SE'] = dengue_df['SE'].astype(str).str.zfill(6)
    weather_df['SE'] = weather_df['SE'].astype(str).str.zfill(6)
    
    merged = pd.merge(dengue_df, weather_df, on='SE', how='left', suffixes=('', '_weather'))
    
    # Drop duplicate columns from merge
    cols_to_drop = [c for c in merged.columns if c.endswith('_weather') and c.replace('_weather', '') in merged.columns]
    merged = merged.drop(columns=cols_to_drop)
    
    # Ensure continuous weeks (Brazilian calendar)
    from data.epiweeks import weeks_in_year
    
    min_se = int(merged['SE'].min())
    max_se = int(merged['SE'].max())
    
    all_weeks = []
    for year in range(min_se // 100, max_se // 100 + 1):
        for week in range(1, weeks_in_year(year) + 1):
            se = f"{year}{week:02d}"
            se_int = int(se)
            if min_se <= se_int <= max_se:
                all_weeks.append(se)
    
    all_weeks_df = pd.DataFrame({'SE': all_weeks})
    all_weeks_df['SE'] = all_weeks_df['SE'].astype(str).str.zfill(6)
    
    merged = pd.merge(all_weeks_df, merged, on='SE', how='left')
    
    # Fill missing cases with 0
    if 'casos' in merged.columns:
        merged['casos'] = merged['casos'].fillna(0).astype(int)
    
    # Forward fill weather variables
    weather_cols = [c for c in merged.columns if c not in ['SE', 'ano', 'semana', 'casos', 'casos_est']]
    for col in weather_cols:
        if merged[col].dtype in ['float64', 'int64']:
            merged[col] = merged[col].ffill()
    
    # Recreate ano and semana
    merged['ano'] = merged['SE'].str[:4].astype(int)
    merged['semana'] = merged['SE'].str[4:].astype(int)
    
    # NOWCAST: treino usa casos puro, inferencia usa nowcast nas ultimas 12 semanas
    # Aplica max(casos, casos_est) SEMPRE (antes só últimas 12, deixava 202633/202634 com 0)
    if 'casos_est' in merged.columns:
        merged = merged.sort_values('SE').reset_index(drop=True)
        merged['casos'] = pd.to_numeric(merged['casos'], errors='coerce').fillna(0).astype(float)
        merged['casos_est'] = pd.to_numeric(merged['casos_est'], errors='coerce').fillna(0).astype(float)
        mask = (merged['casos_est'] > merged['casos'])
        if mask.any():
            merged.loc[mask, 'casos'] = np.round(merged.loc[mask, 'casos_est']).astype(int)
            logger.info(f"Nowcast substitution: {mask.sum()} weeks updated (janela 12 semanas)")
    
    # Base features for feature engineering compatibility
    merged['log_casos'] = np.log1p(merged['casos'])
    # FIX: sazonalidade com weeks_in_year para anos 53 semanas
    try:
        from data.epiweeks import weeks_in_year
        weeks = merged['ano'].apply(lambda y: weeks_in_year(int(y)))
        merged['sin_semana'] = np.sin(2 * np.pi * merged['semana'] / weeks)
        merged['cos_semana'] = np.cos(2 * np.pi * merged['semana'] / weeks)
    except Exception:
        merged['sin_semana'] = np.sin(2 * np.pi * merged['semana'] / 52)
        merged['cos_semana'] = np.cos(2 * np.pi * merged['semana'] / 52)
    
    # Lags corrigidos: shift N (sem vazamento), antes era shift 0 / N-1
    for lag in range(1, 9):
        merged[f'log_lag{lag}'] = merged['log_casos'].shift(lag)
    
    # Weather lags corrigidos: shift 2 e 4 (antes 1 e 3)
    temp_col = None
    for c in ['temp_mean', 'temp_mean_mean', 'tempmed']:
        if c in merged.columns:
            temp_col = c
            break
    if temp_col:
        merged['temp_lag2'] = merged[temp_col].shift(2)
        merged['temp_lag4'] = merged[temp_col].shift(4)
    
    precip_col = None
    for c in ['precip_total', 'precip']:
        if c in merged.columns:
            precip_col = c
            break
    if precip_col:
        merged['precip_lag2'] = merged[precip_col].shift(2)
        merged['precip_lag4'] = merged[precip_col].shift(4)
    
    hum_col = None
    for c in ['humidity_mean', 'relative_humidity_2m_mean', 'umidmed']:
        if c in merged.columns:
            hum_col = c
            break
    if hum_col:
        merged['humidity_lag2'] = merged[hum_col].shift(2)
        merged['humidity_lag4'] = merged[hum_col].shift(4)
    
    return merged


def _get_latest_engineered_data() -> pd.DataFrame:
    """Get fully engineered features, cached for 5 minutes."""
    global _data_cache
    
    now = time.time()
    if _data_cache['df'] is not None and (now - _data_cache['timestamp']) < _data_cache['ttl']:
        logger.info("Using cached engineered data")
        return _data_cache['df']
    
    logger.info("Fetching fresh data from APIs...")
    
    # Fetch from APIs directly (no CSV)
    dengue_df = _fetch_infodengue()
    weather_df = _fetch_openmeteo()
    
    if dengue_df.empty:
        raise HTTPException(status_code=503, detail="Failed to fetch dengue data")
    if weather_df.empty:
        raise HTTPException(status_code=503, detail="Failed to fetch weather data")
    if len(weather_df) < 156:
        logger.warning(f"Weather coverage below minimum: {len(weather_df)} weeks < 156 weeks required")
    
    # Merge and process in memory
    merged = _merge_and_prepare(dengue_df, weather_df)
    
    # Full feature engineering (usa config com expand_* v2)
    from data.features import EpisenseFeatureEngineer
    fe = EpisenseFeatureEngineer(config)
    engineered = fe.run_full_feature_engineering(
        merged,
        convention='advanced',
        target_horizons=TARGET_HORIZONS,
        legacy_mode=False
    )
    
    # Cache
    _data_cache['df'] = engineered
    _data_cache['timestamp'] = now
    
    logger.info(f"Engineered data shape: {engineered.shape}, latest SE: {engineered.iloc[-1].get('SE', 'N/A')}")
    return engineered


class PredictionResponse(BaseModel):
    horizonte_semanas: int = Field(..., description="Prediction horizon in weeks (1-8)")
    semana_epidemiologica: str = Field(..., description="Epidemiological week in YYYYWW format")
    casos_previstos: int = Field(..., description="Predicted number of dengue cases (median q0.50)")
    q05: int = Field(..., description="Quantile 0.05 lower bound 90%")
    q25: int = Field(..., description="Quantile 0.25 lower bound 50%")
    q50: int = Field(..., description="Quantile 0.50 median")
    q75: int = Field(..., description="Quantile 0.75 upper bound 50%")
    q95: int = Field(..., description="Quantile 0.95 upper bound 90%")
    alerta: str = Field(..., description="Alert level: baixo, medio, alto, critico")


class PredictResponse(BaseModel):
    geocode: str
    data_previsao: str
    previsoes: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    data_source: str


@app.on_event("startup")
async def startup_event():
    """Initialize inference engine on startup."""
    global inference_engine
    logger.info("Starting Episense API (Production - No CSV)...")
    
    try:
        inference_engine = EpisenseInference("models", config)
        logger.info("Inference engine loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load inference engine: {e}")
        inference_engine = None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if inference_engine is not None else "degraded",
        timestamp=datetime.now().isoformat(),
        model_loaded=inference_engine is not None,
        data_source="InfoDengue + OpenMeteo APIs (real-time, no CSV storage)"
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: Request):
    """Get dengue predictions for a geocode (real-time data from APIs)."""
    global inference_engine
    
    # Extract geocode from JSON body
    try:
        body = await request.json()
        geocode = body.get("geocode")
    except Exception:
        raise HTTPException(status_code=422, detail="geocode required (JSON body)")
    
    if not geocode:
        raise HTTPException(status_code=422, detail="geocode required (JSON body)")
    
    # Validate geocode
    expected_geocode = config['project']['geocode']
    if geocode != expected_geocode:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid geocode. Expected {expected_geocode} for Campo Grande, MS"
        )
    
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get fresh engineered data (cached 5 min)
        engineered = _get_latest_engineered_data()
        
        if engineered is None or engineered.empty:
            raise HTTPException(status_code=503, detail="Data not available")
        
        # Get predictions
        predictions = inference_engine.get_latest_predictions(engineered)
        
        if not predictions:
            raise HTTPException(status_code=500, detail="Failed to generate predictions")
        
        # Format response
        previsoes = []
        for horizon in sorted(predictions.keys()):
            pred = predictions[horizon]
            quantis = pred.get('quantis', {})
            q05 = quantis.get('q0050', quantis.get('q0025', pred['casos_previstos'])) if 'q0050' in quantis else pred['casos_previstos']
            # try all variants
            # normalize: check for 5q keys
            def get_q(key_variants, default):
                for kv in key_variants:
                    if kv in quantis:
                        return quantis[kv]
                return default
            q05 = get_q(['q0050','q05','0.05',0.05], pred['casos_previstos'])
            q25 = get_q(['q0250','q25','0.25',0.25], q05)
            q50 = get_q(['q0500','q50','0.5',0.5], pred['casos_previstos'])
            q75 = get_q(['q0750','q75','0.75',0.75], q95 if 'q95' in locals() else pred['casos_previstos'])
            q95 = get_q(['q0950','q95','0.95',0.95], pred['casos_previstos'])
            # fallback direct dict if quantis uses float keys
            if isinstance(quantis, dict) and any(isinstance(k,float) for k in quantis.keys()):
                q05 = quantis.get(0.05, q05)
                q25 = quantis.get(0.25, q25)
                q50 = quantis.get(0.5, q50)
                q75 = quantis.get(0.75, q75)
                q95 = quantis.get(0.95, q95)
            previsoes.append(PredictionResponse(
                horizonte_semanas=pred['horizonte_semanas'],
                semana_epidemiologica=pred['semana_epidemiologica'],
                casos_previstos=pred['casos_previstos'],
                q05=int(q05) if q05 is not None else 0,
                q25=int(q25) if q25 is not None else int(q05) if q05 is not None else 0,
                q50=int(q50) if q50 is not None else 0,
                q75=int(q75) if q75 is not None else int(q95) if q95 is not None else 0,
                q95=int(q95) if q95 is not None else 0,
                alerta=pred['alerta']
            ))
        
        return PredictResponse(
            geocode=geocode,
            data_previsao=datetime.now().isoformat(),
            previsoes=previsoes
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/predict/{geocode}", response_model=PredictResponse)
async def predict_get(geocode: str):
    """Get dengue predictions for a geocode (GET endpoint)."""
    from starlette.datastructures import Headers
    
    class MockRequest:
        def __init__(self):
            self.headers = Headers({"content-type": "application/json"})
            self._body = {"geocode": geocode}
        async def json(self):
            return self._body
    
    return await predict(MockRequest())


@app.get("/metrics")
async def get_metrics():
    """Get model validation metrics."""
    global inference_engine
    
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import json
        import math
        metrics_path = Path("models/validation_results.json")
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            # Replace NaN with null for JSON compliance
            def clean_nan(obj):
                if isinstance(obj, float):
                    if math.isnan(obj) or math.isinf(obj):
                        return None
                    return obj
                elif isinstance(obj, dict):
                    return {k: clean_nan(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_nan(v) for v in obj]
                return obj
            metrics = clean_nan(metrics)
            return metrics
        else:
            raise HTTPException(status_code=404, detail="Metrics not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load metrics: {str(e)}")


@app.get("/metadata")
async def get_metadata():
    """Get model and data metadata."""
    global inference_engine
    
    try:
        engineered = _get_latest_engineered_data()
        latest_se = str(engineered.iloc[-1].get('SE', 'N/A')) if not engineered.empty else None
        latest_casos = int(engineered.iloc[-1].get('casos', 0)) if not engineered.empty else None
    except:
        latest_se = None
        latest_casos = None
    
    metadata = {
        "project": config['project']['name'],
        "location": config['project']['location'],
        "geocode": config['project']['geocode'],
        "target_horizons": config['targets']['horizons'],
        "model_type": config['model']['type'],
        "ensemble_seeds": config['model']['ensemble']['seeds'],
        "feature_count": sum(len(v) for v in inference_engine.feature_names.values()) if inference_engine and inference_engine.feature_names else 0,
        "data_source": "Real-time APIs (InfoDengue + OpenMeteo) - no CSV storage",
        "latest_data_se": latest_se,
        "latest_casos": latest_casos,
        "orientacao_horizontes": config.get('api', {}).get('response', {}).get('horizon_guidance', ''),
        "api_version": "1.0.0"
    }
    
    return metadata


def run_api():
    """Run the API server."""
    import uvicorn
    host = config['api']['host']
    port = config['api']['port']
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_api()
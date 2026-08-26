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

# Simple in-memory cache for API data (5 min TTL)
_data_cache = {
    'df': None,
    'timestamp': 0,
    'ttl': 300  # 5 minutes
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
    """Fetch latest dengue data from InfoDengue API directly (no CSV storage)."""
    url = f"{INFODENGUE_URL}?geocode={GEOCODE}&disease=dengue&format=json&ew_format=SE"
    logger.info(f"Fetching InfoDengue data from {url}")
    
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    
    data = response.json()
    df = pd.DataFrame(data)
    logger.info(f"Downloaded {len(df)} rows from InfoDengue")
    
    # Keep only needed columns
    keep_cols = ['SE', 'casos', 'casos_est']
    available_cols = [c for c in keep_cols if c in df.columns]
    df = df[available_cols].copy()
    
    df['SE'] = df['SE'].astype(str).str.zfill(6)
    df['ano'] = df['SE'].str[:4].astype(int)
    df['semana'] = df['SE'].str[4:].astype(int)
    
    if 'casos' in df.columns:
        df['casos'] = pd.to_numeric(df['casos'], errors='coerce').fillna(0).astype(int)
    
    df = df.sort_values('SE').reset_index(drop=True)
    return df


def _fetch_openmeteo() -> pd.DataFrame:
    """Fetch latest weather data from OpenMeteo API directly (no CSV storage)."""
    # Get last 2 years of daily data (enough for lags/rolling up to 26 weeks)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=730)
    
    params = {
        'latitude': LAT,
        'longitude': LON,
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'daily': ','.join(config['data_sources']['openmeteo']['daily_variables']),
        'timezone': TIMEZONE
    }
    
    logger.info(f"Fetching OpenMeteo data from {OPENMETEO_URL}")
    response = requests.get(OPENMETEO_URL, params=params, timeout=60)
    response.raise_for_status()
    
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
    
    # NOWCAST SUBSTITUTION: Use max of confirmados and nowcast
    if 'casos_est' in merged.columns:
        merged['casos'] = pd.to_numeric(merged['casos'], errors='coerce').fillna(0).astype(float)
        merged['casos_est'] = pd.to_numeric(merged['casos_est'], errors='coerce').fillna(0).astype(float)
        mask = merged['casos_est'] > merged['casos']
        if mask.any():
            merged.loc[mask, 'casos'] = np.round(merged.loc[mask, 'casos_est']).astype(int)
            logger.info(f"Nowcast substitution: {mask.sum()} weeks updated")
    
    # Base features for feature engineering compatibility
    merged['log_casos'] = np.log1p(merged['casos'])
    merged['sin_semana'] = np.sin(2 * np.pi * merged['semana'] / 52)
    merged['cos_semana'] = np.cos(2 * np.pi * merged['semana'] / 52)
    
    # Legacy lags (for compatibility with trained models)
    merged['log_lag1'] = merged['log_casos'].shift(0)
    for lag in range(2, 9):
        merged[f'log_lag{lag}'] = merged['log_casos'].shift(lag - 1)
    
    # Weather lags (legacy)
    temp_col = None
    for c in ['temp_mean', 'temp_mean_mean', 'tempmed']:
        if c in merged.columns:
            temp_col = c
            break
    if temp_col:
        merged['temp_lag2'] = merged[temp_col].shift(1)
        merged['temp_lag4'] = merged[temp_col].shift(3)
    
    precip_col = None
    for c in ['precip_total', 'precip']:
        if c in merged.columns:
            precip_col = c
            break
    if precip_col:
        merged['precip_lag2'] = merged[precip_col].shift(1)
        merged['precip_lag4'] = merged[precip_col].shift(3)
    
    hum_col = None
    for c in ['humidity_mean', 'relative_humidity_2m_mean', 'umidmed']:
        if c in merged.columns:
            hum_col = c
            break
    if hum_col:
        merged['humidity_lag2'] = merged[hum_col].shift(1)
        merged['humidity_lag4'] = merged[hum_col].shift(3)
    
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
    
    # Merge and process in memory
    merged = _merge_and_prepare(dengue_df, weather_df)
    
    # Full feature engineering
    from data.features import EpisenseFeatureEngineer
    fe = EpisenseFeatureEngineer()
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
    casos_previstos: int = Field(..., description="Predicted number of dengue cases")
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
            previsoes.append(PredictionResponse(
                horizonte_semanas=pred['horizonte_semanas'],
                semana_epidemiologica=pred['semana_epidemiologica'],
                casos_previstos=pred['casos_previstos'],
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
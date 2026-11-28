"""
Episense API - Production Version
Reads the municipal data prepared by the scheduled collection.
No CSV persistence - all in-memory processing like Aedex All.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
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
import threading
from threading import Lock
# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.inference import EpisenseInference
from episense_security import (
    ApiKeyMiddleware,
    SecurityConfigError,
    get_settings,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load config
with open("config/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Fail closed in production: no API key -> no startup. In development the key
# is optional, but if EPISENSE_API_KEY is set it is always enforced.
try:
    security_settings = get_settings()
    security_settings.validate("api")
except SecurityConfigError as exc:
    logger.critical("Configuração de segurança inválida: %s", exc)
    raise

app = FastAPI(
    title="Episense API",
    description="Dengue case prediction API for municipalities in MS - scheduled municipal data",
    version="1.0.0",
    docs_url="/docs" if security_settings.docs_enabled else None,
    redoc_url="/redoc" if security_settings.docs_enabled else None,
    openapi_url="/openapi.json" if security_settings.docs_enabled else None,
)

# Auth/rate-limit is added first so CORS stays outermost: even 401/429
# responses then carry the CORS headers a browser client needs to read them.
app.add_middleware(ApiKeyMiddleware, settings=security_settings)

if security_settings.cors_origins:
    # Explicit allowlist only. Credentials stay disabled because this API does
    # not use cookies; the API key is sent in a header.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(security_settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-API-Key"],
        expose_headers=["Retry-After"],
    )

if security_settings.api_key:
    logger.info("Autenticação por API key habilitada")
else:
    logger.warning("Autenticação por API key desabilitada (somente desenvolvimento)")


def _custom_openapi():
    """Add the token schemes so Swagger UI shows an Authorize button.

    The middleware already enforces the token; this only documents the two
    accepted headers (X-API-Key and Authorization: Bearer).
    """
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    components = schema.setdefault("components", {})
    components.setdefault("securitySchemes", {}).update(
        {
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
            },
            "BearerAuth": {"type": "http", "scheme": "bearer"},
        }
    )
    schema["security"] = [{"ApiKeyAuth": []}, {"BearerAuth": []}]
    app.openapi_schema = schema
    return schema


app.openapi = _custom_openapi

# ---------------------------------------------------------------------------
# Inference engine — auto-reload por mtime (mesma lógica do dashboard)
# ---------------------------------------------------------------------------
# Antes a API carregava o engine uma única vez no startup e ficava servindo
# artefatos velhos após cada retreino. O dashboard (dashboard/app.py::get_inference)
# compara o mtime dos modelos a cada request e recarrega sem restart; replicamos
# a mesma lógica aqui, com lock para requisições concorrentes no mesmo worker.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(config.get('paths', {}).get('models', 'models'))
if not MODELS_DIR.is_absolute():
    MODELS_DIR = PROJECT_ROOT / MODELS_DIR

# Artefatos consumidos por EpisenseInference além dos .pkl. Incluí-los no
# fingerprint evita servir calibração/feature list antigas quando apenas um
# JSON é regravado (ex.: scripts/recompute_conformal_profile.py).
from models.releases import active_models
from models.forecast_audit import official_forecast

_loaded_release = None
_MODEL_ARTIFACT_GLOBS = (
    "lgbm_*.pkl",
    "feature_list.json",
    "normalization_params.json",
    "quantile_mode.json",
    "quantile_conformal.json",
    "quantile_recalibration.json",
    "quantile_recalibration_regime.json",
)

inference_engine: Optional[EpisenseInference] = None
_inference_mtime: Optional[float] = None
_inference_lock = Lock()

_data_cache = {}
_data_cache_lock = Lock()


def _models_latest_mtime(directory=None) -> float:
    """Maior mtime entre os artefatos que a inferência consome (0 se ausente)."""
    directory = directory or active_models(MODELS_DIR)
    latest = 0.0
    for pattern in _MODEL_ARTIFACT_GLOBS:
        try:
            for path in directory.glob(pattern):
                try:
                    latest = max(latest, path.stat().st_mtime)
                except OSError:
                    continue
        except OSError as e:
            logger.warning(f"[models] falha ao verificar artefatos em {MODELS_DIR}: {e}")
    return latest


def _count_loaded_models(engine: Optional[EpisenseInference]) -> int:
    """Número de modelos (horizonte × quantil × seed) carregados no engine."""
    try:
        return sum(len(seeds) for horizon in engine.models.values() for seeds in horizon.values())
    except Exception:
        return 0


def get_inference(force_reload: bool = False) -> Optional[EpisenseInference]:
    """Retorna o engine de inferência, recarregando quando os artefatos mudam.

    Mesma lógica do dashboard (dashboard/app.py::get_inference): compara o
    mtime mais recente dos modelos a cada chamada e recarrega sem restart.
    O lock protege requisições concorrentes dentro do mesmo worker uvicorn;
    workers são processos separados e mantêm seu próprio engine.
    Em falha de recarga, mantém o último engine válido em memória.
    """
    global inference_engine, _inference_mtime, _loaded_release

    directory = active_models(MODELS_DIR)
    latest = _models_latest_mtime(directory)

    def _is_current() -> bool:
        return (
            not force_reload
            and inference_engine is not None
            and _inference_mtime is not None
            and directory == _loaded_release
            and latest <= _inference_mtime
        )

    if _is_current():
        return inference_engine

    with _inference_lock:
        # Recheca após o lock: outra thread pode ter recarregado enquanto esperávamos.
        directory = active_models(MODELS_DIR)
        latest = _models_latest_mtime(directory)
        if _is_current():
            return inference_engine

        reloading = inference_engine is not None
        try:
            engine = EpisenseInference(str(directory), config)
        except Exception as e:
            logger.error(f"Falha ao recarregar inference de {MODELS_DIR}: {e}", exc_info=True)
            return inference_engine

        _loaded_release = directory
        inference_engine = engine
        _inference_mtime = latest if latest > 0 else time.time()
        logger.info(
            f"Inference {'recarregada' if reloading else 'carregada'}: "
            f"{_count_loaded_models(engine)} modelos, mtime={_inference_mtime:.0f} ({MODELS_DIR})"
        )
        return inference_engine

# Config constants
GEOCODE = config['project']['geocode']
INFODENGUE_URL = config['data_sources']['infodengue']['base_url']
OPENMETEO_URL = config['data_sources']['openmeteo']['base_url']
LAT = config['data_sources']['openmeteo']['latitude']
LON = config['data_sources']['openmeteo']['longitude']
TIMEZONE = config['data_sources']['openmeteo']['timezone']
TARGET_HORIZONS = config['targets']['horizons']


ROOT = Path(__file__).parent.parent


def _get_latest_engineered_data() -> pd.DataFrame:
    """Use the municipal base written by the scheduled dashboard job."""
    engine = get_inference()
    if engine is None:
        raise HTTPException(503, "Modelo municipal indisponível")
    path = ROOT / "data/processed/episense_base.csv"
    with _data_cache_lock:
        revision = (path.stat().st_mtime_ns, id(engine.config))
        if _data_cache.get("revision") != revision:
            base = pd.read_csv(path, dtype={"SE": str})
            _data_cache["df"] = engine.prepare_inference_features(base)
            _data_cache["revision"] = revision
        return _data_cache["df"]


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
    models_mtime: Optional[float] = None
    loaded_models_mtime: Optional[float] = None
    models_count: Optional[int] = None
    last_retrain_se: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize inference engine on startup (and prime mtime tracking)."""
    logger.info("Starting Episense API (Production - No CSV)...")

    engine = get_inference()
    if engine is not None:
        logger.info("Inference engine loaded successfully")
    else:
        logger.error("Failed to load inference engine")

@app.get("/", include_in_schema=False)
async def root():
    """Página inicial simples: aponta para health e Swagger (quando ligado)."""
    return {
        "service": "Episense API",
        "health": "/health",
        "docs": "/docs" if security_settings.docs_enabled else None,
    }


@app.get("/health", response_model=HealthResponse, openapi_extra={"security": []})
async def health_check():
    """Health check endpoint (não dispara recarga; apenas reporta o estado atual)."""
    last_retrain_se = None
    try:
        marker = (_loaded_release or active_models(MODELS_DIR)) / ".retrain_last_se"
        if marker.exists():
            last_retrain_se = marker.read_text().strip() or None
    except OSError:
        pass
    return HealthResponse(
        status="healthy" if inference_engine is not None else "degraded",
        timestamp=datetime.now().isoformat(),
        model_loaded=inference_engine is not None,
        data_source="Base municipal InfoDengue + OpenMeteo",
        models_mtime=_models_latest_mtime(),
        loaded_models_mtime=_inference_mtime,
        models_count=_count_loaded_models(inference_engine),
        last_retrain_se=last_retrain_se,
    )


from config.municipalities import SERVICES


def municipal_request(request, geocode, path, body=None):
    if str(GEOCODE) != "5002704" or geocode not in SERVICES:
        raise HTTPException(400, "Município sem serviço cadastrado")
    headers = {k: v for k, v in request.headers.items() if k in {"authorization", "x-api-key"}}
    try:
        result = requests.request("POST" if body else "GET", f"http://{SERVICES[geocode]}-api:8000{path}",
                                  json=body, headers=headers, timeout=180)
    except requests.RequestException as exc:
        raise HTTPException(503, "Serviço municipal indisponível") from exc
    if result.status_code != 200:
        raise HTTPException(result.status_code, "Recurso municipal indisponível")
    return result.json()


class PredictionRequest(BaseModel):
    geocode: str = Field(pattern=r"^[0-9]{7}$", examples=["5002704", *SERVICES])

    @field_validator("geocode", mode="before")
    @classmethod
    def normalize_geocode(cls, value):
        return str(value) if isinstance(value, int) else value


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictionRequest, request: Request):
    geocode = body.geocode
    if geocode != str(GEOCODE):
        return municipal_request(request, geocode, "/predict", {"geocode": geocode})

    # Recarrega automaticamente se os artefatos mudaram desde o último request.
    engine = get_inference()
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get fresh engineered data (cached 5 min)
        engineered = _get_latest_engineered_data()
        
        if engineered is None or engineered.empty:
            raise HTTPException(status_code=503, detail="Data not available")
        
        # Get predictions
        predictions = engine.get_latest_predictions(engineered)
        
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
            q95 = get_q(['q0950','q95','0.95',0.95], pred['casos_previstos'])
            # FIX Bug6: q95 definido antes de q75; fallback explícito para q95 (paridade com treino q0950)
            q75 = get_q(['q0750','q75','0.75',0.75], q95)
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
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.get("/predict/{geocode}", response_model=PredictResponse)
def predict_get(geocode: str, request: Request):
    if len(geocode) != 7 or not geocode.isascii() or not geocode.isdigit():
        raise HTTPException(422, "Geocode inválido")
    return predict(PredictionRequest(geocode=geocode), request)


@app.get("/forecast-issued/{origin_se}")
def get_issued_forecast(origin_se: str, request: Request, geocode: Optional[str] = None):
    if geocode is not None and geocode != str(GEOCODE):
        if len(origin_se) != 6 or not origin_se.isascii() or not origin_se.isdigit():
            raise HTTPException(422, "Semana epidemiológica inválida")
        return municipal_request(request, geocode, f"/forecast-issued/{origin_se}")
    entry = official_forecast(PROJECT_ROOT / "data/processed/forecast_archive.json", origin_se)
    if entry is None:
        raise HTTPException(status_code=404, detail="Nenhuma emissão programada para esta origem")
    return entry


@app.get("/metrics/walk-forward")
def get_walk_forward_metrics(request: Request, geocode: Optional[str] = None):
    """Relatório de comparação pareada por horizonte e fold."""
    if geocode is not None and geocode != str(GEOCODE):
        return municipal_request(request, geocode, "/metrics/walk-forward")
    import json

    path = active_models(MODELS_DIR) / "walk_forward_evaluation.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Avaliação walk-forward indisponível")
    return json.loads(path.read_text())


@app.get("/metrics")
def get_metrics(request: Request, geocode: Optional[str] = None):
    """Get model validation metrics."""
    if geocode is not None and geocode != str(GEOCODE):
        return municipal_request(request, geocode, "/metrics")
    if get_inference() is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import json
        import math
        metrics_path = active_models(MODELS_DIR) / "validation_results.json"
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
            # Remove MaxAE if present (not to be exposed)
            def strip_maxae(obj):
                if isinstance(obj, dict):
                    return {k: strip_maxae(v) for k, v in obj.items() if k.lower() != "maxae"}
                elif isinstance(obj, list):
                    return [strip_maxae(v) for v in obj]
                return obj
            metrics = strip_maxae(metrics)
            return metrics
        else:
            raise HTTPException(status_code=404, detail="Metrics not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Metrics error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to load metrics")


@app.get("/metadata")
def get_metadata(request: Request, geocode: Optional[str] = None):
    """Get model and data metadata."""
    if geocode is not None and geocode != str(GEOCODE):
        return municipal_request(request, geocode, "/metadata")
    engine = get_inference()
    
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
        "feature_count": sum(len(v) for v in engine.feature_names.values()) if engine and engine.feature_names else 0,
        "data_source": "Base municipal InfoDengue + OpenMeteo",
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
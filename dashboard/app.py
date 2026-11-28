"""Episense Dashboard - FastAPI"""
from fastapi import FastAPI, Query, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import pandas as pd
import numpy as np
import json
import yaml
import sys
from functools import lru_cache
import time
import threading
from threading import Lock
from typing import Dict
import logging
from datetime import datetime, timedelta, timezone
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    HAS_SCHEDULER = True
except ImportError:
    HAS_SCHEDULER = False

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from episense_security import (
    DashboardTokenMiddleware,
    SecurityConfigError,
    get_settings,
)

logger = logging.getLogger(__name__)

try:
    security_settings = get_settings()
    security_settings.validate("dashboard")
except SecurityConfigError as exc:
    logger.critical("Configuração de segurança inválida: %s", exc)
    raise

app = FastAPI(
    title="Episense Dashboard",
    version="1.0",
    docs_url="/docs" if security_settings.docs_enabled else None,
    redoc_url="/redoc" if security_settings.docs_enabled else None,
    openapi_url="/openapi.json" if security_settings.docs_enabled else None,
)

# Auth/rate-limit is added first so CORS stays outermost (see API main.py).
app.add_middleware(DashboardTokenMiddleware, settings=security_settings)

if security_settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(security_settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-API-Key"],
        expose_headers=["Retry-After"],
    )

if security_settings.api_key:
    logger.info("Autenticação por token do dashboard habilitada")
else:
    logger.warning("Autenticação por token do dashboard desabilitada (somente desenvolvimento)")

# Load data - AUTO-RELOAD por mtime (sem cron, sem restart)
BASE_CSV = ROOT / "data/processed/episense_base.csv"
from data.epiweeks import add_epiweeks
from models.validation_calendar import origin_weeks, target_weeks
from models.releases import active_models
from models.forecast_audit import capture_inputs, forecast_rows, official_forecast

MODELS_ROOT = ROOT / yaml.safe_load((ROOT / "config/config.yaml").read_text())["paths"]["models"]
CONFIG_PATH = ROOT / "config/config.yaml"

# Caches com mtime para reload automático quando nova semana entra (sem cron)
_df_base_mtime = None
_val_pred_mtime = None
_val_res_mtime = None
_config_mtime = None
_cases_cache = {"mtime": None, "casos": None}


def _get_casos_by_se():
    """Dict SE(zfill6)->casos do df_base; rebuild quando o CSV muda."""
    df = df_base
    mtime = _df_base_mtime
    c = _cases_cache
    if df is None or mtime is None:
        return {}
    if c["mtime"] == mtime and c["casos"] is not None:
        return c["casos"]
    casos = {}
    try:
        for se, v in zip(df["SE"], df["casos"]):
            fv = float(v)
            if np.isfinite(fv):
                casos[str(se).zfill(6)] = fv
    except Exception as e:
        logger.warning(f"casos_by_se falhou: {e}")
        return {}
    c["mtime"] = mtime
    c["casos"] = casos
    return casos


def _load_df_base():
    global df_base, _df_base_mtime
    try:
        mtime = BASE_CSV.stat().st_mtime
        if df_base is None or _df_base_mtime != mtime:
            df = pd.read_csv(BASE_CSV, usecols=lambda c: c in {"SE","casos","data_inicio_semana","casos_est","casos_notificados","casconf","is_imputed"}, dtype={"SE": str})
            df["SE"] = df["SE"].astype(str).str.replace(".0","", regex=False)
            df = df.sort_values("SE")
            df_base = df
            _df_base_mtime = mtime
            logger.info(f"df_base recarregado: {len(df)} linhas até {df['SE'].iloc[-1]}")
    except Exception as e:
        logger.warning(f"falha ao recarregar df_base: {e}")
    return df_base

def _load_val_pred():
    global val_pred, _val_pred_mtime
    try:
        path = active_models(MODELS_ROOT) / "validation_history.json"
        if not path.exists():
            path = path.with_name("validation_predictions.json")
        mtime = (path, path.stat().st_mtime)
        if val_pred is None or _val_pred_mtime != mtime:
            with open(path) as f:
                val_pred = json.load(f)
            _val_pred_mtime = mtime
            logger.info(f"val_pred recarregado: {len(val_pred)} horizontes")
    except Exception as e:
        logger.warning(f"falha ao recarregar val_pred: {e}")
    return val_pred

def _load_val_res():
    global val_res, _val_res_mtime
    try:
        path = active_models(MODELS_ROOT) / "validation_results.json"
        mtime = (path, path.stat().st_mtime)
        if val_res is None or _val_res_mtime != mtime:
            with open(path) as f:
                val_res = json.load(f)
            _val_res_mtime = mtime
            logger.info(f"val_res recarregado")
    except Exception as e:
        logger.warning(f"falha ao recarregar val_res: {e}")
    return val_res

# Inicialização (primeiro load)
try:
    df_base = pd.read_csv(BASE_CSV, usecols=lambda c: c in {"SE","casos","data_inicio_semana","casos_est","casos_notificados","casconf","is_imputed"}, dtype={"SE": str})
    df_base["SE"] = df_base["SE"].astype(str).str.replace(".0","", regex=False)
    df_base = df_base.sort_values("SE")
    _df_base_mtime = BASE_CSV.stat().st_mtime
except Exception:
    df_base = pd.DataFrame(columns=["SE","casos","data_inicio_semana"])
    _df_base_mtime = None

val_pred = {}
_load_val_pred()

val_res = {}
_load_val_res()

try:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    _config_mtime = CONFIG_PATH.stat().st_mtime
except Exception:
    config = {}
    _config_mtime = None

# Caches for forecast acceleration
_df_full_cache = None
_df_full_mtime = None
_feat_cache = None
_feat_cache_mtime = None
_feat_cache_lock = Lock()
FORECAST_ARCHIVE = ROOT / "data/processed/forecast_archive.json"
_last_refresh_check = 0
_refresh_lock = Lock()
_archive_lock = Lock()
_fetch_lock = Lock()
_fetch_last_error = None
_fetch_stale = False
_df_base_lock = Lock()
GEOCODE = config['project']['geocode']
INFODENGUE_URL = config['data_sources']['infodengue']['base_url']
OPENMETEO_URL = config['data_sources']['openmeteo']['base_url']
LAT = config['data_sources']['openmeteo']['latitude']
LON = config['data_sources']['openmeteo']['longitude']
TIMEZONE = config['data_sources']['openmeteo']['timezone']
TARGET_HORIZONS = config['targets']['horizons']
from data.live_utils import apply_nowcast, anchor_guard_filter

def _get_df_full():
    global _df_full_cache, _df_full_mtime
    try:
        mtime = (ROOT/"data/processed/episense_base.csv").stat().st_mtime
        if _df_full_cache is None or _df_full_mtime != mtime:
            df = pd.read_csv(ROOT/"data/processed/episense_base.csv", dtype={"SE": str})
            df["SE"] = df["SE"].astype(str).str.replace(".0", "", regex=False)
            df = df.sort_values("SE")
            _df_full_cache = df
            _df_full_mtime = mtime
        return _df_full_cache
    except Exception:
        return pd.read_csv(ROOT/"data/processed/episense_base.csv", dtype={"SE": str})

def _get_feat_full():
    global _feat_cache, _feat_cache_mtime
    with _feat_cache_lock:
        try:
            mtime = (ROOT/"data/processed/episense_base.csv").stat().st_mtime
            if _feat_cache is None or _feat_cache_mtime != mtime:
                inf = get_inference()
                if inf is None:
                    return None
                df = _get_df_full()
                # heavy feature engineering once for full history (815 rows, 639 feats)
                _feat_cache = inf.prepare_inference_features(df)
                _feat_cache_mtime = mtime
            return _feat_cache
        except Exception as e:
            print(f"feat cache failed: {e}")
            import traceback; traceback.print_exc()
            return None

def _get_raw_max_se() -> str:
    """Max SE in raw infodengue full (or latest timestamped)."""
    try:
        raw_full = ROOT / "data/raw/infodengue/infodengue_cg_full.csv"
        if raw_full.exists():
            df = pd.read_csv(raw_full, usecols=["SE"], dtype={"SE": str})
            return str(df["SE"].astype(str).str.zfill(6).max())
        # fallback latest timestamped
        import glob
        files = list((ROOT / "data/raw/infodengue").glob("infodengue_cg_*.csv"))
        if files:
            latest = max(files, key=lambda p: p.stat().st_mtime)
            df = pd.read_csv(latest, usecols=["SE"], dtype={"SE": str})
            return str(df["SE"].astype(str).str.zfill(6).max())
    except Exception as e:
        logger.warning(f"_get_raw_max_se failed: {e}")
    return "0"

def _get_base_max_se() -> str:
    try:
        with _df_base_lock:
            if len(df_base):
                return str(df_base["SE"].max()).zfill(6)
    except Exception:
        pass
    return "0"

def _read_forecast_archive():
    if not FORECAST_ARCHIVE.exists():
        return []
    entries = json.loads(FORECAST_ARCHIVE.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError("Arquivo de previsões inválido")
    return entries


def _original_forecasts(*, shape_legacy=False):
    for entry in _read_forecast_archive():
        forecast = entry.get("published_forecast")
        if not forecast:
            if entry.get("reprojected_at") or entry.get("quantiles_backfilled_at"):
                continue
            forecast = entry.get("forecast")
        if forecast:
            source = entry.get("source", "legacy_archive")
            profile = config.get("postprocessing", {}).get("band_shape", {})
            if shape_legacy and source == "legacy_archive" and profile.get("enabled") and profile.get("mode") == "profile":
                from models.band_shape import reshape_fan_logits
                from models.forecast_audit import QUANTILES
                # O legado pode ter apenas três quantis. No modo profile, a
                # mediana arquivada basta para reconstruir as cinco bandas.
                fan = {
                    row["h"]: {tau: np.log1p([row["q50"]]) for tau in QUANTILES}
                    for row in forecast if row.get("q50") is not None
                }
                forecast = forecast_rows(entry["origin_se"], reshape_fan_logits(fan, sorted(fan), profile))
                source = "legacy_profile"
            yield {
                "origin_se": entry["origin_se"],
                "generated_at": entry.get("published_at") or entry.get("generated_at"),
                "forecast": forecast,
                "source": source,
                "issuance_id": entry.get("issuance_id"),
            }


def _archive_forecast(origin_se, forecast, frame, engine, source_collected_at):
    """Persiste uma única emissão programada por origem."""
    from tempfile import NamedTemporaryFile
    with _archive_lock:
        entries = _read_forecast_archive()
        if any(e.get("origin_se") == origin_se and e.get("issuance_id") for e in entries):
            return False
        audit = capture_inputs(FORECAST_ARCHIVE.parent / "forecast_inputs", frame, engine)
        observed = audit["persistence"]
        shadow = []
        if observed is not None and np.isfinite(observed) and observed >= 0:
            shadow = [
                {"h": row["h"], "target_se": row["target_se"], "cases": observed,
                 "observed_se": audit["persistence_week"]}
                for row in forecast if row["h"] in (1, 2)
            ]
        from models.h8_shadow import shadow_forecast
        try:
            h8_shadow = shadow_forecast(engine, frame, origin_se)
        except Exception:
            logger.exception("Falha na variante experimental de H8 na origem %s", origin_se)
            h8_shadow = None
        entries.append({
            "h8_shadow": h8_shadow,
            "origin_se": origin_se,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "forecast": forecast,
            "source": "issued",
            "issuance_id": f"{GEOCODE}-dengue-{origin_se}",
            "source_collected_at": source_collected_at,
            "persistence_shadow": shadow,
            "audit": audit,
        })
        FORECAST_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=FORECAST_ARCHIVE.parent,
                                delete=False) as handle:
            temp = Path(handle.name)
            try:
                json.dump(entries, handle, indent=2, ensure_ascii=False)
                handle.flush()
                temp.replace(FORECAST_ARCHIVE)
            finally:
                temp.unlink(missing_ok=True)
        return True


def _merge_and_prepare_live(dengue_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    dengue_df['SE'] = dengue_df['SE'].astype(str).str.zfill(6)
    weather_df['SE'] = weather_df['SE'].astype(str).str.zfill(6)
    merged = pd.merge(dengue_df, weather_df, on='SE', how='left', suffixes=('','_weather'))
    cols_to_drop = [c for c in merged.columns if c.endswith('_weather') and c.replace('_weather','') in merged.columns]
    merged = merged.drop(columns=cols_to_drop)
    from data.epiweeks import weeks_in_year
    min_se = int(merged['SE'].min())
    max_se = int(merged['SE'].max())
    all_weeks=[]
    for year in range(min_se//100, max_se//100+1):
        for week in range(1, weeks_in_year(year)+1):
            se=f"{year}{week:02d}"
            if min_se <= int(se) <= max_se:
                all_weeks.append(se)
    all_weeks_df=pd.DataFrame({'SE': all_weeks})
    all_weeks_df['SE']=all_weeks_df['SE'].astype(str).str.zfill(6)
    merged=pd.merge(all_weeks_df, merged, on='SE', how='left')
    # ANCHOR GUARD: paridade com data/process_data.py — dropa última SE com clima parcial (<7 dias) via shared helper
    clima_dias = None
    if not weather_df.empty and 'clima_dias' in weather_df.columns:
        clima_dias = dict(zip(weather_df['SE'].astype(str).str.zfill(6), weather_df['clima_dias']))
    elif not weather_df.empty and 'clima_dias' in merged.columns:
        clima_dias = dict(zip(merged['SE'].astype(str).str.zfill(6), merged['clima_dias']))
    elif 'week_start' in merged.columns and 'week_end' in merged.columns:
        try:
            dias = (pd.to_datetime(merged['week_end']) - pd.to_datetime(merged['week_start'])).dt.days + 1
            clima_dias = dict(zip(merged['SE'].astype(str).str.zfill(6), dias))
        except Exception:
            clima_dias = None
    elif not weather_df.empty and 'week_start' in weather_df.columns and 'week_end' in weather_df.columns:
        try:
            dias = (pd.to_datetime(weather_df['week_end']) - pd.to_datetime(weather_df['week_start'])).dt.days + 1
            clima_dias = dict(zip(weather_df['SE'].astype(str).str.zfill(6), dias))
        except Exception:
            clima_dias = None
    merged = anchor_guard_filter(merged, clima_dias) if clima_dias else merged
    merged = apply_nowcast(merged)
    if 'casos' in merged.columns:
        merged['is_imputed'] = merged['casos'].isna()
        merged['casos_observados'] = merged['casos']
        merged['casos']=merged['casos'].fillna(0)
    weather_cols=[c for c in merged.columns if c not in ['SE','ano','semana','casos','casos_est','casos_notificados','casos_observados']]
    for col in weather_cols:
        if merged[col].dtype in ['float64','int64']:
            merged[col]=merged[col].ffill()
    merged['ano']=merged['SE'].str[:4].astype(int)
    merged['semana']=merged['SE'].str[4:].astype(int)
    merged['log_casos']=np.log1p(merged['casos'])
    try:
        from data.epiweeks import weeks_in_year
        weeks=merged['ano'].apply(lambda y: weeks_in_year(int(y)))
        merged['sin_semana']=np.sin(2*np.pi*merged['semana']/weeks)
        merged['cos_semana']=np.cos(2*np.pi*merged['semana']/weeks)
    except Exception:
        merged['sin_semana']=np.sin(2*np.pi*merged['semana']/52)
        merged['cos_semana']=np.cos(2*np.pi*merged['semana']/52)
    for lag in range(1,9):
        merged[f'log_lag{lag}']=merged['log_casos'].shift(lag)
    temp_col=None
    for c in ['temp_mean','temp_mean_mean','tempmed']:
        if c in merged.columns:
            temp_col=c; break
    if temp_col:
        merged['temp_lag2']=merged[temp_col].shift(2)
        merged['temp_lag4']=merged[temp_col].shift(4)
    precip_col=None
    for c in ['precip_total','precip']:
        if c in merged.columns:
            precip_col=c; break
    if precip_col:
        merged['precip_lag2']=merged[precip_col].shift(2)
        merged['precip_lag4']=merged[precip_col].shift(4)
    hum_col=None
    for c in ['humidity_mean','relative_humidity_2m_mean','umidmed']:
        if c in merged.columns:
            hum_col=c; break
    if hum_col:
        merged['humidity_lag2']=merged[hum_col].shift(2)
        merged['humidity_lag4']=merged[hum_col].shift(4)
    return merged

def _get_live_engineered():
    features = _get_feat_full()
    if features is None or features.empty:
        raise HTTPException(503, "Base municipal indisponível")
    return features


def _reload_local_data():
    _load_df_base()
    _load_val_pred()
    _load_val_res()


_inference = None
_inference_release = None
_inference_lock = Lock()


def get_inference():
    global _inference, _inference_release
    with _inference_lock:
        directory = active_models(MODELS_ROOT)
        if _inference is None or directory != _inference_release:
            try:
                from models.inference import EpisenseInference
                engine = EpisenseInference(model_dir=str(directory), config=config)
            except Exception:
                logger.exception("Falha ao carregar modelos de %s", directory)
            else:
                _inference = engine
                _inference_release = directory
        return _inference


def se_to_date(se):
    try:
        row = df_base[df_base["SE"]==str(se)]
        if not row.empty:
            return row.iloc[0]["data_inicio_semana"]
        return se
    except: return se

_dashboard_scheduler = None

_last_scheduled_issuance = None


def _dashboard_scheduled_refresh():
    """Emite após a coleta das duas fontes e registra falhas para a próxima tentativa."""
    global _last_scheduled_issuance, _last_refresh_check
    _last_refresh_check = time.time()
    from data.collect_infodengue import (
        collect_infodengue_data, process_infodengue_data, update_full_history,
        save_infodengue_data,
    )
    from data.archive_weekly_cases import save_counts
    from data.collect_openmeteo import collect_openmeteo_data, aggregate_to_weekly, save_openmeteo_data
    from data.process_data import EpisenseDataProcessor
    from scripts.retrain_pipeline import train_and_publish

    origin_expected = None
    pending = BASE_CSV.with_name(".weekly_update_pending")
    try:
        raw = collect_infodengue_data()
        if raw.empty:
            raise RuntimeError("Resposta InfoDengue vazia")
        dengue = process_infodengue_data(raw)
        origin = origin_expected = str(dengue["SE"].max())
        base_week = _get_df_full()["SE"].max() if BASE_CSV.exists() else "0"
        if origin < base_week or (origin == base_week and not pending.exists()):
            return
        if official_forecast(FORECAST_ARCHIVE, origin) and origin == base_week:
            pending.unlink(missing_ok=True)
            return

        collected_at = datetime.now(timezone.utc).isoformat()
        window = dengue.sort_values("SE").tail(16)
        expected = [add_epiweeks(origin, offset) for offset in range(-15, 1)]
        if window["SE"].tolist() != expected or window["casos"].isna().any():
            raise RuntimeError("Janela de 16 semanas incompleta")
        save_counts(window[["SE", "casos"]].to_dict("records"),
                    ROOT / "data/raw/infodengue/revisions_16")
        pending.write_text(origin)
        directory = active_models(MODELS_ROOT)
        marker = directory / ".retrain_last_se"
        if origin != base_week or not marker.exists() or marker.read_text().strip() != origin:
            daily = collect_openmeteo_data()
            if daily is None or daily.empty:
                raise RuntimeError("Coleta OpenMeteo vazia")
            weekly = aggregate_to_weekly(daily)
            if weekly.empty:
                raise RuntimeError("Dados climáticos semanais indisponíveis")
            save_openmeteo_data(daily, weekly, "2010-01-01", str(pd.Timestamp.now(tz=TIMEZONE).date()))
            save_infodengue_data(dengue)
            path = ROOT / "data/raw/infodengue/infodengue_cg_full.csv"
            history = update_full_history(dengue, path)
            temp = path.with_suffix(".tmp")
            history.to_csv(temp, index=False)
            temp.replace(path)
            frame = EpisenseDataProcessor().run_full_pipeline()
            if frame is None or frame.empty or str(frame["SE"].max()) != origin:
                raise RuntimeError(f"Base municipal ainda não alcançou {origin}")
            directory = train_and_publish(MODELS_ROOT, ROOT / "logs", trained_week=origin, weekly_update=True)
        frame = pd.read_csv(BASE_CSV, dtype={"SE": str}).sort_values("SE")
        engine = get_inference()
        if _inference_release != directory or engine is None:
            raise RuntimeError("Modelo publicado indisponível")
        forecast = forecast_rows(origin, engine.predict(frame))
        if len(forecast) != 8 or any(
            row[q] is None for row in forecast for q in ("q05", "q25", "q50", "q75", "q95")
        ):
            raise RuntimeError("Previsão incompleta")
        _archive_forecast(origin, forecast, frame, engine, collected_at)
        pending.unlink(missing_ok=True)
        _last_scheduled_issuance = {"origin_se": origin, "status": "issued", "checked_at": collected_at}
        logger.info("Emissão programada concluída para %s", origin)
    except Exception as exc:
        _last_scheduled_issuance = {"origin_se": origin_expected, "status": "pending",
                                    "checked_at": datetime.now(timezone.utc).isoformat()}
        logger.warning("Emissão programada pendente para %s: %s", origin_expected, exc)


@app.on_event("startup")
async def dashboard_startup():
    global _dashboard_scheduler
    if HAS_SCHEDULER:
        try:
            _dashboard_scheduler = BackgroundScheduler(timezone=config['data_sources']['openmeteo'].get('timezone', 'America/Campo_Grande'))
            _dashboard_scheduler.add_job(_dashboard_scheduled_refresh, 'cron', hour='1,7,13,19', minute=15,
                                         id='dash_daily', coalesce=True, max_instances=1, misfire_grace_time=60)
            _dashboard_scheduler.start()
            logger.info("Dashboard scheduler iniciado: coleta municipal às 01:15, 07:15, 13:15 e 19:15")
        except Exception as e:
            logger.warning(f"Dashboard scheduler não iniciado: {e}")
@app.on_event("shutdown")
async def dashboard_shutdown():
    global _dashboard_scheduler
    if _dashboard_scheduler:
        try:
            _dashboard_scheduler.shutdown(wait=False)
        except Exception:
            pass

@app.get("/api/auth/check")
def api_auth_check():
    """Endpoint leve usado pela tela de token para validar a chave."""
    return {"ok": True, "required": security_settings.api_key is not None}


@app.get("/api/real")
def api_real(limit: int = Query(52, ge=10, le=200), end_se: str = None):
    _reload_local_data()
    frame = df_base if end_se is None else df_base.loc[df_base["SE"] <= str(end_se).zfill(6)]
    frame = frame.tail(limit).copy()
    if "is_imputed" in frame:
        frame.loc[frame["is_imputed"].fillna(False), "casos"] = np.nan
    records = frame.astype(object).where(frame.notna(), None).to_dict(orient="records")
    return {"data": records, "source": "base_municipal", "stale": False}


@app.get("/api/overview")
def api_overview():
    columns = {"SE", "pop", "is_imputed", "casos", "temp_mean_mean",
               "precip_total", "humidity_mean"}
    frame = pd.read_csv(BASE_CSV, usecols=lambda col: col in columns, dtype={"SE": str})
    frame = frame.sort_values("SE").tail(12).copy()
    if "is_imputed" in frame:
        missing = frame["is_imputed"].fillna(False)
        frame.loc[missing, frame.columns.difference(["SE", "is_imputed"])] = np.nan
    frame["threshold_per_100k"] = config.get("epidemic_threshold_per_100k")
    return {"weeks": frame.astype(object).where(frame.notna(), None).to_dict("records")}


@app.get("/api/weeks")
def api_weeks(limit: int = 100):
    _reload_local_data()
    # return last N SEs for selector
    df = df_base.tail(limit)
    return {"weeks": df["SE"].tolist(), "dates": df["data_inicio_semana"].tolist()}

@app.get("/api/metrics")
def api_metrics(latest: bool = False):
    _reload_local_data()
    model = active_models(MODELS_ROOT)
    manifest = model / "release.json"
    metadata = json.loads(manifest.read_text()) if manifest.exists() else {}
    if latest:
        for candidate in (MODELS_ROOT / "releases").glob("*/release.json"):
            info = json.loads(candidate.read_text())
            if info.get("completed_at", "") > metadata.get("completed_at", ""):
                model, metadata = candidate.parent, info
    provenance = {"release": model.name, "completed_at": metadata.get("completed_at"),
                  "trained_week": metadata.get("trained_week")}
    path = model / "walk_forward_evaluation.json"
    if path.exists():
        report = json.loads(path.read_text())
        if report.get("schema_version") == 2:
            folds = {"avg": {"label": "Total por horizonte", "period": "Linhas pareadas",
                             "se": report["evaluation_cutoff_se"]}}
            for fold, cutoff in report["fold_cutoff_se"].items():
                folds[fold] = {"label": str(cutoff[:4]), "period": "Primeira origem do fold", "se": cutoff}
            result = {**provenance, "folds": folds, "scope": report["scope"], "evaluation_role": "model_development",
                      "training_embargo_verified": report["training_embargo_verified"]}
            for h, avg in report["by_horizon"].items():
                result[h] = {"avg": avg, **report["by_horizon_and_fold"][h]}
            return result
    # Fold periods (epi weeks) derived from validation_predictions (final 4 folds 2022-2025)
    # Keep in sync with dashboard fold labels
    fold_info = {
        "avg": {"label": "Média geral", "period": "2022 a 2025", "se": "202236 a 202632", "note": "4 folds; Fold 2025 parcial 49 sem (até 202632)"},
        "0": {"label": "Fold 2022", "period": "04/09/2022 a 27/08/2023", "se": "202236 a 202335"},
        "1": {"label": "Fold 2023", "period": "03/09/2023 a 25/08/2024", "se": "202336 a 202435"},
        "2": {"label": "Fold 2024", "period": "01/09/2024 a 31/08/2025", "se": "202436 a 202536"},
        "3": {"label": "Fold 2025 (parcial)", "period": "07/09/2025 a 09/08/2026", "se": "202537 a 202632", "note": "49 semanas observadas; 202633+ ainda sem dados"},
    }
    if model != active_models(MODELS_ROOT):
        raise HTTPException(404, "Avaliação indisponível para o último treinamento")
    out = {**provenance, "folds": fold_info, "scope": "legacy"}
    for h in [str(i) for i in range(1,9)]:
        if h not in val_res:
            continue
        avg = val_res[h].get("ensemble_avg", {})
        per = val_res[h].get("per_fold_ensemble", [])
        by_fold = {str(r.get("fold")): r for r in per}
        # conjunto canônico exposto (spec 2026-10, sem MAPE/CRPS) + referência do baseline
        # e métricas de leitura direta (gain vs baseline / WAPE); (MaxAE/dados legados removidos)
        adv_keys = ["mae", "rmse", "bias", "wape",
                    "wis_total", "wis_50", "wis_90", "width_50", "width_90",
                    "coverage_50", "coverage_90",
                    "baseline_wis", "baseline_mae", "baseline_coverage_90",
                    "skill_mae", "skill_wis"]
        def pick(src):
            return {k: src.get(k) for k in adv_keys} if src else None
        out[h] = {
            "avg": pick(avg),
            "0": pick(by_fold.get("0", {})),
            "1": pick(by_fold.get("1", {})),
            "2": pick(by_fold.get("2", {})),
            "3": pick(by_fold.get("3", {})),
        }
    return out

@app.get("/api/history")
def api_history(horizon: str = "1", window: int = 52):
    """Exibe backtest da versão ativa e registros originais por horizonte."""
    _reload_local_data()
    h = str(horizon).replace("h", "")
    _load_val_pred()
    processed = _val_pred_mtime and _val_pred_mtime[0].name == "validation_history.json"
    if h not in val_pred:
        return {"history": [], "horizon": h, "source": "validation_predictions"}

    def final_folds(qkey):
        folds = val_pred.get(h, {}).get(qkey, [])
        filtered = []
        for fold in folds:
            ses = origin_weeks(fold)
            if ses and str(ses[0]) < "202236":
                continue
            filtered.append(fold)
        return filtered or folds

    q_mid = "0.5" if "0.5" in val_pred[h] else sorted(val_pred[h].keys())[0]
    q_low = "0.05" if "0.05" in val_pred[h] else None
    q_low25 = "0.25" if "0.25" in val_pred[h] else None
    q_high75 = "0.75" if "0.75" in val_pred[h] else None
    q_high = "0.95" if "0.95" in val_pred[h] else None

    points = {}
    for fold in final_folds(q_mid):
        for origin, se, pred, true in zip(origin_weeks(fold), target_weeks(fold, int(h)),
                                         fold["y_pred_casos"], fold["y_test_casos"]):
            se = str(se)
            points.setdefault(se, {})
            points[se].update({"SE": se, "origin_se": origin, "pred": float(pred), "true": float(true)})

    if q_low:
        for fold in final_folds(q_low):
            for se, pred in zip(target_weeks(fold, int(h)), fold["y_pred_casos"]):
                se = str(se)
                if se in points:
                    points[se]["pred_low"] = float(pred)
                    points[se]["q05"] = float(pred)
    if q_low25:
        for fold in final_folds(q_low25):
            for se, pred in zip(target_weeks(fold, int(h)), fold["y_pred_casos"]):
                se = str(se)
                if se in points:
                    points[se]["q25"] = float(pred)
    if q_high75:
        for fold in final_folds(q_high75):
            for se, pred in zip(target_weeks(fold, int(h)), fold["y_pred_casos"]):
                se = str(se)
                if se in points:
                    points[se]["q75"] = float(pred)
    if q_high:
        for fold in final_folds(q_high):
            for se, pred in zip(target_weeks(fold, int(h)), fold["y_pred_casos"]):
                se = str(se)
                if se in points:
                    points[se]["pred_high"] = float(pred)
                    points[se]["q95"] = float(pred)

    se_map = dict(zip(df_base["SE"], df_base["data_inicio_semana"]))
    base_window = df_base.tail(window)
    # FIX (v8, relato 03/11): o gráfico histórico nunca deve cortar o INÍCIO do
    # walk-forward. O recorte por `window` (últimas N semanas do frame live/CSV)
    # escondia o começo do Fold 2022 (202236): com window=200 o frame live só
    # alcançava ~202301 — as ~19 primeiras semanas do fold0 não apareciam.
    # Quando a janela pedida começa DEPOIS do primeiro fold exibido, prepend as
    # semanas do CSV base (o tail live é preservado integralmente).
    try:
        _fold_start = min(
            (target_weeks(f, int(h))[0] for f in final_folds(q_mid) if origin_weeks(f)),
            default=None,
        )
        if _fold_start:
            _se_col = base_window["SE"].astype(str).str.zfill(6)
            _starts = list(_se_col)
            if _starts and _fold_start < _starts[0]:
                _base_se = df_base["SE"].astype(str).str.zfill(6)
                _pre = df_base[(_base_se >= _fold_start) & (_base_se < _starts[0])].copy()
                if len(_pre):
                    _pre["SE"] = _pre["SE"].astype(str).str.zfill(6)
                    base_window = pd.concat([_pre, base_window], ignore_index=True)
                    base_window = base_window.sort_values("SE").reset_index(drop=True)
                    logger.info(f"api_history: janela estendida até {_fold_start} — início do Fold 2022 preservado ({len(_pre)} semanas prependidas; window={window})")
    except Exception as e:
        logger.info(f"api_history fold-start extension skip: {e}")
    # Stitch archive live (todos os quantis) — preenche historico além da validacao walk-forward
    # Para cada SE sem pred (gap honesto pos 202632), usa a previsao live mais recente que mirava aquela SE.
    # Ex: 202633 H1 vem de origin 202632 H1; 202634 H1 vem de origin 202633 H1 (mais recente que 202632 H2).
    archive_map = {}  # target_se -> {pred_low, pred, pred_high, origin_se, h, q05, q50, q95}
    try:
        if FORECAST_ARCHIVE.exists():
            arch = list(_original_forecasts(shape_legacy=processed))
            # arch é lista de {origin_se, forecast: [{h,target_se,q05,q50,q95}]}
            # Para um dado horizon h, só nos interessa o item com h == int(h)
            hh = int(h)
            # ordena por origin_se crescente, o ultimo vence (mais recente)
            for entry in sorted(arch, key=lambda x: str(x.get("origin_se")).zfill(6)):
                orig = str(entry.get("origin_se")).zfill(6)
                for fc in entry.get("forecast", []):
                    if int(fc.get("h", -1)) != hh:
                        continue
                    tgt = str(fc.get("target_se")).zfill(6)
                    # salva todos os quantis, e mantém também pred_* para compat
                    archive_map[tgt] = {
                        "pred_low": fc.get("q05"),
                        "pred": fc.get("q50"),
                        "pred_high": fc.get("q95"),
                        "q05": fc.get("q05"),
                        "q25": fc.get("q25"),
                        "q50": fc.get("q50"),
                        "q75": fc.get("q75"),
                        "q95": fc.get("q95"),
                        "origin_se": orig,
                        "h": fc.get("h"),
                        "source": entry["source"],
                    }
    except Exception as e:
        logger.warning(f"archive stitch failed for h={h}: {e}")

    tail_path = active_models(MODELS_ROOT) / "retrospective_tail.json"
    if tail_path.exists():
        for row in json.loads(tail_path.read_text()):
            if row["h"] == int(h):
                archive_map.setdefault(row["target_se"], dict(row, source="retrospective_revised_inputs"))

    # Build history from base window so last week is identical for every horizon (fix shrinking chart / missing Hoje)
    all_points = []
    for _, row in base_window.iterrows():
        se = str(row["SE"]).zfill(6)
        # validation pred may be missing for long horizons at the very end (expected gap)
        pv = points.get(se, {})
        p_low = pv.get("pred_low")
        p_med = pv.get("pred")
        p_high = pv.get("pred_high")
        p_q25 = pv.get("q25")
        p_q75 = pv.get("q75")
        stitched = False
        stitch_meta = None
        am = archive_map.get(se)
        arch_full = am is not None and am.get("q05") is not None and am.get("q50") is not None and am.get("q95") is not None
        if arch_full and (p_med is None or am["source"] == "issued" or not processed):
            p_low = am.get("q05", p_low)
            p_q25 = am.get("q25", p_q25)
            p_med = am.get("q50", p_med)
            p_q75 = am.get("q75", p_q75)
            p_high = am.get("q95", p_high)
            stitched = True
            stitch_meta = am
        pt = {
            "SE": se,
            "pred": float(p_med) if p_med is not None else None,
            "pred_low": float(p_low) if p_low is not None else None,
            "pred_high": float(p_high) if p_high is not None else None,
            # expõe também q05/q25/q50/q75/q95 explicitamente para o frontend (5 quantis)
            "q05": float(p_low) if p_low is not None else None,
            "q25": float(p_q25) if p_q25 is not None else None,
            "q50": float(p_med) if p_med is not None else None,
            "q75": float(p_q75) if p_q75 is not None else None,
            "q95": float(p_high) if p_high is not None else None,
            "true": float(pv["true"]) if pv.get("true") is not None and np.isfinite(pv["true"]) else None,
            "real": float(row["casos"]) if pd.notna(row["casos"]) and not row.get("is_imputed", False) else None,
            "confirmados": float(row["casconf"]) if pd.notna(row.get("casconf")) else None,
            "notificados": float(row["casos_notificados"]) if pd.notna(row.get("casos_notificados")) else None,
            "nowcast": float(row["casos_est"]) if pd.notna(row.get("casos_est")) else None,
            "date": se_map.get(se, se),
            "source": stitch_meta["source"] if stitched else (("backtest_processed" if processed else "backtest_raw") if p_med is not None else None),
            "stitched": stitched,
            "stitch_origin": stitch_meta.get("origin_se") if stitch_meta else None,
            "origin_se": stitch_meta["origin_se"] if stitched else pv.get("origin_se"),
            "target_se": se,
        }
        all_points.append(pt)
    # Gap honesto: ultimas h semanas SEM archive permanecem null (spanGaps false no chart).
    # Com archive, gap é preenchido automaticamente quando nova semana é publicada.
    has_stitched = any(p.get("stitched") for p in all_points)
    return {
        "horizon": h,
        "history": all_points,
        "source": "validation_predictions+archive" if has_stitched else "validation_predictions",
        "scope": ("walk_forward_processed" if processed else "walk_forward_raw") + ("+live_archive" if has_stitched else ""),
        "quantiles": {"low": q_low or "0.05", "median": q_mid or "0.5", "high": q_high or "0.95"},
        "stitched_count": sum(1 for p in all_points if p.get("stitched")),
        "archive_entries": len(archive_map),
    }

@app.get("/api/forecast")
def api_forecast(origin_se: str, horizon: str = None):
    _reload_local_data()
    """Forecast from origin_se - LIVE like API 8001.

    Quando alguém entra no dashboard a previsão futura é feita na hora via inferência
    com dados frescos da InfoDengue+OpenMeteo (mesma lógica da API 8001), não lida
    de arquivo estático. Origens históricas ainda podem usar validação para auditoria,
    mas a origem mais recente SEMPRE é live.
    """
    _load_val_pred()
    processed = _val_pred_mtime and _val_pred_mtime[0].name == "validation_history.json"
    origin_se = str(origin_se)
    issued = official_forecast(FORECAST_ARCHIVE, origin_se)
    if issued and (origin_se != _get_base_max_se() or issued["model_release"] == active_models(MODELS_ROOT).name):
        return {**issued, "origin_date": se_to_date(origin_se), "forecast": [
            {**row, "target_date": se_to_date(row["target_se"]),
             "real": _get_casos_by_se().get(row["target_se"])}
            for row in issued["forecast"]
        ]}
    live_latest = _get_base_max_se()

    if origin_se not in df_base["SE"].values:
        # se não está no CSV mas pode estar no live fresco, tenta buscar live
        try:
            eng = _get_live_engineered()
            if origin_se in eng["SE"].astype(str).str.zfill(6).values:
                pass
            else:
                return {"error": f"SE {origin_se} not found", "origin_se": origin_se}
        except Exception:
            return {"error": f"SE {origin_se} not found", "origin_se": origin_se}

    def lookup_validation():
        """Consulta quantis gravados no backtest temporal."""
        result = {
            "origin_se": origin_se,
            "origin_date": se_to_date(origin_se),
            "forecast": [],
            "source": "validation",
            "calibration_scheme": "walk_forward_processed" if processed else "raw",
            "quantiles": {"low": "0.05", "median": "0.5", "high": "0.95"},
        }
        casos_by_se = _get_casos_by_se()
        any_available = False
        last_data_se = ""
        try:
            if df_base is not None and len(df_base):
                last_data_se = str(df_base["SE"].max()).zfill(6)
        except Exception:
            last_data_se = ""
        for hh in range(1, 9):
            hstr = str(hh)
            target_se = add_epiweeks(origin_se, hh)
            target_se = str(target_se).zfill(6)
            qvals = {"0.05": None, "0.25": None, "0.5": None, "0.75": None, "0.95": None}
            fold_by_tau = {}
            if hstr in val_pred:
                for q in ("0.05", "0.25", "0.5", "0.75", "0.95"):
                    folds = val_pred[hstr].get(q, [])
                    # mesmo filtro do /api/history: descarta fold legacy 2021
                    final = [f for f in folds
                             if not (origin_weeks(f) and origin_weeks(f)[0] < "202236")] or folds
                    for fold in final:
                        ses = [str(x).zfill(6) for x in origin_weeks(fold)]
                        if origin_se in ses:
                            idx = ses.index(origin_se)
                            try:
                                v = float(fold["y_pred_casos"][idx])
                            except Exception:
                                continue
                            if np.isfinite(v):
                                qvals[q] = v
                                fold_by_tau[q] = fold.get("fold")
                    # last-win implícito: itera na ordem salva (fold0..3)
                    # e o último fold que cobre o alvo sobrescreve
            real = casos_by_se.get(target_se)
            if target_se > last_data_se:
                real = None  # semana sem observação publicada
            disponivel = qvals.get("0.5") is not None
            any_available = any_available or disponivel
            row = {
                "h": hh,
                "target_se": target_se,
                "target_date": se_to_date(target_se),
                "q05": qvals.get("0.05"), "q25": qvals.get("0.25"),
                "q50": qvals.get("0.5"), "q75": qvals.get("0.75"), "q95": qvals.get("0.95"),
                "real": real,
                "fold": fold_by_tau.get("0.5"),
                "calibrated": False,
                "disponivel": bool(disponivel),
            }
            result["forecast"].append(row)
        result["any_calibrated"] = False
        return result if any_available else None

    # Se a origem é a mais recente, SEMPRE faz live fresco (mesma lógica da API 8001)
    is_latest = origin_se == live_latest
    if not is_latest:
        try:
            entries = [e for e in _original_forecasts(shape_legacy=processed) if str(e["origin_se"]) == origin_se]
        except (OSError, ValueError):
            logger.exception("Erro ao consultar arquivo de previsões")
            entries = []
        if processed:
            historical = lookup_validation()
            if historical:
                return historical
        if entries:
            entry = entries[-1]
            forecast = [
                {**row, "target_date": se_to_date(row["target_se"]),
                 "real": _get_casos_by_se().get(row["target_se"])}
                for row in entry["forecast"]
            ]
            return {**entry, "origin_date": se_to_date(origin_se), "forecast": forecast}
        return lookup_validation() or {
            "origin_se": origin_se, "origin_date": se_to_date(origin_se),
            "forecast": [], "source": "none",
        }

    # LIVE path - igual API 8001: dados frescos in-memory
    

    inf = get_inference()
    if inf:
        try:
            base = _get_df_full()
            if base.empty or str(base["SE"].iloc[-1]).zfill(6) != origin_se:
                raise ValueError("Dados indisponíveis para a origem solicitada")
            preds = inf.predict(base)
            forecast = forecast_rows(origin_se, preds)
            for row in forecast:
                row["target_date"] = se_to_date(row["target_se"])
            result = {"origin_se": origin_se, "origin_date": se_to_date(origin_se),
                      "forecast": forecast, "source": "live"}
            return result
        except Exception as e:
            print(f"live inference failed {e}, fallback to partial validation")
            import traceback; traceback.print_exc()

    partial = lookup_validation()
    if partial is not None:
        return partial
    return {"origin_se": origin_se, "origin_date": se_to_date(origin_se), "forecast": [], "source": "none"}

@app.get("/api/forecast-history")
def api_forecast_history():
    try:
        if FORECAST_ARCHIVE.exists():
            data = list(_original_forecasts())
            return {"history": data, "count": len(data)}
        return {"history": [], "count": 0}
    except Exception as e:
        logger.error(f"forecast-history error: {e}", exc_info=True)
        return {"history": [], "count": 0, "error": "failed to load forecast history"}

@app.get("/api/status")
def api_status():
    _reload_local_data()
    try:
        raw_max = _get_raw_max_se()
        base_max = _get_base_max_se()
        archive_count = 0
        if FORECAST_ARCHIVE.exists():
            try:
                archive_count = len(json.loads(FORECAST_ARCHIVE.read_text(encoding="utf-8")))
            except: pass
        fetch_stale = globals().get("_fetch_stale", False)
        fetch_error = globals().get("_fetch_last_error", None)
        return {
            "base_max": base_max,
            "raw_max": raw_max,
            "stale": raw_max > base_max or fetch_stale,
            "fetch_stale": fetch_stale,
            "fetch_error": fetch_error,
            "forecast_archive": archive_count,
            "last_check": _last_refresh_check,
            "model_release": active_models(MODELS_ROOT).name,
            "archive_revision": FORECAST_ARCHIVE.stat().st_mtime_ns if FORECAST_ARCHIVE.exists() else 0,
            "base_revision": BASE_CSV.stat().st_mtime_ns if BASE_CSV.exists() else 0,
            "last_scheduled_issuance": _last_scheduled_issuance,
        }
    except Exception as e:
        logger.error(f"status error: {e}", exc_info=True)
        return {"error": "failed to load status"}

@app.on_event("startup")
async def preload():
    # Preload inference models in background to avoid first-request latency
    try:
        get_inference()
        # Warm forecast for latest SE to populate cache
        latest = str(df_base["SE"].iloc[-1]) if len(df_base) else None
        if latest:
            # don't block startup on warm, do it after
            import asyncio
            async def warm():
                await asyncio.sleep(0.5)
                try:
                    # trigger cached forecast (live path) once
                    from fastapi.testclient import TestClient
                except Exception:
                    pass
            # actual warm via direct call in thread to avoid blocking
            import threading
            def do_warm():
                try:
                    inf = get_inference()
                    if inf is None:
                        return
                    # build feat cache once (heavy 1.2s) to make first forecast fast
                    _get_feat_full()
                    print("feat cache warm done")
                except Exception as e:
                    print(f"preload warm failed: {e}")
                    import traceback; traceback.print_exc()
            threading.Thread(target=do_warm, daemon=True).start()
    except Exception as e:
        print(f"preload failed: {e}")

# serve static
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
def index():
    return FileResponse(str(static_dir / "municipios.html"))


from config.municipalities import SERVICES


@app.get("/municipal/{geocode}/{endpoint:path}")
def municipal_data(geocode: str, endpoint: str, request: Request):
    allowed = {"health", "api/auth/check", "api/real", "api/weeks", "api/metrics",
               "api/history", "api/forecast", "api/forecast-history", "api/status", "api/overview"}
    if geocode not in SERVICES or endpoint not in allowed:
        raise HTTPException(404)
    import requests
    try:
        result = requests.get(f"http://{SERVICES[geocode]}-dashboard:8000/{endpoint}",
                              params=request.query_params.multi_items(),
                              headers={k: v for k, v in request.headers.items()
                                       if k in {"authorization", "x-api-key"}}, timeout=180)
    except requests.RequestException as exc:
        raise HTTPException(503, "Dashboard municipal indisponível") from exc
    return Response(result.content, status_code=result.status_code, media_type="application/json")

@app.get("/health")
def health():
    _reload_local_data()
    return {"status": "ok", "models": len(list(active_models(MODELS_ROOT).glob("lgbm_h*_seed*_q*.pkl")))}


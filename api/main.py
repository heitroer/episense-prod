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
import threading
from threading import Lock
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    HAS_SCHEDULER = True
except ImportError:
    HAS_SCHEDULER = False

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
_data_cache_lock = Lock()

# Config constants
GEOCODE = config['project']['geocode']
INFODENGUE_URL = config['data_sources']['infodengue']['base_url']
OPENMETEO_URL = config['data_sources']['openmeteo']['base_url']
LAT = config['data_sources']['openmeteo']['latitude']
LON = config['data_sources']['openmeteo']['longitude']
TIMEZONE = config['data_sources']['openmeteo']['timezone']
TARGET_HORIZONS = config['targets']['horizons']
# Single source of truth for nowcast window (also in data/live_utils.py and config.yaml)
from data.live_utils import NOWCAST_WINDOW, apply_nowcast, anchor_guard_filter


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
    """Fetch latest weather data from OpenMeteo API directly (no CSV storage). Com retry 429 e fallback local.
    Anchor guard parity: busca até último sábado completo (não hoje incompleto) e conta clima_dias por SE.
    """
    # Anchor guard: último sábado completo estritamente antes de hoje (paridade com data/process_data.py)
    today = datetime.now().date()
    raw = (today.weekday() - 5) % 7
    days_ago = 7 if raw == 0 else raw
    end_date = today - timedelta(days=days_ago)
    start_date = end_date - timedelta(days=1200)
    logger.info(f"[anchor guard] OpenMeteo end_date ajustado para último sábado completo: {end_date} (hoje {today})")
    
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
    
    # Usa mesma agregação que data/collect_openmeteo.py para paridade treino/live
    from data.collect_openmeteo import aggregate_to_weekly
    # Renomeia antes para aggregate_to_weekly esperar nomes limpos
    rename_pre = {
        'temperature_2m_max': 'temp_max',
        'temperature_2m_min': 'temp_min',
        'temperature_2m_mean': 'temp_mean',
        'precipitation_sum': 'precip',
        'relative_humidity_2m_mean': 'humidity',
        'wind_speed_10m_max': 'wind_max'
    }
    df = df.rename(columns=rename_pre)
    # conta dias por SE antes de agregar (para anchor guard)
    from data.epiweeks import date_to_epiweek as _d2e
    tmp_se = df['date'].apply(lambda d: _d2e(d)).astype(str).str.zfill(6)
    clima_dias = tmp_se.value_counts().to_dict()
    weekly = aggregate_to_weekly(df)
    weekly['SE'] = weekly['SE'].astype(str).str.zfill(6)
    # weekly já tem week_start/week_end para dias calc, mas garante clima_dias
    weekly['clima_dias'] = weekly['SE'].map(clima_dias)
    # filtra semanas parciais
    while len(weekly) > 1:
        last_se = weekly.iloc[-1]['SE']
        days = clima_dias.get(last_se, 0)
        if pd.isna(days) or int(days) < 7:
            motivo = 'clima ausente' if pd.isna(days) else f'clima parcial ({int(days)}/7 dias)'
            logger.warning(f"Anchor guard (OpenMeteo): SE {last_se} removida ({motivo}) - ancora recua")
            weekly = weekly.iloc[:-1]
            clima_dias.pop(last_se, None)
        else:
            break
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
    
    # NOWCAST: casos_est substitui casos onde casos_est > casos apenas nas últimas NOWCAST_WINDOW semanas
    # Treino usa casos puro; live corrige atraso de notificação (InfoDengue nowcast) nas últimas SEs
    merged = apply_nowcast(merged, window=NOWCAST_WINDOW)
    
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
    """Get fully engineered features, cached for 60s. Usa _ensure_fresh_data para rebuild automático."""
    global _data_cache
    try:
        _ensure_fresh_data()
    except Exception:
        pass
    now = time.time()
    with _data_cache_lock:
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
    
    # Cache (thread-safe)
    with _data_cache_lock:
        _data_cache['df'] = engineered
        _data_cache['timestamp'] = now
    
    logger.info(f"Engineered data shape: {engineered.shape}, latest SE: {engineered.iloc[-1].get('SE', 'N/A')}")
    return engineered


ROOT = Path(__file__).parent.parent
_last_refresh_check = 0
_refresh_lock = Lock()
_refresh_cooldown = 60  # seconds between checks
_last_fetch = 0
_fetch_cooldown = 900  # 15 min between external API fetches
_fetch_lock = Lock()
_fetch_last_error = None
_fetch_stale = False

def _get_raw_max_se() -> str:
    """Max SE in raw infodengue full (or latest timestamped)."""
    try:
        raw_full = ROOT / "data/raw/infodengue/infodengue_cg_full.csv"
        if raw_full.exists():
            df = pd.read_csv(raw_full, usecols=["SE"], dtype={"SE": str})
            return str(df["SE"].astype(str).str.zfill(6).max())
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
        base_path = ROOT / "data/processed/episense_base.csv"
        if base_path.exists():
            df = pd.read_csv(base_path, usecols=["SE"], dtype={"SE": str})
            if not df.empty:
                return str(df["SE"].astype(str).str.zfill(6).max())
    except Exception:
        pass
    return "0"

def _fetch_fresh_raw():
    """Fetch latest InfoDengue + OpenMeteo directly from external APIs and update raw CSVs.
    Called from _ensure_fresh_data when fetch cooldown expired. 15min cooldown.
    """
    global _last_fetch, _fetch_last_error, _fetch_stale
    now = time.time()
    if now - _last_fetch < _fetch_cooldown:
        return False
    if not _fetch_lock.acquire(blocking=False):
        return False
    try:
        if time.time() - _last_fetch < _fetch_cooldown:
            return False
        _last_fetch = time.time()
        fetched = False
        infodengue_error = None
        openmeteo_error = None
        try:
            from data.collect_infodengue import collect_infodengue_data, process_infodengue_data, update_full_history, save_infodengue_data
            logger.info("Fetching fresh InfoDengue from external API (on-demand)...")
            df_raw = collect_infodengue_data()
            if df_raw is not None and not df_raw.empty:
                df_proc = process_infodengue_data(df_raw)
                if not df_proc.empty:
                    full_path = ROOT / "data/raw/infodengue/infodengue_cg_full.csv"
                    df_full = update_full_history(df_proc, full_path)
                    df_full.to_csv(full_path, index=False)
                    try:
                        save_infodengue_data(df_proc)
                    except Exception:
                        pass
                    logger.info(f"InfoDengue fresh fetched: {len(df_proc)} rows range {df_proc['SE'].min()}-{df_proc['SE'].max()} full {len(df_full)}")
                    fetched = True
                else:
                    infodengue_error = RuntimeError("InfoDengue processing returned empty")
            else:
                infodengue_error = RuntimeError("InfoDengue fetch empty")
        except Exception as e:
            infodengue_error = e
            _fetch_last_error = str(e)
            logger.error(f"InfoDengue on-demand fetch failed: {e}", exc_info=True)
        try:
            from data.collect_openmeteo import collect_openmeteo_data, aggregate_to_weekly, save_openmeteo_data
            logger.info("Fetching fresh OpenMeteo from external API (on-demand)...")
            df_daily = collect_openmeteo_data()
            if df_daily is not None and not df_daily.empty:
                df_weekly = aggregate_to_weekly(df_daily)
                if not df_weekly.empty:
                    try:
                        from datetime import datetime as _dt
                        end_date = _dt.now().strftime("%Y-%m-%d")
                        save_openmeteo_data(df_daily, df_weekly, "2010-01-01", end_date)
                    except Exception as e:
                        logger.warning(f"save_openmeteo failed: {e}")
                    logger.info(f"OpenMeteo fresh fetched: {len(df_daily)} daily -> {len(df_weekly)} weeks {df_weekly['SE'].min()}-{df_weekly['SE'].max()}")
                    fetched = True
                else:
                    openmeteo_error = RuntimeError("OpenMeteo weekly empty")
            else:
                openmeteo_error = RuntimeError("OpenMeteo daily empty")
        except Exception as e:
            openmeteo_error = e
            _fetch_last_error = str(e)
            logger.error(f"OpenMeteo on-demand fetch failed: {e}", exc_info=True)
        if not fetched:
            _fetch_stale = True
            _fetch_last_error = _fetch_last_error or f"InfoDengue: {infodengue_error}, OpenMeteo: {openmeteo_error}"
            logger.error(f"_fetch_fresh_raw: ambas as fontes falharam (InfoDengue: {infodengue_error}, OpenMeteo: {openmeteo_error}) — dados stale")
        else:
            _fetch_stale = False
            _fetch_last_error = None
        return fetched
    finally:
        try:
            _fetch_lock.release()
        except: pass

def _ensure_fresh_data():
    """On-demand refresh (no cron) — called at start of each request.
    If raw infodengue has newer SE than processed base (raw_max > base_max), rebuild base via processor.
    Debounced 60s, thread-safe, week_start/week_end anchor guard already in _merge_and_prepare.
    """
    global _last_refresh_check
    now = time.time()
    if now - _last_refresh_check < _refresh_cooldown:
        return
    with _refresh_lock:
        if time.time() - _last_refresh_check < _refresh_cooldown:
            return
        _last_refresh_check = time.time()
        try:
            try:
                _fetch_fresh_raw()
            except Exception as fe:
                logger.warning(f"fetch fresh raw failed (continuing with local raw): {fe}")
            raw_max = _get_raw_max_se()
            base_max = _get_base_max_se()
            raw_path = ROOT / "data/raw/infodengue/infodengue_cg_full.csv"
            base_path = ROOT / "data/processed/episense_base.csv"
            raw_mtime = raw_path.stat().st_mtime if raw_path.exists() else 0
            base_mtime = base_path.stat().st_mtime if base_path.exists() else 0
            need_rebuild = False
            reason = ""
            if raw_max > base_max:
                need_rebuild = True
                reason = f"raw {raw_max} > base {base_max}"
            elif raw_mtime > base_mtime + 5:
                need_rebuild = True
                reason = f"raw mtime {raw_mtime} > base {base_mtime} (possible revisao)"
            if not need_rebuild:
                return
            logger.info(f"On-demand refresh triggered: {reason} — rebuilding episense_base.csv")
            from data.process_data import EpisenseDataProcessor
            proc = EpisenseDataProcessor()
            df_new = proc.run_full_pipeline()
            if df_new is None or df_new.empty:
                logger.warning("Processor returned empty, abort refresh")
                return
            # invalidate live cache
            global _data_cache
            _data_cache['df'] = None
            _data_cache['timestamp'] = 0
            logger.info(f"On-demand refresh done: base {base_max} -> {str(df_new['SE'].max()).zfill(6)}")
        except Exception as e:
            logger.warning(f"_ensure_fresh_data failed: {e}", exc_info=True)


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


_scheduler = None

def _scheduled_refresh():
    """Job semanal automático: busca InfoDengue/OpenMeteo frescos e rebuilda base se nova SE."""
    try:
        logger.info("[scheduler] Verificando nova SE InfoDengue (agendado)")
        # força fetch fresco ignorando cooldown para job agendado
        global _last_fetch
        _last_fetch = 0
        try:
            _fetch_fresh_raw()
        except Exception as e:
            logger.warning(f"[scheduler] fetch_fresh_raw falhou: {e}")
        _ensure_fresh_data()
        # invalida cache live para próxima requisição usar dados novos
        with _data_cache_lock:
            _data_cache['df'] = None
            _data_cache['timestamp'] = 0
        logger.info("[scheduler] Refresh concluído")
    except Exception as e:
        logger.error(f"[scheduler] Erro: {e}", exc_info=True)

@app.on_event("startup")
async def startup_event():
    """Initialize inference engine on startup."""
    global inference_engine, _scheduler
    logger.info("Starting Episense API (Production - No CSV)...")
    
    try:
        inference_engine = EpisenseInference("models", config)
        logger.info("Inference engine loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load inference engine: {e}")
        inference_engine = None

    # Agendamento automático semanal (seg 06:05 + diário 06:05 para capturar revisões)
    if HAS_SCHEDULER:
        try:
            _scheduler = BackgroundScheduler(timezone=config['data_sources']['openmeteo'].get('timezone', 'America/Campo_Grande'))
            _scheduler.add_job(_scheduled_refresh, 'cron', day_of_week='mon', hour=6, minute=5, id='weekly_refresh')
            _scheduler.add_job(_scheduled_refresh, 'cron', hour=6, minute=10, id='daily_refresh')
            _scheduler.start()
            logger.info("Scheduler iniciado: weekly seg 06:05 + daily 06:10")
        except Exception as e:
            logger.warning(f"Scheduler não iniciado: {e}")
    else:
        logger.warning("APScheduler não disponível, atualização depende de requisições on-demand")

@app.on_event("shutdown")
async def shutdown_event():
    global _scheduler
    if _scheduler:
        try:
            _scheduler.shutdown(wait=False)
        except Exception:
            pass


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
    
    # Validate geocode (accept int or string, compare zfill 7)
    expected_geocode = str(config['project']['geocode']).zfill(7)
    if str(geocode).zfill(7) != expected_geocode:
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
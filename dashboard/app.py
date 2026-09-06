"""Episense Dashboard - FastAPI"""
from fastapi import FastAPI, Query
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
import logging
from datetime import datetime, timedelta

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

app = FastAPI(title="Episense Dashboard", version="1.0")
logger = logging.getLogger(__name__)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load data - AUTO-RELOAD por mtime (sem cron, sem restart)
BASE_CSV = ROOT / "data/processed/episense_base.csv"
VAL_PRED = ROOT / "models/validation_predictions.json"
VAL_RES = ROOT / "models/validation_results.json"
CONFIG_PATH = ROOT / "config/config.yaml"

# Caches com mtime para reload automático quando nova semana entra (sem cron)
_df_base_mtime = None
_val_pred_mtime = None
_val_res_mtime = None
_config_mtime = None

def _load_df_base():
    global df_base, _df_base_mtime
    try:
        mtime = BASE_CSV.stat().st_mtime
        if df_base is None or _df_base_mtime != mtime:
            df = pd.read_csv(BASE_CSV, usecols=["SE","casos","data_inicio_semana"], dtype={"SE": str})
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
        mtime = VAL_PRED.stat().st_mtime
        if val_pred is None or _val_pred_mtime != mtime:
            with open(VAL_PRED) as f:
                val_pred = json.load(f)
            _val_pred_mtime = mtime
            logger.info(f"val_pred recarregado: {len(val_pred)} horizontes")
    except Exception as e:
        logger.warning(f"falha ao recarregar val_pred: {e}")
    return val_pred

def _load_val_res():
    global val_res, _val_res_mtime
    try:
        mtime = VAL_RES.stat().st_mtime
        if val_res is None or _val_res_mtime != mtime:
            with open(VAL_RES) as f:
                val_res = json.load(f)
            _val_res_mtime = mtime
            logger.info(f"val_res recarregado")
    except Exception as e:
        logger.warning(f"falha ao recarregar val_res: {e}")
    return val_res

# Inicialização (primeiro load)
try:
    df_base = pd.read_csv(BASE_CSV, usecols=["SE","casos","data_inicio_semana"], dtype={"SE": str})
    df_base["SE"] = df_base["SE"].astype(str).str.replace(".0","", regex=False)
    df_base = df_base.sort_values("SE")
    _df_base_mtime = BASE_CSV.stat().st_mtime
except Exception:
    df_base = pd.DataFrame(columns=["SE","casos","data_inicio_semana"])
    _df_base_mtime = None

try:
    with open(VAL_PRED) as f:
        val_pred = json.load(f)
    _val_pred_mtime = VAL_PRED.stat().st_mtime
except Exception:
    val_pred = {}
    _val_pred_mtime = None

try:
    with open(VAL_RES) as f:
        val_res = json.load(f)
    _val_res_mtime = VAL_RES.stat().st_mtime
except Exception:
    val_res = {}
    _val_res_mtime = None

try:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    _config_mtime = CONFIG_PATH.stat().st_mtime
except Exception:
    config = {}
    _config_mtime = None

# Caches for forecast acceleration
_forecast_cache = {}
_forecast_cache_lock = Lock()
_forecast_cache_max = 64
_df_full_cache = None
_df_full_mtime = None
_feat_cache = None
_feat_cache_mtime = None
_feat_cache_lock = Lock()
FORECAST_ARCHIVE = ROOT / "data/processed/forecast_archive.json"
_last_refresh_check = 0
_refresh_lock = Lock()
_refresh_cooldown = 60  # seconds between checks
_last_fetch = 0
_fetch_cooldown = 900  # 15 min between external API fetches (infodengue+openmeteo)
_fetch_lock = Lock()
_df_base_lock = Lock()
# Live in-memory cache (same as API 8001) for dashboard forecast - 60s para garantir atualização a cada abertura do site (não estático)
_live_cache = {'df': None, 'timestamp': 0, 'ttl': 60}
GEOCODE = config['project']['geocode']
INFODENGUE_URL = config['data_sources']['infodengue']['base_url']
OPENMETEO_URL = config['data_sources']['openmeteo']['base_url']
LAT = config['data_sources']['openmeteo']['latitude']
LON = config['data_sources']['openmeteo']['longitude']
TIMEZONE = config['data_sources']['openmeteo']['timezone']
TARGET_HORIZONS = config['targets']['horizons']

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

def _reload_df_base():
    """Reload global df_base from BASE_CSV."""
    global df_base
    try:
        df = pd.read_csv(BASE_CSV, usecols=["SE","casos","data_inicio_semana"], dtype={"SE": str})
        df["SE"] = df["SE"].astype(str).str.replace(".0","", regex=False).str.zfill(6)
        df = df.sort_values("SE")
        with _df_base_lock:
            df_base = df
        logger.info(f"df_base reloaded: {len(df_base)} rows, last {df_base['SE'].iloc[-1] if len(df_base) else 'empty'}")
    except Exception as e:
        logger.error(f"_reload_df_base failed: {e}")

def _archive_forecast(origin_se: str):
    """Persist live forecast for origin_se to forecast_archive.json if not already stored."""
    try:
        origin_se = str(origin_se).zfill(6)
        archive = FORECAST_ARCHIVE
        # load existing
        data = []
        if archive.exists():
            try:
                data = json.loads(archive.read_text(encoding="utf-8"))
                if not isinstance(data, list):
                    data = []
            except Exception:
                data = []
        if any(str(x.get("origin_se")).zfill(6) == origin_se for x in data):
            return  # already archived
        inf = get_inference()
        if inf is None:
            return
        # need feat for origin
        feat_full = _get_feat_full()
        # generate forecast via inference (reuse logic from api_forecast live path)
        # Use _get_df_full sliced to origin
        df_full = _get_df_full()
        df_full = df_full[df_full["SE"].astype(str).str.zfill(6) <= origin_se].copy().sort_values("SE")
        if len(df_full) == 0:
            return
        # prepare features sliced
        if feat_full is not None and "SE" in feat_full.columns:
            mask = feat_full["SE"].astype(str).str.zfill(6) <= origin_se
            df_feat = feat_full[mask].copy()
        else:
            df_feat = inf.prepare_inference_features(df_full)
        preds = inf.predict(df_feat)
        forecast = []
        for hh in range(1,9):
            if hh not in preds:
                continue
            fut_se = add_epiweeks(origin_se, hh)
            q05=q50=q95=None
            for tau, arr in preds[hh].items():
                if len(arr)==0:
                    continue
                val = float(arr[-1])
                casos = int(round(float(np.expm1(val)))) if not np.isnan(val) else None
                if abs(tau-0.05)<0.01: q05=casos
                elif abs(tau-0.50)<0.01: q50=casos
                elif abs(tau-0.95)<0.01: q95=casos
            forecast.append({"h": hh, "target_se": str(fut_se), "q05": q05, "q50": q50, "q95": q95})
        entry = {"origin_se": origin_se, "origin_date": se_to_date(origin_se), "generated_at": datetime.now().isoformat(), "forecast": forecast, "base_max_at_gen": _get_base_max_se()}
        data.append(entry)
        # keep sorted and limit to last 200
        data = sorted(data, key=lambda x: str(x.get("origin_se")))
        if len(data) > 200:
            data = data[-200:]
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Archived forecast for origin {origin_se} -> {forecast[0] if forecast else 'empty'}")
    except Exception as e:
        logger.warning(f"_archive_forecast {origin_se} failed: {e}")
        import traceback; traceback.print_exc()

def _fetch_fresh_raw():
    """Fetch latest InfoDengue + OpenMeteo directly from external APIs and update raw CSVs.
    Called from _ensure_fresh_data when fetch cooldown expired. Never throws.
    """
    global _last_fetch
    now = time.time()
    if now - _last_fetch < _fetch_cooldown:
        return False
    # try double-checked locking
    if not _fetch_lock.acquire(blocking=False):
        return False
    try:
        if time.time() - _last_fetch < _fetch_cooldown:
            return False
        _last_fetch = time.time()
        fetched = False
        # --- InfoDengue ---
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
                    logger.warning("InfoDengue fetch returned empty after processing")
            else:
                logger.warning("InfoDengue fetch empty or failed")
        except Exception as e:
            logger.warning(f"InfoDengue on-demand fetch failed: {e}")
            import traceback; traceback.print_exc()
        # --- OpenMeteo ---
        try:
            from data.collect_openmeteo import collect_openmeteo_data, aggregate_to_weekly, save_openmeteo_data
            logger.info("Fetching fresh OpenMeteo from external API (on-demand)...")
            # collect with auto end_date = yesterday
            df_daily = collect_openmeteo_data()
            if df_daily is not None and not df_daily.empty:
                df_weekly = aggregate_to_weekly(df_daily)
                if not df_weekly.empty:
                    # save with dated filenames
                    try:
                        from datetime import datetime
                        end_date = datetime.now().strftime("%Y-%m-%d")
                        save_openmeteo_data(df_daily, df_weekly, "2010-01-01", end_date)
                    except Exception as e:
                        logger.warning(f"save_openmeteo failed: {e}")
                    logger.info(f"OpenMeteo fresh fetched: {len(df_daily)} daily -> {len(df_weekly)} weeks {df_weekly['SE'].min()}-{df_weekly['SE'].max()}")
                    fetched = True
                else:
                    logger.warning("OpenMeteo weekly empty")
            else:
                logger.warning("OpenMeteo daily empty")
        except Exception as e:
            logger.warning(f"OpenMeteo on-demand fetch failed: {e}")
            import traceback; traceback.print_exc()
        return fetched
    finally:
        try:
            _fetch_lock.release()
        except: pass


def _fetch_infodengue_live() -> pd.DataFrame:
    """Live fetch: InfoDengue /alertcity retorna apenas as últimas ~3 semanas.
    Para histórico completo (S18+ correto), carrega o full history local
    (infodengue_cg_full.csv) e mescla as fresh rows por SE em memória.
    Assim o dashboard sempre mostra dados ATUALIZADOS na abertura do site,
    sem depender de arquivo estático desatualizado, mas com histórico completo.
    """
    import requests
    url = f"{INFODENGUE_URL}?geocode={GEOCODE}&disease=dengue&format=json&ew_format=SE"
    # 1) carrega full history do disco (persistido por _fetch_fresh_raw)
    full_path = ROOT / "data/raw/infodengue/infodengue_cg_full.csv"
    df_full = None
    if full_path.exists():
        try:
            tmp = pd.read_csv(full_path, dtype={"SE": str})
            keep_full = [c for c in ['SE','casos','casos_est'] if c in tmp.columns]
            tmp = tmp[keep_full].copy()
            tmp['SE'] = tmp['SE'].astype(str).str.replace(".0","", regex=False).str.zfill(6)
            if 'casos' in tmp.columns:
                tmp['casos'] = pd.to_numeric(tmp['casos'], errors='coerce').fillna(0).astype(int)
            if 'casos_est' in tmp.columns:
                tmp['casos_est'] = pd.to_numeric(tmp['casos_est'], errors='coerce').fillna(0)
            tmp = tmp.drop_duplicates(subset=['SE']).sort_values('SE').reset_index(drop=True)
            df_full = tmp
            logger.info(f"[dashboard live] full history loaded: {len(df_full)} rows {df_full['SE'].min()}-{df_full['SE'].max()}")
        except Exception as e:
            logger.warning(f"[dashboard live] failed to load full history {full_path}: {e}")
            df_full = None
    # 2) fetch fresh ~3 rows da API (sempre atualizado ao abrir o site)
    try:
        logger.info(f"[dashboard live] Fetching InfoDengue fresh {url}")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        data = r.json()
        df_fresh = pd.DataFrame(data)
        if df_fresh.empty:
            if df_full is not None and not df_full.empty:
                logger.warning("[dashboard live] fresh empty, usando full history")
                df = df_full.copy()
                df['ano'] = df['SE'].str[:4].astype(int)
                df['semana'] = df['SE'].str[4:].astype(int)
                return df.sort_values('SE').reset_index(drop=True)
            return df_fresh
        keep = [c for c in ['SE','casos','casos_est'] if c in df_fresh.columns]
        df_fresh = df_fresh[keep].copy()
        df_fresh['SE'] = df_fresh['SE'].astype(str).str.zfill(6)
        if 'casos' in df_fresh.columns:
            df_fresh['casos'] = pd.to_numeric(df_fresh['casos'], errors='coerce').fillna(0).astype(int)
        if 'casos_est' in df_fresh.columns:
            df_fresh['casos_est'] = pd.to_numeric(df_fresh['casos_est'], errors='coerce').fillna(0)
        # 3) merge: full + fresh (fresh sobrescreve SEs recentes com nowcast atualizado)
        if df_full is not None and not df_full.empty:
            df_full['SE'] = df_full['SE'].astype(str).str.zfill(6)
            df_fresh['SE'] = df_fresh['SE'].astype(str).str.zfill(6)
            df_full_filtered = df_full[~df_full['SE'].isin(df_fresh['SE'])]
            df_merged = pd.concat([df_full_filtered, df_fresh], ignore_index=True)
            df_merged = df_merged.drop_duplicates(subset=['SE']).sort_values('SE').reset_index(drop=True)
            df_merged['ano'] = df_merged['SE'].str[:4].astype(int)
            df_merged['semana'] = df_merged['SE'].str[4:].astype(int)
            logger.info(f"[dashboard live] merged full {len(df_full)} + fresh {len(df_fresh)} -> {len(df_merged)} fresh SEs {sorted(df_fresh['SE'].tolist())}")
            return df_merged
        else:
            df_fresh['ano'] = df_fresh['SE'].str[:4].astype(int)
            df_fresh['semana'] = df_fresh['SE'].str[4:].astype(int)
            return df_fresh.sort_values('SE').reset_index(drop=True)
    except Exception as e:
        logger.warning(f"[dashboard live] fetch fresh failed {e}, fallback full history")
        if df_full is not None and not df_full.empty:
            df = df_full.copy()
            df['ano'] = df['SE'].str[:4].astype(int)
            df['semana'] = df['SE'].str[4:].astype(int)
            return df.sort_values('SE').reset_index(drop=True)
        raise

def _fetch_openmeteo_live() -> pd.DataFrame:
    import requests, time
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
    logger.info(f"[dashboard live] Fetching OpenMeteo {OPENMETEO_URL}")
    last_exc = None
    response = None
    for attempt in range(3):
        try:
            r = requests.get(OPENMETEO_URL, params=params, timeout=60)
            r.raise_for_status()
            response = r
            break
        except requests.exceptions.RequestException as e:
            last_exc = e
            status = getattr(getattr(e, 'response', None), 'status_code', None)
            if status is None and response is not None:
                status = getattr(response, 'status_code', None)
            if status == 429 and attempt < 2:
                wait = 2 ** attempt * 5
                logger.warning(f"[dashboard live] OpenMeteo 429 retry {attempt+1}/3 em {wait}s")
                time.sleep(wait)
                continue
            if attempt == 2:
                break
            raise
    if response is None or (hasattr(response, 'status_code') and response.status_code != 200):
        logger.warning(f"[dashboard live] OpenMeteo falhou após retries ({last_exc}), fallback local")
        try:
            from data.collect_openmeteo import latest_weekly_file
            wf = latest_weekly_file(ROOT / "data/raw/openmeteo")
            if wf and wf.exists():
                logger.info(f"[dashboard live] Fallback local {wf}")
                df_fallback = pd.read_csv(wf, dtype={"SE": str})
                df_fallback["SE"] = df_fallback["SE"].astype(str).str.zfill(6)
                return df_fallback.sort_values("SE").reset_index(drop=True)
        except Exception as fe:
            logger.warning(f"[dashboard live] Fallback falhou: {fe}")
        if last_exc:
            # não propaga 429 como 500, retorna vazio para permitir merge só com dengue (nowcast ainda funciona)
            logger.warning(f"[dashboard live] Retornando vazio para permitir merge dengue-only: {last_exc}")
            return pd.DataFrame()
    data = response.json()
    daily = data.get('daily', {})
    if not daily:
        logger.warning("No daily OpenMeteo")
        return pd.DataFrame()
    df = pd.DataFrame(daily)
    df['date'] = pd.to_datetime(df['time'])
    df = df.drop(columns=['time'])
    from data.epiweeks import date_to_epiweek
    df['SE'] = df['date'].apply(lambda d: date_to_epiweek(d))
    df['SE'] = df['SE'].astype(str).str.zfill(6)
    df['ano'] = df['SE'].str[:4].astype(int)
    df['semana'] = df['SE'].str[4:].astype(int)
    agg = {}
    for col in df.columns:
        if col in ['date','SE','ano','semana']:
            continue
        if 'precip' in col.lower() or 'precipitation' in col.lower():
            agg[col] = 'sum'
        elif 'wind' in col.lower() or 'speed' in col.lower():
            agg[col] = 'max'
        else:
            agg[col] = 'mean'
    weekly = df.groupby(['SE','ano','semana']).agg(agg).reset_index()
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
    logger.info(f"[dashboard live] OpenMeteo weeks {len(weekly)}")
    return weekly

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
    if 'casos' in merged.columns:
        merged['casos']=merged['casos'].fillna(0).astype(int)
    weather_cols=[c for c in merged.columns if c not in ['SE','ano','semana','casos','casos_est']]
    for col in weather_cols:
        if merged[col].dtype in ['float64','int64']:
            merged[col]=merged[col].ffill()
    merged['ano']=merged['SE'].str[:4].astype(int)
    merged['semana']=merged['SE'].str[4:].astype(int)
    if 'casos_est' in merged.columns:
        merged=merged.sort_values('SE').reset_index(drop=True)
        merged['casos']=pd.to_numeric(merged['casos'], errors='coerce').fillna(0).astype(float)
        merged['casos_est']=pd.to_numeric(merged['casos_est'], errors='coerce').fillna(0).astype(float)
        # FIX histórico: casos = max(casos, casos_est) SEMPRE (antes só últimas 12, deixava 202633/202634 com 0)
        mask=(merged['casos_est']>merged['casos'])
        if mask.any():
            merged.loc[mask,'casos']=np.round(merged.loc[mask,'casos_est']).astype(int)
            logger.info(f"[dashboard live] Nowcast {mask.sum()} semanas")
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
    global _live_cache
    now=time.time()
    if _live_cache['df'] is not None and (now - _live_cache['timestamp']) < _live_cache['ttl']:
        logger.info("[dashboard live] using cached engineered")
        return _live_cache['df']
    logger.info("[dashboard live] Fetching fresh data from APIs (same as 8001)...")
    dengue=_fetch_infodengue_live()
    weather=_fetch_openmeteo_live()
    if dengue.empty:
        raise RuntimeError("fresh fetch dengue empty")
    if weather.empty:
        logger.warning(f"[dashboard live] weather empty ({len(weather)}), usando dengue-only com ffill histórico")
        # weather vazio por 429 - merge ainda funciona com left join e ffill
    if len(weather) < 156 and not weather.empty:
        logger.warning(f"weather weeks {len(weather)} <156")
    merged=_merge_and_prepare_live(dengue, weather)
    from data.features import EpisenseFeatureEngineer
    fe=EpisenseFeatureEngineer(config)
    engineered=fe.run_full_feature_engineering(merged, convention='advanced', target_horizons=TARGET_HORIZONS, legacy_mode=False)
    _live_cache['df']=engineered
    _live_cache['timestamp']=now
    logger.info(f"[dashboard live] engineered {engineered.shape} last {engineered.iloc[-1].get('SE')}")
    # invalidate forecast cache when live data refreshes (new week)
    with _forecast_cache_lock:
        _forecast_cache.clear()
    return engineered


def _ensure_fresh_data():
    """On-demand refresh (no cron) — called at start of each API request / dashboard load.
    If raw infodengue has newer SE than processed base, rebuild base via processor.
    Debounced 60s, thread-safe, preserves forecast history.
    Também recarrega df_base/val_pred/val_res por mtime (sem restart) e dispara retreino em background se nova SE além da validação.
    """
    global _last_refresh_check, _df_full_cache, _feat_cache, _forecast_cache, _df_full_mtime, _feat_cache_mtime
    # Auto-reload estático por mtime (sem cron, sem restart) - garante histórico atualizado quando nova semana entra
    try:
        _load_df_base()
        _load_val_pred()
        _load_val_res()
    except Exception:
        pass
    now = time.time()
    if now - _last_refresh_check < _refresh_cooldown:
        return
    with _refresh_lock:
        if time.time() - _last_refresh_check < _refresh_cooldown:
            return
        _last_refresh_check = time.time()
        try:
            # First, try to update raw files from external APIs if cooldown expired
            # This makes infodengue_cg_full.csv / openmeteo_weekly*.csv stay current without cron
            try:
                _fetch_fresh_raw()
            except Exception as fe:
                logger.warning(f"fetch fresh raw failed (continuing with local raw): {fe}")
            raw_max = _get_raw_max_se()
            base_max = _get_base_max_se()
            # also check mtime of raw vs base for revised weeks (same max but newer mtime)
            raw_path = ROOT / "data/raw/infodengue/infodengue_cg_full.csv"
            base_path = ROOT / "data/processed/episense_base.csv"
            raw_mtime = raw_path.stat().st_mtime if raw_path.exists() else 0
            base_mtime = base_path.stat().st_mtime if base_path.exists() else 0
            # Need rebuild if raw has newer SE or raw mtime newer than base mtime (revised weeks)
            need_rebuild = False
            reason = ""
            if raw_max > base_max:
                need_rebuild = True
                reason = f"raw {raw_max} > base {base_max}"
            elif raw_mtime > base_mtime + 5:  # 5s grace
                # Check if content actually changed (hash) to avoid rebuild on touch
                # quick: compare file sizes or SE content; for now treat mtime as signal for revisions
                need_rebuild = True
                reason = f"raw mtime {raw_mtime} > base {base_mtime} (possible revisao)"
            if not need_rebuild:
                # Also check openmeteo freshness: base anchor guard may have dropped last week due to partial weather,
                # waiting for full 7 days. No rebuild needed until weather completes.
                return
            logger.info(f"On-demand refresh triggered: {reason} — rebuilding episense_base.csv")
            from data.process_data import EpisenseDataProcessor
            proc = EpisenseDataProcessor()
            df_new = proc.run_full_pipeline()
            if df_new is None or df_new.empty:
                logger.warning("Processor returned empty, abort refresh")
                return
            new_max = str(df_new["SE"].max()).zfill(6)
            # reload global df_base
            _reload_df_base()
            # invalidate caches
            with _feat_cache_lock:
                _df_full_cache = None
                _feat_cache = None
                _df_full_mtime = None
                _feat_cache_mtime = None
            with _forecast_cache_lock:
                _forecast_cache.clear()
            # Archive forecast for new latest (and also for previous if not yet archived)
            # Archive new latest
            try:
                _archive_forecast(new_max)
                # Also ensure previous base_max archived (if missing)
                if base_max != new_max and int(base_max) > 201400:
                    _archive_forecast(base_max)
            except Exception as e:
                logger.warning(f"archive after rebuild failed: {e}")
            logger.info(f"On-demand refresh done: base {base_max} -> {new_max}, forecast archived")
            # Auto-retreino em background se nova SE além da validação (sem bloquear request, sem cron)
            try:
                _load_val_pred()
                max_val_se = "0"
                for h in val_pred.values():
                    if not isinstance(h, dict):
                        continue
                    for q in h.values():
                        if not isinstance(q, list):
                            continue
                        for fold in q:
                            ses = fold.get("test_se", [])
                            if ses:
                                m = max(str(s).zfill(6) for s in ses)
                                if m > max_val_se:
                                    max_val_se = m
                if new_max > max_val_se:
                    logger.info(f"Auto-retreino disparado: new_max {new_max} > val_max {max_val_se} (sem cron)")
                    def do_retrain():
                        try:
                            from scripts.train import EpisenseTrainer, load_config, load_processed_data
                            cfg = load_config()
                            df = load_processed_data(cfg)
                            if df is None or df.empty:
                                logger.warning("Auto-retreino: df vazio, abort")
                                return
                            trainer = EpisenseTrainer(cfg)
                            trainer.train_all_horizons(df)
                            _load_val_pred()
                            _load_val_res()
                            with _forecast_cache_lock:
                                _forecast_cache.clear()
                            global _inference
                            _inference = None
                            get_inference()
                            logger.info("Auto-retreino concluído e caches invalidados")
                        except Exception as e:
                            logger.warning(f"Auto-retreino falhou: {e}")
                            import traceback; traceback.print_exc()
                    threading.Thread(target=do_retrain, daemon=True).start()
            except Exception as e:
                logger.warning(f"checagem auto-retreino falhou: {e}")
        except Exception as e:
            logger.warning(f"_ensure_fresh_data failed: {e}")
            import traceback; traceback.print_exc()

# lazy inference engine
_inference = None
_inference_mtime = None
def get_inference():
    global _inference, _inference_mtime
    # verifica se modelos mudaram (retreino) - recarrega automaticamente sem restart
    try:
        models_dir = ROOT / "models"
        # pega mtime mais recente dos pkls
        latest = max((p.stat().st_mtime for p in models_dir.glob("lgbm_*.pkl")), default=0)
        if _inference is not None and _inference_mtime is not None and latest <= _inference_mtime:
            return _inference if _inference else None
        if latest > (_inference_mtime or 0):
            logger.info(f"Modelos mudaram (mtime {latest}), recarregando inference")
            _inference = None
    except Exception:
        pass
    if _inference is None:
        try:
            from models.inference import EpisenseInference
            _inference = EpisenseInference(model_dir=str(ROOT/"models"), config=config)
            try:
                _inference_mtime = max((p.stat().st_mtime for p in (ROOT/"models").glob("lgbm_*.pkl")), default=time.time())
            except Exception:
                _inference_mtime = time.time()
        except Exception as e:
            print(f"Inference load failed: {e}")
            _inference = False
    return _inference if _inference else None

def se_to_date(se):
    try:
        row = df_base[df_base["SE"]==str(se)]
        if not row.empty:
            return row.iloc[0]["data_inicio_semana"]
        return se
    except: return se

def add_epiweeks(se: str, n: int) -> str:
    try:
        from data.epiweeks import epiweek_to_date, date_to_epiweek
        import pandas as pd
        se = str(se)
        y = int(se[:4]); w = int(se[4:])
        d = epiweek_to_date(y, w) + pd.Timedelta(weeks=n)
        return date_to_epiweek(d)
    except Exception:
        # fallback
        return str(int(str(se)) + n)

@app.get("/api/real")
def api_real(limit: int = Query(52, ge=10, le=200), end_se: str = None):
    _ensure_fresh_data()
    # usa live fresh para "real" (nowcast nos ultimos 12, igual API 8001) se disponivel
    try:
        eng = _get_live_engineered()
        if eng is not None and "SE" in eng.columns and "casos" in eng.columns:
            eng = eng.copy()
            eng["SE"] = eng["SE"].astype(str).str.zfill(6)
            eng = eng.sort_values("SE")
            # filtra por end_se se pedido
            if end_se:
                eng = eng[eng["SE"] <= str(end_se).zfill(6)]
            df = eng.tail(limit)[["SE","casos","data_inicio_semana"]].copy() if "data_inicio_semana" in eng.columns else eng.tail(limit)[["SE","casos"]].copy()
            # garante data_inicio_semana via df_base map se faltar
            if "data_inicio_semana" not in df.columns:
                m = dict(zip(df_base["SE"].astype(str).str.zfill(6), df_base["data_inicio_semana"]))
                # fallback para SE futuras: usa SE como string
                df["data_inicio_semana"] = df["SE"].map(m).fillna(df["SE"])
            return {"data": df.to_dict(orient="records"), "source": "live"}
    except Exception as e:
        logger.warning(f"api_real live fallback to df_base: {e}")
    if end_se:
        df = df_base[df_base["SE"] <= str(end_se)].tail(limit)
    else:
        df = df_base.tail(limit)
    return {"data": df.to_dict(orient="records"), "source": "csv"}

@app.get("/api/weeks")
def api_weeks(limit: int = 100):
    _ensure_fresh_data()
    # return last N SEs for selector
    df = df_base.tail(limit)
    return {"weeks": df["SE"].tolist(), "dates": df["data_inicio_semana"].tolist()}

@app.get("/api/metrics")
def api_metrics():
    _ensure_fresh_data()
    # Fold periods (epi weeks) derived from validation_predictions (final 4 folds 2022-2025)
    # Keep in sync with dashboard fold labels
    fold_info = {
        "avg": {"label": "Média geral", "period": "2022 a 2025", "se": "202236 a 202632", "note": "4 folds; Fold 2025 parcial 49 sem (até 202632)"},
        "0": {"label": "Fold 2022", "period": "04/09/2022 a 27/08/2023", "se": "202236 a 202335"},
        "1": {"label": "Fold 2023", "period": "03/09/2023 a 25/08/2024", "se": "202336 a 202435"},
        "2": {"label": "Fold 2024", "period": "01/09/2024 a 31/08/2025", "se": "202436 a 202536"},
        "3": {"label": "Fold 2025 (parcial)", "period": "07/09/2025 a 09/08/2026", "se": "202537 a 202632", "note": "49 semanas observadas; 202633+ ainda sem dados"},
    }
    out = {"folds": fold_info}
    for h in [str(i) for i in range(1,9)]:
        if h not in val_res:
            continue
        avg = val_res[h].get("ensemble_avg", {})
        per = val_res[h].get("per_fold_ensemble", [])
        by_fold = {str(r.get("fold")): r for r in per}
        # advanced keys exposed for toggle
        adv_keys = ["mae","wis","coverage_90","coverage_50","maxae","wis_sharpness","mase","wis_over","wis_under","baseline_wis","etp","fbias","fbias_rel","fbias_frac","fbias_surto","fbias_calmaria"]
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
    _ensure_fresh_data()
    """Historical walk-forward predictions for one horizon.

    The dashboard must use the same final evaluation scope as validation_results.json:
    4 epidemiological folds, 2022-2025, gap 8, expanding. validation_predictions.json
    still carries one legacy 2021 fold, so it is filtered here before stitching.
    """
    h = str(horizon).replace("h", "")
    if h not in val_pred:
        return {"history": [], "horizon": h, "source": "validation_predictions"}

    def final_folds(qkey):
        folds = val_pred.get(h, {}).get(qkey, [])
        filtered = []
        for fold in folds:
            ses = fold.get("test_se", [])
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
        for se, pred, true in zip(fold["test_se"], fold["y_pred_casos"], fold["y_test_casos"]):
            se = str(se)
            points.setdefault(se, {})
            points[se].update({"SE": se, "pred": float(pred), "true": float(true)})

    if q_low:
        for fold in final_folds(q_low):
            for se, pred in zip(fold["test_se"], fold["y_pred_casos"]):
                se = str(se)
                if se in points:
                    points[se]["pred_low"] = float(pred)
                    points[se]["q05"] = float(pred)
    if q_low25:
        for fold in final_folds(q_low25):
            for se, pred in zip(fold["test_se"], fold["y_pred_casos"]):
                se = str(se)
                if se in points:
                    points[se]["q25"] = float(pred)
    if q_high75:
        for fold in final_folds(q_high75):
            for se, pred in zip(fold["test_se"], fold["y_pred_casos"]):
                se = str(se)
                if se in points:
                    points[se]["q75"] = float(pred)
    if q_high:
        for fold in final_folds(q_high):
            for se, pred in zip(fold["test_se"], fold["y_pred_casos"]):
                se = str(se)
                if se in points:
                    points[se]["pred_high"] = float(pred)
                    points[se]["q95"] = float(pred)

    # Real deve ser live (mesma lógica API 8001 com nowcast), não CSV defasado
    try:
        eng_live = _get_live_engineered()
        if eng_live is not None and "SE" in eng_live.columns:
            eng_live = eng_live.copy()
            eng_live["SE"] = eng_live["SE"].astype(str).str.zfill(6)
            live_map = dict(zip(eng_live["SE"], eng_live["casos"]))
            live_date = dict(zip(eng_live["SE"], eng_live.get("data_inicio_semana", eng_live["SE"])))
            # usa live como base_window quando live está mais recente que CSV
            live_max = eng_live["SE"].max()
            base_max = str(df_base["SE"].max()).zfill(6)
            if live_max > base_max:
                # live tem semanas novas ainda não em df_base (caso CSV ainda não rebuildou)
                base_window = eng_live.sort_values("SE").tail(window)[["SE","casos","data_inicio_semana"]].copy() if "data_inicio_semana" in eng_live.columns else eng_live.sort_values("SE").tail(window)[["SE","casos"]].copy()
                if "data_inicio_semana" not in base_window.columns:
                    base_window["data_inicio_semana"] = base_window["SE"]
                se_map = dict(zip(base_window["SE"], base_window["data_inicio_semana"]))
                case_map = dict(zip(base_window["SE"], base_window["casos"]))
            else:
                # live cobre mesmas SEs, mas com nowcast (casos corrigidos nos ultimos 12)
                # mescla: usa live para real nas ultimas 12, CSV para historico antigo (igual API)
                se_map = dict(zip(df_base["SE"].astype(str).str.zfill(6), df_base["data_inicio_semana"]))
                # sobrescreve com live onde houver
                for k,v in live_date.items():
                    se_map[k]=v
                case_map = dict(zip(df_base["SE"].astype(str).str.zfill(6), df_base["casos"]))
                for k,v in live_map.items():
                    # só sobrescreve se live tem valor diferente (nowcast) e SE existe no window
                    case_map[k]=v
                base_window = df_base.sort_values("SE").tail(window).copy()
                base_window["SE"] = base_window["SE"].astype(str).str.zfill(6)
                # aplica live casos no base_window para exibição
                base_window["casos"] = base_window["SE"].map(case_map).fillna(base_window["casos"])
                # também atualiza se_map já feito
        else:
            raise RuntimeError("no live")
    except Exception as e:
        logger.info(f"api_history live map fallback to csv: {e}")
        se_map = dict(zip(df_base["SE"].astype(str).str.zfill(6), df_base["data_inicio_semana"]))
        case_map = dict(zip(df_base["SE"].astype(str).str.zfill(6), df_base["casos"]))
        base_window = df_base.sort_values("SE").tail(window)
    # Stitch archive live (todos os quantis) — preenche historico além da validacao walk-forward
    # Para cada SE sem pred (gap honesto pos 202632), usa a previsao live mais recente que mirava aquela SE.
    # Ex: 202633 H1 vem de origin 202632 H1; 202634 H1 vem de origin 202633 H1 (mais recente que 202632 H2).
    archive_map = {}  # target_se -> {pred_low, pred, pred_high, origin_se, h, q05, q50, q95}
    try:
        if FORECAST_ARCHIVE.exists():
            import json as _json
            arch = _json.loads(FORECAST_ARCHIVE.read_text(encoding="utf-8"))
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
                    }
    except Exception as e:
        logger.warning(f"archive stitch failed for h={h}: {e}")

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
        # Se gap honesto (sem validacao), preenche com archive live mais recente para esse horizonte
        if (p_med is None or p_low is None or p_high is None) and se in archive_map:
            am = archive_map[se]
            # só usa se archive tem os 3 quantis completos (q05/q50/q95)
            if am.get("q05") is not None and am.get("q50") is not None and am.get("q95") is not None:
                p_low = am.get("q05", p_low)
                p_q25 = am.get("q25", p_q25)
                p_med = am.get("q50", p_med)
                p_q75 = am.get("q75", p_q75)
                p_high = am.get("q95", p_high)
                stitched = True
                stitch_meta = am
        # Fallback: se ainda gap e archive não tinha essa SE/horizonte, gera live na hora para origin=SE-h e arquiva
        if (p_med is None or p_low is None or p_high is None):
            try:
                # origin que prediz 'se' no horizonte hh
                hh_int = int(h)
                # calcula origin = se - hh semanas (via epiweeks)
                from data.epiweeks import epiweek_to_date, date_to_epiweek
                import pandas as _pd
                se_str = str(se).zfill(6)
                y = int(se_str[:4]); w = int(se_str[4:])
                origin_try = date_to_epiweek(epiweek_to_date(y, w) - _pd.Timedelta(weeks=hh_int))
                origin_try = str(origin_try).zfill(6)
                # só tenta se origin existe no historico (não futuro) e não é além do base/live
                try:
                    # verifica se origin existe em base ou live
                    exists = False
                    if origin_try in df_base["SE"].astype(str).str.zfill(6).values:
                        exists = True
                    else:
                        eng_chk = _live_cache.get('df')
                        if eng_chk is not None and "SE" in eng_chk.columns:
                            if origin_try in eng_chk["SE"].astype(str).str.zfill(6).values:
                                exists = True
                    if exists:
                        # tenta gerar e arquivar (salva todos os quantis q05/q50/q95)
                        _archive_forecast(origin_try)
                        # recarrega archive_map para esta SE
                        if FORECAST_ARCHIVE.exists():
                            import json as _js2
                            arch2 = _js2.loads(FORECAST_ARCHIVE.read_text(encoding="utf-8"))
                            for entry in arch2:
                                if str(entry.get("origin_se")).zfill(6) != origin_try:
                                    continue
                                for fc in entry.get("forecast", []):
                                    if int(fc.get("h",-1)) != hh_int:
                                        continue
                                    if str(fc.get("target_se")).zfill(6) == se:
                                        if fc.get("q05") is not None and fc.get("q50") is not None and fc.get("q95") is not None:
                                            p_low = fc.get("q05", p_low); p_q25 = fc.get("q25", p_q25); p_med = fc.get("q50", p_med); p_q75 = fc.get("q75", p_q75); p_high = fc.get("q95", p_high)
                                            stitched = True
                                            stitch_meta = {"origin_se": origin_try, "h": hh_int, "q05": p_low, "q25": p_q25, "q50": p_med, "q75": p_q75, "q95": p_high, "pred_low": p_low, "pred": p_med, "pred_high": p_high}
                                            # atualiza archive_map para próximas iterações
                                            archive_map[se] = stitch_meta
                                        break
                except Exception as ie:
                    logger.info(f"on-demand stitch fallback skip {se} h={h}: {ie}")
            except Exception as fe:
                logger.info(f"stitch fallback calc failed {se} h={h}: {fe}")
        # Correcao non-crossing: garante Q05 <= Q50 <= Q95, e Q25/Q75 se disponiveis
        if p_low is not None and p_med is not None and p_high is not None:
            try:
                vals_in = [float(p_low), float(p_med), float(p_high)]
                if p_q25 is not None:
                    vals_in.append(float(p_q25))
                if p_q75 is not None:
                    vals_in.append(float(p_q75))
                vals = sorted(vals_in)
                # reatribui ordenado: preserva ordem 90% e 50% quando disponiveis
                if len(vals)==5:
                    p_low, p_q25, p_med, p_q75, p_high = vals[0], vals[1], vals[2], vals[3], vals[4]
                elif len(vals)==4:
                    # missing one of 25/75
                    p_low, p_med, p_high = vals[0], vals[1 if p_q25 is None else 2], vals[-1]
                else:
                    p_low, p_med, p_high = vals[0], vals[1], vals[2]
                if stitched and stitch_meta:
                    # mantém q05/q50/q95 coerentes com ordenação
                    stitch_meta = dict(stitch_meta)
                    if len(vals)==5:
                        stitch_meta["q05"], stitch_meta["q25"], stitch_meta["q50"], stitch_meta["q75"], stitch_meta["q95"] = vals[0], vals[1], vals[2], vals[3], vals[4]
                    else:
                        stitch_meta["q05"], stitch_meta["q50"], stitch_meta["q95"] = vals[0], vals[1], vals[2]
                        if p_q25 is not None:
                            stitch_meta["q25"]=p_q25
                        if p_q75 is not None:
                            stitch_meta["q75"]=p_q75
            except Exception:
                pass
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
            "true": float(pv.get("true", row["casos"])) if "true" in pv else float(row["casos"]),
            "real": float(row["casos"]),
            "date": se_map.get(se, se),
            "stitched": stitched,
            "stitch_origin": stitch_meta.get("origin_se") if stitch_meta else None,
        }
        all_points.append(pt)
    # Gap honesto: ultimas h semanas SEM archive permanecem null (spanGaps false no chart).
    # Com archive, gap é preenchido automaticamente quando nova semana é publicada.
    has_stitched = any(p.get("stitched") for p in all_points)
    return {
        "horizon": h,
        "history": all_points,
        "source": "validation_predictions+archive" if has_stitched else "validation_predictions",
        "scope": "wf_2022_2025_gap8_expanding+live_archive" if has_stitched else "wf_2022_2025_gap8_expanding",
        "quantiles": {"low": q_low or "0.05", "median": q_mid or "0.5", "high": q_high or "0.95"},
        "stitched_count": sum(1 for p in all_points if p.get("stitched")),
        "archive_entries": len(archive_map),
    }

@app.get("/api/forecast")
def api_forecast(origin_se: str, horizon: str = None):
    _ensure_fresh_data()
    """Forecast from origin_se - LIVE like API 8001.

    Quando alguém entra no dashboard a previsão futura é feita na hora via inferência
    com dados frescos da InfoDengue+OpenMeteo (mesma lógica da API 8001), não lida
    de arquivo estático. Origens históricas ainda podem usar validação para auditoria,
    mas a origem mais recente SEMPRE é live.
    """
    origin_se = str(origin_se)
    cache_key = origin_se
    # verifica se origin existe em df_base OU no live fresco (para SE recém-publicada)
    # tenta live primeiro para freshness check
    live_latest = None
    try:
        # peek live cache without forcing fetch if already fresh
        if _live_cache['df'] is not None:
            live_latest = str(_live_cache['df']['SE'].iloc[-1]).zfill(6)
        else:
            # fallback ao df_base
            live_latest = _get_base_max_se()
    except Exception:
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
        result = {
            "origin_se": origin_se,
            "origin_date": se_to_date(origin_se),
            "forecast": [],
            "source": "validation",
            "quantiles": {"low": "0.05", "median": "0.5", "high": "0.95"},
        }
        complete = True
        for hh in range(1, 9):
            hstr = str(hh)
            target_se = add_epiweeks(origin_se, hh)
            qvals = {"q05": None, "q25": None, "q50": None, "q75": None, "q95": None}
            if hstr in val_pred:
                for q, out_key in [("0.05", "q05"), ("0.25", "q25"), ("0.5", "q50"), ("0.75", "q75"), ("0.95", "q95")]:
                    for fold in val_pred[hstr].get(q, []):
                        ses = [str(x) for x in fold.get("test_se", [])]
                        if target_se in ses:
                            idx = ses.index(target_se)
                            qvals[out_key] = int(round(float(fold["y_pred_casos"][idx])))
                            break
            if qvals["q50"] is None:
                complete = False
            result["forecast"].append({
                "h": hh,
                "target_se": target_se,
                "target_date": se_to_date(target_se),
                **qvals,
            })
        return result if complete else None

    # Se a origem é a mais recente, SEMPRE faz live fresco (mesma lógica da API 8001)
    is_latest = (origin_se == live_latest) or (origin_se == _get_base_max_se())
    # também considera live_latest após fetch fresco
    try:
        # força checagem de freshness se for latest (TTL 5min igual API)
        if is_latest:
            # tenta live primeiro - ignora cache de forecast validado
            pass
        else:
            # para origens antigas, mantém cache rápido (validação)
            with _forecast_cache_lock:
                if cache_key in _forecast_cache:
                    # se cache é live para latest, não retorna validado antigo
                    cached = _forecast_cache[cache_key]
                    if is_latest and cached.get("source") == "validation":
                        pass
                    else:
                        return cached
            val_lookup = lookup_validation()
            if val_lookup is not None and not is_latest:
                with _forecast_cache_lock:
                    if cache_key not in _forecast_cache:
                        if len(_forecast_cache) >= _forecast_cache_max:
                            oldest = next(iter(_forecast_cache))
                            del _forecast_cache[oldest]
                        _forecast_cache[cache_key] = val_lookup
                return val_lookup
    except Exception:
        pass
    # LIVE path - igual API 8001: dados frescos in-memory
    

    inf = get_inference()
    if inf:
        try:
            # LIVE: mesma lógica da API 8001 - dados frescos InfoDengue+OpenMeteo in-memory
            # Tenta primeiro o cache live fresco (TTL 300s igual API)
            try:
                engineered = _get_live_engineered()
            except Exception as e:
                logger.warning(f"live engineered failed, fallback to CSV cache: {e}")
                engineered = None
            if engineered is not None and "SE" in engineered.columns:
                # slice engineered até origin_se inclusive
                engineered["SE"] = engineered["SE"].astype(str).str.zfill(6)
                mask = engineered["SE"] <= origin_se
                if mask.sum() == 0:
                    # origin além do live (SE futura), usa tudo
                    df_feat = engineered.copy()
                else:
                    # inf.predict espera full engineered mas usa só última linha;
                    # precisamos dar o slice até origin para prever a partir dali
                    # Truque: passa o slice, o predict vai engenheirar novamente? Não, já está engenheirado
                    # Mas predict() chama prepare_inference_features que detecta já-engenheirado via target_h1
                    # Então podemos passar o slice direto
                    df_feat = engineered[mask].copy()
                preds = inf.predict(df_feat)
            else:
                # fallback CSV
                feat_full = _get_feat_full()
                if feat_full is not None and "SE" in feat_full.columns:
                    mask = feat_full["SE"].astype(str).str.zfill(6) <= origin_se
                    if mask.sum() == 0:
                        feat_full = _get_feat_full()
                        mask = feat_full["SE"].astype(str).str.zfill(6) <= origin_se
                    df_feat = feat_full[mask].copy()
                    preds = inf.predict(df_feat)
                else:
                    df_full = _get_df_full()
                    df_full = df_full[df_full["SE"].astype(str).str.zfill(6) <= origin_se].copy().sort_values("SE")
                    df_feat = inf.prepare_inference_features(df_full)
                    preds = inf.predict(df_feat)
            result = {"origin_se": origin_se, "origin_date": se_to_date(origin_se), "forecast": [], "source": "live"}
            for hh in range(1, 9):
                if hh not in preds:
                    continue
                fut_se = add_epiweeks(origin_se, hh)
                q05 = q25 = q50 = q75 = q95 = None
                for tau, arr in preds[hh].items():
                    if len(arr) == 0:
                        continue
                    val = float(arr[-1])
                    casos = int(round(float(np.expm1(val)))) if not np.isnan(val) else None
                    if abs(tau - 0.05) < 0.01:
                        q05 = casos
                    elif abs(tau - 0.25) < 0.01:
                        q25 = casos
                    elif abs(tau - 0.50) < 0.01:
                        q50 = casos
                    elif abs(tau - 0.75) < 0.01:
                        q75 = casos
                    elif abs(tau - 0.95) < 0.01:
                        q95 = casos
                result["forecast"].append({"h": hh, "target_se": str(fut_se), "target_date": se_to_date(fut_se), "q05": q05, "q25": q25, "q50": q50, "q75": q75, "q95": q95})
            # cache live result
            with _forecast_cache_lock:
                if len(_forecast_cache) >= _forecast_cache_max:
                    # evict oldest
                    oldest = next(iter(_forecast_cache))
                    del _forecast_cache[oldest]
                _forecast_cache[cache_key] = result
            # persist live forecast for origin_se (on-demand archive)
            try:
                _archive_forecast(origin_se)
            except Exception:
                pass
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
    _ensure_fresh_data()
    try:
        if FORECAST_ARCHIVE.exists():
            data = json.loads(FORECAST_ARCHIVE.read_text(encoding="utf-8"))
            return {"history": data, "count": len(data)}
        return {"history": [], "count": 0}
    except Exception as e:
        return {"history": [], "count": 0, "error": str(e)}

@app.get("/api/status")
def api_status():
    _ensure_fresh_data()
    try:
        raw_max = _get_raw_max_se()
        base_max = _get_base_max_se()
        archive_count = 0
        if FORECAST_ARCHIVE.exists():
            try:
                archive_count = len(json.loads(FORECAST_ARCHIVE.read_text(encoding="utf-8")))
            except: pass
        return {
            "base_max": base_max,
            "raw_max": raw_max,
            "stale": raw_max > base_max,
            "forecast_archive": archive_count,
            "last_check": _last_refresh_check,
        }
    except Exception as e:
        return {"error": str(e)}

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
    return FileResponse(str(static_dir / "index.html"))

@app.get("/health")
def health():
    _ensure_fresh_data()
    return {"status": "ok", "models": len(list((ROOT/"models").glob("lgbm_h*_seed*_q*.pkl")))}


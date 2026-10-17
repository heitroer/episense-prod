"""Episense Dashboard - FastAPI"""
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import fcntl
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
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    HAS_SCHEDULER = True
except ImportError:
    HAS_SCHEDULER = False

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

app = FastAPI(title="Episense Dashboard", version="1.0")
logger = logging.getLogger(__name__)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load data - AUTO-RELOAD por mtime (sem cron, sem restart)
BASE_CSV = ROOT / "data/processed/episense_base.csv"
VAL_PRED = ROOT / "models/validation_predictions.json"
VAL_RES = ROOT / "models/validation_results.json"
CONFORMAL_JSON = ROOT / "models/quantile_conformal.json"
CONFIG_PATH = ROOT / "config/config.yaml"

# Caches com mtime para reload automático quando nova semana entra (sem cron)
_df_base_mtime = None
_val_pred_mtime = None
_val_res_mtime = None
_config_mtime = None
_conformal = None
_conformal_mtime = None

TAU_ORDER = ("0.05", "0.25", "0.5", "0.75", "0.95")


def _load_conformal():
    """Artefato de calibração conformal de produção (pool 4 folds)."""
    global _conformal, _conformal_mtime
    try:
        if CONFORMAL_JSON.exists():
            mtime = CONFORMAL_JSON.stat().st_mtime
            if _conformal is None or _conformal_mtime != mtime:
                with open(CONFORMAL_JSON) as f:
                    _conformal = json.load(f)
                _conformal_mtime = mtime
                logger.info(f"conformal recarregado (production h={sorted((_conformal.get('production') or {}).keys())})")
    except Exception as e:
        logger.warning(f"falha ao recarregar conformal: {e}")
    return _conformal


def _qc_cfg():
    """Config quantile_conformal (enabled, apply_horizons, regime_thresholds)."""
    qc = config.get("quantile_conformal") if isinstance(config, dict) else None
    return qc if isinstance(qc, dict) else {}


_regime_hook_cache = {"mtime": None, "hook": None, "casos": None}


def _get_casos_by_se():
    """Dict SE(zfill6)->casos do df_base; rebuild quando o CSV muda."""
    df = df_base
    mtime = _df_base_mtime
    c = _regime_hook_cache
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


def _get_regime_hook():
    """Hook (h, target_se) -> regime-base, como na produção/recompute."""
    c = _regime_hook_cache
    if c["hook"] is not None and c["mtime"] == _df_base_mtime:
        return c["hook"]
    try:
        from models.calibration import make_regime_hook

        hook = make_regime_hook(_get_casos_by_se(), _qc_cfg().get("regime_thresholds") or None)
        c["hook"] = hook
        c["mtime"] = _df_base_mtime
        return hook
    except Exception as e:
        logger.warning(f"regime hook indisponível: {e}")
        return None


def _calibrate_qvals(h_int: int, target_se: str, qvals: dict):
    """Aplica deltas conformal de PRODUÇÃO (pool 4 folds) aos quantis de uma
    semana-alvo (log1p + median_shift + non-crossing).

    qvals: {"0.05": v|None, ..., "0.95": v|None} -> (qvals_out, applied)
    """
    qc = _qc_cfg()
    horizons = set(int(x) for x in (qc.get("apply_horizons") or []) if str(x).strip().lstrip("-").isdigit())
    if not qc.get("enabled") or int(h_int) not in horizons:
        return qvals, False
    conf = _load_conformal()
    prod = (conf or {}).get("production") or {}
    if not prod:
        return qvals, False
    entry = {}
    try:
        from models.calibration import entry_for_key

        hook = _get_regime_hook()
        regime = hook(int(h_int), str(target_se)) if hook else "global"
        entry = entry_for_key(prod, int(h_int), regime)
    except Exception as e:
        logger.info(f"calibração indisponível h={h_int} t={target_se}: {e}")
    if not entry:
        return qvals, False
    shift = float(entry.get("median_shift") or 0.0)
    deltas = entry.get("deltas") or {}
    out = {}
    for tau, v in qvals.items():
        if v is None:
            out[tau] = None
            continue
        lg = np.log1p(max(float(v), 0.0)) + shift
        if abs(float(tau) - 0.5) > 1e-9:
            try:
                d = float(deltas.get(f"{float(tau):.2f}"))
            except Exception:
                d = 0.0
            if np.isfinite(d):
                lg += d
        out[tau] = float(np.clip(np.expm1(lg), 0.0, None))
    # non-crossing por semana entre os taus presentes
    present = [(t, out[t]) for t in TAU_ORDER if out.get(t) is not None]
    if len(present) >= 2:
        ordered = sorted(present, key=lambda x: x[1])
        if any(a[0] != b[0] for a, b in zip(present, ordered)):
            # reatribui os valores ordenados seguindo a ordem dos taus
            for (t_orig, _), (_, val) in zip(present, ordered):
                out[t_orig] = val
    return out, True

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
_fetch_last_error = None
_fetch_stale = False
_df_base_lock = Lock()
# Live in-memory cache (same as API 8001) for dashboard forecast - 60s para garantir atualização a cada abertura do site (não estático)
_live_cache = {'df': None, 'timestamp': 0, 'ttl': 60}
_live_cache_lock = Lock()
GEOCODE = config['project']['geocode']
INFODENGUE_URL = config['data_sources']['infodengue']['base_url']
OPENMETEO_URL = config['data_sources']['openmeteo']['base_url']
LAT = config['data_sources']['openmeteo']['latitude']
LON = config['data_sources']['openmeteo']['longitude']
TIMEZONE = config['data_sources']['openmeteo']['timezone']
TARGET_HORIZONS = config['targets']['horizons']
from data.live_utils import NOWCAST_WINDOW, apply_nowcast, anchor_guard_filter

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
            q05=q25=q50=q75=q95=None
            for tau, arr in preds[hh].items():
                if len(arr)==0:
                    continue
                val = float(arr[-1])
                casos = int(round(float(np.expm1(val)))) if not np.isnan(val) else None
                if abs(tau-0.05)<0.01: q05=casos
                elif abs(tau-0.25)<0.01: q25=casos
                elif abs(tau-0.50)<0.01: q50=casos
                elif abs(tau-0.75)<0.01: q75=casos
                elif abs(tau-0.95)<0.01: q95=casos
            forecast.append({"h": hh, "target_se": str(fut_se), "q05": q05, "q25": q25, "q50": q50, "q75": q75, "q95": q95})
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

def _backfill_archive_quantiles():
    """Completa q25/q75 ausentes nas entradas antigas do forecast_archive.

    Entradas antigas gravavam apenas q05/q50/q95 — o intervalo de 50% ficava
    sem banda para as semanas pós-validação (histórico de previsões arquivadas).
    NÃO altera valores já persistidos: recalcula a previsão para o MESMO
    origin_se com os modelos atuais e preenche somente os quantis faltantes,
    marcando a entrada com quantiles_backfilled_at. Idempotente: roda no
    startup do dashboard e pode ser reexecutado sem efeito quando completo.
    """
    try:
        if not FORECAST_ARCHIVE.exists():
            return
        try:
            data = json.loads(FORECAST_ARCHIVE.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, list) or not data:
            return

        inf = get_inference()
        if inf is None:
            return
        feat_full = _get_feat_full()

        def _needs_backfill(entry):
            fc = entry.get("forecast", [])
            return any((f.get("q25") is None or f.get("q75") is None) for f in fc)

        pending = [str(e.get("origin_se")).zfill(6) for e in data if _needs_backfill(e)]
        if not pending:
            return
        logger.info(f"backfill q25/q75: {len(pending)} entradas incompletas no archive")

        for origin_se in pending:
            try:
                reloaded = json.loads(FORECAST_ARCHIVE.read_text(encoding="utf-8"))
                if not isinstance(reloaded, list):
                    return
                entry = next((e for e in reloaded if str(e.get("origin_se")).zfill(6) == origin_se), None)
                if entry is None or not _needs_backfill(entry):
                    continue
                df_full = _get_df_full()
                df_full = df_full[df_full["SE"].astype(str).str.zfill(6) <= origin_se].copy().sort_values("SE")
                if len(df_full) == 0:
                    continue
                if feat_full is not None and "SE" in feat_full.columns:
                    mask = feat_full["SE"].astype(str).str.zfill(6) <= origin_se
                    df_feat = feat_full[mask].copy()
                else:
                    df_feat = inf.prepare_inference_features(df_full)
                preds = inf.predict(df_feat)
                for f in entry.get("forecast", []):
                    try:
                        hh = int(f.get("h", 0))
                    except (TypeError, ValueError):
                        continue
                    if hh not in preds:
                        continue
                    for tau, arr in preds[hh].items():
                        if len(arr) == 0:
                            continue
                        val = float(arr[-1])
                        casos = int(round(float(np.expm1(val)))) if not np.isnan(val) else None
                        if abs(tau-0.25) < 0.01 and f.get("q25") is None:
                            f["q25"] = casos
                        elif abs(tau-0.75) < 0.01 and f.get("q75") is None:
                            f["q75"] = casos
                entry["quantiles_backfilled_at"] = datetime.now().isoformat()
                archive_tmp = FORECAST_ARCHIVE.with_suffix(".json.tmp")
                archive_tmp.write_text(json.dumps(reloaded, indent=2, ensure_ascii=False), encoding="utf-8")
                archive_tmp.replace(FORECAST_ARCHIVE)
                logger.info(f"backfill q25/q75 concluído para origin {origin_se}")
            except Exception as e:
                logger.warning(f"backfill q25/q75 {origin_se} falhou: {e}")
        logger.info("backfill q25/q75 finalizado")
    except Exception as e:
        logger.warning(f"backfill archive: {e}")

def _fetch_fresh_raw():
    """Fetch latest InfoDengue + OpenMeteo directly from external APIs and update raw CSVs.
    Called from _ensure_fresh_data when fetch cooldown expired. Falhas são explícitas (error log + stale flag).
    """
    global _last_fetch, _fetch_last_error, _fetch_stale
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
        infodengue_error = None
        openmeteo_error = None
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
                    infodengue_error = RuntimeError("InfoDengue processing returned empty")
            else:
                logger.warning("InfoDengue fetch empty or failed")
                infodengue_error = RuntimeError("InfoDengue fetch empty")
        except Exception as e:
            infodengue_error = e
            _fetch_last_error = str(e)
            logger.error(f"InfoDengue on-demand fetch failed: {e}")
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
                    openmeteo_error = RuntimeError("OpenMeteo weekly empty")
            else:
                logger.warning("OpenMeteo daily empty")
                openmeteo_error = RuntimeError("OpenMeteo daily empty")
        except Exception as e:
            openmeteo_error = e
            _fetch_last_error = str(e)
            logger.error(f"OpenMeteo on-demand fetch failed: {e}")
            import traceback; traceback.print_exc()
        if not fetched:
            _fetch_stale = True
            _fetch_last_error = _fetch_last_error or f"InfoDengue: {infodengue_error}, OpenMeteo: {openmeteo_error}"
            logger.error(f"_fetch_fresh_raw: ambas as fontes falharam (InfoDengue: {infodengue_error}, OpenMeteo: {openmeteo_error}) — dados stale, fallback local será usado; se fallback também indisponível caller deve levantar 503")
        else:
            _fetch_stale = False
            _fetch_last_error = None
        return fetched
    finally:
        try:
            _fetch_lock.release()
        except: pass


def _fetch_infodengue_live() -> pd.DataFrame:
    """Live fetch: com range ew/ey a InfoDengue devolve a série consolidada
    revisada (~18 semanas). O full history local (infodengue_cg_full.csv)
    fornece o histórico anterior; as fresh rows sobrescrevem SEs recentes com
    as revisões. Assim o dashboard sempre mostra dados ATUALIZADOS e 100%
    paridade com o site InfoDengue na abertura do site.
    """
    import requests
    from data.collect_infodengue import build_range_url
    # Range ew/ey: série consolidada revisada (~18 semanas), paridade com site
    url = build_range_url(
        f"{INFODENGUE_URL}?geocode={GEOCODE}&disease=dengue&format=json&ew_format=SE"
    )
    # 1) carrega full history do disco (persistido por _fetch_fresh_raw)
    full_path = ROOT / "data/raw/infodengue/infodengue_cg_full.csv"
    df_full = None
    if full_path.exists():
        try:
            tmp = pd.read_csv(full_path, dtype={"SE": str})
            keep_full = [c for c in ['SE','casos','casos_est','Rt','p_rt1'] if c in tmp.columns]
            tmp = tmp[keep_full].copy()
            tmp['SE'] = tmp['SE'].astype(str).str.replace(".0","", regex=False).str.zfill(6)
            if 'casos' in tmp.columns:
                tmp['casos'] = pd.to_numeric(tmp['casos'], errors='coerce').fillna(0).astype(int)
            if 'casos_est' in tmp.columns:
                tmp['casos_est'] = pd.to_numeric(tmp['casos_est'], errors='coerce').fillna(0)
            if 'Rt' in tmp.columns:
                tmp['Rt'] = pd.to_numeric(tmp['Rt'], errors='coerce')
            if 'p_rt1' in tmp.columns:
                tmp['p_rt1'] = pd.to_numeric(tmp['p_rt1'], errors='coerce')
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
        keep = [c for c in ['SE','casos','casos_est','Rt','p_rt1'] if c in df_fresh.columns]
        df_fresh = df_fresh[keep].copy()
        df_fresh['SE'] = df_fresh['SE'].astype(str).str.zfill(6)
        if 'casos' in df_fresh.columns:
            df_fresh['casos'] = pd.to_numeric(df_fresh['casos'], errors='coerce').fillna(0).astype(int)
        if 'casos_est' in df_fresh.columns:
            df_fresh['casos_est'] = pd.to_numeric(df_fresh['casos_est'], errors='coerce').fillna(0)
        if 'Rt' in df_fresh.columns:
            df_fresh['Rt'] = pd.to_numeric(df_fresh['Rt'], errors='coerce')
        if 'p_rt1' in df_fresh.columns:
            df_fresh['p_rt1'] = pd.to_numeric(df_fresh['p_rt1'], errors='coerce')
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
    # Anchor guard parity: último sábado completo estritamente antes de hoje
    today = datetime.now().date()
    raw = (today.weekday() - 5) % 7
    days_ago = 7 if raw == 0 else raw
    end_date = today - timedelta(days=days_ago)
    start_date = end_date - timedelta(days=1200)
    logger.info(f"[anchor guard] OpenMeteo live end_date ajustado para último sábado completo: {end_date} (hoje {today})")
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
        logger.error(f"[dashboard live] OpenMeteo falhou após retries ({last_exc}), tentando fallback local")
        try:
            from data.collect_openmeteo import latest_weekly_file
            wf = latest_weekly_file(ROOT / "data/raw/openmeteo")
            if wf and wf.exists():
                logger.info(f"[dashboard live] Fallback local {wf}")
                df_fallback = pd.read_csv(wf, dtype={"SE": str})
                df_fallback["SE"] = df_fallback["SE"].astype(str).str.zfill(6)
                logger.warning(f"[dashboard live] Usando fallback local stale: {wf} — dados de clima podem estar desatualizados")
                return df_fallback.sort_values("SE").reset_index(drop=True)
        except Exception as fe:
            logger.error(f"[dashboard live] Fallback local também falhou: {fe}")
        if last_exc:
            # Falha explícita: fallback também falhou — retorna vazio mas marca stale para caller levantar 503 ou flag
            logger.error(f"[dashboard live] OpenMeteo fresh e fallback falharam — retornando vazio (stale) para permitir merge dengue-only: {last_exc}")
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
    # Usa mesma agregação que data/collect_openmeteo.py para paridade treino/live
    from data.collect_openmeteo import aggregate_to_weekly
    rename_pre = {
        'temperature_2m_max': 'temp_max',
        'temperature_2m_min': 'temp_min',
        'temperature_2m_mean': 'temp_mean',
        'precipitation_sum': 'precip',
        'relative_humidity_2m_mean': 'humidity',
        'wind_speed_10m_max': 'wind_max'
    }
    df = df.rename(columns=rename_pre)
    from data.epiweeks import date_to_epiweek as _d2e
    tmp_se = df['date'].apply(lambda d: _d2e(d)).astype(str).str.zfill(6)
    clima_dias = tmp_se.value_counts().to_dict()
    weekly = aggregate_to_weekly(df)
    weekly['SE'] = weekly['SE'].astype(str).str.zfill(6)
    weekly['clima_dias'] = weekly['SE'].map(clima_dias)
    while len(weekly) > 1:
        last_se = weekly.iloc[-1]['SE']
        days = clima_dias.get(last_se, 0)
        if pd.isna(days) or int(days) < 7:
            motivo = 'clima ausente' if pd.isna(days) else f'clima parcial ({int(days)}/7 dias)'
            logger.warning(f"Anchor guard (OpenMeteo live): SE {last_se} removida ({motivo}) - ancora recua")
            weekly = weekly.iloc[:-1]
            clima_dias.pop(last_se, None)
        else:
            break
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
    if 'casos' in merged.columns:
        merged['casos']=merged['casos'].fillna(0).astype(int)
    weather_cols=[c for c in merged.columns if c not in ['SE','ano','semana','casos','casos_est']]
    for col in weather_cols:
        if merged[col].dtype in ['float64','int64']:
            merged[col]=merged[col].ffill()
    merged['ano']=merged['SE'].str[:4].astype(int)
    merged['semana']=merged['SE'].str[4:].astype(int)
    merged = apply_nowcast(merged, window=NOWCAST_WINDOW)
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
    global _live_cache, _fetch_stale
    now=time.time()
    with _live_cache_lock:
        if _live_cache['df'] is not None and (now - _live_cache['timestamp']) < _live_cache['ttl']:
            logger.info("[dashboard live] using cached engineered")
            return _live_cache['df']
    logger.info("[dashboard live] Fetching fresh data from APIs (same as 8001)...")
    dengue=_fetch_infodengue_live()
    weather=_fetch_openmeteo_live()
    if dengue.empty:
        logger.error("[dashboard live] dengue empty e fallback full também falhou — sem dados para predição (stale)")
        raise HTTPException(status_code=503, detail="Falha ao buscar dados InfoDengue (fresh e fallback falharam) — dados stale")
    if weather.empty:
        logger.error(f"[dashboard live] weather empty ({len(weather)}), fallback local também falhou — usando dengue-only com ffill histórico (stale)")
        _fetch_stale = True
        # merge ainda funciona com left join e ffill, mas marca stale para /api/status e callers
    if len(weather) < 156 and not weather.empty:
        logger.warning(f"weather weeks {len(weather)} <156")
    merged=_merge_and_prepare_live(dengue, weather)
    if merged is None or merged.empty:
        logger.error("[dashboard live] merged vazio após _merge_and_prepare_live — falha explícita")
        raise HTTPException(status_code=503, detail="Falha ao preparar dados live (merged vazio) — dados stale")
    from data.features import EpisenseFeatureEngineer
    fe=EpisenseFeatureEngineer(config)
    engineered=fe.run_full_feature_engineering(merged, convention='advanced', target_horizons=TARGET_HORIZONS, legacy_mode=False)
    if engineered is None or engineered.empty:
        logger.error("[dashboard live] engineered vazio após feature engineering — falha explícita")
        raise HTTPException(status_code=503, detail="Falha ao gerar features live (engineered vazio) — dados stale")
    with _live_cache_lock:
        _live_cache['df']=engineered
        _live_cache['timestamp']=now
    logger.info(f"[dashboard live] engineered {engineered.shape} last {engineered.iloc[-1].get('SE')}")
    # invalidate forecast cache when live data refreshes (new week)
    with _forecast_cache_lock:
        _forecast_cache.clear()
    return engineered


def _try_retrain_lock():
    """Lock cross-processo/thread para auto-retreino (flock em models/.retrain_lock).

    Impede dois retreinos simultâneos sobrescrevendo artefatos (ex.: thread do
    uvicorn disparada por _ensure_fresh_data + exec manual no container).
    Retorna o file handle que MANTÉM o lock (feche para liberar) ou None.
    """
    try:
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        fp = open(models_dir / ".retrain_lock", "w")
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fp
    except OSError:
        return None

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
                    # V6b: marcador persistido de última SE treinada — sem ele o
                    # gatilho re-treina em todo refresh (new_max 202640 > val_max
                    # 202635 permanece TRUE, pois a validação nunca alcança a SE
                    # mais recente), virando loop de retreino a cada revisão.
                    _last_trained = "0"
                    try:
                        _last_trained = Path("models/.retrain_last_se").read_text().strip() or "0"
                    except OSError:
                        pass
                    if new_max <= _last_trained:
                        logger.info(f"Auto-retreino ignorado: new_max {new_max} já treinado (marker {_last_trained})")
                    else:
                        _rl = _try_retrain_lock()
                        if _rl is None:
                            logger.info("Auto-retreino ignorado: outro retreino já em curso (lock models/.retrain_lock)")
                        else:
                            logger.info(f"Auto-retreino disparado: new_max {new_max} > val_max {max_val_se} e > marker {_last_trained} (sem cron)")
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
                                    try:
                                        Path("models/.retrain_last_se").write_text(str(new_max))
                                    except Exception as e2:
                                        logger.warning(f"marker .retrain_last_se não gravado: {e2}")
                                    logger.info("Auto-retreino concluído e caches invalidados")
                                except Exception as e:
                                    logger.warning(f"Auto-retreino falhou: {e}")
                                    import traceback; traceback.print_exc()
                                finally:
                                    try:
                                        _rl.close()
                                    except Exception:
                                        pass
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
    """Soma n semanas epidemiológicas a SE (YYYYWW) usando calendário brasileiro (domingo).
    Falha explicitamente se SE inválida — nunca usa fallback int(SE)+n que ignora virada de ano e semana 53.
    """
    from data.epiweeks import epiweek_to_date, date_to_epiweek
    import pandas as pd
    se_str = str(se).zfill(6)
    try:
        y = int(se_str[:4]); w = int(se_str[4:])
        d = epiweek_to_date(y, w) + pd.Timedelta(weeks=n)
        result = date_to_epiweek(d)
        if not isinstance(result, str) or len(result) != 6 or not result.isdigit():
            raise ValueError(f"SE inválida gerada: {result} de {se}+{n}")
        return result
    except Exception as e:
        logger.error(f"add_epiweeks falhou para se={se} n={n}: {e}")
        raise ValueError(f"Falha ao calcular SE {se}+{n} semanas: {e}") from e

_dashboard_scheduler = None

def _dashboard_scheduled_refresh():
    try:
        logger.info("[dashboard scheduler] Verificando nova SE")
        global _last_fetch
        _last_fetch = 0
        try:
            _fetch_fresh_raw()
        except Exception as e:
            logger.warning(f"[dashboard scheduler] fetch falhou: {e}")
        _ensure_fresh_data()
        with _live_cache_lock:
            _live_cache['df'] = None
            _live_cache['timestamp'] = 0
        logger.info("[dashboard scheduler] Refresh concluído")
    except Exception as e:
        logger.error(f"[dashboard scheduler] Erro: {e}", exc_info=True)

@app.on_event("startup")
async def dashboard_startup():
    global _dashboard_scheduler
    if HAS_SCHEDULER:
        try:
            _dashboard_scheduler = BackgroundScheduler(timezone=config['data_sources']['openmeteo'].get('timezone', 'America/Campo_Grande'))
            _dashboard_scheduler.add_job(_dashboard_scheduled_refresh, 'cron', day_of_week='mon', hour=6, minute=7, id='dash_weekly')
            _dashboard_scheduler.add_job(_dashboard_scheduled_refresh, 'cron', hour=6, minute=12, id='dash_daily')
            _dashboard_scheduler.start()
            logger.info("Dashboard scheduler iniciado: seg 06:07 + daily 06:12")
        except Exception as e:
            logger.warning(f"Dashboard scheduler não iniciado: {e}")
    # Backfill q25/q75 nas entradas antigas do forecast_archive (so gravavam
    # q05/q50/q95 — intervalo de 50% sem banda no histórico de previsões).
    try:
        threading.Thread(target=_backfill_archive_quantiles, daemon=True).start()
    except Exception as e:
        logger.warning(f"backfill q25/q75 não agendado: {e}")

@app.on_event("shutdown")
async def dashboard_shutdown():
    global _dashboard_scheduler
    if _dashboard_scheduler:
        try:
            _dashboard_scheduler.shutdown(wait=False)
        except Exception:
            pass

@app.get("/api/real")
def api_real(limit: int = Query(52, ge=10, le=200), end_se: str = None):
    _ensure_fresh_data()
    # usa live fresh para "real" (nowcast nas ultimas 16 semanas, igual API 8001) se disponivel
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
            # flag stale: se _fetch_stale True, indica clima/dengue fallback stale
            stale = globals().get("_fetch_stale", False)
            return {"data": df.to_dict(orient="records"), "source": "live", "stale": stale}
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"api_real live fallback to df_base (stale): {e}")
    if end_se:
        df = df_base[df_base["SE"] <= str(end_se)].tail(limit)
    else:
        df = df_base.tail(limit)
    # fallback csv é stale por definição quando live falhou
    return {"data": df.to_dict(orient="records"), "source": "csv", "stale": True, "warning": "dados live indisponíveis, usando CSV stale"}

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
        # conjunto canônico exposto (spec 2026-10, sem MAPE/CRPS) + referência do baseline
        # (MaxAE/dados legados removidos)
        adv_keys = ["mae", "rmse", "bias", "wis_total",
                    "wis_50", "wis_90", "width_50", "width_90",
                    "coverage_50", "coverage_90",
                    "baseline_wis", "baseline_mae", "baseline_coverage_90"]
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

    # Calibração conformal DE PRODUÇÃO (pool 4 folds, regimes pela semana-base)
    # sobre os pontos de validação: mesmas bandas que o modelo implantado teria
    # publicado — coerente com o fan do /api/forecast e com as linhas live do
    # archive (que já saem calibradas da inferência). h fora de apply_horizons
    # permanecem raw.
    try:
        for se, pv in points.items():
            taus = {"0.05": pv.get("q05"), "0.25": pv.get("q25"), "0.5": pv.get("pred"),
                    "0.75": pv.get("q75"), "0.95": pv.get("q95")}
            cal, applied = _calibrate_qvals(int(h), str(se).zfill(6), taus)
            if not applied:
                continue
            pv["pred"] = cal.get("0.5")
            pv["pred_low"] = cal.get("0.05")
            pv["pred_high"] = cal.get("0.95")
            pv["q05"] = cal.get("0.05")
            pv["q25"] = cal.get("0.25")
            pv["q75"] = cal.get("0.75")
            pv["q95"] = cal.get("0.95")
    except Exception as e:
        logger.warning(f"calibração history falhou (mantendo raw): {e}")

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
        """Fan histórico walk-forward por ORIGEM.

        Para cada h∈1..8 a previsão vinda da semana-origem é a predição de
        validação do alvo origin+h (fold que cobre o alvo; no cruzamento de
        folds vale 'último vence', mesma regra do /api/history). Os quantis
        recebem a calibração conformal DE PRODUÇÃO (pool dos 4 folds — o que
        o modelo implantado de fato produziria), com chips de fold e flag
        `calibrated`. Linhas sem predição validada ficam `disponivel=false`
        (fan parcial perto do fim do histórico / 2026).
        """
        result = {
            "origin_se": origin_se,
            "origin_date": se_to_date(origin_se),
            "forecast": [],
            "source": "validation",
            "calibration_scheme": "production_pool_cqr_delta_only",
            "quantiles": {"low": "0.05", "median": "0.5", "high": "0.95"},
        }
        casos_by_se = _get_casos_by_se()
        any_available = False
        any_calibrated = False
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
                             if not (f.get("test_se") and str(f["test_se"][0]) < "202236")] or folds
                    for fold in final:
                        ses = [str(x).zfill(6) for x in fold.get("test_se", [])]
                        if target_se in ses:
                            idx = ses.index(target_se)
                            try:
                                v = float(fold["y_pred_casos"][idx])
                            except Exception:
                                continue
                            if np.isfinite(v):
                                qvals[q] = v
                                fold_by_tau[q] = fold.get("fold")
                    # last-win implícito: itera na ordem salva (fold0..3)
                    # e o último fold que cobre o alvo sobrescreve
            cal, applied = _calibrate_qvals(hh, target_se, qvals)
            any_calibrated = any_calibrated or applied
            real = casos_by_se.get(target_se)
            if target_se > last_data_se:
                real = None  # semana sem observação publicada
            disponivel = cal.get("0.5") is not None
            any_available = any_available or disponivel
            row = {
                "h": hh,
                "target_se": target_se,
                "target_date": se_to_date(target_se),
                "q05": cal.get("0.05"), "q25": cal.get("0.25"),
                "q50": cal.get("0.5"), "q75": cal.get("0.75"), "q95": cal.get("0.95"),
                "real": real,
                "fold": fold_by_tau.get("0.5"),
                "calibrated": bool(applied),
                "disponivel": bool(disponivel),
            }
            result["forecast"].append(row)
        result["any_calibrated"] = any_calibrated
        return result if any_available else None

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
            # stamp: invalida quando val_pred/df_base/conformal mudam (mtime)
            stamp = (_val_pred_mtime, _df_base_mtime, _conformal_mtime)
            with _forecast_cache_lock:
                if cache_key in _forecast_cache:
                    # se cache é live para latest, não retorna validado antigo
                    cached = _forecast_cache[cache_key]
                    if is_latest and cached.get("source") == "validation":
                        pass
                    elif cached.get("_stamp") == stamp:
                        return cached
            val_lookup = lookup_validation()
            if val_lookup is not None and not is_latest:
                with _forecast_cache_lock:
                    val_lookup["_stamp"] = stamp
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
        logger.error(f"forecast-history error: {e}", exc_info=True)
        return {"history": [], "count": 0, "error": "failed to load forecast history"}

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
    return FileResponse(str(static_dir / "index.html"))

@app.get("/health")
def health():
    _ensure_fresh_data()
    return {"status": "ok", "models": len(list((ROOT/"models").glob("lgbm_h*_seed*_q*.pkl")))}


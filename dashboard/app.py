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
from threading import Lock

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

app = FastAPI(title="Episense Dashboard", version="1.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load data
BASE_CSV = ROOT / "data/processed/episense_base.csv"
VAL_PRED = ROOT / "models/validation_predictions.json"
VAL_RES = ROOT / "models/validation_results.json"
CONFIG_PATH = ROOT / "config/config.yaml"

df_base = pd.read_csv(BASE_CSV, usecols=["SE","casos","data_inicio_semana"], dtype={"SE": str})
# ensure SE is string YYYYWW
df_base["SE"] = df_base["SE"].astype(str).str.replace(".0","", regex=False)
df_base = df_base.sort_values("SE")

with open(VAL_PRED) as f:
    val_pred = json.load(f)

with open(VAL_RES) as f:
    val_res = json.load(f)

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

# Caches for forecast acceleration
_forecast_cache = {}
_forecast_cache_lock = Lock()
_forecast_cache_max = 64
_df_full_cache = None
_df_full_mtime = None
_feat_cache = None
_feat_cache_mtime = None
_feat_cache_lock = Lock()

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

# lazy inference engine
_inference = None
def get_inference():
    global _inference
    if _inference is None:
        try:
            from models.inference import EpisenseInference
            _inference = EpisenseInference(model_dir=str(ROOT/"models"), config=config)
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
    if end_se:
        # filter up to end_se inclusive
        df = df_base[df_base["SE"] <= str(end_se)].tail(limit)
    else:
        df = df_base.tail(limit)
    return {"data": df.to_dict(orient="records")}

@app.get("/api/weeks")
def api_weeks(limit: int = 100):
    # return last N SEs for selector
    df = df_base.tail(limit)
    return {"weeks": df["SE"].tolist(), "dates": df["data_inicio_semana"].tolist()}

@app.get("/api/metrics")
def api_metrics():
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
        out[h] = {
            "avg": {
                "wmape": avg.get("wmape", avg.get("wmape_weighted")),
                "wis": avg.get("wis"),
                "rWIS": avg.get("rWIS"),
                "coverage_90": avg.get("coverage_90"),
                "maxae": avg.get("maxae"),
                "etp": avg.get("etp"),
            },
            "0": {k: by_fold.get("0", {}).get(k) for k in ["wmape","wis","rWIS","coverage_90","maxae","etp"]} if "0" in by_fold else None,
            "1": {k: by_fold.get("1", {}).get(k) for k in ["wmape","wis","rWIS","coverage_90","maxae","etp"]} if "1" in by_fold else None,
            "2": {k: by_fold.get("2", {}).get(k) for k in ["wmape","wis","rWIS","coverage_90","maxae","etp"]} if "2" in by_fold else None,
            "3": {k: by_fold.get("3", {}).get(k) for k in ["wmape","wis","rWIS","coverage_90","maxae","etp"]} if "3" in by_fold else None,
        }
    return out

@app.get("/api/history")
def api_history(horizon: str = "1", window: int = 52):
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
    if q_high:
        for fold in final_folds(q_high):
            for se, pred in zip(fold["test_se"], fold["y_pred_casos"]):
                se = str(se)
                if se in points:
                    points[se]["pred_high"] = float(pred)

    # Real must be horizon-independent: use df_base window for stable rightmost week and Hoje marker
    se_map = dict(zip(df_base["SE"], df_base["data_inicio_semana"]))
    case_map = dict(zip(df_base["SE"], df_base["casos"]))
    base_window = df_base.sort_values("SE").tail(window)
    # Build history from base window so last week is identical for every horizon (fix shrinking chart / missing Hoje)
    all_points = []
    for _, row in base_window.iterrows():
        se = str(row["SE"])
        # validation pred may be missing for long horizons at the very end (expected gap)
        pv = points.get(se, {})
        # Correcao non-crossing: validation_predictions.json foi salvo sem sort (bug train.py 781-788), mediana ficava > q95 em 8 semanas do pico 2022-2023.
        # Garante Q05 <= Q50 <= Q95 aqui sem leakage (apenas ordena os 3 valores ja previstos para a mesma SE).
        p_low = pv.get("pred_low")
        p_med = pv.get("pred")
        p_high = pv.get("pred_high")
        if p_low is not None and p_med is not None and p_high is not None:
            try:
                vals = sorted([float(p_low), float(p_med), float(p_high)])
                p_low, p_med, p_high = vals[0], vals[1], vals[2]
            except Exception:
                pass
        pt = {
            "SE": se,
            "pred": p_med,
            "pred_low": p_low,
            "pred_high": p_high,
            "true": float(pv.get("true", row["casos"])) if "true" in pv else float(row["casos"]),
            "real": float(row["casos"]),
            "date": se_map.get(se, se),
        }
        all_points.append(pt)
    # Gap honesto: ultimas h semanas sem validacao permanecem null (spanGaps false no chart).
    # Nao faz forward-fill: evita linha falsa e preserva erro real de cobertura para horizontes longos.
    return {
        "horizon": h,
        "history": all_points,
        "source": "validation_predictions",
        "scope": "wf_2022_2025_gap8_expanding",
        "quantiles": {"low": q_low, "median": q_mid, "high": q_high},
    }

@app.get("/api/forecast")
def api_forecast(origin_se: str, horizon: str = None):
    """Forecast from origin_se.

    Historical origins are served from validation_predictions.json first. That keeps chart
    interactions fast and uses the same walk-forward predictions displayed in /api/history.
    If the selected origin needs targets beyond the validation file, fall back to live inference.
    """
    origin_se = str(origin_se)
    cache_key = origin_se
    if origin_se not in df_base["SE"].values:
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
            qvals = {"q05": None, "q50": None, "q95": None}
            if hstr in val_pred:
                for q, out_key in [("0.05", "q05"), ("0.5", "q50"), ("0.95", "q95")]:
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

    # fast cache check
    with _forecast_cache_lock:
        if cache_key in _forecast_cache:
            return _forecast_cache[cache_key]
    val_lookup = lookup_validation()
    if val_lookup is not None:
        with _forecast_cache_lock:
            if cache_key not in _forecast_cache:
                if len(_forecast_cache) >= _forecast_cache_max:
                    oldest = next(iter(_forecast_cache))
                    del _forecast_cache[oldest]
                _forecast_cache[cache_key] = val_lookup
        return val_lookup

    inf = get_inference()
    if inf:
        try:
            # Fast path: use cached engineered full and slice to origin
            feat_full = _get_feat_full()
            if feat_full is not None:
                # slice engineered frame up to origin_se inclusive
                # feat_full already sorted by SE
                if "SE" in feat_full.columns:
                    mask = feat_full["SE"].astype(str) <= origin_se
                    # ensure at least one row
                    if mask.sum() == 0:
                        feat_full = _get_feat_full()
                        mask = feat_full["SE"].astype(str) <= origin_se
                    df_feat = feat_full[mask].copy()
                else:
                    df_feat = feat_full.copy()
                # predict directly on sliced engineered frame without re-engineering
                preds = inf.predict(df_feat)
            else:
                df_full = _get_df_full()
                df_full = df_full[df_full["SE"] <= origin_se].copy().sort_values("SE")
                df_feat = inf.prepare_inference_features(df_full)
                preds = inf.predict(df_feat)
            result = {"origin_se": origin_se, "origin_date": se_to_date(origin_se), "forecast": [], "source": "live"}
            for hh in range(1, 9):
                if hh not in preds:
                    continue
                fut_se = add_epiweeks(origin_se, hh)
                q05 = q50 = q95 = None
                for tau, arr in preds[hh].items():
                    if len(arr) == 0:
                        continue
                    val = float(arr[-1])
                    casos = int(round(float(np.expm1(val)))) if not np.isnan(val) else None
                    if abs(tau - 0.05) < 0.01:
                        q05 = casos
                    elif abs(tau - 0.50) < 0.01:
                        q50 = casos
                    elif abs(tau - 0.95) < 0.01:
                        q95 = casos
                result["forecast"].append({"h": hh, "target_se": str(fut_se), "target_date": se_to_date(fut_se), "q05": q05, "q50": q50, "q95": q95})
            # cache live result
            with _forecast_cache_lock:
                if len(_forecast_cache) >= _forecast_cache_max:
                    # evict oldest
                    oldest = next(iter(_forecast_cache))
                    del _forecast_cache[oldest]
                _forecast_cache[cache_key] = result
            return result
        except Exception as e:
            print(f"live inference failed {e}, fallback to partial validation")
            import traceback; traceback.print_exc()

    partial = lookup_validation()
    if partial is not None:
        return partial
    return {"origin_se": origin_se, "origin_date": se_to_date(origin_se), "forecast": [], "source": "none"}

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
    return {"status": "ok", "models": len(list((ROOT/"models").glob("lgbm_h*_seed*_q*.pkl")))}


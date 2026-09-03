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
    out = {}
    for h in ["1","4","8"]:
        if h in val_res:
            avg = val_res[h].get("ensemble_avg", {})
            out[h] = {
                "wmape": avg.get("wmape", avg.get("wmape_weighted")),
                "wis": avg.get("wis"),
                "rWIS": avg.get("rWIS"),
                "coverage_90": avg.get("coverage_90"),
                "maxae": avg.get("maxae"),
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

    all_points = sorted(points.values(), key=lambda x: x["SE"])
    all_points = all_points[-window:]
    se_map = dict(zip(df_base["SE"], df_base["data_inicio_semana"]))
    case_map = dict(zip(df_base["SE"], df_base["casos"]))
    for point in all_points:
        se = point["SE"]
        point["date"] = se_map.get(se, se)
        point["real"] = float(case_map[se]) if se in case_map else point["true"]
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

    val_lookup = lookup_validation()
    if val_lookup is not None:
        return val_lookup

    inf = get_inference()
    if inf:
        try:
            df_full = pd.read_csv(ROOT/"data/processed/episense_base.csv", dtype={"SE": str})
            df_full["SE"] = df_full["SE"].astype(str).str.replace(".0", "", regex=False)
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
            return result
        except Exception as e:
            print(f"live inference failed {e}, fallback to partial validation")
            import traceback; traceback.print_exc()

    partial = lookup_validation()
    if partial is not None:
        return partial
    return {"origin_se": origin_se, "origin_date": se_to_date(origin_se), "forecast": [], "source": "none"}

# serve static
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
def index():
    return FileResponse(str(static_dir / "index.html"))

@app.get("/health")
def health():
    return {"status": "ok", "models": len(list((ROOT/"models").glob("lgbm_h*_q0500.pkl")))}


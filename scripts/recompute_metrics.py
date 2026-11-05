"""Regenera models/validation_results.json (+ models/metrics_dataframe.csv)
com o conjunto canônico de métricas (models/metrics.py) a partir das previsões
walk-forward JÁ SALVAS em models/validation_predictions.json — sem retreino.

Escopo idêntico ao validação final: folds 2022-2025 (SE >= 202236; fold legado
2021 descartado), ensemble = média das seeds (já persistida nas previsões).
ensemble_avg primário = média dos folds completos (52 semanas, 2022-2024);
full_4folds_avg = referência com os 4 folds.

Removidos na spec 2026-10: MASE (e agregados), ETP, SPL, MAPE, CRPS,
quantile_crossed, fbias.
Métricas: mae, rmse, bias, wape, wis_total, wis_50, wis_90,
width_50, width_90, coverage_50, coverage_90 (+ baseline_wis/baseline_mae/
baseline_coverage_90 como referência sazonal e skill_mae/skill_wis = ganho
vs baseline, razão das médias na agregação).

Uso:  ./.venv/bin/python3 scripts/recompute_metrics.py
"""

import json
import math
import os
import sys
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.metrics import compute_all_metrics, QUANTILES, PAIR_50, PAIR_90, pinball_mean, skill_score  # noqa: E402
from models.calibration import compute_conformal  # noqa: E402
from data.epiweeks import epiweek_to_date, date_to_epiweek  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# EPISENSE_VP_DIR: override do diretório de predições/results (experimentos sem
# tocar models/ de produção; artefatos de saída seguem o mesmo diretório).
_VP_DIR = Path(os.environ.get('EPISENSE_VP_DIR', "models"))
if not _VP_DIR.is_absolute():
    _VP_DIR = ROOT / _VP_DIR
VAL_PRED = _VP_DIR / "validation_predictions.json"
VAL_RES = _VP_DIR / "validation_results.json"
BASE = ROOT / "data/processed/episense_base.csv"
FIRST_TEST_SE = "202236"          # escopo final de avaliação
FOLD_GAP_WEEKS = 8                # gap do walk-forward (config training.walk_forward)
OUTBREAK_MONTHS = {10, 11, 12, 1, 2, 3, 4, 5}  # spec epidemiológica Out-Mai


def se_shift(se: str, weeks: int) -> str:
    """SE + weeks (negativo permitido) via data do calendário epidemiológico."""
    se = str(se).zfill(6)
    d = epiweek_to_date(int(se[:4]), int(se[4:]))
    return str(date_to_epiweek(d + timedelta(days=weeks * 7))).zfill(6)


def se_to_month(se: str) -> int:
    se = str(se).zfill(6)
    return int(epiweek_to_date(int(se[:4]), int(se[4:])).month)


def build_baseline_preds(pool: pd.DataFrame, test_se: list, min_se: str = "000000") -> dict:
    """Baseline sazonal ingênuo (persistência Ano-1): quantis empíricos por
    semana-do-ano do pool de treino; fallback ±k semanas via datas e depois
    pool global — mesma filosofia do baseline do trainer (sem vazamento)."""
    pool = pool.copy()
    pool["SE"] = pool["SE"].astype(str).str.zfill(6).str.replace(".0", "", regex=False)
    pool["SE_num"] = pd.to_numeric(pool["SE"], errors="coerce")
    pool = pool[(pool["SE_num"].notna()) & (pool["SE"] >= str(min_se).zfill(6))]
    pool = pool.sort_values("SE")
    vals = pool["casos"].astype(float).to_numpy()
    yrs = pool["SE"].str[:4].to_numpy()
    wks = pool["SE"].str[4:].str.zfill(2).to_numpy()
    wk_hist: dict = {}
    for wk, y in zip(wks, vals):
        if np.isfinite(y):
            wk_hist.setdefault(wk, []).append(float(y))
    global_hist = [float(v) for v in vals if np.isfinite(v)]
    out = []
    for se in test_se:
        se = str(se).zfill(6)
        q_row = {}
        for tau in QUANTILES:
            hist = wk_hist.get(se[4:])
            if hist is None:
                try:
                    d0 = epiweek_to_date(int(se[:4]), int(se[4:]))
                    for k in range(1, 5):
                        for dt in (d0 - timedelta(days=7 * k), d0 + timedelta(days=7 * k)):
                            cand = str(date_to_epiweek(dt)).zfill(6)
                            hist = wk_hist.get(cand[4:])
                            if hist:
                                break
                        if hist:
                            break
                except Exception:
                    hist = None
            if hist is None:
                hist = global_hist if global_hist else [0.0]
            hist_arr = np.asarray(hist, dtype=float)
            hist_arr = hist_arr[np.isfinite(hist_arr)]
            if len(hist_arr) == 0:
                hist_arr = np.array([0.0])
            q_row[tau] = float(np.quantile(hist_arr, tau))
        out.append(q_row)
    return out


def baseline_metrics(y_true: np.ndarray, base_rows: list) -> dict:
    pred_med = np.array([r[0.50] for r in base_rows], dtype=float)
    _m = np.isfinite(y_true) & np.isfinite(pred_med)
    b_mae = (float(np.mean(np.abs(y_true[_m] - pred_med[_m])))
             if np.any(_m) else float("nan"))
    pins = [pinball_mean(y_true, np.array([r[t] for r in base_rows], dtype=float), t)
            for t in QUANTILES]
    pins = [v for v in pins if np.isfinite(v)]
    b_wis = float(np.mean(pins)) if pins else float("nan")
    lo = np.array([r[min(PAIR_90)] for r in base_rows], dtype=float)
    hi = np.array([r[max(PAIR_90)] for r in base_rows], dtype=float)
    mm = np.isfinite(y_true) & np.isfinite(lo) & np.isfinite(hi)
    b_cov = (float(np.mean((y_true[mm] >= lo[mm]) & (y_true[mm] <= hi[mm])))
             if np.any(mm) else float("nan"))
    return {"baseline_wis": b_wis, "baseline_mae": b_mae, "baseline_coverage_90": b_cov}


def clean_nan(obj):
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nan(v) for v in obj]
    return obj


def nanmean_dict(dicts: list) -> dict:
    if not dicts:
        return {}
    out = {}
    for key in dicts[0]:
        vals = []
        for d in dicts:
            v = d.get(key, np.nan)
            if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
                continue  # ignora dicts/listas/strings (ex.: 'stratified')
            vals.append(float(v))
        arr = np.array(vals, dtype=float) if vals else np.array([], dtype=float)
        finite = arr[np.isfinite(arr)] if len(arr) else np.array([])
        if len(finite):
            out[key] = float(np.mean(finite))
        elif len(arr):
            out[key] = float('nan')
    return out


def main():
    if not VAL_PRED.exists():
        print("validation_predictions.json inexistente — nada a fazer")
        return
    vp = json.loads(VAL_PRED.read_text(encoding="utf-8"))
    with open(ROOT / "config" / "config.yaml") as _f:
        cfg = yaml.safe_load(_f)
    start_year = int(cfg.get('training', {}).get('start_year', 2014))
    min_se = f"{start_year}01"
    qc_cfg = cfg.get('quantile_conformal', {}) if isinstance(cfg.get('quantile_conformal', {}), dict) else {}
    QC_ENABLED = bool(qc_cfg.get('enabled', False))
    QC = {'half_life_years': float(qc_cfg.get('half_life_years', 2.5)),
          'max_delta': float(qc_cfg.get('max_abs_delta', 1.0)),
          'max_shift': float(qc_cfg.get('max_abs_median_shift', 0.8)),
          'use_median_shift': bool(qc_cfg.get('apply_median_shift', False))}
    try:
        QC['apply_horizons'] = set(int(x) for x in qc_cfg.get('apply_horizons', [1, 2, 3]))
    except Exception:
        QC['apply_horizons'] = {1, 2, 3}
    # Post-processamento EXP ci (v7 experimentos — exp v7 nov/2026; desligado por padrão)
    POST_CFG = cfg.get('postprocessing', {}) or {}
    if not isinstance(POST_CFG, dict):
        POST_CFG = {}
    base = pd.read_csv(BASE, usecols=["SE", "casos"], dtype={"SE": str})
    base["SE"] = base["SE"].str.replace(".0", "", regex=False)

    old_raw = None
    if VAL_RES.exists():
        try:
            old_raw = json.loads(VAL_RES.read_text(encoding="utf-8"))
        except Exception:
            old_raw = None

    results = {}
    df_rows = []
    for h in sorted(vp.keys()):
        hd = vp[h]  # {tau: [ {fold, test_se, y_test_casos, y_pred_casos} ]}
        fold_map: dict = {}
        for tau_key, blocks in hd.items():
            tau = float(tau_key)
            for blk in blocks:
                fid = str(blk["fold"])
                fm = fold_map.setdefault(fid, {})
                for se, pred in zip(blk["test_se"], blk["y_pred_casos"]):
                    se = str(se).zfill(6)
                    if np.isfinite(float(pred)):
                        fm.setdefault(se, {})[tau] = float(pred)
                if "0.5" not in fm.get("_truth", {}):
                    truth = fm.setdefault("_truth", {})
                    for se, true_v in zip(blk["test_se"], blk["y_test_casos"]):
                        truth[str(se).zfill(6)] = float(true_v) if np.isfinite(float(true_v)) else float('nan')

        per_fold = []
        # ==============================================================
        # Pós-processamento v7/v8 (experimentos, config-driven)
        # Camada A (nível): C4 combinação modelo+baseline (λ) e C3/C6/C5
        #   correções de nível — banda recentrada no novo nível (geometria preservada).
        # Camada B: conformal CQR delta-only por regime-base (fit pós-camada A).
        # Camada C (paridade live): phantom gating do q95 (cap = max(med·fator,
        #   med+5) por regime do MÊS-ALVO), espelhado de models/inference.py.
        # validation_scheme define o pool de calibração de CADA fold f:
        #   'expanding': folds ANTERIORES (implantação incremental; fold0 fica sem
        #     nenhum ajuste — é a razão do fold 2022-23 aparentar quantis "malucos").
        #   'loo': folds ≠ f (sem auto-resíduo; espelha o artefato de produção,
        #     que calibra com todos os folds disponíveis) — fold0 recebe ajuste real.
        # Todas as flags false + 'expanding' = estado canônico (provado no-op).
        # ==============================================================
        APPLY_SCHEME = str(POST_CFG.get('validation_scheme', 'expanding'))
        apply_horizons_eff = (list(range(1, 9)) if POST_CFG.get('conformal_all_horizons')
                              else sorted(QC['apply_horizons']))
        regime_hook = None
        if QC_ENABLED:
            try:
                from models.calibration import make_regime_hook, compute_conformal, entry_for_key
                casos_by_se = {str(se).zfill(6): float(v) for se, v in zip(base['SE'], base['casos'])
                               if np.isfinite(float(v))}
                regime_hook = make_regime_hook(casos_by_se, qc_cfg.get('regime_thresholds') or None)
            except Exception as e:
                print(f"AVISO: regime hook indisponível ({e})")
        # Camada C: paridade do phantom gating (mesmo cfg do inference)
        PH_CFG = cfg.get('phantom_gating', {}) or {}
        phantom_parity = bool(POST_CFG.get('phantom_parity', False))
        nc_protect_median = bool(POST_CFG.get('nc_protect_median', False))
        # EXP-Q0 (v8.1): knobs do re-forma da banda 90% em torno da mediana
        QR_CFG = POST_CFG.get('quantile_reshape', {}) if isinstance(POST_CFG.get('quantile_reshape', {}), dict) else {}
        QR_ENABLED = bool(QR_CFG.get('enabled', False))
        QR = {'roll': max(1, int(QR_CFG.get('smooth_roll4_weeks', 4))),
              'target_cov90': float(QR_CFG.get('target_cov90', 0.90)),
              'tol_pp': float(QR_CFG.get('miss_tol_pp', 1.0)),
              'grid_step': float(QR_CFG.get('grid_step', 0.05)),
              'k_max': float(QR_CFG.get('k_max', 3.0)),
              'nmin_regime': int(QR_CFG.get('pool_nmin_regime', 12))}
        try:
            QR['apply_horizons'] = set(int(x) for x in (QR_CFG.get('apply_horizons') or list(range(1, 9))))
        except Exception:
            QR['apply_horizons'] = set(range(1, 9))
        ph_disabled = set(int(x) for x in (PH_CFG.get('disabled_horizons') or []))
        ph_cal_fac = float(PH_CFG.get('calm_q95_max_factor', 1.5))
        ph_out_fac = float(PH_CFG.get('outbreak_q95_max_factor', 2.0))
        ph_outbreak_months = set(cfg.get('epidemiological_periods', {}).get(
            'outbreak_months', [10, 11, 12, 1, 2, 3, 4, 5]))

        # ---- Passo 0: primitivas por fold (raw já non-crossing, baseline, cru) ----
        units: dict = {}
        for fid in sorted(fold_map.keys(), key=int):
            fm = fold_map[fid]
            truth = fm.pop("_truth", {})
            ses = sorted(se for se in fm if se >= FIRST_TEST_SE)
            if not ses:
                continue
            y_true = np.array([truth[se] for se in ses], dtype=float)
            preds = {t: np.array([fm[se].get(t, np.nan) for se in ses], dtype=float)
                     for t in QUANTILES}
            med = preds[0.50]
            # non-crossing no ensemble salvo (por semana)
            for w in range(len(ses)):
                vals = [preds[t][w] for t in QUANTILES if np.isfinite(preds[t][w])]
                if len(vals) == len(QUANTILES) and np.any(np.diff(vals, axis=0) < 0):
                    s_vals = sorted(vals)
                    for i, t in enumerate(QUANTILES):
                        preds[t][w] = s_vals[i]
            m_raw = compute_all_metrics(y_true, preds)
            # baseline do pool anterior ao teste (gap de 8 semanas)
            start = ses[0]
            pool_end = se_shift(start, -FOLD_GAP_WEEKS)
            pool_b = base[base["SE"].str.zfill(6) < pool_end]
            b_rows = build_baseline_preds(pool_b, ses, min_se=min_se)
            base_med = np.array([r[0.50] for r in b_rows], dtype=float)
            units[fid] = {'ses': ses, 'y_true': y_true, 'preds': preds,
                          'y_log': np.log1p(np.clip(y_true, 0.0, None)),
                          'med_log': np.log1p(np.clip(med, 0.0, None)),
                          'base_log': np.log1p(np.clip(base_med, 0.0, None)),
                          'm_raw': m_raw, 'b_rows': b_rows}

        if not units:
            continue

        def _pool_fids(fid):
            if APPLY_SCHEME == 'loo':
                return [x for x in units.keys() if x != fid]
            return [x for x in units.keys() if int(x) < int(fid)]

        # ---- Camada A passo 1: λ da combinação (C4) por fold ----
        c4_scope = [int(x) for x in (POST_CFG.get('baseline_combination_horizons') or list(range(1, 9)))]
        for fid, u in units.items():
            pm: dict = {}
            u['pm'] = pm
            x0 = u['med_log']
            if POST_CFG.get('baseline_combination') and int(h) in c4_scope:
                _pf = _pool_fids(fid)
                lam = 1.0
                if _pf:
                    _Yp = np.concatenate([units[k]['y_log'] for k in _pf])
                    _Mp = np.concatenate([units[k]['med_log'] for k in _pf])
                    _Bp = np.concatenate([units[k]['base_log'] for k in _pf])
                    _fin = np.isfinite(_Yp) & np.isfinite(_Mp) & np.isfinite(_Bp)
                    if int(_fin.sum()) >= 8:
                        _den = float(np.sum((_Mp[_fin] - _Bp[_fin]) ** 2))
                        if _den > 1e-9:
                            lam = float(np.clip(
                                np.sum((_Mp[_fin] - _Bp[_fin]) * (_Yp[_fin] - _Bp[_fin])) / _den,
                                0.0, 1.0))
                    # v7.2 FIX: λ foi descoberto in-fold (self-fit) na v7.1 — versão
                    # honesta (pool expanding/loo) fica ~1.0 ⇒ C4 sai da promoção.
                    # Mantém compat de meta (tag + combo_lambda) p/ rastrear.
                pm['combo_lambda'] = float(lam)
                x0 = lam * u['med_log'] + (1.0 - lam) * u['base_log']
            u['x0'] = np.asarray(x0, dtype=float)

        # ---- Camada A passo 2: correções de nível (C3 slope / C6 amp / C5 bins) ----
        for fid, u in units.items():
            pm = u['pm']
            _pf = _pool_fids(fid)
            if _pf:
                _Yp = np.concatenate([units[k]['y_log'] for k in _pf])
                _Xp = np.concatenate([units[k]['x0'] for k in _pf])
                _Bp = np.concatenate([units[k]['base_log'] for k in _pf])
            else:
                _Yp = _Xp = _Bp = np.array([], dtype=float)
            _fin = np.isfinite(_Yp) & np.isfinite(_Xp)
            b_s = a_s = None
            if POST_CFG.get('level_slope_correction') and len(_Xp) >= 8 and int(_fin.sum()) >= 8:
                _xbar, _ybar = float(_Xp[_fin].mean()), float(_Yp[_fin].mean())
                _den = float(np.sum((_Xp[_fin] - _xbar) ** 2))
                if _den > 1e-9:
                    b_s = float(np.clip(np.sum((_Xp[_fin] - _xbar) * (_Yp[_fin] - _ybar)) / _den,
                                        float(POST_CFG.get('level_slope_min', 0.6)),
                                        float(POST_CFG.get('level_slope_max', 1.5))))
                    a_s = float(_ybar - b_s * _xbar)
            pm['slope'] = {'a': a_s, 'b': b_s}
            med_proc_log = (a_s + b_s * u['x0']) if b_s is not None else u['x0']
            # EXP-C6: amplificação do desvio vs nível sazonal (1 parâmetro/fold)
            m_amp = None
            if POST_CFG.get('level_amplification'):
                h_amp_scope = [int(v) for v in (POST_CFG.get('level_amplification_horizons') or list(range(1, 9)))]
                if int(h) in h_amp_scope and len(_Xp) >= 12:
                    Dp = _Xp - _Bp
                    Rp = _Yp - _Bp
                    f_amp = _fin & np.isfinite(_Bp)
                    side = str(POST_CFG.get('level_amplification_side', 'both'))
                    m_max = float(POST_CFG.get('level_amplification_max', 1.5))
                    m_min = float(POST_CFG.get('level_amplification_min', 1.0))
                    if side == 'up':
                        sel = f_amp & (Dp > 0)
                        if int(sel.sum()) >= 12:
                            den = float(np.sum(Dp[sel] ** 2))
                            if den > 1e-9:
                                m_amp = float(np.clip(np.sum(Dp[sel] * Rp[sel]) / den, m_min, m_max))
                        if m_amp is not None:
                            d_i = med_proc_log - u['base_log']
                            amp = np.where(d_i > 0, m_amp, 1.0)
                            med_proc_log = u['base_log'] + amp * d_i
                    else:
                        if int(f_amp.sum()) >= 12:
                            den = float(np.sum(Dp[f_amp] ** 2))
                            if den > 1e-9:
                                m_amp = float(np.clip(np.sum(Dp[f_amp] * Rp[f_amp]) / den, m_min, m_max))
                        if m_amp is not None:
                            med_proc_log = u['base_log'] + m_amp * (med_proc_log - u['base_log'])
            pm['level_amp'] = {'slope': m_amp, 'side': POST_CFG.get('level_amplification_side', 'both')}
            # EXP-C5: correção por quintil de nível previsto (ganho g(bin) shrink)
            if POST_CFG.get('level_bin_correction') and not POST_CFG.get('level_amplification'):
                h_bin_scope = POST_CFG.get('level_bin_correction_horizons') or list(range(1, 9))
                if int(h) in [int(x) for x in h_bin_scope] and len(_Xp) >= 20:
                    bin_e = np.quantile(_Xp[_fin], [0.2, 0.4, 0.6, 0.8])
                    nb = len(bin_e) + 1
                    idx = np.clip(np.searchsorted(bin_e, _Xp[_fin]), 0, nb - 1)
                    shrink = float(POST_CFG.get('level_bin_shrink', 0.5))
                    nmin = int(POST_CFG.get('level_bin_nmin', 8))
                    g = np.zeros(nb)
                    for k in range(nb):
                        sel = idx == k
                        if int(sel.sum()) >= nmin:
                            g[k] = shrink * float(np.mean(_Yp[_fin][sel] - _Xp[_fin][sel]))
                    pm['level_bin_g'] = [float(v) for v in g]
                    gilt = np.clip(np.searchsorted(bin_e, u['x0']), 0, nb - 1)
                    med_proc_log = u['x0'] + g[gilt]
                else:
                    pm['level_bin_g'] = None
            # banda recentrada no novo nível (preserva a geometria do modelo)
            logs_a = {}
            for t in QUANTILES:
                lt = np.log1p(np.clip(u['preds'][t], 0.0, None))
                logs_a[t] = (med_proc_log if t == 0.50
                             else med_proc_log + (lt - u['med_log']))
            pred_post = {}
            for t in QUANTILES:
                pred_post[t] = np.clip(np.expm1(logs_a[t]), 0.0, None)
            for w in range(len(u['ses'])):
                vals = [pred_post[t][w] for t in QUANTILES if np.isfinite(pred_post[t][w])]
                if len(vals) == len(QUANTILES) and np.any(np.diff(vals, axis=0) < 0):
                    s_vals = sorted(vals)
                    for i, t in enumerate(QUANTILES):
                        pred_post[t][w] = s_vals[i]
            u['logs_a'] = logs_a
            u['pred_post'] = pred_post

        # ---- Camada B: conformal CQR delta-only por fold (pool por esquema) ----
        # EXP-Q0: histórico por fold (bandas pré/pós-reshape) p/ suavização e clamps
        _qr_pc: dict = {}
        _qr_pub: dict = {}
        all_blocks = {float(t): [] for t in QUANTILES}
        for fid in sorted(units.keys(), key=int):
            u = units[fid]
            for t in QUANTILES:
                all_blocks[float(t)].append({
                    'fold': int(fid),
                    'test_se': [str(s).zfill(6) for s in u['ses']],
                    'y_test_casos': [float(v) if np.isfinite(v) else float('nan') for v in u['y_true']],
                    'y_pred_casos': [float(v) if np.isfinite(v) else float('nan') for v in u['pred_post'][t]],
                })
        for fid, u in units.items():
            fold_conf = {}
            if QC_ENABLED and regime_hook is not None:
                _pf = _pool_fids(fid)
                _pb = {float(t): [dict(b) for b in all_blocks[float(t)] if int(b['fold']) != int(fid)]
                       if APPLY_SCHEME == 'loo' else
                       [dict(b) for b in all_blocks[float(t)] if int(b['fold']) < int(fid)]
                       for t in QUANTILES}
                if any(_pb.values()):
                    pool_fs = {int(h): {str(tau): blocks for tau, blocks in _pb.items()}}
                    try:
                        fold_conf = compute_conformal(
                            pool_fs, regime_hook,
                            half_life_years=QC['half_life_years'],
                            max_delta=QC['max_delta'],
                            max_shift=QC['max_shift'],
                            use_median_shift=QC['use_median_shift']
                                and int(h) in apply_horizons_eff)
                    except Exception as e:
                        print(f"AVISO: conformal fold {fid} falhou ({e})")
            conf_all = fold_conf if int(h) in apply_horizons_eff else {}
            preds_cal = {}
            for t in QUANTILES:
                logs = np.array(u['logs_a'][t], dtype=float)
                if conf_all and regime_hook is not None:
                    for i, se in enumerate(u['ses']):
                        try:
                            entry = entry_for_key(conf_all, int(h), regime_hook(int(h), str(se)))
                        except Exception:
                            entry = {}
                        if not entry:
                            continue
                        sh = float(entry.get('median_shift', 0.0) or 0.0)
                        if np.isfinite(sh):
                            logs[i] += sh
                        if abs(t - 0.5) > 1e-9:
                            try:
                                d = float((entry.get('deltas') or {}).get(f'{t:.2f}'))
                            except Exception:
                                d = 0.0
                            if np.isfinite(d):
                                logs[i] += d
                preds_cal[t] = np.clip(np.expm1(logs), 0.0, None)
            # snapshot pré-reshape (pós-conformal/Camada A) p/ histórico de suavização
            _med_l_pc = np.log1p(np.clip(preds_cal[0.50], 0.0, None))
            _q05_l_pc = np.log1p(np.clip(preds_cal[0.05], 0.0, None))
            _q95_l_pc = np.log1p(np.clip(preds_cal[0.95], 0.0, None))
            _qr_pc[int(fid)] = {'med': _med_l_pc, 'q05': _q05_l_pc, 'q95': _q95_l_pc}
            # ---- EXP-Q0 (v8.1): re-forma da banda 90% em torno da mediana ----
            # Larguras em log1p medidas pós-conformal; trailing rollN GLOBAL (ordem
            # SE dos folds anteriores — só passado publicado); re-centragem:
            # q95' = med + w_hi*, q05' = med − w_lo*; clamps K(h×regime) calibrados
            # expanding no pool de bandas PUBLICADAS dos folds anteriores (menor K
            # da grade com miss do lado ≤ miss_raw + tol_pp/100 — no pior caso
            # degrada a cobertura em tol_pp p.p.). Mediana intocada ⇒ MAE intocado.
            # fold0 (sem pool): sem clamp (só suavização) — honesto expanding.
            qr_meta = None
            if QR_ENABLED and int(h) in QR['apply_horizons']:
                try:
                    _regime_of = (lambda _se: regime_hook(int(h), _se)) if regime_hook is not None else (lambda _se: 'global')
                    # widths pré-reshape dos folds anteriores + fold atual eixo SE
                    _hist_conf = []
                    for _kf in sorted(_qr_pc.keys(), key=int):
                        if int(_kf) >= int(fid):
                            break
                        _e = _qr_pc[_kf]
                        _u = units[str(_kf)]
                        for _j, _se in enumerate(_u['ses']):
                            _hist_conf.append((float(_e['q95'][_j] - _e['med'][_j]),
                                               float(_e['med'][_j] - _e['q05'][_j])))
                    _n_hist = len(_hist_conf)
                    _wh_all = pd.Series([t[0] for t in _hist_conf] + [float(v) for v in (_q95_l_pc - _med_l_pc)])
                    _wl_all = pd.Series([t[1] for t in _hist_conf] + [float(v) for v in (_med_l_pc - _q05_l_pc)])
                    _roll = int(QR['roll'])
                    _minp = min(2, _roll)
                    whi_s = _wh_all.rolling(_roll, min_periods=_minp).mean().iloc[_n_hist:].to_numpy()
                    wlo_s = _wl_all.rolling(_roll, min_periods=_minp).mean().iloc[_n_hist:].to_numpy()
                    whi_s = np.where(np.isfinite(whi_s), whi_s, np.maximum(_q95_l_pc - _med_l_pc, 0.0))
                    wlo_s = np.where(np.isfinite(wlo_s), wlo_s, np.maximum(_med_l_pc - _q05_l_pc, 0.0))
                    # caps expanding no pool de bandas PUBLISHED dos folds anteriores
                    pool_pub = []
                    for _kf in sorted(_qr_pub.keys(), key=int):
                        if int(_kf) >= int(fid):
                            break
                        _p = _qr_pub[_kf]
                        _u = units[str(_kf)]
                        for _j, _se in enumerate(_u['ses']):
                            _se = str(_se).zfill(6)
                            _yl = float(_u['y_log'][_j])
                            if not np.isfinite(_yl):
                                continue
                            pool_pub.append((_regime_of(_se), _yl, float(_p['med'][_j]),
                                             float(_p['wh'][_j]), float(_p['wl'][_j])))
                    caps = {}
                    if pool_pub:
                        _grid = np.arange(QR['grid_step'], QR['k_max'] + 1e-9, QR['grid_step'])
                        _tol = float(QR['tol_pp']) / 100.0

                        def _cap_for(_sel, _side):
                            _yl = np.array([pool_pub[i][1] for i in _sel])
                            _ml = np.array([pool_pub[i][2] for i in _sel])
                            if _side == 'hi':
                                _wv = np.array([pool_pub[i][3] for i in _sel])
                                _raw_miss = float(np.mean(_yl > _ml + _wv))
                            else:
                                _wv = np.array([pool_pub[i][4] for i in _sel])
                                _raw_miss = float(np.mean(_yl < _ml - _wv))
                            _lim = _raw_miss + _tol
                            for _K in _grid:
                                _wK = np.minimum(_wv, float(_K))
                                _miss = float(np.mean(_yl > _ml + _wK)) if _side == 'hi' \
                                    else float(np.mean(_yl < _ml - _wK))
                                if _miss <= _lim:
                                    return float(_K)
                            return None

                        _reg_counts = {}
                        for _t in pool_pub:
                            _reg_counts[_t[0]] = _reg_counts.get(_t[0], 0) + 1
                        for _r in ('high', 'mid', 'low'):
                            _sel = [i for i, _t in enumerate(pool_pub) if _t[0] == _r]
                            if len(_sel) >= QR['nmin_regime']:
                                caps[_r] = (_cap_for(_sel, 'hi'), _cap_for(_sel, 'lo'))
                        if not any(caps.get(_r) for _r in ('high', 'mid', 'low')):
                            _all = list(range(len(pool_pub)))
                            caps['global'] = (_cap_for(_all, 'hi'), _cap_for(_all, 'lo'))
                    # aplicação: re-centragem dos bordos em torno da mediana
                    _n_hi = _n_lo = 0
                    for i in range(len(u['ses'])):
                        _rg = _regime_of(str(u['ses'][i]).zfill(6))
                        _K = caps.get(_rg) if caps.get(_rg) else caps.get('global')
                        w_hi_i = float(whi_s[i])
                        w_lo_i = float(wlo_s[i])
                        if _K:
                            if _K[0] is not None and w_hi_i > _K[0]:
                                w_hi_i = _K[0]
                                _n_hi += 1
                            if _K[1] is not None and w_lo_i > _K[1]:
                                w_lo_i = _K[1]
                                _n_lo += 1
                        preds_cal[0.95][i] = np.clip(np.expm1(_med_l_pc[i] + w_hi_i), 0.0, None)
                        preds_cal[0.05][i] = np.clip(np.expm1(_med_l_pc[i] - w_lo_i), 0.0, None)
                    qr_meta = {'enabled': True, 'roll': _roll, 'n_hi_clamped': int(_n_hi),
                               'n_lo_clamped': int(_n_lo)}
                    if caps:
                        qr_meta['caps'] = {r: [None if v is None else round(float(v), 3)
                                               for v in _t] for r, _t in caps.items()}
                except Exception as e:
                    print(f"AVISO: quantile_reshape fold {fid} falhou ({e})")
            # ---- Camada C: paridade do phantom gating (cap q95 por regime do mês-alvo) ----
            ph_n = 0
            if phantom_parity and int(h) not in ph_disabled and conf_all is not None:
                if 0.95 in preds_cal and 0.5 in preds_cal:
                    try:
                        from data.epiweeks import epiweek_to_date as _e2d
                        med_c = np.clip(preds_cal[0.5], 0.0, None)   # casos
                        q95_c = np.clip(preds_cal[0.95], 0.0, None)  # casos
                        for i, se in enumerate(u['ses']):
                            _se = str(se).zfill(6)
                            try:
                                _mo = _e2d(int(_se[:4]), int(_se[4:])).month
                            except Exception:
                                _mo = None
                            fac = ph_out_fac if (_mo in ph_outbreak_months) else ph_cal_fac
                            cap_c = max(float(med_c[i]) * fac, float(med_c[i]) + 5.0)
                            if np.isfinite(q95_c[i]) and q95_c[i] > cap_c:
                                q95_c[i] = cap_c
                                ph_n += 1
                        preds_cal[0.95] = np.clip(q95_c, 0.0, None)
                    except Exception as e:
                        print(f"AVISO: phantom parity fold {fid} falhou ({e})")
            # non-crossing pós-calibração/phantom (por semana)
            # EXP-NC: nc_protect_median repaira cruzamentos com CLIP em torno da
            #   mediana pós-A (banda nunca reatribui o slot q50 ⇒ deltas de banda
            #   não vazam no MAE). false = sort (paridade live do inference).
            _med_prot = np.clip(preds_cal[0.50], 0.0, None)
            for w in range(len(u['ses'])):
                vals = [preds_cal[t][w] for t in QUANTILES if np.isfinite(preds_cal[t][w])]
                if len(vals) == len(QUANTILES) and np.any(np.diff(vals, axis=0) < 0):
                    if nc_protect_median:
                        med0 = _med_prot[w]
                        q = {t: float(preds_cal[t][w]) for t in QUANTILES}
                        q[0.25] = min(q[0.25], med0)
                        q[0.05] = min(q[0.05], q[0.25], med0)
                        q[0.75] = max(q[0.75], med0)
                        q[0.95] = max(q[0.95], q[0.75], med0)
                        for t in QUANTILES:
                            preds_cal[t][w] = q[t]
                    else:
                        s_vals = sorted(vals)
                        for i, t in enumerate(QUANTILES):
                            preds_cal[t][w] = s_vals[i]
            m = compute_all_metrics(u['y_true'], preds_cal) if QC_ENABLED else dict(u['m_raw'])
            if QR_ENABLED:
                _ml_p = np.log1p(np.clip(preds_cal[0.50], 0.0, None))
                _q05_p = np.log1p(np.clip(preds_cal[0.05], 0.0, None))
                _q95_p = np.log1p(np.clip(preds_cal[0.95], 0.0, None))
                _qr_pub[int(fid)] = {'med': _ml_p, 'wh': (_q95_p - _ml_p), 'wl': (_ml_p - _q05_p)}
            scheme = 'expanding_cqr_delta_only' if APPLY_SCHEME != 'loo' else 'loo_cqr_delta_only'
            if phantom_parity and int(h) not in ph_disabled:
                scheme += '+phantom'
            if u['pm'].get('level_amp', {}).get('slope') is not None:
                scheme += '+level_amp'
            elif u['pm'].get('level_bin_g') is not None:
                scheme += '+level_bin'
            elif u['pm'].get('slope', {}).get('b') is not None:
                scheme += '+level_slope'
            if 'combo_lambda' in u['pm']:
                scheme += '+baseline_combo'
            if nc_protect_median:
                scheme += '+ncmed'
            if qr_meta:
                scheme += '+qreshape'
            m['raw_uncalibrated'] = {k: u['m_raw'].get(k) for k in
                                     ('mae', 'rmse', 'bias', 'wape', 'wis_total', 'wis_50', 'wis_90',
                                      'width_50', 'width_90', 'coverage_50', 'coverage_90')}
            m['calibration'] = {'applied': bool(conf_all), 'scheme': scheme,
                                'pool': APPLY_SCHEME,
                                'phantom_applied': int(ph_n),
                                'half_life_years': QC['half_life_years'],
                                'apply_horizons': sorted(apply_horizons_eff),
                                'post': {'validation_scheme': APPLY_SCHEME,
                                         'conformal_all_horizons': bool(POST_CFG.get('conformal_all_horizons')),
                                         'level_slope_correction': bool(POST_CFG.get('level_slope_correction')),
                                         'level_bin_correction': bool(POST_CFG.get('level_bin_correction')),
                                         'baseline_combination': bool(POST_CFG.get('baseline_combination')),
                                         'phantom_parity': bool(phantom_parity),
                                         'nc_protect_median': bool(nc_protect_median),
                                         **{k: v for k, v in u['pm'].items()}}}
            if qr_meta:
                m['calibration']['qreshape'] = dict(qr_meta)
            m.update(baseline_metrics(u['y_true'], u['b_rows']))
            # Ganho vs baseline (skill): por fold, razão direta
            m['skill_mae'] = skill_score(m.get('mae'), m.get('baseline_mae'))
            m['skill_wis'] = skill_score(m.get('wis_total'), m.get('baseline_wis'))
            m['fold'] = int(fid)
            m['horizon'] = int(h)
            m['n_weeks'] = len(u['ses'])
            m['stratified'] = stratified_for_fold(u['y_true'], u['preds'], u['ses'])
            per_fold.append(m)

        if not per_fold:
            continue
        complete = [m for m in per_fold if m.get('n_weeks', 0) >= 52]
        avg = nanmean_dict(complete if complete else per_fold)
        avg.pop('fold', None)  # identificador do fold não faz sentido em média
        avg.pop('n_weeks', None)
        avg['horizon'] = int(h)
        avg['n_folds'] = len(complete if complete else per_fold)
        avg['excluded_partial_folds'] = [str(m['fold']) for m in per_fold if m.get('n_weeks', 0) < 52]
        full = nanmean_dict(per_fold)
        full.pop('fold', None)
        full.pop('n_weeks', None)
        full['horizon'] = int(h)
        avg['full_4folds_avg'] = full
        # skills como razão das MÉDIAS (consistente com as médias canônicas);
        # a média das razões por fold fica em per_fold_ensemble como diagnóstico
        for _agg in (avg, full):
            _agg['skill_mae'] = skill_score(_agg.get('mae'), _agg.get('baseline_mae'))
            _agg['skill_wis'] = skill_score(_agg.get('wis_total'), _agg.get('baseline_wis'))
        # referência de seeds do JSON antigo (relatório não recalcula seeds)
        if old_raw and h in old_raw:
            old_avg = old_raw[h].get('ensemble_avg', {})
            if 'n_seeds' in old_avg:
                avg['n_seeds'] = old_avg['n_seeds']
        results[h] = {
            'quantiles': list(QUANTILES),
            'per_fold_ensemble': per_fold,
            'ensemble_avg': avg,
            'stratified': aggregate_stratified(per_fold),
        }

        def _g(k, d=avg):
            v = d.get(k)
            return v if isinstance(v, (int, float)) else float('nan')
        df_rows.append({
            'Horizonte': int(h), 'MAE': _g('mae'), 'RMSE': _g('rmse'),
            'Bias': _g('bias'), 'WAPE': _g('wape'),
            'WIS_total': _g('wis_total'), 'WIS_50': _g('wis_50'),
            'WIS_90': _g('wis_90'), 'Width_50': _g('width_50'),
            'Width_90': _g('width_90'), 'Coverage_50': _g('coverage_50'),
            'Coverage_90': _g('coverage_90'),
            'Skill_MAE': _g('skill_mae'), 'Skill_WIS': _g('skill_wis'),
        })
        print(f"  h{h}{' [CQR-EXP]' if QC_ENABLED else ''}: mae={_g('mae'):.2f} rmse={_g('rmse'):.2f} "
              f"bias={_g('bias'):+.2f} wape={_g('wape'):.1%} wis_total={_g('wis_total'):.2f} "
              f"(50 {_g('wis_50'):.2f} / 90 {_g('wis_90'):.2f}) "
              f"cov50={_g('coverage_50'):.2f} cov90={_g('coverage_90'):.2f} "
              f"w50={_g('width_50'):.1f} w90={_g('width_90'):.1f} "
              f"skill_mae={_g('skill_mae'):+.1%} skill_wis={_g('skill_wis'):+.1%}")

    # backup do legado (apenas na primeira regeneração) e escrita do novo
    legacy = VAL_RES.with_name("validation_results_legacy.json.bak")
    if VAL_RES.exists() and not legacy.exists():
        VAL_RES.replace(legacy)
    VAL_RES.write_text(json.dumps(clean_nan(results), indent=2, ensure_ascii=False),
                       encoding="utf-8")
    df = pd.DataFrame(df_rows).set_index('Horizonte')
    df.to_csv(_VP_DIR / "metrics_dataframe.csv")
    print(f"\nSalvo: {VAL_RES.name} (backup: validation_results_legacy.json.bak)")
    print(f"Salvo: {_VP_DIR / 'metrics_dataframe.csv'}")
    print(f"Salvo: models/metrics_dataframe.csv ({list(df.columns)})")


def stratified_for_fold(y_true: np.ndarray, preds: dict, ses: list) -> dict:
    from models.metrics import compute_all_metrics as _cam
    out = {}
    months = np.array([se_to_month(se) for se in ses])
    for period, sel in (("outbreak", np.isin(months, list(OUTBREAK_MONTHS))),
                        ("calm", ~np.isin(months, list(OUTBREAK_MONTHS)))):
        if not np.any(sel):
            continue
        m = _cam(y_true[sel], {t: np.array(v, dtype=float)[sel] for t, v in preds.items()})
        out[period] = {'n_samples': int(sel.sum()),
                       'mae': m.get('mae', float('nan')),
                       'wis': m.get('wis_total', float('nan'))}
    return out


def aggregate_stratified(per_fold: list) -> dict:
    result = {}
    for period in ('outbreak', 'calm'):
        rows = [m['stratified'][period] for m in per_fold
                if period in m.get('stratified', {})]
        if not rows:
            continue
        n = sum(r['n_samples'] for r in rows)

        def _w(key):
            vals = np.array([r.get(key, np.nan) for r in rows], dtype=float)
            w = np.array([r['n_samples'] if np.isfinite(r.get(key, np.nan)) else 0
                          for r in rows], dtype=float)
            if np.any(np.isfinite(vals)) and np.sum(w) > 0:
                return float(np.nansum(vals * w) / (np.sum(w) + 1e-10))
            return float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else float('nan')
        result[period] = {'n_samples': n, 'mae': _w('mae'), 'wis': _w('wis')}
    return result


if __name__ == "__main__":
    main()

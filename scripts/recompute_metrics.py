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
VAL_PRED = ROOT / "models/validation_predictions.json"
VAL_RES = ROOT / "models/validation_results.json"
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
        # Pós-processamento EXPANDING (experimentos v7, config-driven)
        # Camada A (nível): EXP-C4 combinação modelo+baseline e EXP-C3 correção
        #   de nível por slope (anti "regressão à média": y_log ~ a + b·pred_log,
        #   b fitado nos folds anteriores). Banda recentrada no novo nível
        #   (deltas de banda do modelo preservados em log — mesma geometria).
        # Camada B (conformal CQR delta-only): fitada PÓS-camada A, SÓ com os
        #   folds anteriores (mesma honestidade do esquema expanding v3).
        # Flags em cfg['postprocessing'] — todas false mantém o estado canônico.
        # ==============================================================
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
        processed_blocks: dict = {float(t): [] for t in QUANTILES}  # pós-camada A (fit do conformal)
        level_pool: list = []                                       # {'y': arr, 'x_c4': arr} em log1p
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

            # ---- baseline do pool anterior ao teste (gap de 8 semanas) ----
            start = ses[0]
            pool_end = se_shift(start, -FOLD_GAP_WEEKS)
            pool = base[base["SE"].str.zfill(6) < pool_end]
            b_rows = build_baseline_preds(pool, ses, min_se=min_se)
            base_med = np.array([r[0.50] for r in b_rows], dtype=float)

            # ---- Camada A (nível) — parâmetros fitados nos folds ANTERIORES ----
            y_log = np.log1p(np.clip(y_true, 0.0, None))
            med_log = np.log1p(np.clip(med, 0.0, None))
            base_log = np.log1p(np.clip(base_med, 0.0, None))
            post_meta: dict = {}
            x0 = med_log
            c4_scope = [int(x) for x in (POST_CFG.get('baseline_combination_horizons') or list(range(1, 9)))]
            if POST_CFG.get('baseline_combination') and int(h) in c4_scope:
                fin = np.isfinite(y_log) & np.isfinite(med_log) & np.isfinite(base_log)
                lam = 1.0
                if int(fin.sum()) >= 8:
                    yv, mv, bv = y_log[fin], med_log[fin], base_log[fin]
                    den = float(np.sum((mv - bv) ** 2))
                    if den > 1e-9:
                        lam = float(np.clip(np.sum((mv - bv) * (yv - bv)) / den, 0.0, 1.0))
                post_meta['combo_lambda'] = float(lam)
                x0 = lam * med_log + (1.0 - lam) * base_log
            b_s = a_s = None
            if POST_CFG.get('level_slope_correction') and level_pool:
                X = np.concatenate([e['x_c4'] for e in level_pool])
                Y = np.concatenate([e['y'] for e in level_pool])
                fin = np.isfinite(X) & np.isfinite(Y)
                if int(fin.sum()) >= 8:
                    xbar, ybar = float(X[fin].mean()), float(Y[fin].mean())
                    den = float(np.sum((X[fin] - xbar) ** 2))
                    if den > 1e-9:
                        b_s = float(np.clip(np.sum((X[fin] - xbar) * (Y[fin] - ybar)) / den,
                                            float(POST_CFG.get('level_slope_min', 0.6)),
                                            float(POST_CFG.get('level_slope_max', 1.5))))
                        a_s = float(ybar - b_s * xbar)
            post_meta['slope'] = {'a': a_s, 'b': b_s}
            med_proc_log = (a_s + b_s * x0) if b_s is not None else x0
            # EXP-C6: amplificação do desvio vs nível sazonal (anti-regressão à média,
            # 1 parâmetro m por horizonte fitado nos folds anteriores):
            #   pred'_log = base_log + m·(pred_log − base_log), m ∈ [min, max].
            m_amp = None
            if POST_CFG.get('level_amplification'):
                h_amp_scope = [int(v) for v in (POST_CFG.get('level_amplification_horizons') or list(range(1, 9)))]
                if int(h) in h_amp_scope and level_pool:
                    Dp = np.concatenate([np.asarray(e['x_c4'], float) - np.asarray(e['b'], float)
                                         for e in level_pool])
                    Rp = np.concatenate([np.asarray(e['y'], float) - np.asarray(e['b'], float)
                                         for e in level_pool])
                    fin = np.isfinite(Dp) & np.isfinite(Rp)
                    side = str(POST_CFG.get('level_amplification_side', 'both'))
                    m_max = float(POST_CFG.get('level_amplification_max', 1.5))
                    m_min = float(POST_CFG.get('level_amplification_min', 1.0))
                    if side == 'up':
                        sel = fin & (Dp > 0)
                        if int(sel.sum()) >= 12:
                            den = float(np.sum(Dp[sel] ** 2))
                            if den > 1e-9:
                                m_amp = float(np.clip(np.sum(Dp[sel] * Rp[sel]) / den, m_min, m_max))
                        if m_amp is not None:
                            d_i = med_proc_log - base_log
                            amp = np.where(d_i > 0, m_amp, 1.0)
                            med_proc_log = base_log + amp * d_i
                    else:
                        if int(fin.sum()) >= 12:
                            den = float(np.sum(Dp[fin] ** 2))
                            if den > 1e-9:
                                m_amp = float(np.clip(np.sum(Dp[fin] * Rp[fin]) / den, m_min, m_max))
                        if m_amp is not None:
                            med_proc_log = base_log + m_amp * (med_proc_log - base_log)
            post_meta['level_amp'] = {'slope': m_amp,
                                      'side': POST_CFG.get('level_amplification_side', 'both')}
            # EXP-C5: correção por quintil de NÍVEL PREVISTO (ataca o erro condicional
            # nos dois extremos: pico subestimado / nível baixo superestimado).
            # g(bin) = shrink * média(y − med) em log nos folds ANTERIORES por bin.
            bin_edges_cal = None
            if POST_CFG.get('level_bin_correction') and not POST_CFG.get('level_amplification'):
                h_bin_scope = POST_CFG.get('level_bin_correction_horizons') or list(range(1, 9))
                if int(h) in [int(x) for x in h_bin_scope] and level_pool:
                    Xp = np.concatenate([e['x_c4'] for e in level_pool])
                    Yp = np.concatenate([e['y'] for e in level_pool])
                    fin = np.isfinite(Xp) & np.isfinite(Yp)
                    bin_edges_cal = np.quantile(Xp[fin], [0.2, 0.4, 0.6, 0.8]) if fin.sum() >= 20 else None
                    if bin_edges_cal is not None:
                        nb = len(bin_edges_cal) + 1
                        idx = np.clip(np.searchsorted(bin_edges_cal, Xp[fin]), 0, nb - 1)
                        shrink = float(POST_CFG.get('level_bin_shrink', 0.5))
                        nmin = int(POST_CFG.get('level_bin_nmin', 8))
                        g = np.zeros(nb)
                        for k in range(nb):
                            sel = idx == k
                            if int(sel.sum()) >= nmin:
                                g[k] = shrink * float(np.mean(Yp[fin][sel] - Xp[fin][sel]))
                        post_meta['level_bin_g'] = [float(v) for v in g]
                        gilt = np.clip(np.searchsorted(bin_edges_cal, x0), 0, nb - 1)
                        med_proc_log = x0 + g[gilt]
                    else:
                        post_meta['level_bin_g'] = None
                else:
                    post_meta['level_bin_g'] = None
            # banda recentrada no novo nível (preserva a geometria do modelo)
            logs_a = {}
            for t in QUANTILES:
                lt = np.log1p(np.clip(preds[t], 0.0, None))
                logs_a[t] = (med_proc_log if t == 0.50
                             else med_proc_log + (lt - med_log))
            pred_post = {}
            for t in QUANTILES:
                pred_post[t] = np.clip(np.expm1(logs_a[t]), 0.0, None)
            # non-crossing pós-camada A (por semana)
            for w in range(len(ses)):
                vals = [pred_post[t][w] for t in QUANTILES if np.isfinite(pred_post[t][w])]
                if len(vals) == len(QUANTILES) and np.any(np.diff(vals, axis=0) < 0):
                    s_vals = sorted(vals)
                    for i, t in enumerate(QUANTILES):
                        pred_post[t][w] = s_vals[i]

            # ---- Camada B: conformal fitado nas previsões PÓS-camada A anteriores ----
            fold_conformal_prev = {}
            if QC_ENABLED and regime_hook is not None:
                has_pool = any(blocks for blocks in processed_blocks.values())
                if has_pool:
                    pool_fs = {h: {str(tau): [dict(b) for b in blocks]
                                   for tau, blocks in processed_blocks.items()}
                               for h in [int(h)]}
                    try:
                        fold_conformal_prev = compute_conformal(
                            pool_fs, regime_hook,
                            half_life_years=QC['half_life_years'],
                            max_delta=QC['max_delta'],
                            max_shift=QC['max_shift'],
                            use_median_shift=QC['use_median_shift']
                                and int(h) in apply_horizons_eff)
                    except Exception as e:
                        print(f"AVISO: conformal expanding fold {fid} falhou ({e})")
            conf_all = fold_conformal_prev if int(h) in apply_horizons_eff else {}
            preds_cal = {}
            for t in QUANTILES:
                logs = np.array(logs_a[t], dtype=float)
                if conf_all and regime_hook is not None:
                    for i, se in enumerate(ses):
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
            # non-crossing pós-calibração (por semana)
            for w in range(len(ses)):
                vals = [preds_cal[t][w] for t in QUANTILES if np.isfinite(preds_cal[t][w])]
                if len(vals) == len(QUANTILES) and np.any(np.diff(vals, axis=0) < 0):
                    s_vals = sorted(vals)
                    for i, t in enumerate(QUANTILES):
                        preds_cal[t][w] = s_vals[i]
            m = compute_all_metrics(y_true, preds_cal) if QC_ENABLED else dict(m_raw)
            scheme = 'expanding_cqr_delta_only'
            if post_meta.get('level_amp', {}).get('slope') is not None:
                scheme += '+level_amp'
            elif post_meta.get('level_bin_g') is not None:
                scheme += '+level_bin'
            elif post_meta.get('slope', {}).get('b') is not None:
                scheme += '+level_slope'
            if 'combo_lambda' in post_meta:
                scheme += '+baseline_combo'
            m['raw_uncalibrated'] = {k: m_raw.get(k) for k in
                                     ('mae', 'rmse', 'bias', 'wape', 'wis_total', 'wis_50', 'wis_90',
                                      'width_50', 'width_90', 'coverage_50', 'coverage_90')}
            m['calibration'] = {'applied': bool(conf_all), 'scheme': scheme,
                                'half_life_years': QC['half_life_years'],
                                'apply_horizons': sorted(apply_horizons_eff),
                                'post': {'conformal_all_horizons': bool(POST_CFG.get('conformal_all_horizons')),
                                         'level_slope_correction': bool(POST_CFG.get('level_slope_correction')),
                                         'level_bin_correction': bool(POST_CFG.get('level_bin_correction')),
                                         'baseline_combination': bool(POST_CFG.get('baseline_combination')),
                                         **post_meta}}
            m.update(baseline_metrics(y_true, b_rows))
            # Ganho vs baseline (skill): por fold, razão direta
            m['skill_mae'] = skill_score(m.get('mae'), m.get('baseline_mae'))
            m['skill_wis'] = skill_score(m.get('wis_total'), m.get('baseline_wis'))
            m['fold'] = int(fid)
            m['horizon'] = int(h)
            m['n_weeks'] = len(ses)
            m['stratified'] = stratified_for_fold(y_true, preds, ses)
            per_fold.append(m)
            # ---- registra exemplos/blocos p/ os folds seguintes (fit expanding) ----
            level_pool.append({'y': y_log, 'x_c4': np.array(x0, dtype=float),
                               'b': np.array(base_log, dtype=float)})
            for t in QUANTILES:
                processed_blocks[float(t)].append({
                    'fold': int(fid),
                    'test_se': [str(s).zfill(6) for s in ses],
                    'y_test_casos': [float(v) if np.isfinite(v) else float('nan') for v in y_true],
                    'y_pred_casos': [float(v) if np.isfinite(v) else float('nan') for v in pred_post[t]],
                })

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
    df.to_csv(ROOT / "models" / "metrics_dataframe.csv")
    print(f"\nSalvo: {VAL_RES.name} (backup: validation_results_legacy.json.bak)")
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

"""
Análise EXP-Q2: re-forma da banda quantílica em torno da mediana (live e validação).

Problema (dashboard/produção 2026-43+): q05 "cola no chão" (0-3 casos) e q95
explode 3-5x a mediana em pontos aleatórios do fan — os modelos de cauda
independentes têm ruído muito maior que a mediana. Objetivo: banda que segue o
desenho da mediana (âncora multiplicativa por horizonte, suavizada ao longo
do fan h1..h8, monotônica — clássico "fan chart").

Este script mede, nos folds walk-forward (models/validation_predictions.json):
  * estado atual: cov90/cov50, WIS total/50/90, corr(log q05 vs log q50),
    distribuição dos fatores q*/q50;
  * simulações da re-forma (suavização + monotonia + caps) e impacto em
    cobertura/WIS — para escolher os caps do config.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.metrics import weighted_interval_score, coverage_interval, mean_interval_width  # noqa: E402

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]


def _aligned(vp, h):
    """(fold, se) -> {tau: (y_true, y_pred)} para o horizonte h."""
    d = vp[str(h)]
    out = {}
    for tau in QUANTILES:
        d = vp[str(h)]
        key = f"{tau:.2f}" if f"{tau:.2f}" in d else str(tau)
        for blk in d[key]:
            fold = int(blk['fold'])
            for se, yt, yp in zip(blk['test_se'], blk['y_test_casos'], blk['y_pred_casos']):
                se = str(se).zfill(6)
                key = (fold, se)
                rec = out.setdefault(key, {})
                if tau == 0.50:
                    rec['y'] = float(yt)
                rec[tau] = float(yp)
    return out


def build_fans(vp):
    from data.epiweeks import epiweek_to_date, date_to_epiweek
    rec = {h: _aligned(vp, h) for h in range(1, 9)}
    fans = []  # (fold, origin_se, {h: {tau: pred, 'y': y}})
    for (fold, se_origin), r1 in rec[1].items():
        fan = {1: r1}
        ok = True
        for h in range(2, 9):
            y, w = int(se_origin[:4]), int(se_origin[4:])
            se_t = date_to_epiweek(epiweek_to_date(y, w) + pd.Timedelta(weeks=h))
            r = rec[h].get((fold, str(se_t).zfill(6)))
            if r is None:
                ok = False
                break
            fan[h] = r
        if ok:
            fans.append((fold, se_origin, fan))
    return rec, fans


def metrics_from(y, q):
    out = {}
    out['cov90'] = coverage_interval(y, q[0.05], q[0.95])
    out['cov50'] = coverage_interval(y, q[0.25], q[0.75])
    out['w90'] = mean_interval_width(y, q[0.05], q[0.95])
    out['w50'] = mean_interval_width(y, q[0.25], q[0.75])
    out['wis_proxy'] = weighted_interval_score(y, q)
    out['mae'] = float(np.mean(np.abs(y - q[0.50])))
    return out


def flatten(fans):
    ys = []
    qd = {t: [] for t in QUANTILES}
    for fold, o, fan in fans:
        for h in range(1, 9):
            ys.append(fan[h]['y'])
            for t in QUANTILES:
                qd[t].append(fan[h][t])
    y = np.array(ys)
    q = {t: np.array(qd[t]) for t in QUANTILES}
    return y, q


def nc_sort(q):
    arr = np.vstack([q[t] for t in QUANTILES])
    arr = np.sort(arr, axis=0)
    return {t: arr[i] for i, t in enumerate(QUANTILES)}


def metrics_raw(fans):
    y, q = flatten(fans)
    q = nc_sort(q)
    m = metrics_from(y, q)
    with np.errstate(divide='ignore', invalid='ignore'):
        corr05 = np.corrcoef(np.log1p(np.clip(q[0.05], 0, None)), np.log1p(np.clip(q[0.50], 0, None)))[0, 1]
        corr95 = np.corrcoef(np.log1p(np.clip(q[0.95], 0, None)), np.log1p(np.clip(q[0.50], 0, None)))[0, 1]
    m['shape_corr_q05_q50'] = float(corr05)
    m['shape_corr_q95_q50'] = float(corr95)
    for t, name in [(0.05, 'f05'), (0.25, 'f25'), (0.75, 'f75'), (0.95, 'f95')]:
        med = np.clip(q[0.50], 0.0, None)
        fac = np.where(med > 0.5, q[t] / np.maximum(med, 1e-9), np.nan)
        fac = fac[np.isfinite(fac)]
        m[f'{name}_p50'] = float(np.median(fac))
        m[f'{name}_p25'] = float(np.percentile(fac, 25))
        m[f'{name}_p75'] = float(np.percentile(fac, 75))
        m[f'{name}_p95'] = float(np.percentile(fac, 95))
    return m


def smooth_series(x, window=3):
    """Rolling median centrada, bordas com kernel encolhido."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n <= 2 or window <= 1:
        return x.copy()
    out = x.copy()
    for i in range(n):
        k = min(window, 2 * min(i, n - 1 - i) + 1)
        if k <= 1:
            continue
        lo = i - k // 2
        hi = i + k // 2 + 1
        out[i] = np.median(x[lo:hi])
    return out


RAW_CAPS = {'lo': (0.00, 1.00), 'hi': (1.00, 50.0)}


def simulate_reshape(fans, caps, smooth=True, monotone=True, profile=None):
    """Aplica re-forma em cada fan (eixo h) e devolve métricas agregadas.

    profile: dict tau -> array[8] de fatores fixos (calibrados) — sobrepõe os
    fatores raw: q* = med × f*[h] determinístico (fan clássico).
    """
    ys = []
    qd = {t: [] for t in QUANTILES}
    for fold, o, fan in fans:
        med = np.array([max(fan[h][0.50], 0.0) for h in range(1, 9)])
        f = {}
        for t, side in [(0.05, 'lo'), (0.25, 'lo'), (0.75, 'hi'), (0.95, 'hi')]:
            if profile is not None:
                raw = np.array(profile[t], dtype=float)
            else:
                raw = np.array([fan[h][t] / med[h - 1] if med[h - 1] > 0.5 else np.nan
                                for h in range(1, 9)])
                if smooth:
                    fin = np.isfinite(raw)
                    if fin.any():
                        raw = np.where(fin, raw, np.nanmedian(raw))
                    else:
                        raw = np.ones(8)
                    raw = smooth_series(raw, 3)
                if monotone:
                    if side == 'hi':
                        raw = np.maximum.accumulate(raw)
                    else:
                        raw = np.minimum.accumulate(raw)
                c = caps[side]
                raw = np.clip(raw, c[0], c[1])
            if side == 'lo':
                raw = np.minimum(raw, 0.98)
            else:
                raw = np.maximum(raw, 1.02)
            if profile is not None:
                c = caps[side]
                raw = np.clip(raw, c[0], c[1])
            f[t] = raw
        for i, h in enumerate(range(1, 9)):
            ys.append(fan[h]['y'])
            qd[0.50].append(med[i])
            for t in [0.05, 0.25, 0.75, 0.95]:
                qd[t].append(med[i] * f[t][i])
    y = np.array(ys)
    q = nc_sort({t: np.array(qd[t]) for t in QUANTILES})
    return metrics_from(y, q)


if __name__ == '__main__':
    vp = json.load(open(ROOT / 'models/validation_predictions.json'))
    rec, fans = build_fans(vp)
    print(f"fans completos reconstruídos: {len(fans)} (horizonte-1 rows: {len(rec[1])})")
    mr = metrics_raw(fans)
    print("== RAW (estado atual, não-crossing) ==")
    for k, v in mr.items():
        print(f"  {k}: {v:.3f}")
    print()
    print("== fatores RAW por horizonte (cases) ==")
    for h in range(1, 9):
        f05, f95 = [], []
        for fold, o, fan in fans:
            med = max(fan[h][0.50], 0.0)
            if med > 0.5:
                f05.append(fan[h][0.05] / med)
                f95.append(fan[h][0.95] / med)
        print(f"  h{h}: f05 p50={np.median(f05):.2f} p25={np.percentile(f05,25):.2f} p75={np.percentile(f05,75):.2f} "
              f"p05={np.percentile(f05,5):.2f} | f95 p50={np.median(f95):.2f} p25={np.percentile(f95,25):.2f} "
              f"p75={np.percentile(f95,75):.2f} p95={np.percentile(f95,95):.2f}")
    print()
    variants = [
        ('C:  lo[0.25,0.80] hi[1.20,3.5]', {'lo': (0.25, 0.80), 'hi': (1.20, 3.50)}, True),
        ('C+: lo[0.25,0.80] hi[1.20,3.8]', {'lo': (0.25, 0.80), 'hi': (1.20, 3.80)}, True),
        ('C-: lo[0.25,0.82] hi[1.20,4.0]', {'lo': (0.25, 0.82), 'hi': (1.20, 4.00)}, True),
        ('F:  lo[0.30,0.80] hi[1.20,3.8] (floor f05 0.30)', {'lo': (0.30, 0.80), 'hi': (1.20, 3.80)}, True),
        ('G:  lo[0.30,0.82] hi[1.25,4.2] (floor f05 0.30)', {'lo': (0.30, 0.82), 'hi': (1.25, 4.20)}, True),
    ]
    base_wis = mr['wis_proxy']
    for name, caps, use_smooth in variants:
        m = simulate_reshape(fans, caps, smooth=use_smooth, monotone=True)

    # ---- PERFIL fixo por horizonte (isotônico PAV) ----
    def pav_sorted(vals, side):
        """Pool-adjacent-violators (isotônica) sobre mediana por horizonte."""
        v = np.asarray(vals, dtype=float)
        if side == 'hi':
            pass  # não-decrescente
        else:
            v = 1.0 / v  # para lo: trabalhar em não-decrescente de 1/f
        w = [1] * len(v)
        blocks = v.astype(float).tolist()
        ws = w
        out_v = []
        i = 0
        from collections import deque
        stack = []
        for x, wt in zip(blocks, [1.0] * len(blocks)):
            stack.append([x, wt])
            while len(stack) > 1 and stack[-2][0] >= stack[-1][0]:
                x1, w1 = stack.pop()
                x2, w2 = stack.pop()
                stack.append([(x1 * w1 + x2 * w2) / (w1 + w2), w1 + w2])
        levels = []
        for x, wt in stack:
            levels += [x] * int(wt)
        levels = np.array(levels)
        if side == 'hi':
            return levels
        return 1.0 / levels

    print()
    print("== PERFIL (fatores mediana por h, isotônico PAV, caps) ==")
    profile = {}
    capsprof = {'lo': (0.25, 0.82), 'hi': (1.15, 3.80)}
    for t_u, t_v, side in [(0.05, 'f05', 'lo'), (0.25, 'f25', 'lo'), (0.75, 'f75', 'hi'), (0.95, 'f95', 'hi')]:
        med_h = []
        for h in range(1, 9):
            fx = []
            for fold, o, fan in fans:
                m_ = max(fan[h][0.50], 0.0)
                if m_ > 0.5:
                    fx.append(fan[h][t_u] / m_)
            med_h.append(float(np.median(fx)))
        pav = pav_sorted(med_h, side)
        prof = np.clip(pav, capsprof[side][0], capsprof[side][1] if side == 'hi' else capsprof[side][1])
        if side == 'lo':
            prof = np.minimum(prof, 0.82)
        profile[t_u] = [round(float(p), 3) for p in prof]
        print(f"  {t_v} raw_med={[round(x,2) for x in med_h]} -> profile={profile[t_u]}")

    import copy as _copy
    for name in ['P1 perfil (mediana+h, PAV, caps lo.25/.82 hi1.15/3.8)',
                 'P2 perfil expandido (caps lo.27/.82 hi1.18/3.8 +10% hi)']:
        prof = _copy.deepcopy(profile)
        if name.startswith('P2'):
            for t_u in prof:
                prof[t_u] = [round(v * (1.10 if v > 1.02 else 0.955), 3) if t_u in (0.75, 0.95) else v
                             for v in prof[t_u]]
            caps_use = {'lo': (0.27, 0.80), 'hi': (1.18, 4.10)}
        else:
            caps_use = capsprof
        m = simulate_reshape(fans, caps_use, smooth=use_smooth, monotone=True, profile=prof)
        print(f"== {name} ==")
        for k in ['cov90', 'cov50', 'w90', 'w50', 'wis_proxy', 'mae']:
            d = m[k] - mr[k]
            print(f"  {k}: {m[k]:.3f} ({d:+.3f} | {(m[k]/base_wis-1)*100:+.1f}%" if k == 'wis_proxy' else f"  {k}: {m[k]:.3f} ({d:+.3f})")
        print()
        print(f"== {name} ==")
        for k in ['cov90', 'cov50', 'w90', 'w50', 'wis_proxy', 'mae']:
            d = m[k] - mr[k]
            print(f"  {k}: {m[k]:.3f} ({d:+.3f} | {(m[k]/base_wis-1)*100:+.1f}%" if k == 'wis_proxy' else f"  {k}: {m[k]:.3f} ({d:+.3f})")
        print()

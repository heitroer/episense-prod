"""EXP-P0-3 — variantes de correção do colapso CQR x band_shape.

Compara, no walk-forward 2022-25 com a convenção live (mediana x viés 1.08 x
perfil x CQR), o estado atual contra guardrails que impedem o CQR de destruir
a forma da banda:

  current        perfil + delta CQR como está hoje
  caps           delta CQR + re-aplica caps do band_shape (hi_min/lo_min/lo_max)
  upper_nonneg   deltas de q75/q95 nunca negativos (banda superior só alarga)
  upper_shrink   deltas de q75/q95 encolhidos por lambda
  shift          median shift CQR (produção, pool 4 folds)
  no_conf        perfil puro (sem CQR)

Métricas: MAE (q50), WIS padrão, cov50/cov90, w50/w90.
Uso: .venv/bin/python scripts/experiment_p0_3_cqr.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.calibration import (compute_conformal, make_regime_hook,  # noqa: E402
                                entry_for_key)
from models.metrics import (coverage_interval, mean_interval_width,  # noqa: E402
                            weighted_interval_score)

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
BAND_TAUS = (0.05, 0.25, 0.75, 0.95)


def load_cfg():
    with open(ROOT / 'config/config.yaml') as f:
        return yaml.safe_load(f)


def load_vp():
    with open(ROOT / 'models/validation_predictions.json') as f:
        return json.load(f)


def get_blocks(vp, h, tau):
    d = vp[str(h)]
    key = f"{tau:.2f}" if f"{tau:.2f}" in d else str(tau)
    return d[key]


def profile_blocks(vp, h, tau, profile, mb_factor, mb_hs):
    """Banda perfilada em casos: q* = med x profile[h-1] x (1.08 se h>=2)."""
    pf = profile[tau][h - 1]
    blocks = []
    for b in get_blocks(vp, h, 0.50):
        p50 = np.asarray(b['y_pred_casos'], dtype=float)
        scale = pf * (mb_factor if int(h) in mb_hs else 1.0)
        nb = dict(b)
        nb['y_pred_casos'] = list(np.clip(p50 * scale, 0.0, None))
        blocks.append(nb)
    return blocks


def build_fold_series(vp, profile, mb_factor, mb_hs, apply_hs):
    fs = {}
    for h in apply_hs:
        fs_h = {0.50: get_blocks(vp, h, 0.50)}
        fs_h.update({tau: profile_blocks(vp, h, tau, profile, mb_factor, mb_hs)
                     for tau in BAND_TAUS})
        fs[h] = fs_h
    return fs


def flat_live(vp, h, mb_factor, mb_hs):
    """Rows (fold, se) -> y e quantis LIVE (viés 1.08 em todos os quantis)."""
    d = vp[str(h)]
    out = {}
    bias = mb_factor if int(h) in mb_hs else 1.0
    for tau in QUANTILES:
        key = f"{tau:.2f}" if f"{tau:.2f}" in d else str(tau)
        for blk in d[key]:
            fold = int(blk['fold'])
            for se, yt, yp in zip(blk['test_se'], blk['y_test_casos'], blk['y_pred_casos']):
                r = out.setdefault((fold, str(se).zfill(6)), {})
                if tau == 0.50:
                    r['y'] = float(yt)
                r[tau] = float(yp) * bias
    return out


def apply_entry(qp, entry, *, upper_nonneg=False, upper_shrink=1.0,
                shift_scale=1.0):
    """Aplica entry (deltas + median_shift) sobre a banda em casos qp."""
    q = {t: np.array(v, dtype=float) for t, v in qp.items()}
    if not entry:
        return q
    shift = float(entry.get('median_shift', 0.0) or 0.0) * float(shift_scale)
    deltas = entry.get('deltas') or {}
    for tau in QUANTILES:
        d = 0.0
        if tau != 0.50:
            try:
                d = float(deltas.get(f'{tau:.2f}', 0.0))
            except Exception:
                d = 0.0
            if tau in (0.75, 0.95):
                if upper_shrink != 1.0:
                    d *= upper_shrink
                if upper_nonneg:
                    d = max(d, 0.0)
        q[tau] = np.expm1(np.log1p(np.clip(q[tau], 0.0, None)) + shift + d)
    return q


def apply_caps(q, hi_min=1.18, lo_max=0.80, lo_min=0.27):
    """Re-aplica os caps do band_shape após o conformal (por coluna)."""
    q = {t: np.array(v, dtype=float) for t, v in q.items()}
    med = q[0.50]
    q[0.75] = np.maximum(q[0.75], med * hi_min)
    q[0.95] = np.maximum(q[0.95], q[0.75])
    q[0.25] = np.minimum(q[0.25], med * lo_max)
    q[0.05] = np.maximum(q[0.05], med * lo_min)
    q[0.05] = np.minimum(q[0.05], q[0.25])
    return q


def metrics(y, q):
    return {
        'cov90': coverage_interval(y, q[0.05], q[0.95]),
        'cov50': coverage_interval(y, q[0.25], q[0.75]),
        'w90': mean_interval_width(y, q[0.05], q[0.95]),
        'w50': mean_interval_width(y, q[0.25], q[0.75]),
        'wis': weighted_interval_score(y, q),
        'mae': float(np.mean(np.abs(y - q[0.50]))),
    }


def main():
    cfg = load_cfg()
    vp = load_vp()
    bs = (cfg.get('postprocessing') or {}).get('band_shape') or {}
    profile = {float(k): [float(x) for x in v] for k, v in (bs.get('profile') or {}).items()}
    mb_factor = float((cfg.get('postprocessing') or {}).get('median_bias_factor', 1.08))
    mb_hs = set(int(x) for x in (cfg.get('postprocessing') or {}).get('median_bias_horizons', range(2, 9)))
    qc = cfg.get('quantile_conformal') or {}
    apply_hs = sorted(int(x) for x in qc.get('apply_horizons', [1, 2, 3, 5]))

    base = pd.read_csv(ROOT / 'data/processed/episense_base.csv')
    base['SE'] = base['SE'].astype(str).str.replace('.0', '', regex=False).str.zfill(6)
    casos_by_se = {s: float(v) for s, v in zip(base['SE'], base['casos']) if np.isfinite(float(v))}
    hook = make_regime_hook(casos_by_se, qc.get('regime_thresholds') or None)

    fs = build_fold_series(vp, profile, mb_factor, mb_hs, apply_hs)
    prod = compute_conformal(fs, hook, half_life_years=float(qc.get('half_life_years', 2.5)),
                             max_delta=float(qc.get('max_abs_delta', 1.0)),
                             use_median_shift=False)
    prod_shift = compute_conformal(fs, hook, half_life_years=float(qc.get('half_life_years', 2.5)),
                                   max_delta=float(qc.get('max_abs_delta', 1.0)),
                                   use_median_shift=True,
                                   max_shift=float(qc.get('max_abs_median_shift', 0.25)))

    n_folds = len(get_blocks(vp, 1, 0.50))
    exp = {}
    exp_shift = {}
    for f in range(n_folds):
        fs_f = {}
        for h in apply_hs:
            fs_f[h] = {tau: [b for b in fs[h][tau] if int(b.get('fold', f)) < f]
                       for tau in (0.50, *BAND_TAUS)}
        exp[str(f)] = compute_conformal(fs_f, hook, half_life_years=float(qc.get('half_life_years', 2.5)),
                                        max_delta=float(qc.get('max_abs_delta', 1.0)),
                                        use_median_shift=False) if any(fs_f[h][0.50] for h in apply_hs) else {}
        exp_shift[str(f)] = compute_conformal(fs_f, hook, half_life_years=float(qc.get('half_life_years', 2.5)),
                                              max_delta=float(qc.get('max_abs_delta', 1.0)),
                                              use_median_shift=True,
                                              max_shift=float(qc.get('max_abs_median_shift', 0.25))) if any(fs_f[h][0.50] for h in apply_hs) else {}

    variants = {
        'current': dict(),
        'caps': dict(caps=True),
        'upper_nonneg': dict(upper_nonneg=True),
        'upper_shrink50': dict(upper_shrink=0.5),
        'upper_shrink25': dict(upper_shrink=0.25),
        'shift': dict(shift_doc=True),
        'no_conf': dict(no_conf=True),
    }

    def entry_for(doc, h, se, shift_doc=False):
        d = prod_shift if shift_doc else (prod if doc == 'prod' else exp.get(str(se_fold), {}))
        return entry_for_key(d, int(h), hook(int(h), str(se)))

    per = {name: {} for name in variants}
    for h in range(1, 9):
        rec = flat_live(vp, h, mb_factor, mb_hs)
        rows = sorted(rec.keys())
        y = np.array([rec[k]['y'] for k in rows])
        # banda perfilada live (mediana já com viés 1.08)
        med = np.array([rec[k][0.50] for k in rows], dtype=float)
        qp = {0.50: med}
        for tau in BAND_TAUS:
            qp[tau] = np.clip(med * profile[tau][h - 1], 0.0, None)
        for name, opt in variants.items():
            q = {t: np.array(v, dtype=float) for t, v in qp.items()}
            if h in apply_hs and name != 'no_conf':
                for i, k in enumerate(rows):
                    fold, se = k
                    doc = prod if not opt.get('shift_doc') else prod_shift
                    e = entry_for_key(doc, int(h), hook(int(h), str(se)))
                    if not e:
                        continue
                    shift = float(e.get('median_shift', 0.0) or 0.0)
                    deltas = e.get('deltas') or {}
                    for tau in QUANTILES:
                        d = 0.0
                        if tau != 0.50:
                            try:
                                d = float(deltas.get(f'{tau:.2f}', 0.0))
                            except Exception:
                                d = 0.0
                            if tau in (0.75, 0.95):
                                if opt.get('upper_shrink'):
                                    d *= float(opt['upper_shrink'])
                                if opt.get('upper_nonneg'):
                                    d = max(d, 0.0)
                        q[tau][i] = np.expm1(np.log1p(max(q[tau][i], 0.0)) + shift + d)
            if opt.get('caps'):
                q = apply_caps(q)
            per[name][h] = metrics(y, q)
    # agregados
    for name in variants:
        per[name]['agg'] = {k: float(np.nanmean([per[name][h][k] for h in range(1, 9)]))
                            for k in ('cov90', 'cov50', 'w90', 'w50', 'wis', 'mae')}
    order = ['current', 'caps', 'upper_nonneg', 'upper_shrink50', 'upper_shrink25', 'shift', 'no_conf']
    print("== agregado h1-h8 ==")
    print(f"{'variant':>15} | {'MAE':>8} {'WIS':>8} {'cov50':>7} {'cov90':>7} {'w50':>7} {'w90':>8}")
    for name in order:
        a = per[name]['agg']
        print(f"{name:>15} | {a['mae']:8.2f} {a['wis']:8.2f} {a['cov50']:7.3f} {a['cov90']:7.3f} {a['w50']:7.1f} {a['w90']:8.1f}")
    print("\n== por horizonte (MAE / WIS / cov50 / cov90) ==")
    for h in range(1, 9):
        print(f"\n-- h{h} --")
        for name in order:
            m = per[name][h]
            print(f"  {name:>15}: mae={m['mae']:8.2f} wis={m['wis']:8.2f} cov50={m['cov50']:.3f} cov90={m['cov90']:.3f} w50={m['w50']:7.1f} w90={m['w90']:8.1f}")
    # ratio live no regime mid (base 202644~23 casos) para cada variante
    print("\n== ratio q75/q50 e q95/q50 no regime 'mid' (deltas produção) ==")
    for h in apply_hs:
        print(f"h{h}:")
        for name in order:
            if name == 'no_conf':
                e = {}
            else:
                e = entry_for_key(prod_shift if name == 'shift' else prod, int(h), 'mid')
            ratio75 = profile[0.75][h - 1] * np.exp(float((e.get('deltas') or {}).get('0.75', 0.0)) + float(e.get('median_shift', 0.0) or 0.0))
            ratio95 = profile[0.95][h - 1] * np.exp(float((e.get('deltas') or {}).get('0.95', 0.0)) + float(e.get('median_shift', 0.0) or 0.0))
            # shift aplica em q50 também; ratio correto = (perfil*exp(delta+shift)) / exp(shift)
            if name == 'shift':
                ratio75 = profile[0.75][h - 1] * np.exp(float((e.get('deltas') or {}).get('0.75', 0.0)))
                ratio95 = profile[0.95][h - 1] * np.exp(float((e.get('deltas') or {}).get('0.95', 0.0)))
            if name in ('upper_nonneg', 'upper_shrink50', 'upper_shrink25'):
                d75 = float((e.get('deltas') or {}).get('0.75', 0.0))
                d95 = float((e.get('deltas') or {}).get('0.95', 0.0))
                if name == 'upper_nonneg':
                    d75, d95 = max(d75, 0.0), max(d95, 0.0)
                else:
                    lam = 0.5 if name == 'upper_shrink50' else 0.25
                    d75, d95 = d75 * lam, d95 * lam
                ratio75 = profile[0.75][h - 1] * np.exp(d75)
                ratio95 = profile[0.95][h - 1] * np.exp(d95)
            if name == 'caps':
                ratio75 = max(ratio75, 1.18)
            print(f"  {name:>15}: q75/q50={ratio75:.3f} q95/q50={ratio95:.3f}")


if __name__ == '__main__':
    main()

"""
EXP-Q2 — validação do pipeline live completo:
  raw  |  perfil (band_shape)  |  perfil+conf(pool)  | perfil + conformal (expanding)
Métricas em escala casos (models.metrics), pool walk-forward 2022-25.
O phantom gating live (cap q95 1.5x/2.0x por mês-alvo) não entra aqui —
convenção dos validation results históricos (parity flag off).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.metrics import pinball_mean, coverage_interval, mean_interval_width  # noqa: E402
from models.calibration import make_regime_hook  # noqa: E402

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]


def load_vp():
    with open(ROOT / 'models/validation_predictions.json') as f:
        return json.load(f)


def get_cfg():
    with open(ROOT / 'config/config.yaml') as f:
        return yaml.safe_load(f)


def flat(vp, h):
    d = vp[str(h)]
    out = {}
    for tau in QUANTILES:
        key = f"{tau:.2f}" if f"{tau:.2f}" in d else str(tau)
        for blk in d[key]:
            fold = int(blk['fold'])
            for se, yt, yp in zip(blk['test_se'], blk['y_test_casos'], blk['y_pred_casos']):
                r = out.setdefault((fold, str(se).zfill(6)), {})
                if tau == 0.50:
                    r['y'] = float(yt)
                r[tau] = float(yp)
    return out


def metrics(y, q):
    return {
        'cov90': coverage_interval(y, q[0.05], q[0.95]),
        'cov50': coverage_interval(y, q[0.25], q[0.75]),
        'w90': mean_interval_width(y, q[0.05], q[0.95]),
        'w50': mean_interval_width(y, q[0.25], q[0.75]),
        'wis': sum(pinball_mean(y, q[t], t) for t in QUANTILES),
        'mae': float(np.mean(np.abs(y - q[0.50]))),
    }


def main():
    vp = load_vp()
    cfg = get_cfg()
    bs = (cfg.get('postprocessing') or {}).get('band_shape') or {}
    profile = {float(k): [float(x) for x in v] for k, v in (bs.get('profile') or {}).items()}
    mb_factor = float((cfg.get('postprocessing') or {}).get('median_bias_factor', 1.08))
    mb_hs = set(int(x) for x in (cfg.get('postprocessing') or {}).get('median_bias_horizons', range(2, 9)))
    qc = cfg.get('quantile_conformal') or {}
    apply_hs = set(int(x) for x in qc.get('apply_horizons', [1, 2, 3, 5]))

    with open(ROOT / 'models/quantile_conformal.json') as f:
        conf_doc = json.load(f)
    expanding = conf_doc.get('expanding_by_fold') or {}

    base = pd.read_csv(ROOT / 'data/processed/episense_base.csv')
    base['SE'] = base['SE'].astype(str).str.replace('.0', '', regex=False).str.zfill(6)
    casos_by_se = {s: float(v) for s, v in zip(base['SE'], base['casos']) if np.isfinite(float(v))}
    hook = make_regime_hook(casos_by_se, qc.get('regime_thresholds') or None)

    from models.calibration import entry_for_key

    # entry_for_key espera {str(h): ...}; expanding usa doc {'0': {...}} por fold
    def deltas_for(exp_doc, h, se):
        try:
            entry = entry_for_key(exp_doc, int(h), hook(int(h), str(se)))
        except Exception:
            entry = None
        return (entry or {}).get('deltas') or {}

    import logging
    logging.disable(logging.CRITICAL)

    per_var = {'raw': {}, 'perfil': {}, 'perfil+conformal': {}, 'perfil+conf(pool)': {}}
    with open(ROOT / 'models/quantile_conformal.json') as f:
        prod_doc = json.load(f)
    prod = prod_doc.get('production') or {}
    lines = []
    for h in range(1, 9):
        rec = flat(vp, h)
        rows = sorted(rec.keys())
        y = np.array([rec[k]['y'] for k in rows])
        med_cases = np.array([max(rec[k][0.50], 0.0) for k in rows])
        # perfis
        pf = profile
        med_bias = med_cases * (mb_factor if h in mb_hs else 1.0)
        qp = {0.50: med_cases}
        for tau in [0.05, 0.25, 0.75, 0.95]:
            qp[tau] = np.clip(med_bias * pf[tau][h - 1], 0.0, None)
        # conformal sobre os re-shapeds (h em apply_hs):
        #   pool      = production (pool 4 folds) — comportamento live real
        #   expanding = expanding_by_fold — honesto, por fold
        qc_pool = {t: list(v) for t, v in qp.items()}
        qc_exp = {t: list(v) for t, v in qp.items()}
        if h in apply_hs:
            for i, k in enumerate(rows):
                fold, se = k
                d_exp = deltas_for(expanding.get(str(fold), {}), h, se)
                d_pool = deltas_for(prod, h, se)
                for tau in [0.05, 0.25, 0.75, 0.95]:
                    v = float((d_exp or {}).get(f'{tau:.2f}', np.nan))
                    if np.isfinite(v):
                        qc_exp[tau][i] = np.expm1(np.log1p(qc_exp[tau][i]) + v)
                    v = float((d_pool or {}).get(f'{tau:.2f}', np.nan))
                    if np.isfinite(v):
                        qc_pool[tau][i] = np.expm1(np.log1p(qc_pool[tau][i]) + v)
        for name, q in [('raw', {t: np.array([rec[k][t] for k in rows]) for t in QUANTILES}),
                        ('perfil', {t: np.array(qp[t]) for t in QUANTILES}),
                        ('perfil+conf(pool)', {t: np.array(qc_pool[t]) for t in QUANTILES}),
                        ('perfil+conformal', {t: np.array(qc_exp[t]) for t in QUANTILES})]:
            per_var[name][h] = metrics(y, q)
    # agregado
    for name, by_h in per_var.items():
        hs = [h for h in by_h if h != 'agg']
        by_h['agg'] = {k: float(np.nanmean([by_h[h][k] for h in hs])) for k in
                       ['cov90', 'cov50', 'w90', 'w50', 'wis', 'mae']}
    order = ['raw', 'perfil', 'perfil+conf(pool)', 'perfil+conformal']
    hdr = f"{'métrica':>6} | {'per-h (raw → perfil → pool → expanding)'}"
    print(hdr)
    for k in ['cov90', 'cov50', 'wis', 'mae', 'w90']:
        print(f"\n== {k} ==")
        for h in range(1, 9):
            vals = " | ".join(f"{per_var[n][h][k]:9.3f}" for n in order)
            print(f"  h{h}: {vals}")
        aggs = " | ".join(f"{per_var[n]['agg'][k]:9.3f}" for n in order)
        print(f"  agg média: {aggs}")


if __name__ == '__main__':
    main()

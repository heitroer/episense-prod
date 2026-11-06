"""
EXP-Q2 — Recalibração conformal sobre a banda PERFILADA (post band_shape)

A banda live agora é q* = med × f*[h] (perfil calibrado, models/band_shape.py
+ config postprocessing.band_shape). Os deltas de quantile_conformal.json
estavam calibrados sobre a banda RAW (pré-re-forma): após o perfil, os deltas
antigos deslocam caudas para posições incoerentes (ex.: q75 do h1 abaixo da
mediana com delta −0.27 no regime low). Esta passada regenera os deltas CQR
(por horizonte x regime-base, meia-vida 2.5a, pool dos 4 folds) medindo os
resíduos da BANDA PERFILADA espelhando o pipeline live:
    band_cases = med × profile[h] × (1.08 se h>=2)      # viés FBias +0.05
    delta_τ    = Quantil_w(τ; log1p(y) − log1p(band_cases))
Gera production (pool 4 folds, convenção atual do arquivo) + expanding_by_fold
(fold f calibrado só com folds < f — honesto).

Uso:  .venv/bin/python scripts/recompute_conformal_profile.py [--write]
Sem --write: só imprime (dry-run).
"""
import sys
import copy
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.calibration import (compute_conformal, make_regime_hook,  # noqa
                                save_json)

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
BAND_TAUS = (0.05, 0.25, 0.75, 0.95)

SRC = ROOT / 'models/validation_predictions.json'


def load_blocks(h, tau):
    with open(SRC) as f:
        src = json.load(f)
    d = src[str(h)]
    key = f"{tau:.2f}" if f"{tau:.2f}" in d else str(tau)
    return d[key]


def profile_blocks(h, tau_quant, profile, mb_factor, mb_hs):
    """Blocos {fold,test_se,y_test_casos,y_pred_casos} com a banda reformulada:
    q*_cases = med × profile[h-1] × (1.08 se h in mb_hs) — espelho do live."""
    idx = h - 1
    pf = profile[tau_quant][idx]
    blocks = copy.deepcopy(load_blocks(h, 0.50))
    for b in blocks:
        p50 = np.asarray(b['y_pred_casos'], dtype=float)
        scale = pf * (mb_factor if int(h) in mb_hs else 1.0)
        b['y_pred_casos'] = list(np.clip(p50 * scale, 0.0, None))
    return blocks


def main(write=False):
    cfg = yaml.safe_load(open(ROOT / 'config/config.yaml'))
    bs_cfg = cfg.get('postprocessing', {}).get('band_shape', {}) or {}
    profile = {float(k): [float(x) for x in v] for k, v in (bs_cfg.get('profile') or {}).items()}
    mb_factor = float(cfg.get('postprocessing', {}).get('median_bias_factor', 1.08))
    mb_hs = set(int(x) for x in cfg.get('postprocessing', {}).get('median_bias_horizons', range(2, 9)) or [])
    qc_cfg = cfg.get('quantile_conformal', {}) or {}
    half_life = float(qc_cfg.get('half_life_years', 2.5))
    max_delta = float(qc_cfg.get('max_abs_delta', 1.0))
    apply_hs = sorted(int(x) for x in qc_cfg.get('apply_horizons', [1, 2, 3, 5]))

    base = pd.read_csv(ROOT / 'data/processed/episense_base.csv')
    base['SE'] = base['SE'].astype(str).str.replace('.0', '', regex=False).str.zfill(6)
    casos_by_se = {str(s).zfill(6): float(v) for s, v in zip(base['SE'], base['casos'])
                   if np.isfinite(float(v))}
    hook = make_regime_hook(casos_by_se, (qc_cfg.get('regime_thresholds') or None))

    print("== Recalibração conformal sobre banda perfilada ==")
    print(f"   apply_horizons={apply_hs} half_life={half_life}a max_delta={max_delta}")

    fold_series = {}
    for h in apply_hs:
        fs_h = {0.50: load_blocks(h, 0.50)}
        fs_h.update({tau: profile_blocks(h, tau, profile, mb_factor, mb_hs)
                     for tau in BAND_TAUS})
        fold_series[h] = fs_h

    prod = compute_conformal(fold_series, hook,
                             half_life_years=half_life, max_delta=max_delta,
                             use_median_shift=False)

    n_folds = len(load_blocks(1, 0.50))
    expanding_by_fold = {}
    for f in range(n_folds):
        fs = {}
        for h in apply_hs:
            fs_h = {0.50: [b for b in load_blocks(h, 0.50) if int(b.get('fold', f)) < f]}
            fs_h.update({tau: [b for b in fold_series[h][tau]
                               if int(b.get('fold', f)) < f] for tau in BAND_TAUS})
            fs[h] = fs_h
        n_rows = sum(len(fs[h][tau]) for h in apply_hs for tau in (0.50, *BAND_TAUS))
        if n_rows == 0:
            expanding_by_fold[str(f)] = {}
            continue
        expanding_by_fold[str(f)] = compute_conformal(
            fs, hook, half_life_years=half_life, max_delta=max_delta,
            use_median_shift=False)

    print("\n== deltas PRODUÇÃO (pool 4 folds) ==")
    for h in apply_hs:
        e = prod.get(str(h), {})
        for g in ['high', 'mid', 'low', 'global']:
            if g in e:
                ents = e[g]['deltas']
                print(f"  h{h} {g:>7}: " + "  ".join(f"{k}={v:+.3f}" for k, v in sorted(ents.items()))
                      + f"  (n={e[g]['n_cal']})")

    print("\n== deltas EXPANDING (honesto) ==")
    for h in apply_hs:
        for f in range(n_folds):
            e = expanding_by_fold.get(str(f), {}).get(str(h), {})
            for g in ['high', 'mid', 'low', 'global']:
                if g in e:
                    ents = e[g]['deltas']
                    print(f"  fold{f} h{h} {g:>7}: " + "  ".join(f"{k}={v:+.3f}" for k, v in sorted(ents.items())))

    if write:
        target = ROOT / 'models/quantile_conformal.json'
        bak = target.with_suffix('.json.bak_prereshape')
        if not bak.exists():
            bak.write_text(target.read_text(encoding='utf-8'), encoding='utf-8')
            print(f"\nbackup: {bak}")
        save_json(target, prod, expanding_by_fold)
        cur = json.loads(target.read_text(encoding='utf-8'))
        cur.update({
            'generated_at': datetime.now().isoformat(),
            'version': 3,
            'description': ("CQR delta-only por horizonte x regime-base RECALIBRADO sobre a "
                            "banda PERFILADA (EXP-Q2 band_shape) — espelho do pipeline live "
                            "(banda = med x perfil x viés 1.08 p/ h>=2). Regime-base = "
                            "intensidade da semana-base; meia-vida de pesos; mediana sem delta."),
        })
        target.write_text(json.dumps(cur, indent=1, ensure_ascii=False), encoding='utf-8')
        print(f"\nescrito: {target}")
    else:
        print("\n(dry-run — use --write para gravar models/quantile_conformal.json)")


if __name__ == '__main__':
    main(write='--write' in sys.argv)

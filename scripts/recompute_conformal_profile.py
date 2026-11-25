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
Gera production (alvos maduros, convenção atual do arquivo) + expanding_by_fold
(fold f calibrado com desfechos maduros de folds anteriores).

Uso:  .venv/bin/python scripts/recompute_conformal_profile.py [--write]
Sem --write: só imprime (dry-run).
"""
import sys
import json
import os
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

MODEL_DIR = ROOT / os.environ.get('EPISENSE_MODELS_DIR', 'models')
SRC = MODEL_DIR / 'validation_predictions.json'


def calibrate_profile(vp, cfg, base):
    from data.epiweeks import add_epiweeks
    from models.validation_calendar import mature_calibration_series, origin_weeks

    from models.validation_calendar import check_observed_targets
    check_observed_targets(vp, base)
    post = cfg["postprocessing"]
    profile = {float(k): v for k, v in post["band_shape"]["profile"].items()}
    qc = cfg["quantile_conformal"]
    maturity = cfg["training"]["walk_forward"]["label_maturity_weeks"]
    cases = dict(zip(base["SE"].astype(str).str.zfill(6), base["casos"]))
    hook = make_regime_hook(cases, qc.get("regime_thresholds"))
    cutoffs = {}
    series = {}
    for key, table in vp.items():
        h = int(key)
        medians = next(blocks for tau, blocks in table.items() if float(tau) == 0.5)
        for block in medians:
            fold = int(block["fold"])
            start = min(origin_weeks(block))
            cutoffs[fold] = min(start, cutoffs.get(fold, start))
        if h not in qc["apply_horizons"]:
            continue
        series[h] = {0.5: medians}
        bias = post.get("median_bias_factor", 1.08) if h in post.get("median_bias_horizons", range(2, 9)) else 1.
        for tau in BAND_TAUS:
            scale = profile[tau][h - 1] * bias
            series[h][tau] = [
                {**b, "y_pred_casos": np.maximum(0., np.asarray(b["y_pred_casos"]) * scale).tolist()}
                for b in medians
            ]

    def fit(cutoff, before_fold=None):
        pool = mature_calibration_series(series, cutoff, maturity, before_fold)
        return compute_conformal(pool, hook, half_life_years=qc.get("half_life_years", 2.5),
                                 max_delta=qc.get("max_abs_delta", 1.), use_median_shift=False)

    expanding = {str(f): fit(start, f) for f, start in cutoffs.items()}
    return fit(add_epiweeks(max(cases), 1)), expanding


def main(write=False):
    cfg = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    base = pd.read_csv(ROOT / os.environ.get("EPISENSE_TRAINING_BASE", "data/processed/episense_base.csv"))
    prod, expanding_by_fold = calibrate_profile(json.loads(SRC.read_text()), cfg, base)
    apply_hs = cfg["quantile_conformal"]["apply_horizons"]
    print("\n== deltas PRODUÇÃO (alvos maduros) ==")
    for h in apply_hs:
        e = prod.get(str(h), {})
        for g in ['high', 'mid', 'low', 'global']:
            if g in e:
                ents = e[g]['deltas']
                print(f"  h{h} {g:>7}: " + "  ".join(f"{k}={v:+.3f}" for k, v in sorted(ents.items()))
                      + f"  (n={e[g]['n_cal']})")

    print("\n== deltas EXPANDING (honesto) ==")
    for h in apply_hs:
        for f in expanding_by_fold:
            e = expanding_by_fold.get(str(f), {}).get(str(h), {})
            for g in ['high', 'mid', 'low', 'global']:
                if g in e:
                    ents = e[g]['deltas']
                    print(f"  fold{f} h{h} {g:>7}: " + "  ".join(f"{k}={v:+.3f}" for k, v in sorted(ents.items())))

    if write:
        target = MODEL_DIR / 'quantile_conformal.json'
        bak = target.with_suffix('.json.bak_prereshape')
        if target.exists() and not bak.exists():
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

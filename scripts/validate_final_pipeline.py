"""Compara quantis brutos, perfilados e finais usando a transformação da publicação."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collections import defaultdict
from scripts.evaluate_walk_forward import calibration_blocks, evaluate
from data.epiweeks import add_epiweeks
from data.origin_inputs import issued_inputs
from models.calibration import compute_conformal
from models.forecast_audit import forecast_rows, QUANTILES as NAMES
from models.inference import EpisenseInference
from models.releases import active_models
from models.validation_calendar import origin_weeks, label_available_week
from models.metrics import weighted_interval_score, coverage_interval, mean_interval_width  # noqa: E402
from models.calibration import make_regime_hook  # noqa: E402

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]


def flat(vp, h):
    d = vp[str(h)]
    out = {}
    for tau in QUANTILES:
        key = f"{tau:.2f}" if f"{tau:.2f}" in d else str(tau)
        for blk in d[key]:
            fold = int(blk['fold'])
            for se, yt, yp in zip(origin_weeks(blk), blk['y_test_casos'], blk['y_pred_casos']):
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
        'wis': weighted_interval_score(y, q),
        'mae': float(np.mean(np.abs(y - q[0.50]))),
    }


def stage_rows(vp, cfg, base):
    from models.validation_calendar import check_observed_targets
    check_observed_targets(vp, base)
    engine = EpisenseInference(config=cfg, load_models=False)
    maturity = cfg["training"]["walk_forward"]["label_maturity_weeks"]
    cutoff = add_epiweeks(max(base["SE"]), 1)
    fans = defaultdict(dict)
    records, shaped, starts = {}, {}, {}
    for h in map(int, vp):
        for (fold, origin), record in flat(vp, h).items():
            fans[fold, origin][h] = {t: record[t] for t in QUANTILES}
            target = add_epiweeks(origin, h)
            records[fold, origin, h] = {"actual": record["y"], "target": target,
                                        "available": label_available_week(target, maturity)}
            starts[fold] = min(origin, starts.get(fold, origin))

    def logits(fan):
        return {h: {t: np.log1p([value]) for t, value in band.items()} for h, band in fan.items()}

    issued = any("input_sha256" in b for table in vp.values() for blocks in table.values() for b in blocks)
    if issued:
        _, _, expanding = evaluate(vp, base, cfg)
        cases = {origin: float(frame["casos"].iloc[-1])
                 for origin, _, _, frame, _ in issued_inputs(cfg, {o for _, o in fans})}
    else:
        # Replay of historical releases, using their frozen consolidated input convention.
        cases = base.set_index("SE")["casos"].to_dict()
        qc = engine.quantile_conformal_config
        for (fold, origin), fan in fans.items():
            for h, band in engine.shape_predictions(logits(fan), origin).items():
                if h in qc.get("apply_horizons", []):
                    shaped[fold, origin, h] = {t: float(np.expm1(v[0])) for t, v in band.items()}
        hook = make_regime_hook(cases, qc.get("regime_thresholds"))
        expanding = {str(f): compute_conformal(
            calibration_blocks(shaped, records, cutoff=start, before_fold=f), hook,
            half_life_years=qc.get("half_life_years", 2.5), max_delta=qc.get("max_abs_delta", 1.0),
            use_median_shift=False) for f, start in starts.items()}

    for (fold, origin), fan in sorted(fans.items()):
        engine.conformal = expanding[str(fold)]
        profile = engine.shape_predictions(logits(fan), origin)
        final = engine.postprocess_predictions(logits(fan), origin, cases[origin])
        for name, values in (("raw", logits(fan)), ("perfil", profile), ("publicado", final)):
            for row in forecast_rows(origin, values):
                rec = records[fold, origin, row["h"]]
                if rec["available"] <= cutoff:
                    yield name, fold, origin, rec["actual"], row


def main():
    import argparse
    import logging
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=active_models(ROOT / "models"))
    directory = parser.parse_args().models_dir
    vp = json.loads((directory / "validation_predictions.json").read_text())
    cfg = yaml.safe_load((directory / "training_config.yaml").read_text())
    base = pd.read_csv(directory / "training_base.csv", dtype={"SE": str})
    logging.getLogger("models.inference").setLevel(logging.ERROR)
    if not any("input_sha256" in b for t in vp.values() for blocks in t.values() for b in blocks):
        print("Diagnóstico histórico com dados consolidados da versão.")
    grouped = defaultdict(list)
    for name, _, _, actual, row in stage_rows(vp, cfg, base):
        grouped[name, row["h"]].append((actual, row))
    per_var = defaultdict(dict)
    for (name, h), rows in grouped.items():
        y = np.array([y for y, _ in rows])
        q = {t: np.array([r[key] for _, r in rows], dtype=float) for t, key in NAMES.items()}
        per_var[name][h] = metrics(y, q)
    # agregado
    for name, by_h in per_var.items():
        hs = [h for h in by_h if h != 'agg']
        by_h['agg'] = {k: float(np.nanmean([by_h[h][k] for h in hs])) for k in
                       ['cov90', 'cov50', 'w90', 'w50', 'wis', 'mae']}
    order = ['raw', 'perfil', 'publicado']
    hdr = f"{'métrica':>6} | {'per-h (raw → perfil → publicado)'}"
    print(hdr)
    for k in ['cov90', 'cov50', 'wis', 'mae', 'w90']:
        print(f"\n== {k} ==")
        for h in sorted(h for h in per_var["publicado"] if h != "agg"):
            vals = " | ".join(f"{per_var[n][h][k]:9.3f}" for n in order)
            print(f"  h{h}: {vals}")
        aggs = " | ".join(f"{per_var[n]['agg'][k]:9.3f}" for n in order)
        print(f"  agg média: {aggs}")


if __name__ == '__main__':
    main()

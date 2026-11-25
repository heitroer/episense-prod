"""Avalia os quantis do walk-forward com a cadeia de publicação do Episense."""
import argparse
from collections import defaultdict
import json
import logging
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.config import load_config
from data.epiweeks import add_epiweeks
from data.origin_inputs import issued_inputs
from models.calibration import compute_conformal, make_regime_hook, save_json
from models.forecast_audit import QUANTILES, forecast_rows
from models.inference import EpisenseInference
from models.metrics import paired_forecast_metrics
from models.validation_calendar import label_available_week, origin_weeks, target_weeks


def read_records(artifact, maturity, *, revised=False):
    forecasts = {}
    for h, table in artifact.items():
        h = int(h)
        for quantile, blocks in table.items():
            tau = float(quantile)
            for block in blocks:
                fold = int(block["fold"])
                origins = origin_weeks(block)
                targets = target_weeks(block, h)
                expected = [label_available_week(target, maturity) for target in targets]
                available = block.get("target_available_se", expected)
                if list(available) != expected:
                    raise ValueError(f"Disponibilidade incompatível no fold {fold}, H{h}")
                fit_cutoff = block.get("fit_cutoff_se")
                outer_cutoff = block.get("test_cutoff_se")
                if fit_cutoff and block["last_fit_target_available_se"] > fit_cutoff:
                    raise ValueError(f"Rótulo futuro no treino interno, fold {fold}, H{h}")
                if outer_cutoff and (outer_cutoff != origins[0]
                                     or block["last_validation_target_available_se"] > outer_cutoff):
                    raise ValueError(f"Rótulo futuro na validação interna, fold {fold}, H{h}")
                digests = block.get("input_sha256")
                if revised:
                    if digests is not None or block.get("historical_inputs") != "revised":
                        raise ValueError("Proveniência incompatível com o treino revisado")
                    digests = [None] * len(origins)
                if not isinstance(digests, list) or len(digests) != len(origins):
                    raise ValueError("Previsões históricas sem snapshots de origem verificáveis")
                values = block["y_pred_casos"]
                actuals = block["y_test_casos"]
                if len({len(origins), len(targets), len(available), len(values), len(actuals)}) != 1:
                    raise ValueError(f"Séries de tamanhos diferentes no fold {fold}, H{h}")
                for origin, target, ready, value, actual, digest in zip(origins, targets, available, values, actuals, digests):
                    key = (fold, origin, h)
                    row = forecasts.setdefault(key, {
                        "target": target, "available": ready, "actual": float(actual),
                        "digest": digest, "raw": {}, "embargo_manifest": bool(fit_cutoff and outer_cutoff),
                    })
                    if row["digest"] != digest or tau in row["raw"] or not np.isclose(row["actual"], float(actual), equal_nan=True):
                        raise ValueError(f"Quantil duplicado ou alvo divergente: {key}, q{tau}")
                    row["raw"][tau] = float(value)
    return forecasts


def calibration_blocks(shaped, forecasts, *, cutoff, before_fold=None):
    grouped = defaultdict(list)
    for (fold, origin, h), bands in shaped.items():
        row = forecasts[(fold, origin, h)]
        if row["available"] > cutoff or (before_fold is not None and fold >= before_fold):
            continue
        grouped[(h, fold)].append((origin, row, bands))
    series = defaultdict(lambda: defaultdict(list))
    for (h, fold), rows in sorted(grouped.items()):
        for tau in QUANTILES:
            series[h][tau].append({
                "fold": fold,
                "origin_se": [origin for origin, _, _ in rows],
                "target_se": [row["target"] for _, row, _ in rows],
                "y_test_casos": [row["actual"] for _, row, _ in rows],
                "y_pred_casos": [bands[tau] for _, _, bands in rows],
            })
    return series


def evaluate(artifact, base, config, *, scored_rows=None, history=None):
    from models.validation_calendar import check_observed_targets
    check_observed_targets(artifact, base)
    maturity = int(config["training"]["walk_forward"].get("label_maturity_weeks", 16))
    cutoff = add_epiweeks(max(base["SE"]), 1)
    if base["SE"].duplicated().any():
        raise ValueError("A base contém semanas duplicadas")
    observed = base if "is_imputed" not in base else base.loc[~base["is_imputed"].fillna(False)]
    cases = observed.dropna(subset=["casos"]).set_index("SE")["casos"].to_dict()
    engine = EpisenseInference(config=config, load_models=False)
    if engine.quantile_recalibration_enabled:
        raise ValueError("Disponibilize calibração legada por fold antes de avaliar essa configuração")
    revised = config["training"].get("historical_inputs") == "revised"
    forecasts = read_records(artifact, maturity, revised=revised)
    fans = defaultdict(dict)
    cutoffs = {}
    for (fold, origin, h), row in sorted(forecasts.items()):
        cutoffs[fold] = min(origin, cutoffs.get(fold, origin))
        if (set(row["raw"]) != QUANTILES.keys()
                or not all(np.isfinite(value) and value > -1 for value in row["raw"].values())
                or not np.isfinite(row["actual"])):
            continue
        fans[(fold, origin)][h] = row["raw"]

    def to_logits(fan):
        return {h: {tau: np.log1p([value]) for tau, value in band.items()} for h, band in fan.items()}

    qc = engine.quantile_conformal_config
    horizons = {int(h) for h in qc.get("apply_horizons", [1, 2, 3])}
    shaped = {}
    if engine.quantile_conformal_enabled:
        for (fold, origin), fan in fans.items():
            bands = engine.shape_predictions(to_logits(fan), origin)
            for h, quantiles in bands.items():
                if h in horizons:
                    shaped[(fold, origin, h)] = {
                        tau: float(np.expm1(values[0])) for tau, values in quantiles.items()
                    }

    origins = {origin for _, origin, _ in forecasts}
    regimes, persistence, seasonal_pools = {}, {}, {}
    by_origin = defaultdict(list)
    for (fold, origin, h), row in forecasts.items():
        by_origin[origin].append((fold, row))
    def revised_inputs():
        from data.observations import observed_values
        for origin in sorted(origins):
            frame = base.loc[base["SE"] <= origin].sort_values("SE")
            yield origin, origin, None, frame, observed_values(frame).dropna()

    inputs = revised_inputs() if revised else issued_inputs(config, origins)
    for origin, issued_week, digest, frame, observed_cases in inputs:
        regimes[origin] = float(frame["casos"].iloc[-1])
        persistence[origin] = float(observed_cases.iloc[-1])
        for fold, row in by_origin[origin]:
            if row["digest"] != digest or issued_week >= row["target"]:
                raise ValueError(f"Snapshot divergente ou previsão tardia em {origin}")
            key = (fold, origin)
            if key not in seasonal_pools:
                pools = defaultdict(list)
                for idx, value in observed_cases.items():
                    week = frame.at[idx, "SE"]
                    if label_available_week(week, maturity) <= cutoffs[fold] and np.isfinite(value) and value >= 0:
                        pools[week[4:]].append(value)
                seasonal_pools[key] = pools
    if origins != set(regimes):
        raise ValueError("Faltam snapshots das emissões usadas no walk-forward")
    hook = make_regime_hook(regimes, qc.get("regime_thresholds"))

    def calibration(at, before_fold=None):
        series = calibration_blocks(shaped, forecasts, cutoff=at, before_fold=before_fold)
        return compute_conformal(series, hook, half_life_years=float(qc.get("half_life_years", 2.5)),
                                 max_delta=float(qc.get("max_abs_delta", 1.0)),
                                 use_median_shift=False)

    expanding = {str(fold): calibration(at, fold) for fold, at in cutoffs.items()}
    production = calibration(cutoff)
    published_rows = {}
    for (fold, origin), fan in fans.items():
        engine.conformal = expanding[str(fold)]
        published = forecast_rows(origin, engine.postprocess_predictions(to_logits(fan), origin, regimes[origin]))
        for row in published:
            published_rows[(fold, origin, row["h"])] = row

    if history is not None:
        for h, table in artifact.items():
            history[h] = {}
            for quantile, blocks in table.items():
                name = QUANTILES[float(quantile)]
                history[h][quantile] = [
                    {**block, "y_pred_casos": [
                        published_rows[(int(block["fold"]), origin, int(h))][name]
                        for origin in origin_weeks(block)
                    ]}
                    for block in blocks
                ]

    scores = defaultdict(list)
    for (fold, origin, h), row in sorted(forecasts.items()):
        if row["available"] > cutoff:
            continue
        published = published_rows.get((fold, origin, h), {})
        pool = seasonal_pools[(fold, origin)].get(row["target"][4:], [])
        seasonal = np.quantile(pool, list(QUANTILES)) if pool else [np.nan] * len(QUANTILES)
        actual = cases.get(row["target"], np.nan)
        if np.isfinite(actual) and not np.isclose(row["actual"], actual, rtol=1e-9, atol=1e-7):
            raise ValueError(f"Desfecho divergente da base congelada: {row['target']}")
        if scored_rows is not None and published and row["target"] in cases:
            scored_rows.append((fold, origin, row["available"], published, actual))
        scores[(h, fold)].append((
            actual,
            [published.get(name) for name in QUANTILES.values()],
            persistence[origin],
            seasonal,
        ))

    def summarize(rows):
        return paired_forecast_metrics(*zip(*rows))

    results = defaultdict(dict)
    by_horizon = defaultdict(list)
    for (h, fold), rows in sorted(scores.items()):
        results[h][fold] = summarize(rows)
        by_horizon[h].extend(rows)

    aggregate = {h: summarize(rows) for h, rows in sorted(by_horizon.items())}
    report = {
        "schema_version": 2,
        "scope": "retrospective_revised_inputs" if revised else "issued_inputs_consolidated_outcomes",
        "evaluation_role": "model_development",
        "baseline_persistence": "última contagem observada na emissão",
        "baseline_seasonal": "quantis empíricos da semana-alvo em anos anteriores, maduros no corte do fold e observados no snapshot da emissão",
        "wis_scale": "standard",
        "aggregation": "pooled_rows",
        "label_availability": "proxy_week_after_target_end_plus_maturity",
        "label_maturity_weeks": maturity,
        "evaluation_cutoff_se": cutoff,
        "fold_cutoff_se": cutoffs,
        "training_embargo_verified": bool(forecasts) and all(
            row["embargo_manifest"] for row in forecasts.values()
        ),
        "n_input_records": len(forecasts),
        "n_eligible": sum(len(rows) for rows in scores.values()),
        "n_scored": sum(row["n_scored"] for row in aggregate.values()),
        "by_horizon": aggregate,
        "by_horizon_and_fold": results,
    }
    if revised:
        report["baseline_persistence"] = "última contagem observada na base revisada até a origem"
        report["baseline_seasonal"] = "quantis sazonais da base revisada com maturação no corte do fold"
    return report, production, expanding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=ROOT / os.environ.get("EPISENSE_VP_DIR", "models"))
    parser.add_argument("--base", type=Path, default=ROOT / os.environ.get(
        "EPISENSE_TRAINING_BASE", "data/processed/episense_base.csv"))
    parser.add_argument("--write", action="store_true", help="Grava os resultados e a calibração no diretório indicado")
    parser.add_argument("--report-only", type=Path, help="Salva o relatório sem recalibrar a versão ativa")
    args = parser.parse_args()
    for module in ("models.inference", "models.band_shape"):
        logging.getLogger(module).setLevel(logging.ERROR)
    base = pd.read_csv(args.base, dtype={"SE": str})
    artifact = json.loads((args.models_dir / "validation_predictions.json").read_text())
    cfg = load_config(os.environ.get("EPISENSE_TRAINING_CONFIG", str(ROOT / "config/config.yaml")))
    history = {} if args.write else None
    report, production, expanding = evaluate(artifact, base, cfg, history=history)
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.report_only:
        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.report_only.parent,
                                delete=False) as handle:
            handle.write(payload)
            path = Path(handle.name)
        try:
            path.chmod(0o644)
            path.replace(args.report_only)
        finally:
            path.unlink(missing_ok=True)
    if args.write:
        if report["n_scored"] == 0 or (
            cfg.get("quantile_conformal", {}).get("enabled") and not production
        ):
            raise ValueError("Avaliação ou calibração sem dados elegíveis")
        (args.models_dir / "walk_forward_evaluation.json").write_text(payload)
        (args.models_dir / "validation_history.json").write_text(json.dumps(history, allow_nan=False))
        save_json(args.models_dir / "quantile_conformal.json", production, expanding)
    print(payload)


if __name__ == "__main__":
    main()

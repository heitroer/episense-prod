"""Treina o candidato retrospectivo nos municípios configurados. Execute como módulo a partir da raiz."""

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

from data.epiweeks import add_epiweeks
from data.observations import observed_values
from models.growth_calibration import QCOLS, replay_calibration
from models.growth_forecast import (
    GrowthModel,
    PanelGrowth,
    combine_bands,
)
from models.metrics import QUANTILES, compute_all_metrics
from models.validation_calendar import label_available_week, target_weeks

ROOT = Path(__file__).resolve().parents[1]
CITIES = {
    "campo_grande": (
        "data/processed/episense_base.csv",
        "config/config.yaml",
        "models/current",
    ),
    "corumba": (
        "cities/5003207/processed/episense_base.csv",
        "cities/5003207/config.yaml",
        "cities/5003207/pilot",
    ),
    "dourados": (
        "cities/5003702/processed/episense_base.csv",
        "cities/5003702/config.yaml",
        "cities/5003702/pilot",
    ),
}
REGION = (
    "5001102",
    "5003207",
    "5005400",
    "5005707",
    "5006200",
    "5006606",
    "5007901",
    "5008305",
)


def snapshot_references(inputs):
    for city, (_, _, model_dir) in CITIES.items():
        dest = inputs / f"reference_{city}"
        dest.mkdir()
        source = (ROOT / model_dir).resolve()
        if (source / "current").is_symlink():
            source = (source / "current").resolve()
        for name in ("validation_history.json", "walk_forward_evaluation.json"):
            shutil.copyfile(source / name, dest / name)


def snapshot_inputs(output):
    from io import BytesIO

    inputs = output / "inputs"
    inputs.mkdir()
    snapshot_references(inputs)
    for city, (base, config, _) in CITIES.items():
        shutil.copyfile(ROOT / base, inputs / f"{city}.csv")
        cfg = yaml.safe_load((ROOT / config).read_text())
        cfg["training"]["historical_inputs"] = "revised"
        (inputs / f"{city}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        cfg["training"]["walk_forward"]["test_years"] = [2019, 2020, 2021]
        (inputs / f"{city}_warmup.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False)
        )
    weeks = pd.read_csv(inputs / "campo_grande.csv").SE
    for code in REGION:
        if code == "5003207":  # O painel já usa a base operacional de Corumbá.
            continue
        params = {
            "geocode": code,
            "disease": "dengue",
            "format": "csv",
            "ew_start": weeks.min() % 100,
            "ey_start": weeks.min() // 100,
            "ew_end": weeks.max() % 100,
            "ey_end": weeks.max() // 100,
        }
        response = requests.get(
            "https://info.dengue.mat.br/api/alertcity", params=params, timeout=60
        )
        response.raise_for_status()
        peer = pd.read_csv(
            BytesIO(response.content), usecols=["SE", "casos", "municipio_nome", "pop"]
        )
        peer.to_csv(inputs / f"{code}.csv", index=False)


def train_trees(output):
    for city in CITIES:
        for suffix in ("", "_warmup"):
            name = city + suffix
            cfg = output / "inputs" / f"{name}.yaml"
            base = output / "inputs" / f"{city}.csv"
            dest = output / name
            env = {k: v for k, v in os.environ.items() if not k.startswith("EPISENSE_")}
            env.update(
                EPISENSE_MODELS_DIR=str(dest),
                EPISENSE_TRAINING_CONFIG=str(cfg),
                EPISENSE_TRAINING_BASE=str(base),
            )
            commands = [
                ("scripts.train", [], name),
                (
                    "scripts.evaluate_walk_forward",
                    ["--models-dir", str(dest), "--base", str(base), "--write"],
                    name + "-evaluation",
                ),
            ]
            for module, args, log in commands:
                with (output / f"{log}.log").open("w") as handle:
                    subprocess.run(
                        [sys.executable, "-m", module, *args],
                        cwd=ROOT,
                        env=env,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )


def read_base(path):
    frame = (
        pd.read_csv(path, dtype={"SE": str}).sort_values("SE").reset_index(drop=True)
    )
    if frame.empty or any(
        add_epiweeks(a, 1) != b for a, b in zip(frame.SE, frame.SE[1:])
    ):
        raise ValueError(f"Grade semanal irregular em {path}")
    return frame


def regional_inputs(inputs):
    cities = {c: read_base(inputs / f"{c}.csv") for c in CITIES if (inputs / f"{c}.csv").exists()}
    counts = {c: observed_values(f.set_index("SE")) for c, f in cities.items()}
    for code in REGION:
        if code == "5003207" and "corumba" in cities:
            continue
        peer = read_base(inputs / f"{code}.csv")
        counts[code] = peer.set_index("SE").casos
    panel = pd.DataFrame(counts).sort_index()
    return cities, panel


def read_history(directory):
    history = json.loads((directory / "validation_history.json").read_text())
    report = json.loads((directory / "walk_forward_evaluation.json").read_text())
    return history, report


def history_rows(history, h):
    table = history[str(h)]
    for b, med in enumerate(table["0.5"]):
        origins = med["origin_se"]
        targets = target_weeks(med, h)
        columns = []
        for q in QUANTILES:
            block = table[str(q)][b]
            if (
                block["fold"] != med["fold"]
                or block["origin_se"] != origins
                or target_weeks(block, h) != targets
                or block["y_test_casos"] != med["y_test_casos"]
            ):
                raise ValueError("Quantis em coortes diferentes")
            columns.append(block["y_pred_casos"])
        bands = np.column_stack(columns)
        if (
            not np.isfinite(bands).all()
            or (bands < 0).any()
            or (np.diff(bands, axis=1) < 0).any()
        ):
            raise ValueError("Quantis de referência inválidos")
        for i, origin in enumerate(origins):
            yield (
                med["fold"],
                origin,
                targets[i],
                round(med["y_test_casos"][i]),
                bands[i],
            )


def evaluate_city(city, frame, panel, output):
    index = {w: i for i, w in enumerate(frame.SE)}
    published, report = read_history(output / "inputs" / f"reference_{city}")
    runs = [read_history(output / (city + "_warmup")), read_history(output / city)]
    maturity = report["label_maturity_weeks"]
    for _, meta in runs:
        if (
            meta["label_maturity_weeks"] != maturity
            or not meta["training_embargo_verified"]
            or meta["scope"] != "retrospective_revised_inputs"
        ):
            raise ValueError("Corte ou procedência incompatível")
    peer = "dourados" if city == "campo_grande" else "campo_grande"
    destination = output / "candidates" / city
    destination.mkdir(parents=True, exist_ok=True)
    candidate_history = copy.deepcopy(runs[1][0])
    observed = observed_values(frame).to_numpy()
    audit = {}
    records = []
    components = []
    for h in range(1, 9):
        regional = PanelGrowth(frame, panel, city, h, maturity)
        trend = GrowthModel(frame, panel[peer], h, maturity)
        benchmark = {
            (f, o, t): (y, band) for f, o, t, y, band in history_rows(published, h)
        }
        seen = set()
        for phase, (history, meta) in zip(("warmup", "evaluation"), runs):
            for fold, origin, target, y, local in history_rows(history, h):
                cutoff = meta["fold_cutoff_se"][str(fold)]
                if origin < cutoff or observed[index[target]] != y:
                    raise ValueError(f"Alinhamento inválido {city} {origin} H{h}")
                i = index[origin]
                transfer = regional.predict(i, index[cutoff])
                trajectory = trend.predict(i, index[cutoff])
                raw = combine_bands(
                    local, transfer, trajectory, trend.regional_agreement(i)
                )
                ready = label_available_week(target, maturity)
                for method, band in (
                    ("municipal", local),
                    ("regional", transfer),
                    ("tendencia", trajectory),
                    ("combinado", raw),
                ):
                    components.append(
                        [
                            phase,
                            method,
                            h,
                            fold,
                            origin,
                            target,
                            ready,
                            cutoff,
                            y,
                            *band,
                        ]
                    )
                if phase == "evaluation":
                    key = (fold, origin, target)
                    if key in seen or key not in benchmark or benchmark[key][0] != y:
                        raise ValueError(f"Coorte divergente {key}")
                    seen.add(key)
                    for method, band in [
                        ("atual", benchmark[key][1]),
                        ("retreino_local", local),
                    ]:
                        records.append(
                            [
                                city,
                                method,
                                h,
                                fold,
                                origin,
                                target,
                                y,
                                ready <= report["evaluation_cutoff_se"],
                                cutoff,
                                None,
                                None,
                                *band,
                            ]
                        )
        if seen != set(benchmark):
            raise ValueError(f"Semanas ausentes {city} H{h}")
        # Fit the final components on all mature outcomes, after scoring the folds.
        last = len(frame) - 1
        current = frame.SE.iloc[last]
        regional.predict(last, last)
        trend.center(last, last)
        stop = last - trend.delay + 1
        np.savez_compressed(
            destination / f"growth_h{h}.npz",
            fit_cutoff_se=current,
            coefficients=trend.coefficients[last],
            historical_features=trend.features[:stop],
            calibration_errors=trend.errors[:stop],
        )
        for q, booster in zip(QUANTILES, regional.models[current]):
            booster.save_model(
                str(destination / f"regional_h{h}_q{int(100 * q):02d}.txt")
            )
    component_frame = pd.DataFrame(
        components,
        columns=[
            "phase",
            "method",
            "h",
            "fold",
            "origin",
            "target",
            "available",
            "cutoff",
            "y",
            *QCOLS,
        ],
    )
    component_frame.to_csv(destination / "component_forecasts.csv", index=False)
    calibrated = replay_calibration(component_frame, frame, maturity)
    bands = {}
    for r in calibrated.itertuples(index=False):
        band = [getattr(r, c) for c in QCOLS]
        bands[r.h, r.fold, r.origin] = band
        records.append(
            [
                city,
                "candidato",
                r.h,
                r.fold,
                r.origin,
                r.target,
                r.y,
                r.available <= report["evaluation_cutoff_se"],
                r.cutoff,
                r.calibration_n,
                r.last_calibration_available_se,
                *band,
            ]
        )
    for h in range(1, 9):
        for k, q in enumerate(QUANTILES):
            for block in candidate_history[str(h)][str(q)]:
                fold = block["fold"]
                block["y_pred_casos"] = [
                    int(bands[h, fold, o][k]) for o in block["origin_se"]
                ]
                block["forecast_method"] = "regional_growth_ensemble"
                block["historical_inputs"] = "revised"
                if k == 0:
                    rows = calibrated[(calibrated.h == h) & (calibrated.fold == fold)]
                    audit.setdefault(str(h), {})[str(fold)] = {
                        "model_fit_cutoff": block["test_cutoff_se"],
                        "last_training_target_se": add_epiweeks(
                            block["test_cutoff_se"], -maturity - 1
                        ),
                        "calibration_update": "weekly_matured_only",
                        "calibration_n_first": int(rows.calibration_n.iloc[0]),
                        "last_calibration_at_first_origin": int(
                            rows.last_calibration_available_se.iloc[0]
                        ),
                    }
    (destination / "validation_history.json").write_text(
        json.dumps(candidate_history, allow_nan=False)
    )
    (destination / "temporal_audit.json").write_text(json.dumps(audit, indent=2))
    result = pd.DataFrame(
        records,
        columns=[
            "city",
            "method",
            "h",
            "fold",
            "origin",
            "target",
            "y",
            "mature",
            "cutoff",
            "calibration_n",
            "last_calibration_available_se",
            *QCOLS,
        ],
    )
    result.to_csv(output / f"{city}_forecasts.csv", index=False)
    return result


def score(group):
    q = dict(zip(QUANTILES, group[QCOLS].to_numpy().T))
    met = compute_all_metrics(group.y.to_numpy(), q)
    return {
        "n": len(group),
        **{
            k: met[k]
            for k in (
                "mae",
                "wis_total",
                "bias",
                "coverage_50",
                "coverage_90",
                "width_50",
                "width_90",
            )
        },
    }


def save_metrics(rows, output):
    mature = rows.loc[rows.mature].copy()
    threshold = mature.groupby(["city", "method", "h", "fold"]).y.transform(
        lambda x: x.quantile(0.9)
    )
    records = []
    for scope, data in [("all", mature), ("top10", mature[mature.y >= threshold])]:
        for dims, groups in [
            (["city", "method", "h", "fold"], data),
            (["city", "method", "h"], data),
            (["city", "method"], data),
        ]:
            for keys, group in groups.groupby(dims):
                records.append(
                    dict(scope=scope, **dict(zip(dims, keys)), **score(group))
                )
    pd.DataFrame(records).to_csv(output / "metrics.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="Diretório novo do experimento"
    )
    parser.add_argument(
        "--evaluate-existing",
        action="store_true",
        help="Reusar os treinos municipais e as entradas congeladas",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if not args.evaluate_existing:
        output.mkdir(parents=True, exist_ok=False)
        snapshot_inputs(output)
        train_trees(output)
    inputs = output / "inputs"
    cities, panel = regional_inputs(inputs)
    forecasts = pd.concat(
        [evaluate_city(c, frame, panel, output) for c, frame in cities.items()]
    )
    save_metrics(forecasts, output)
    write_manifest(output)


def write_manifest(output):
    inputs = output / "inputs"
    files = [p for p in inputs.rglob("*") if p.is_file()]
    for city in CITIES:
        if not (inputs / f"{city}.csv").exists():
            continue
        for directory in (
            output / city,
            output / (city + "_warmup"),
        ):
            files.extend(
                directory / name
                for name in ("validation_history.json", "walk_forward_evaluation.json")
            )
    files.extend(p for p in (output / "candidates").rglob("*") if p.is_file())
    files.extend(
        [
            Path(__file__),
            ROOT / "models/growth_forecast.py",
            ROOT / "models/growth_calibration.py",
            ROOT / "scripts/train.py",
            ROOT / "scripts/evaluate_walk_forward.py",
            ROOT / "data/features.py",
            ROOT / "models/metrics.py",
            output / "metrics.csv",
        ]
    )
    files.extend(output / f"{city}_forecasts.csv" for city in CITIES if (inputs / f"{city}.csv").exists())
    (output / "manifest.json").write_text(
        json.dumps(
            {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Diagnostica caudas H8 e testa um único ajuste assimétrico no último fold."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_walk_forward import QUANTILES, label_available_week, paired_forecast_metrics, read_records


def scores(frame, predictions):
    y = frame.actual.to_numpy()
    result = paired_forecast_metrics(y, predictions, frame.persistence.to_numpy())
    if result["n_scored"] != len(y) or not len(y):
        raise ValueError("Coorte vazia ou previsões inválidas")
    return {
        "n": len(y),
        "below": int((y < predictions[:, 0]).sum()),
        "above": int((y > predictions[:, -1]).sum()),
        **{key: result[key] for key in ("coverage_90", "width_90", "wis_total", "mae")},
    }


def calibrate(calibration, test):
    # Congela os dois quantis unilaterais antes do primeiro corte de teste.
    y = np.log1p(calibration.actual.to_numpy())
    rank = math.ceil(.95 * (len(y) + 1))
    if rank > len(y):
        raise ValueError("Calibração insuficiente para o percentil de 95%")
    deltas = [max(0., np.sort(residual)[rank - 1]) for residual in
              (np.log1p(calibration.q05) - y, y - np.log1p(calibration.q95))]
    result = test[list(QUANTILES.values())].to_numpy(copy=True)
    if deltas[0] > 0:
        result[:, 0] = np.maximum(0, np.floor(np.expm1(np.log1p(result[:, 0]) - deltas[0])))
    if deltas[1] > 0:
        result[:, -1] = np.ceil(np.expm1(np.log1p(result[:, -1]) + deltas[1]))
    return result, deltas


def recent_candidate(data, test):
    """Estima razões das caudas com os últimos 52 alvos maduros em cada origem."""
    result = []
    for row in test.itertuples():
        past = data[data.available <= row.origin].tail(52)
        if len(past) < 26:
            raise ValueError("Poucos desfechos maduros para o ajuste recente")
        ratios = np.log1p(past.actual) - np.log1p(past.q50)
        calibration = np.quantile(ratios, [.05, .95])
        current = np.log1p([row.q05, row.q95]) - np.log1p(row.q50)
        weight = len(past) / (26 + len(past))
        lower, upper = np.expm1(np.log1p(row.q50) + weight * calibration + (1 - weight) * current)
        result.append([max(0, min(row.q25, math.floor(lower))), row.q25, row.q50,
                       row.q75, max(row.q75, math.ceil(upper))])
    return np.asarray(result)


def analyze(directory):
    directory = directory.resolve(strict=True)
    history = json.loads((directory / "validation_history.json").read_text())
    report = json.loads((directory / "walk_forward_evaluation.json").read_text())
    base = pd.read_csv(directory / "training_base.csv", dtype={"SE": str})
    observed = base.loc[~base.is_imputed].set_index("SE")
    fold_cutoffs = {int(fold): week for fold, week in report["fold_cutoff_se"].items()}
    maturity = report["label_maturity_weeks"]
    first = min(fold_cutoffs.values())
    historic = observed.loc[[label_available_week(week, maturity) <= first for week in observed.index]]
    threshold = (100000 * historic.casos / historic["pop"]).quantile(.75)
    records = read_records({"8": history["8"]}, maturity)
    rows = []
    for (fold, origin, _), row in records.items():
        target = row["target"]
        if not row["embargo_manifest"]:
            raise ValueError("Manifesto de embargo ausente")
        if row["available"] > report["evaluation_cutoff_se"] or target not in observed.index:
            continue
        rows.append(dict(fold=fold, origin=origin, target=target, available=row["available"],
                         actual=observed.at[target, "casos"], stored_actual=row["actual"], persistence=observed.at[origin, "casos"],
                         high=100000 * row["actual"] / observed.at[target, "pop"] >= threshold,
                         **{name: row["raw"][tau] for tau, name in QUANTILES.items()}))
    data = pd.DataFrame(rows).sort_values("origin")
    if data.target.duplicated().any():
        raise ValueError("Alvos repetidos")
    bands = list(QUANTILES.values())

    def original(frame):
        return scores(frame, frame[bands].to_numpy())

    aggregate = original(data)
    reference = report["by_horizon"]["8"]
    if not np.allclose(data.actual, data.stored_actual, rtol=0, atol=1e-7):
        raise ValueError("Contagens incompatíveis com a base congelada")
    legacy = original(data.assign(actual=data.stored_actual))
    if aggregate["n"] != reference["n_scored"] or not any(np.allclose(
        [check[k] for k in ("wis_total", "coverage_90", "width_90")],
        [reference[k] for k in ("wis_total", "coverage_90", "width_90")],
    ) for check in (aggregate, legacy)):
        raise ValueError("Coorte divergente da avaliação canônica da release")
    fold = max(fold_cutoffs)
    cutoff = fold_cutoffs[fold]
    calibration = data[(data.fold < fold) & (data.available <= cutoff)]
    test = data[data.fold == fold]
    candidate, deltas = calibrate(calibration, test)
    return {
        "release": directory.name, "aggregate": aggregate, "threshold_per_100k": threshold,
        "folds": {int(k): original(v) for k, v in data.groupby("fold")},
        "phase": {"alta": original(data[data.high]), "baixa": original(data[~data.high])},
        "holdout": {"origin_start": cutoff, "origin_end": test.origin.max(),
                    "calibration_n": len(calibration), "last_available": calibration.available.max(),
                    "delta_lo": float(deltas[0]), "delta_hi": float(deltas[1]),
                    "published": original(test), "candidate": scores(test, candidate),
                    "recent_candidate": scores(test, recent_candidate(data, test))},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(analyze(parser.parse_args().directory), indent=2, allow_nan=False))

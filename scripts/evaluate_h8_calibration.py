"""Avaliação temporal de uma expansão conformal dos limites externos de H8."""
import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_walk_forward import QUANTILES, evaluate, load_config, paired_forecast_metrics


def compare(directory):
    rows = []
    base = pd.read_csv(directory.parent / "training_base.csv", dtype={"SE": str})
    report, _, _ = evaluate(json.loads((directory / "validation_predictions.json").read_text()),
                            base, load_config(str(directory / "config.yaml")), scored_rows=rows)
    if not report["training_embargo_verified"]:
        raise ValueError("A avaliação exige embargo de treino verificado")
    fold = max(report["fold_cutoff_se"])
    cutoff = report["fold_cutoff_se"][fold]
    scores, fans, outcomes, origins = [], [], [], []
    for block, origin, available, row, actual in rows:
        if row["h"] != 8:
            continue
        fan = np.array([row[name] for name in QUANTILES.values()], dtype=float)
        if not np.isfinite(fan).all() or min(fan) < 0 or np.any(np.diff(fan) < 0):
            raise ValueError("Quantis inválidos")
        if block < fold and available <= cutoff:
            scores.append(max(0, fan[0] - actual, actual - fan[-1]) / math.sqrt(1 + fan[2]))
        if block == fold:
            fans.append(fan)
            outcomes.append(actual)
            origins.append(origin)
    rank = math.ceil(0.90 * (len(scores) + 1))
    if rank > len(scores) or not fans:
        raise ValueError("Amostra insuficiente")
    delta = sorted(scores)[rank - 1]
    original = np.array(fans)
    calibrated = original.copy()
    margin = delta * np.sqrt(1 + original[:, 2])
    calibrated[:, 0] = np.maximum(0, np.floor(original[:, 0] - margin))
    calibrated[:, -1] = np.ceil(original[:, -1] + margin)
    result = dict(cutoff=cutoff, last_origin=max(origins), calibration_n=len(scores),
                  test_n=len(outcomes), delta=delta)
    for name, predictions in [('original', original), ('calibrated', calibrated)]:
        metrics = paired_forecast_metrics(outcomes, predictions, base.set_index("SE").loc[origins, "casos"])
        if metrics["n_scored"] != len(outcomes):
            raise ValueError("Há semanas de teste sem pontuação")
        result[name] = {k: metrics[k] for k in ("coverage_90", "width_90", "wis_total", "mae")}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    logging.disable(logging.INFO)
    print(json.dumps(compare(parser.parse_args().directory), indent=2))

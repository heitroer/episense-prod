"""Diagnóstico retrospectivo de H4/H8 no benchmark de cinco seeds."""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_walk_forward import add_epiweeks, evaluate, label_available_week, load_config


def analyze(directory):
    base = pd.read_csv(directory.parent / "training_base.csv", dtype={"SE": str})
    rows = []
    report, _, _ = evaluate(json.loads((directory / "validation_predictions.json").read_text()),
                            base, load_config(str(directory / "config.yaml")), scored_rows=rows)
    series = base.loc[~base.is_imputed].set_index("SE")
    incidence = 100000 * series.casos / series["pop"]
    first = min(report["fold_cutoff_se"].values())
    maturity = report["label_maturity_weeks"]
    threshold = incidence[[label_available_week(se, maturity) <= first for se in incidence.index]].quantile(.75)
    print(f"Limiar histórico: {threshold:.2f} casos por 100 mil")
    for h in (4, 8):
        records = []
        for _, origin, _, row, actual in rows:
            if row["h"] != h:
                continue
            target = row["target_se"]
            previous = series.at[origin, "casos"]
            forecast = row["q50"]
            neighbors = [incidence.get(add_epiweeks(target, i), np.nan)
                         for i in range(-4, 5) if i]
            high = incidence[target] >= threshold
            peak = high and all(incidence[target] > value for value in neighbors)
            records.append(dict(target=target, high=high, peak=peak, rise=actual > previous,
                                predicts_rise=forecast > previous, below=incidence[origin] < threshold,
                                alert=100000 * forecast / series.at[origin, "pop"] >= threshold,
                                model=abs(forecast-actual), persistence=abs(previous-actual)))
        data = pd.DataFrame(records)
        reference = report["by_horizon"][h]
        assert len(data) == data.target.nunique() == reference["n_scored"]
        assert np.isclose(data.model.mean(), reference["mae"])
        assert np.isclose(data.persistence.mean(), reference["persistence_mae"])
        for name, part in [('baixa', data[~data.high]), ('alta', data[data.high]),
                           ('subida', data[data.rise]), ('pico', data[data.peak])]:
            print(h, name, len(part), round(part.model.mean(), 2), round(part.persistence.mean(), 2),
                  'previu subida', int(part.predicts_rise.sum()))
        subset = data[data.below]
        print(h, 'cruzamentos detectados/reais e falsos/negativos reais',
              (subset.alert & subset.high).sum(), subset.high.sum(),
              (subset.alert & ~subset.high).sum(), (~subset.high).sum())
        print('picos', data.loc[data.peak, 'target'].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    logging.disable(logging.WARNING)
    analyze(parser.parse_args().directory)

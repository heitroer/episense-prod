"""Temporal release checks on the forecast rows actually served by the dashboard."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.epiweeks import add_epiweeks
from data.observations import observed_values
from data.origin_inputs import issued_inputs
from models.metrics import QUANTILES, paired_forecast_metrics
from models.validation_calendar import label_available_week, origin_weeks, target_weeks


def _history(directory):
    table = json.loads((directory / "validation_history.json").read_text())
    result = {}
    for h, quantiles in table.items():
        by_origin = {}
        for tau in QUANTILES:
            for block in quantiles[str(tau)]:
                origins = origin_weeks(block)
                targets = target_weeks(block, int(h))
                values = block["y_pred_casos"]
                if len(origins) != len(values) or len(targets) != len(origins):
                    raise ValueError(f"Histórico truncado em H{h}")
                for origin, target, value in zip(origins, targets, values):
                    row = by_origin.setdefault((origin, target), {})
                    if tau in row or not np.isfinite(value):
                        raise ValueError(f"Previsão duplicada ou inválida em H{h} {origin}")
                    row[tau] = value
        result[h] = {key: [row[t] for t in QUANTILES] for key, row in by_origin.items()}
    return result


def evaluate_promotion(candidate: Path, incumbent: Path, config: dict):
    policy = config["promotion_policy"]
    evaluation = json.loads((candidate / "walk_forward_evaluation.json").read_text())
    if evaluation.get("schema_version") != 2 or not evaluation.get("training_embargo_verified"):
        raise ValueError("Avaliação pareada ou embargo inválido")
    if evaluation["n_scored"] != evaluation["n_eligible"]:
        raise ValueError("Linhas maduras sem pontuação")
    history = _history(candidate)
    previous = None
    if (incumbent / "validation_history.json").exists():
        previous = _history(incumbent)
    elif (incumbent / "release.json").exists() or any(incumbent.glob("lgbm_*.pkl")):
        raise ValueError("Versão anterior sem histórico comparável para promoção automática")

    base = pd.read_csv(candidate / "training_base.csv", dtype={"SE": str}).set_index("SE")
    observed = base.loc[~base.is_imputed] if "is_imputed" in base else base
    cases = observed.casos.dropna().to_dict()
    origins = {origin for table in history.values() for origin, _ in table}
    revised = config["training"].get("historical_inputs") == "revised"
    expected_scope = "retrospective_revised_inputs" if revised else "issued_inputs_consolidated_outcomes"
    if evaluation.get("scope") != expected_scope:
        raise ValueError("Origem dos dados incompatível com a configuração do candidato")
    if revised:
        latest_observed = observed_values(base).sort_index().ffill()
        persistence = latest_observed.reindex(sorted(origins)).dropna().to_dict()
    else:
        persistence = {origin: float(values.iloc[-1])
                       for origin, _, _, _, values in issued_inputs(config, origins)}
    population = observed["pop"]
    incidence = (100000 * observed.casos / population.where(population > 0)).dropna().to_dict()
    maturity = evaluation["label_maturity_weeks"]
    cutoff = evaluation["evaluation_cutoff_se"]
    first = min(evaluation["fold_cutoff_se"].values())
    high = np.quantile([v for k, v in incidence.items() if label_available_week(k, maturity) <= first], .75)
    # Peak classification is retrospective and uses only neighbors mature at evaluation.
    peaks = {k for k, v in incidence.items() if v >= high
             and label_available_week(add_epiweeks(k, 4), maturity) <= cutoff
             and all(v > incidence.get(add_epiweeks(k, d), np.nan) for d in (-4,-3,-2,-1,1,2,3,4))}
    report = {"evaluation_role": "model_selection", "scope": expected_scope, "incumbent": incumbent.name if previous else None, "policy": policy,
              "by_horizon": {}, "failures": []}
    for horizon in config["targets"]["horizons"]:
        h = str(horizon)
        current = history[h]
        keys = sorted(k for k in current if label_available_week(k[1], maturity) <= cutoff
                      and (revised or k[0] in cases) and k[1] in cases)

        def metrics(fans, cohort):
            y = [cases[t] for _, t in cohort]
            q = np.array([fans[k] for k in cohort])
            if not len(y):
                raise ValueError(f"Coorte vazia H{h}")
            if np.any(np.diff(q, axis=1) < 0) or np.min(q) < 0:
                raise ValueError(f"Quantis inválidos em H{h}")
            return paired_forecast_metrics(y, q, [persistence[o] for o, _ in cohort])

        full = metrics(current, keys)
        canonical = evaluation["by_horizon"][h]
        checks = ("n_scored", "mae", "wis_total", "width_90", "coverage_90", "persistence_mae")
        if any(not np.isclose(full[key], canonical[key], rtol=1e-9, atol=1e-9) for key in checks):
            raise ValueError(f"Paridade com avaliação publicada falhou em H{h}")
        overlap = [k for k in keys if previous is None or k in previous[h]]
        peak_keys = [k for k in overlap if k[1] in peaks]
        row = {"n_full": len(keys), "n_paired": len(overlap), "n_peaks": len(peak_keys),
               "paired_sha256": hashlib.sha256(json.dumps(overlap).encode()).hexdigest()}
        reasons = []
        if len(overlap) < policy["min_paired_weeks"] or len(peak_keys) < policy["min_peak_weeks"]:
            reasons.append("amostra madura ou de picos insuficiente")
        else:
            a = metrics(current, overlap)
            b = metrics(previous[h], overlap) if previous else None
            row["candidate"] = {key: a[key] for key in checks}
            row["candidate"]["peak_mae"] = metrics(current, peak_keys)["mae"]
            if a["mae"] > policy["max_mae_persistence_ratio"][horizon] * a["persistence_mae"]:
                reasons.append("MAE acima do orçamento frente à persistência")
            if a["coverage_90"] < policy["coverage_90_floor"]:
                reasons.append("cobertura abaixo do piso")
            if b is not None:
                row["incumbent"] = {key: b[key] for key in checks}
                row["incumbent"]["peak_mae"] = metrics(previous[h], peak_keys)["mae"]
                for metric in ("mae", "wis_total", "width_90", "peak_mae"):
                    if row["candidate"][metric] > row["incumbent"][metric] * policy["max_ratio"][metric] + 1e-9:
                        reasons.append(f"regressão de {metric}")
                if abs(a["coverage_90"] - .90) > abs(b["coverage_90"] - .90) + policy["max_coverage_error_increase"]:
                    reasons.append("piora da calibração de 90%")
        report["by_horizon"][h] = row
        report["failures"].extend(f"H{h}: {s}" for s in reasons)
    report["pass"] = not report["failures"]
    return report

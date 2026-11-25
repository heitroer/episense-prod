"""Calendário dos blocos de previsão usados na validação temporal."""
from data.epiweeks import add_epiweeks


def origin_weeks(block):
    # Nos artefatos legados, test_se identifica a linha de entrada.
    origins = block.get("origin_se")
    if origins is None:
        origins = block["test_se"]
    return [str(se).zfill(6) for se in origins]


def target_weeks(block, horizon):
    expected = [add_epiweeks(se, horizon) for se in origin_weeks(block)]
    declared = block.get("target_se")
    if declared is not None and [str(se).zfill(6) for se in declared] != expected:
        raise ValueError(f"Semanas-alvo incompatíveis com as origens no horizonte {horizon}")
    return expected


def label_available_week(target, maturity_weeks):
    """Corte semanal estimado, após encerramento e maturação do alvo."""
    if maturity_weeks < 0:
        raise ValueError("A maturação dos desfechos deve ser não negativa")
    return add_epiweeks(target, 1 + maturity_weeks)


def available_label_positions(origins, horizon, cutoff, maturity_weeks):
    return [
        i for i, origin in enumerate(origins)
        if label_available_week(add_epiweeks(origin, horizon), maturity_weeks) <= cutoff
    ]


def mature_calibration_series(series, cutoff, maturity, before_fold=None):
    """Return aligned calibration blocks available at the start of a forecast."""
    result = {}
    for h, quantiles in series.items():
        for tau, blocks in quantiles.items():
            for block in blocks:
                fold = int(block["fold"])
                if before_fold is not None and fold >= before_fold:
                    continue
                origins = origin_weeks(block)
                targets = target_weeks(block, int(h))
                actual, pred = block["y_test_casos"], block["y_pred_casos"]
                if len(actual) != len(origins) or len(pred) != len(origins):
                    raise ValueError(f"Bloco de calibração truncado em H{h}, fold {fold}")
                ready = [label_available_week(t, maturity) for t in targets]
                if block.get("target_available_se", ready) != ready:
                    raise ValueError(f"Maturação divergente em H{h}, fold {fold}")
                idx = [i for i, week in enumerate(ready) if week <= cutoff]
                if idx:
                    rows = {"origin_se": origins, "target_se": targets,
                            "y_test_casos": actual, "y_pred_casos": pred}
                    selected = {key: [values[i] for i in idx] for key, values in rows.items()}
                    result.setdefault(h, {}).setdefault(tau, []).append({"fold": fold, **selected})
    return result


def check_observed_targets(artifact, base):
    """Recusa artefatos históricos que usaram uma semana imputada como desfecho."""
    from data.observations import observed_values
    absent = set(base.loc[observed_values(base).isna(), "SE"].astype(str))
    if absent:
        for h, table in artifact.items():
            for blocks in table.values():
                for block in blocks:
                    invalid = absent.intersection(target_weeks(block, int(h)))
                    if invalid:
                        raise ValueError(f"H{h}: desfecho sem observação em {min(invalid)}; regenere as previsões")

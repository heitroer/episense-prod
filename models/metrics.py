"""Conjunto canônico de métricas de avaliação do EpiSense.

Fonte única das definições — usada por scripts/train.py (validação
walk-forward) e scripts/recompute_metrics.py (regeneração de
validation_results.json sem retreino).

Convenções (tudo em escala de CASOS, pós-expm1):
- Predição pontual = quantil 0.50 (mediana).
- mae        = média |y − p| (mediana)
- rmse       = sqrt( média (y − p)² )
- bias       = média( p − y )  [>0 superestima, <0 subestima]
- pinball(τ) = max( τ·(y−q), (τ−1)·(y−q) ), com diff = y − q
- wis_total  = duas vezes a perda pinball média sobre os 5 quantis {05,25,50,75,95}
- wis_50     = pinball médio sobre o par {q25, q75}   (componente 50%)
- wis_90     = pinball médio sobre o par {q05, q95}   (componente 90%)
- width_50 / width_90 = largura média dos intervalos [q25,q75] / [q05,q95]
- coverage_50 / coverage_90 = fração de semanas reais dentro do intervalo
               (nominais 50% / 90%)
- wape = Σ|y − p| / Σ|y| — erro absoluto ponderado pelo volume real (peso
               linear; não explode em semanas de poucos casos como o MAPE)
- skill_mae / skill_wis = 1 − mae/baseline_mae e 1 − wis_total/baseline_wis
               (derivadas nos scripts onde o baseline existe; na agregação
               entram como razão das médias — helper skill_score() abaixo)
Removidos na spec 2026-10: MASE (e agregados), ETP, SPL, MAPE, CRPS,
quantile_crossed e fbias.
Todas as métricas usam máscara finita e retornam NaN quando não computáveis.
"""

from typing import Dict, Mapping
import numpy as np

QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
PAIR_50 = (0.25, 0.75)
PAIR_90 = (0.05, 0.95)


def pinball_mean(y_true: np.ndarray, q_pred: np.ndarray, tau: float,
                 mask: np.ndarray = None) -> float:
    """Pinball loss média para um quantil τ (máscara finita aplicada)."""
    y = np.asarray(y_true, dtype=float)
    q = np.asarray(q_pred, dtype=float)
    m = np.isfinite(y) & np.isfinite(q)
    if mask is not None:
        m = m & np.asarray(mask, dtype=bool)
    if not np.any(m):
        return float('nan')
    diff = y[m] - q[m]
    pin = np.maximum(tau * diff, (tau - 1.0) * diff)
    pin = pin[np.isfinite(pin)]
    return float(np.mean(pin)) if len(pin) else float('nan')


def _quantile_rows(actual, columns):
    y = np.asarray(actual, dtype=float)
    quantiles = [np.asarray(column, dtype=float) for column in columns]
    if y.ndim != 1 or any(column.shape != y.shape for column in quantiles):
        raise ValueError("Cada quantil deve ter uma previsão por observação")
    valid = np.isfinite(y) & (y >= 0)
    for column in quantiles:
        valid &= np.isfinite(column) & (column >= 0)
    for lower, upper in zip(quantiles, quantiles[1:]):
        valid &= lower <= upper
    return y, quantiles, valid


def _mean_quantile_loss(y, predictions, levels):
    if any(t not in predictions for t in levels):
        return float("nan")
    y, columns, valid = _quantile_rows(y, [predictions[t] for t in levels])
    if not valid.any():
        return float("nan")
    return float(np.mean([pinball_mean(y, q, t, valid) for t, q in zip(levels, columns)]))


def weighted_interval_score(y_true: np.ndarray, predictions: Mapping[float, np.ndarray]) -> float:
    """WIS em casos nas linhas com cinco quantis válidos e ordenados."""
    return 2 * _mean_quantile_loss(y_true, predictions, QUANTILES)


def mean_interval_width(y_true: np.ndarray, q_lo: np.ndarray,
                        q_hi: np.ndarray) -> float:
    """Largura média do intervalo [q_lo, q_hi] (apenas pares finitos)."""
    _, (lo, hi), m = _quantile_rows(y_true, [q_lo, q_hi])
    if not np.any(m):
        return float('nan')
    return float(np.mean(hi[m] - lo[m]))


def coverage_interval(y_true: np.ndarray, q_lo: np.ndarray,
                      q_hi: np.ndarray) -> float:
    """Fração de y dentro do intervalo [q_lo, q_hi] (pares finitos)."""
    y, (lo, hi), m = _quantile_rows(y_true, [q_lo, q_hi])
    if not np.any(m):
        return float('nan')
    inside = (y[m] >= lo[m]) & (y[m] <= hi[m])
    return float(np.mean(inside))


def wape(y_true: np.ndarray, pred: np.ndarray) -> float:
    """WAPE: Σ|y − p| / Σ|y| — erro absoluto em % do volume real.

    Percentual ponderado pelo volume (peso linear |y|): semanas com muitos
    casos pesam proporcionalmente mais; não explode em semanas de poucos
    casos como o MAPE removido na spec 2026-10. NaN quando não computável.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    if not np.any(m):
        return float('nan')
    denom = float(np.sum(np.abs(y[m])))
    if not (np.isfinite(denom) and denom > 0):
        return float('nan')
    return float(np.sum(np.abs(y[m] - p[m])) / denom)


def skill_score(value: float, baseline: float) -> float:
    """Ganho vs baseline (skill): 1 − value/baseline.

    >0 = corta X% do erro do baseline sazonal; 0 = empata; <0 = pior que
    colar o histórico. NaN quando baseline ou value não computáveis
    (NaN/∞) ou baseline ≤ 0.
    """
    try:
        v = float(value)
        b = float(baseline)
    except (TypeError, ValueError):
        return float('nan')
    if not (np.isfinite(v) and np.isfinite(b)) or b <= 0:
        return float('nan')
    return float(1.0 - v / b)


def compute_all_metrics(y_true_cases: np.ndarray,
                        pred_cases: Mapping[float, np.ndarray]) -> Dict[str, float]:
    """Métricas canônicas para uma janela de avaliação (escala de casos).

    pred_cases: dict {tau: vetor de previsões em casos}; requer 0.05, 0.25,
    0.50, 0.75, 0.95 (falta de um quantil ⇒ métricas relacionadas viram NaN).
    """
    y = np.asarray(y_true_cases, dtype=float)
    out: Dict[str, float] = {}

    # ---- Pontuais (mediana q50) ----
    y, (median,), m_fin = _quantile_rows(y, [pred_cases.get(0.50, np.full(y.shape, np.nan))])
    if np.any(m_fin):
        err = y[m_fin] - median[m_fin]
        out['mae'] = float(np.mean(np.abs(err)))
        out['rmse'] = float(np.sqrt(np.mean(err ** 2)))
        out['bias'] = float(np.mean(median[m_fin] - y[m_fin]))
        out['wape'] = wape(y[m_fin], median[m_fin])
    else:
        out['mae'] = out['rmse'] = out['bias'] = out['wape'] = float('nan')

    # ---- Familia probabilística (pinball) ----
    pins = {tau: np.asarray(pred_cases[tau], dtype=float)
            for tau in QUANTILES if tau in pred_cases}

    out['wis_total'] = weighted_interval_score(y, pins)

    for pair, key in ((PAIR_50, 'wis_50'), (PAIR_90, 'wis_90')):
        out[key] = _mean_quantile_loss(y, pins, pair)

    q05 = pins.get(0.05)
    q25 = pins.get(0.25)
    q75 = pins.get(0.75)
    q95 = pins.get(0.95)
    out['width_50'] = (mean_interval_width(y, q25, q75)
                       if q25 is not None and q75 is not None else float('nan'))
    out['width_90'] = (mean_interval_width(y, q05, q95)
                       if q05 is not None and q95 is not None else float('nan'))
    out['coverage_50'] = (coverage_interval(y, q25, q75)
                          if q25 is not None and q75 is not None else float('nan'))
    out['coverage_90'] = (coverage_interval(y, q05, q95)
                          if q05 is not None and q95 is not None else float('nan'))
    return out


def pooled_fold_metrics(folds):
    """Aggregate forecast rows, preserving the legacy complete-fold and all-fold cohorts."""
    def summarize(ids):
        actual = np.concatenate([folds[i][0] for i in ids])
        def metrics(position):
            predictions = {t: np.concatenate([folds[i][position][t] for i in ids]) for t in QUANTILES}
            return compute_all_metrics(actual, predictions)

        model, seasonal = metrics(1), metrics(2)
        model.update(baseline_mae=seasonal["mae"], baseline_wis=seasonal["wis_total"],
                     baseline_coverage_90=seasonal["coverage_90"],
                     skill_mae=skill_score(model["mae"], seasonal["mae"]),
                     skill_wis=skill_score(model["wis_total"], seasonal["wis_total"]),
                     n_weeks=len(actual), n_folds=len(ids), aggregation="pooled_rows")
        return model

    complete = [i for i in folds if len(folds[i][0]) >= 52]
    primary = complete or list(folds)
    result = summarize(primary)
    result["excluded_partial_folds"] = [i for i in folds if i not in primary]
    result["n_folds_full"] = len(folds)
    result["full_4folds_avg"] = summarize(list(folds)) if len(primary) < len(folds) else dict(result)
    return result


def paired_forecast_metrics(actual, predictions, persistence, seasonal=None):
    """Score all methods on the same rows, using the standard WIS scale."""
    y = np.asarray(actual, dtype=float)
    p = np.asarray(persistence, dtype=float)
    model = np.asarray(predictions, dtype=float).reshape((-1, len(QUANTILES)))
    reference = None if seasonal is None else np.asarray(seasonal, dtype=float).reshape(model.shape)
    if y.shape != p.shape or y.ndim != 1 or len(model) != len(y):
        raise ValueError("Forecast and outcome arrays must have matching rows")

    def valid_fan(fan):
        return np.all(np.isfinite(fan) & (fan >= 0), axis=1) & np.all(np.diff(fan, axis=1) >= 0, axis=1)

    valid = np.isfinite(y) & (y >= 0) & np.isfinite(p) & (p >= 0) & valid_fan(model)
    if reference is not None:
        valid &= valid_fan(reference)
    if not valid.any():
        return {"n_eligible": len(y), "n_scored": 0, "n_excluded": len(y)}

    y = y[valid]
    p = p[valid]

    def summarize(fan):
        values = compute_all_metrics(y, dict(zip(QUANTILES, fan[valid].T)))
        # Replace legacy pair pinball scores with the full interval scores.
        values["interval_score_50"] = 4 * values.pop("wis_50") / 0.5
        values["interval_score_90"] = 4 * values.pop("wis_90") / 0.1
        return {key: value if np.isfinite(value) else None for key, value in values.items()}

    def gain(value, baseline):
        score = skill_score(value, baseline)
        return score if np.isfinite(score) else None

    forecast = summarize(model)
    persistence_mae = float(np.mean(np.abs(y - p)))
    n = int(valid.sum())
    result = {
        **forecast, "n_eligible": len(valid), "n_scored": n, "n_excluded": len(valid) - n,
        "persistence_mae": persistence_mae,
        "skill_mae_persistence": gain(forecast["mae"], persistence_mae),
    }
    if reference is not None:
        seasonal_metrics = summarize(reference)
        result.update(
            baseline_mae=seasonal_metrics["mae"], baseline_wis=seasonal_metrics["wis_total"],
            baseline_coverage_90=seasonal_metrics["coverage_90"],
            skill_mae=gain(forecast["mae"], seasonal_metrics["mae"]),
            skill_wis=gain(forecast["wis_total"], seasonal_metrics["wis_total"]),
        )
    return result

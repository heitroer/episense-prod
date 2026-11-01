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
- wis_total  = pinball médio sobre os 5 quantis {05,25,50,75,95}
               (aprox. padrão de Bracher et al. 2021)
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


def mean_interval_width(y_true: np.ndarray, q_lo: np.ndarray,
                        q_hi: np.ndarray) -> float:
    """Largura média do intervalo [q_lo, q_hi] (apenas pares finitos)."""
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(q_lo, dtype=float)
    hi = np.asarray(q_hi, dtype=float)
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if not np.any(m):
        return float('nan')
    return float(np.mean(hi[m] - lo[m]))


def coverage_interval(y_true: np.ndarray, q_lo: np.ndarray,
                      q_hi: np.ndarray) -> float:
    """Fração de y dentro do intervalo [q_lo, q_hi] (pares finitos)."""
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(q_lo, dtype=float)
    hi = np.asarray(q_hi, dtype=float)
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
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
    median = np.asarray(pred_cases.get(0.50, np.array([])), dtype=float)
    m_fin = np.isfinite(y) & np.isfinite(median)
    if np.any(m_fin):
        err = y[m_fin] - median[m_fin]
        out['mae'] = float(np.mean(np.abs(err)))
        out['rmse'] = float(np.sqrt(np.mean(err ** 2)))
        out['bias'] = float(np.mean(median[m_fin] - y[m_fin]))
        out['wape'] = wape(y, median)
    else:
        out['mae'] = out['rmse'] = out['bias'] = out['wape'] = float('nan')

    # ---- Familia probabilística (pinball) ----
    pins = {tau: np.asarray(pred_cases[tau], dtype=float)
            for tau in QUANTILES if tau in pred_cases}
    ok_y = np.isfinite(y)

    totals = []
    for tau in QUANTILES:
        if tau in pins:
            v = pinball_mean(y, pins[tau], tau, ok_y)
            if np.isfinite(v):
                totals.append(v)
    out['wis_total'] = float(np.mean(totals)) if totals else float('nan')

    for pair, key in ((PAIR_50, 'wis_50'), (PAIR_90, 'wis_90')):
        vals = []
        for tau in pair:
            if tau in pins:
                v = pinball_mean(y, pins[tau], tau, ok_y)
                if np.isfinite(v):
                    vals.append(v)
        out[key] = float(np.mean(vals)) if vals else float('nan')

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

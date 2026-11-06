"""
EXP-Q2 (nov/2026) — Re-forma da banda quantílica em torno da mediana ("fan chart").

Motivação (produção/dashboard 2026-43+): os modelos de cauda independentes
(q0050..q0950) têm ruído próprio grande — q05 "cola no chão" (fatores q/med ~0.05,
p05 do walk-forward) e q95 explode 3–8× a mediana em semanas aleatórias
(f95 p95 6.6–8.0). O desenho da banda dessincroniza do desenho da mediana
(shape_corr q05×q50 = 0.43; live: 0.78). A banda inferior sobe pouco quando a
mediana sobe e a superior salta sem motivo visível.

Solução: âncora multiplicativa por horizonte — q* = med × f*[h] (factores em
cases-scale), mediana NUNCA alterada (MAE/gate de nível preservados).

Dois modos:
  * 'profile' (default): factors fixos por horizonte calibrados no walk-forward
    (mediana dos fatores raw por h, suavizada isotônica PAV — hi não-decrescente,
    lo não-crescente — e caps). Fan determinístico e limpo: o desenho da banda É
    a mediana escalada; ruído dos modelos de cauda eliminado por construção.
  * 'raw': factors por fan (pós-camadas de nível), suavização roll-median no
    eixo h + monotonia soft (damped accumulate) + caps. Preservado para
    comparação futura.

Calibração (models/validation_predictions.json, 176 fans h1..h8 completos,
scripts/analyze_band_shape.py):
  perfil perfil-P2 (f75/f95 ×1.10 sobre PAV, caps lo[.27,.80] hi[1.18,4.1]):
    cov90 0.835 → 0.876 (+4.2pp; nominal 0.90; live após este passo ainda
    recebe deltas conformal por regime);
    cov50 0.428 → 0.509 (nominal 0.50 — banda 50% era subcoberta);
    WIS −2.1% (MELHOR que raw), MAE inalterado.
"""

from typing import Dict, List

import numpy as np

import logging

logger = logging.getLogger(__name__)

QUANTILE_SIDES = {
    0.05: 'lo',
    0.25: 'lo',
    0.75: 'hi',
    0.95: 'hi',
}


def _smooth_cols(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling median centrada (eixo 0 = horizontes), bordas com kernel encolhido."""
    n = x.shape[0]
    if n <= 2 or window <= 1:
        return x.copy()
    out = x.copy()
    for i in range(n):
        k = min(window, 2 * min(i, n - 1 - i) + 1)
        if k <= 1:
            continue
        lo = i - k // 2
        hi = i + k // 2 + 1
        out[i] = np.nanmedian(x[lo:hi], axis=0)
    return out


def _isotonic_soft(x: np.ndarray, up: bool, damp: float) -> np.ndarray:
    """Monotonia soft (acumulação amortecida): permite deriva ≤ (1-damp)/step.

    up=True: séquence deve ser não-decrescente (lado hi); up=False: não-crescente.
    Diferente do hard accumulate (np.maximum.accumulate), um único pico no h1
    não "contamina" todo o fan — a propagação é amortecida por 'damp'.
    """
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        if i == 0:
            out[i] = x[i]
            continue
        prev = out[i - 1]
        if up:
            out[i] = np.maximum(x[i] if np.isfinite(x[i]).all() else prev, damp * prev)
        else:
            out[i] = np.minimum(x[i] if np.isfinite(x[i]).all() else prev, prev / max(damp, 1e-6))
    return out


def reshape_fan_logits(preds: Dict[int, Dict[float, np.ndarray]],
                       horizons: List[int],
                       cfg: Dict = None) -> Dict[int, Dict[float, np.ndarray]]:
    """Aplica a re-forma da banda em logits (log1p de casos).

    preds: {horizonte: {tau: array(len N) em log1p}} — aceita arrays len 1 (live)
    ou len N (batch). Trata os horizontes como eixo do fan e mantém colunas (N).
    A mediana (tau=0.5) nunca é alterada.
    """
    cfg = cfg or {}
    mode = str(cfg.get('mode', 'profile'))
    window = max(1, int(cfg.get('smooth_window', 3)))
    monotone = str(cfg.get('monotone', 'soft')).lower() if cfg.get('monotone') is not None else 'soft'
    damp = float(cfg.get('monotone_damp', 0.85))
    lo_min = float(cfg.get('lo_min', 0.27))
    lo_max = float(cfg.get('lo_max', 0.80))
    hi_min = float(cfg.get('hi_min', 1.18))
    hi_max = float(cfg.get('hi_max', 4.10))
    eps = float(cfg.get('median_eps_cases', 0.5))
    profile = cfg.get('profile') or {}
    apply_hs = set(int(x) for x in cfg.get('apply_horizons', horizons) or horizons)

    hs = [h for h in horizons if h in preds]
    if len(hs) < 2:
        return preds

    # ---- empilha cases + mediana [H, N] ----
    n = 0
    for h in hs:
        for tau, arr in preds[h].items():
            if arr is not None and getattr(arr, 'size', 0) > 0:
                n = max(n, int(arr.size))
    if n == 0:
        return preds

    def _cases_matrix(tau):
        rows = []
        for h in hs:
            arr = preds[h].get(tau)
            if arr is None or getattr(arr, 'size', 0) == 0:
                rows.append(np.full(n, np.nan))
                continue
            a = np.asarray(arr, dtype=float).ravel()
            cases = np.expm1(np.clip(a, 0.0, None))
            cases[~np.isfinite(cases)] = np.nan
            if a.size < n:
                pad = cases[-1:] if cases.size else np.full(1, np.nan)
                cases = np.concatenate([cases, np.resize(pad, n - a.size)])
            rows.append(cases[:n])
        return np.vstack(rows)  # [H, N]

    med = _cases_matrix(0.50)

    def _safe_div(num, den):
        with np.errstate(all='ignore'):
            out = np.where((den > eps) & np.isfinite(den) & np.isfinite(num), num / np.maximum(den, 1e-9), np.nan)
        return out

    out = {h: dict(t) for h, t in preds.items()}

    new_factors: Dict[float, np.ndarray] = {}
    for tau, side in QUANTILE_SIDES.items():
        if any(tau not in preds[h] for h in hs):
            continue
        if mode == 'profile':
            prof = profile.get(f'{tau:.2f}') or profile.get(str(tau))
            if prof is None or len(prof) < len(hs):
                logger.warning(f"band_shape profile ausente/incompleto para tau={tau} — tau pulado")
                continue
            f = np.tile(np.asarray(prof, dtype=float)[len(prof) - len(hs):][:, None], (1, n))
        else:
            q = _cases_matrix(tau)
            f = _safe_div(q, med)
            # fill NaN por coluna com mediana dos finitos; coluna NaN -> neutro
            with np.errstate(all='ignore'):
                col_median = np.nanmedian(f, axis=0)
                col_median = np.where(np.isfinite(col_median), col_median, 1.0)
                f = np.where(np.isfinite(f), f, col_median[None, :])
            f = _smooth_cols(f, window)
            f = np.where(np.isfinite(f), f, col_median[None, :])
            if monotone == 'hard':
                if side == 'hi':
                    f = np.maximum.accumulate(f, axis=0)
                else:
                    f = np.minimum.accumulate(f, axis=0)
            elif monotone == 'soft':
                if side == 'hi':
                    f = np.vstack([_isotonic_soft(f[:, c], True, damp) for c in range(f.shape[1])]).T
                else:
                    f = np.vstack([_isotonic_soft(f[:, c], False, damp) for c in range(f.shape[1])]).T
            c = (lo_min, lo_max) if side == 'lo' else (hi_min, hi_max)
            f = np.clip(f, c[0], c[1])
        # clampas de sanidade (ambos os modos)
        c = (lo_min, 0.98) if side == 'lo' else (hi_min if hi_min >= 1.0 else 1.001, hi_max)
        f = np.clip(f, c[0], c[1])
        new_factors[tau] = f

    # ordenação intra-coluna dos fatores (f05<=f25 e f75<=f95 salvaguarda)
    if 0.25 in new_factors and 0.05 in new_factors:
        new_factors[0.05] = np.minimum(new_factors[0.05], new_factors[0.25])
    if 0.75 in new_factors and 0.95 in new_factors:
        new_factors[0.95] = np.maximum(new_factors[0.95], new_factors[0.75])

    # ---- reconstroi quantis: q' = med × f' (cases) → log1p [H, N] ----
    n_applied = 0
    for tau, f in new_factors.items():
        q_cases = np.clip(med * f, 0.0, None)
        q_log = np.log1p(q_cases)
        for i, h in enumerate(hs):
            if int(h) not in apply_hs:
                continue
            arr = preds[h][tau]
            if arr is None or getattr(arr, 'size', 0) == 0:
                continue
            out_size = int(np.asarray(arr).ravel().size)
            out[h][tau] = q_log[i, :out_size].copy()
            n_applied += 1

    logger.info(f"band_shape modo={mode} aplicada ({n_applied} bandas de horizonte; "
                f"mediana intocada; {len(new_factors)} taus)")
    return out

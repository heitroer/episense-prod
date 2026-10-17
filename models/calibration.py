"""Calibração conformal de quantis por horizonte e regime-base (EpiSense).

Substitui a recalibração holdout-único (fraqueza P8 do relatório out/2026).

Evidência dos experimentos out/2026 (rascunho em Banco de testes A/B — ver
git log):
- Deltas conformal (CQR, sem median shift) aplicados em h1-h3 são estáveis
  entre esquemas de avaliação (expanding e LOO): cov90 h1 0.82 -> 0.90 com
  MAE inalterado; melhora a cobertura condicional (P2: piso q05 furado).
- Median shift global em log1p transfere MAL entre temporadas (amplitude
  10x entre folds): degrada MAE/bias nos folds extremos => DESABILITADO
  por padrão.
- Em h4-h8 o pool cruza regimes contraditórios (fold2 bandas largas demais
  vs fold3 apertadas demais) => deltas ficam DESLIGADOS (apply_horizons).
- Regime por MÊS do alvo mistura temporadas fracas e fortes; regime por
  INTENSIDADE observável na SEMANA-BASE (casos da base) é o que separa os
  erros sistematicamente (fold3 lows: q05 acima do real; fold0 highs:
  subestimação) e é DECIDÍVEL em produção (a base é a última semana
  observada).

Desenho final:
- ONLY deltas conformal por (horizonte h<=3, regime-base clusters
  high/mid/low pelos casos da semana-base) com pesos temporais exponenciais;
- pool produção = todos os folds walk-forward (2022-2025);
- validação honesta = scheme 'expanding' (fold f calibrado SÓ com folds
  anteriores; fold0 histórico fica sem calibração) — replica o que o
  artefato verá na implantação sequencial;
- non-crossing reaplicado após os deltas.

Formulação (CQR com pesos — Avati et al. 2018 / Romano et al. 2019):
    r_τ,i = log1p(y_i) − log1p(pred_τ,i)
    delta_τ = Quantil_w(τ; r_τ, w)  ⇒  q_τ_novo = q_τ + delta_τ
com w_i = 2^(−idade_i / meia-vida) (idade em anos desde a semana mais
recente do pool). Tudo em escala log1p; cap por |delta| (max_abs_delta).
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

BAND_TAUS = (0.05, 0.25, 0.75, 0.95)   # deltas conformal (τ=0.50 fica sem delta)
REGIMES = ('outbreak', 'calm')


def _se_to_date(se: str):
    from data.epiweeks import epiweek_to_date
    se = str(se).zfill(6)
    return epiweek_to_date(int(se[:4]), int(se[4:]))


def add_epiweeks(se: str, weeks: int) -> str:
    from data.epiweeks import date_to_epiweek
    return str(date_to_epiweek(_se_to_date(se) + timedelta(days=weeks * 7))).zfill(6)


def weighted_quantile(values: np.ndarray, q: float, weights: np.ndarray) -> float:
    """Quantil ponderado (ordenação estável; NaN e peso ≤0 removidos)."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.shape != w.shape:
        return float('nan')
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return float('nan')
    v, w = v[m], w[m]
    order = np.argsort(v, kind='mergesort')
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    total = cw[-1]
    if total <= 0:
        return float(v[-1])
    pos = np.searchsorted(cw, q * total - 1e-12, side='left')
    pos = min(pos, len(v) - 1)
    return float(v[pos])


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    return weighted_quantile(values, 0.5, weights)


def season_weights(se_list: Sequence[str], ref_se: str, half_life_years: float) -> np.ndarray:
    """Peso exponencial pela idade da semana em relação à mais recente (ref_se)."""
    ref_date = _se_to_date(ref_se)
    lam = np.log(2.0) / max(half_life_years, 0.25)
    ages = []
    for se in se_list:
        try:
            age_years = (ref_date - _se_to_date(se)).days / 365.25
        except Exception:
            age_years = 6.0          # fora da janela usual: peso mínimo
        ages.append(max(age_years, 0.0))
    return np.exp(-lam * np.asarray(ages, dtype=float))


def make_regime_hook(casos_by_se: Dict[str, float],
                     thresholds: Optional[Dict[str, float]] = None) -> Callable:
    """Hook de regime por INTENSIDADE OBSERVÁVEL na semana-base (casos).

    thresholds: {'high': 100.0, 'mid': 20.0} — clusters absolutos (invariantes
    à estação). base_se = semana em que a previsão é feita; casos observados
    até ela decidem o regime. Retorna callable (h, target_se) -> 'high'|'mid'|'low'
    (ou 'global' quando o valor da base é desconhecido).
    """
    th = dict(thresholds or {})
    high = float(th.get('high', 100.0))
    mid = float(th.get('mid', 20.0))

    def hook(h: int, target_se: str) -> str:
        try:
            base = add_epiweeks(str(target_se), -int(h))
        except Exception:
            return 'global'
        c = casos_by_se.get(str(base).zfill(6), None)
        try:
            c = float(c)
        except Exception:
            return 'global'
        if not np.isfinite(c):
            return 'global'
        if c >= high:
            return 'high'
        if c >= mid:
            return 'mid'
        return 'low'
    return hook


def compute_conformal(fold_series: Dict, regime_hook: Callable,
                      exclude_folds: Optional[set] = None,
                      half_life_years: float = 2.5,
                      max_delta: float = 1.0,
                      use_median_shift: bool = False,
                      max_shift: float = 0.8,
                      min_cluster_weeks: int = 12) -> Dict:
    """CQR (delta por τ de banda, opcional median shift) por horizonte/regime.

    fold_series: {h: {tau: [{'fold', 'test_se', 'y_test_casos', 'y_pred_casos'}]}}
    regime_hook: callable(h, target_se) -> chave de regime (decidível no base time).
    exclude_folds: remove folds do pool (LOO); para 'expanding' o chamador passa
      o fold_series apenas com os folds anteriores.
    Retorna {str(h): {regime_key: {'deltas', 'median_shift', 'n_cal', 'w_sum'}}}
    + meta {'ref_se', 'half_life_years', 'excluded_folds', 'n_folds_calibration'}.
    """
    if not fold_series:
        return {}
    fs_norm: Dict[int, Dict[float, list]] = {}
    for h, td in fold_series.items():
        blocks_norm = {float(t): blocks for t, blocks in td.items()
                       if str(t).replace('.', '', 1).replace('-', '', 1).isdigit()}
        if blocks_norm:
            fs_norm[int(h)] = blocks_norm
    if not fs_norm:
        return {}
    ref_se = max(str(s).zfill(6) for td in fs_norm.values()
                 for b in td.get(0.5, []) for s in b.get('test_se', []))
    out: Dict = {'ref_se': ref_se, 'half_life_years': half_life_years,
                 'excluded_folds': sorted(int(f) for f in (exclude_folds or set())),
                 'n_folds_calibration': None}

    for h, tau_dict in fs_norm.items():
        n_folds_cal = len(tau_dict.get(0.5, []))
        fold_list = [i for i in range(n_folds_cal)
                     if not (exclude_folds and (i in exclude_folds
                             or int(tau_dict[0.5][i].get('fold', i)) in exclude_folds))]
        if out.get('n_folds_calibration') is None:
            out['n_folds_calibration'] = n_folds_cal
        if not fold_list:
            continue
        rows_se: List[str] = []
        rows_y: List[float] = []
        rows_pred: Dict[float, List[float]] = {t: [] for t in BAND_TAUS}
        for f_idx in fold_list:
            b50 = tau_dict[0.5][f_idx]
            f_ses = [str(s).zfill(6) for s in b50['test_se']]
            rows_se += f_ses
            rows_y.append(np.asarray([float(v) for v in b50['y_test_casos']], float))
            for tau in BAND_TAUS:
                blocks = tau_dict.get(tau)
                if blocks is None:
                    rows_pred[tau] += [float('nan')] * len(f_ses)
                    continue
                b = blocks[f_idx]
                m = {str(s).zfill(6): float(p) for s, p in zip(b['test_se'], b['y_pred_casos'])}
                rows_pred[tau] += [m.get(s, float('nan')) for s in f_ses]
        if len(rows_se) < 5:
            continue
        y_log = np.log1p(np.clip(np.concatenate(rows_y), 0.0, None))
        w = season_weights(rows_se, ref_se, half_life_years)
        keys = np.array([str(regime_hook(int(h), s) or 'global') for s in rows_se])

        pL = {t: np.log1p(np.clip(np.asarray(rows_pred[t], float), 0.0, None))
              for t in BAND_TAUS}
        h_out: Dict[str, Dict] = {}
        groups = list(dict.fromkeys(list(keys))) + ['global']
        for g in groups:
            sel = np.ones(len(keys), dtype=bool) if g == 'global' else (keys == g)
            n_sel = int(np.sum(sel & np.isfinite(y_log)))
            if g != 'global' and n_sel < min_cluster_weeks:
                continue  # recorte pequeno: fica fora (chamador usa fallback global)
            shift = 0.0
            if use_median_shift:
                p50 = np.log1p(np.clip(np.asarray(rows_pred.get(0.5, [np.nan] * len(rows_se)), float), 0.0, None)) \
                    if 0.5 in rows_pred else np.zeros(len(rows_se))
                if 0.5 not in rows_pred:
                    p50 = None
                # median shift (desabilitado por padrão — ver docstring)
                if p50 is not None:
                    r50 = y_log - p50
                    _sh = weighted_median(r50[sel], w[sel])
                    shift = float(np.clip(_sh, -max_shift, max_shift)) if np.isfinite(_sh) else 0.0
            entry = {'median_shift': shift if use_median_shift else 0.0,
                     'deltas': {}, 'n_cal': n_sel,
                     'w_sum': float(np.sum(w[sel]))}
            for tau in BAND_TAUS:
                r_tau = y_log - (pL[tau] + shift)
                d = weighted_quantile(r_tau[sel], tau, w[sel])
                entry['deltas'][f'{tau:.2f}'] = float(np.clip(d, -max_delta, max_delta)
                                                     if np.isfinite(d) else d)
            h_out[g] = entry
        if h_out:
            out[str(int(h))] = h_out
    return out


def entry_for_key(conformal: Dict, horizon: int, regime_key: str) -> Dict:
    """Entrada do horizonte/regime pedido com fallback 'global' -> {}."""
    h_entry = conformal.get(str(int(horizon)))
    if not h_entry:
        return {}
    key = str(regime_key or 'global')
    return h_entry.get(key) or h_entry.get('global', {})


def apply_deltas_log(pred_log: np.ndarray, entry: Dict, tau: float) -> np.ndarray:
    """Aplica delta do tau (e median_shift se existir) em escala log1p."""
    out = np.asarray(pred_log, dtype=float).copy()
    shift = entry.get('median_shift', 0.0)
    try:
        shift = float(shift)
    except Exception:
        shift = 0.0
    if np.isfinite(shift):
        out = out + shift
    if abs(float(tau) - 0.5) > 1e-9:
        delta = (entry.get('deltas') or {}).get(f'{float(tau):.2f}')
        try:
            delta = float(delta)
        except Exception:
            delta = 0.0
        if np.isfinite(delta):
            out = out + delta
    return out


def save_json(path: Path, production: Dict, expanding_by_fold: Optional[Dict] = None) -> None:
    payload = {
        'generated_at': datetime.utcnow().isoformat(),
        'version': 3,
        'description': ('CQR delta-only por horizonte (h1-h3) x regime-base (intensidade '
                        'da semana-base), calibrado nos folds walk-forward 2022-2025 com '
                        'pesos temporais; median shift desabilitado (instável entre '
                        'temporadas); validação expanding em expanding_by_fold'),
        'production': production,
        'expanding_by_fold': expanding_by_fold or {},
    }
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                          encoding='utf-8')
    logger.info(f"Saved conformal calibration: {Path(path)}")


def load_json(path: Path) -> Dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
        return data.get('production', {}) if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning(f"conformal load failed: {e}")
        return {}

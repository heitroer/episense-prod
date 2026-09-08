"""
Shared live utilities for API and Dashboard - single source of truth.
Evita duplicação de _merge_and_prepare / nowcast / anchor guard (RD-ALTO-02).
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Janela nowcasting InfoDengue: casos_est substitui casos onde > casos apenas nas últimas N SEs
NOWCAST_WINDOW = 16


def apply_nowcast(merged: pd.DataFrame, window: int = NOWCAST_WINDOW) -> pd.DataFrame:
    """Aplica nowcast InfoDengue nas últimas `window` SEs onde casos_est > casos."""
    if 'casos_est' not in merged.columns:
        return merged
    merged = merged.sort_values('SE').reset_index(drop=True)
    merged['casos'] = pd.to_numeric(merged['casos'], errors='coerce').fillna(0).astype(float)
    merged['casos_est'] = pd.to_numeric(merged['casos_est'], errors='coerce').fillna(0).astype(float)
    try:
        sorted_ses = sorted(merged['SE'].astype(str).str.zfill(6).unique())
        threshold_se = sorted_ses[-window] if len(sorted_ses) >= window else sorted_ses[0]
        mask = (merged['casos_est'] > merged['casos']) & (merged['SE'].astype(str).str.zfill(6) >= threshold_se)
    except Exception as e:
        logger.warning(f"Nowcast threshold failed {e}, fallback sem janela")
        mask = (merged['casos_est'] > merged['casos'])
    if mask.any():
        merged.loc[mask, 'casos'] = np.round(merged.loc[mask, 'casos_est']).astype(int)
        logger.info(f"Nowcast substitution: {int(mask.sum())} weeks updated (janela {window} semanas, thr {threshold_se if 'threshold_se' in locals() else 'fallback'})")
    return merged


def anchor_guard_filter(merged: pd.DataFrame, clima_dias: dict) -> pd.DataFrame:
    """Remove trailing SEs com clima parcial (<7 dias) - paridade treino/live."""
    if not clima_dias or len(clima_dias) == 0:
        return merged
    while len(merged) > 1:
        last_se = str(merged.iloc[-1]['SE']).zfill(6)
        days = clima_dias.get(last_se, pd.NA)
        if pd.isna(days) or int(days) < 7:
            motivo = 'clima ausente' if pd.isna(days) else f'clima parcial ({int(days)}/7 dias)'
            logger.warning(f"Anchor guard (shared): SE {last_se} removida ({motivo}) - ancora recua")
            merged = merged.iloc[:-1]
        else:
            break
    return merged

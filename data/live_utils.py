"""
Shared live utilities for API and Dashboard - single source of truth.
Evita duplicação de _merge_and_prepare / nowcast / anchor guard (RD-ALTO-02).
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def apply_nowcast(frame: pd.DataFrame) -> pd.DataFrame:
    """Usa a estimativa publicada de cada SE, com fallback à notificação."""
    frame = frame.copy()
    reported = pd.to_numeric(frame['casos'], errors='coerce')
    frame['casos_notificados'] = reported
    estimated = pd.to_numeric(frame.get('casos_est', reported), errors='coerce')
    frame['casos'] = estimated.where(np.isfinite(estimated) & estimated.ge(0), reported)
    return frame


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

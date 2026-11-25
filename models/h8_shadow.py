"""Variante prospectiva H8 que preserva o q95 antes do limitador superior."""
from copy import copy

from models.forecast_audit import forecast_rows

VARIANT = "h8_without_q95_cap_v1"


def shadow_forecast(engine, frame, origin):
    shadow = copy(engine)
    shadow.phantom_gating_disabled_horizons = set(engine.phantom_gating_disabled_horizons) | {8}
    predictions = shadow.predict(frame)
    row, = forecast_rows(origin, {8: predictions[8]})
    if any(row[key] is None for key in ("q05", "q25", "q50", "q75", "q95")):
        raise ValueError("Quantis H8 incompletos na comparação prospectiva")
    return {**row, "variant": VARIANT}

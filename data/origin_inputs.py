"""Historical inputs frozen at issuance, separate from consolidated outcomes."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.epiweeks import add_epiweeks, date_to_epiweek
from models.forecast_audit import load_inputs


def issued_inputs(config, origins):
    directory = Path(config["paths"]["data_processed"])
    archive = directory / "forecast_archive.json"
    entries = json.loads(archive.read_text())
    seen = set()
    for entry in entries:
        origin = entry["origin_se"]
        if origin not in origins or entry.get("source") != "issued" or not entry.get("issuance_id"):
            continue
        if origin in seen:
            raise ValueError(f"Emissão oficial duplicada para {origin}")
        seen.add(origin)
        audit = entry["audit"]
        digest = audit["input_sha256"]
        issued_week = date_to_epiweek(pd.Timestamp(entry["generated_at"]).tz_convert("America/Campo_Grande").date())
        frame = load_inputs(directory / "forecast_inputs", digest)
        weeks = frame["SE"].tolist()
        if (not weeks or weeks[-1] != origin or issued_week < origin
                or any(add_epiweeks(a, 1) != b for a, b in zip(weeks, weeks[1:]))):
            raise ValueError(f"Calendário do snapshot inválido para {origin}")
        observed = frame.get("casos_observados", frame["casos"]).dropna()
        if "is_imputed" in frame:
            observed = observed.loc[~frame.loc[observed.index, "is_imputed"]]
        if (observed.empty or not np.isfinite(observed.iloc[-1])
                or observed.iloc[-1] != audit["persistence"]
                or frame.loc[observed.index[-1], "SE"] != audit["persistence_week"]):
            raise ValueError(f"Persistência incompatível com o snapshot de {origin}")
        yield origin, issued_week, digest, frame, observed
